import time
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
from tqdm import tqdm as _tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import shutil
import json
from tqdm import tqdm
import math
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModel,
)
from transformers.integrations import HfDeepSpeedConfig
from Classification.arguments import get_args
from Classification.distiller import Distiller
from Classification.data_utils.distill_datasets import DistillDataset
from Classification.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from Classification.criterions import build_criterion
import json
import os
import time
import torch


def prepare_dataset(args, distiller):
    data = {}
    if args.do_train:
        data["train"] = DistillDataset(
            args, "train", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = DistillDataset(
            args, "dev", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test.csv")):
            data["test"] = DistillDataset(
                args, "test", distiller.student_tokenizer,
                distiller.teacher_tokenizers
            )
            log_rank("Num of test data: {}".format(len(data["test"])))
    elif args.do_eval:
        data["test"] = DistillDataset(
            args, "test", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of test data: {}".format(len(data["test"])))
    else:
        raise ValueError("Do train and do eval must set one")
    return data


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device):
    log_rank("Start Fine-tuning")
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)
        # unwrap distiller (DeepSpeed wraps the Distiller module as model.module)
        try:
            distiller = model.module
        except Exception:
            distiller = None

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_loader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )

    # ------------------------------------------------------------------
    # FKD PRE-PASS: compute Block Influence (BI) on calibration set using teacher
    # BI_l = 1 - mean(cos(X_l, Y_l)), where X_l is hidden_states[l] and Y_l is hidden_states[l+1]

    if getattr(args, 'criterion', None) in ['fkd_a', 'fkd_h']:
        distiller = model.module  # unwrap Distiller
        if dist.get_rank() == 0:
            print("[FKD] Starting BI pre-pass on calibration set...")
            # For FKD_H, print an Offline->Online transition banner and vocab overlap stats
            if getattr(args, 'criterion', None) == 'fkd_h':
                try:
                    t_vocab = set(distiller.teacher_tokenizers.get_vocab().keys()) if hasattr(distiller, 'teacher_tokenizers') else set()
                    s_vocab = set(distiller.student_tokenizer.get_vocab().keys()) if hasattr(distiller, 'student_tokenizer') else set()
                    inter = t_vocab.intersection(s_vocab)
                    inter_n = len(inter)
                    t_n, s_n = len(t_vocab), len(s_vocab)
                    pct_t = (inter_n / t_n * 100.0) if t_n else 0.0
                    pct_s = (inter_n / s_n * 100.0) if s_n else 0.0
                    print(f"[FKD_H][Offline] Vocab overlap: |T∩S|={inter_n} |T|={t_n} |S|={s_n} => {pct_t:.2f}% of T, {pct_s:.2f}% of S")
                except Exception as e:
                    print(f"[FKD_H][Offline][WARN] Failed to compute vocab overlap: {e}")
                # Announce global alignment and projection usage
                try:
                    ga_path = getattr(args, 'global_alignment_path', None)
                    proj_path = getattr(args, 'offline_projection_path', None)
                    msg = "[FKD_H][Offline→Online] Using: "
                    parts = []
                    parts.append(f"global_alignment={ga_path if ga_path else 'None'}")
                    parts.append(f"offline_projection={proj_path if proj_path else 'None'}")
                    print(msg + ", ".join(parts))
                except Exception:
                    pass

        pre_sampler = DistributedSampler(
            dataset['train'], shuffle=False, drop_last=False,
            rank=dp_rank, num_replicas=dp_world_size
        )
        pre_loader = DataLoader(
            dataset['train'], sampler=pre_sampler,
            batch_size=args.eval_batch_size or args.batch_size, num_workers=args.num_workers,
            collate_fn=dataset['train'].collate
        )
        teacher = distiller.teacher_model
        teacher.eval()

        bi_sums = None
        token_counts = None
        max_batches = getattr(args, 'fkd_calib_max_batches', 0)
        it = 0
        iterator = pre_loader
        if dist.get_rank() == 0:
            iterator = _tqdm(pre_loader, desc="FKD BI pre-pass", dynamic_ncols=True)
        with torch.no_grad():
            for input_batch, output_batch in iterator:
                dataset['train'].move_to_device([input_batch, output_batch], device)
                t_out = teacher(
                    input_batch["teacher_input_ids"],
                    attention_mask=input_batch.get("teacher_attention_mask", None),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs = t_out.hidden_states  # tuple len L+1
                L = len(hs) - 1
                if bi_sums is None:
                    bi_sums = torch.zeros(L, device=device, dtype=torch.float64)
                    token_counts = torch.zeros(L, device=device, dtype=torch.float64)
                mask = input_batch.get("teacher_attention_mask", None)
                if mask is not None:
                    mask = mask.to(device=device, dtype=torch.float32)
                for l in range(L):
                    X = hs[l]      # (B, T, H)
                    Y = hs[l+1]    # (B, T, H)
                    cos = F.cosine_similarity(X, Y, dim=-1)  # (B, T)
                    bi = 1.0 - cos
                    if mask is not None:
                        bi = bi * mask
                        count = mask.sum()
                    else:
                        count = torch.tensor(bi.numel(), device=device, dtype=torch.float32)
                    bi_sums[l] += bi.sum(dtype=torch.float64)
                    token_counts[l] += count.to(torch.float64)
                it += 1
                if max_batches and it >= max_batches:
                    break
        # reduce across ranks
        dist.all_reduce(bi_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_counts, op=dist.ReduceOp.SUM)
        bi_scores = (bi_sums / token_counts.clamp(min=1.0)).to(torch.float32)
        # top-k select
        k = min(getattr(args, 'fkd_k', 4), bi_scores.numel())
        sorted_vals, sorted_idx = torch.sort(bi_scores, descending=True)
        top_idx = sorted_idx[:k].tolist()
        # store to distiller
        # compute softmax weights for selected top-k teacher layers
        sel_scores = bi_scores[sorted_idx[:k]] if k > 0 else torch.tensor([], device=device)
        try:
            topk_weights = torch.softmax(sel_scores, dim=0).detach().cpu().tolist()
        except Exception:
            topk_weights = [1.0 / max(1, k)] * len(top_idx)
        # determine number of student layers if possible to map by relative depth
        s_layers = None
        if hasattr(distiller.student_model, 'config'):
            s_layers = getattr(distiller.student_model.config, 'num_hidden_layers', None)
        if s_layers is None:
            try:
                s_layers = len(distiller.student_model.base_model.encoder.layer)
            except Exception:
                s_layers = None
        mapped_student = []
        t_layers = bi_scores.numel()
        for l in top_idx:
            ratio = (l + 1) / float(t_layers) if t_layers else 1.0
            if s_layers is not None and s_layers > 0:
                s_l = max(0, min(s_layers - 1, int(round(ratio * s_layers)) - 1))
            else:
                s_l = None
            mapped_student.append(s_l)
        distiller.fkd_info = {
            "bi_scores": bi_scores.detach().cpu().tolist(),
            "top_indices": top_idx,
            "teacher_top_k_weights": topk_weights,
            "mapped_student_layers": mapped_student,
        }
        if dist.get_rank() == 0:
            printable = [(int(i), float(bi_scores[i].item())) for i in sorted_idx.tolist()]
            print("[FKD] BI scores (desc):", printable)
            print(f"[FKD] Selected top-{k} layers:", top_idx)
            # Persist BI results and mapped student layers to a JSON file for visualization
            try:
                save_path = os.path.join(args.save_dir, "fkd_layer_weights.json")
                out = {
                    "teacher_bi_scores": bi_scores.detach().cpu().tolist(),
                    "teacher_top_k": top_idx,
                    "teacher_top_k_weights": topk_weights,
                    "mapped_student_layers": mapped_student,
                    "per_epoch_student_means": {}
                }
                with open(save_path, 'w') as f:
                    json.dump(out, f, indent=2)
            except Exception:
                pass
        # recreate training sampler with shuffle
        sampler = DistributedSampler(
            dataset["train"], 
            shuffle=True, 
            drop_last=True, 
            rank=dp_rank, 
            num_replicas=dp_world_size
        )
        train_loader = DataLoader(
            dataset['train'], 
            sampler=sampler, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            collate_fn=dataset["train"].collate
        )
    
    step = 0
    model_list = []
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [], 
        "nll_loss": [],
        "kd_loss": [],
        "accuracy": [],
        "micro_step_time": [],
        "step_time": []
    }
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        print("Training mode?", model.student_model.training)  # True

        epoch_start_time = time.time()
        step = 0
        total_samples = 0
        total_time = 0.0

        data_iter = train_loader
        if dist.get_rank() == 0:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for batch in data_iter:
            st_time = time.time()
            input_batch, output_batch = batch
            dataset["train"].move_to_device([input_batch, output_batch], device)

            loss, logging_output = model(
                criterion,
                {"input_batch": input_batch, "output_batch": output_batch},
                logging_output,
                loss_denom=1, #deepspeed support sync gradient, no need to calculate loss_denom
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue

            # Add losses to logging lists
            logging_output.get('loss', None)
            if "ce_loss" in logging_output:
                logging_output.get('ce_loss', None)
            if "distill_loss" in logging_output:
                logging_output.get('distill_loss', None)

            
            model.backward(loss)
            model.step()
            torch.cuda.synchronize()  # correctlyc compute time

            elapsed_time = time.time() - st_time
            num_samples = input_batch["input_ids"].size(0)
            total_samples += num_samples
            total_time += elapsed_time
            step += 1
            
            logging_output["global_step"] += 1
            logging_output["micro_step_time"].append(elapsed_time)
            logging_output["step_time"].append(elapsed_time)

            if dist.get_rank() == 0:
                ce = logging_output.get("ce_loss", None)
                kd = logging_output.get("distill_loss", None)
                if ce is not None and kd is not None:
                    data_iter.set_postfix(loss=loss.item(), ce=round(float(ce), 4), kd=round(float(kd), 4))
                    # Also print detailed loss breakdown every 10 steps
                    if step % 10 == 0:
                        print(f"[Step {step}] Total: {loss.item():.4f}, CE: {float(ce):.4f}, KD: {float(kd):.4f}")
                else:
                    data_iter.set_postfix(loss=loss.item())
                    if step % 10 == 0:
                        print(f"[Step {step}] Total: {loss.item():.4f} (no component breakdown available)")


    # Log per-epoch mean of student attention weights for FKD_A/FKD_H
        if distiller is not None and getattr(distiller, 'epoch_student_attn_weights', None) and dist.get_rank() == 0:
            save_path = os.path.join(args.save_dir, "fkd_layer_weights.json")
            try:
                with open(save_path, 'r') as f:
                    log_data = json.load(f)
            except (IOError, json.JSONDecodeError):
                log_data = {}

            # Aggregate weights from all batches in the epoch
            epoch_weights = distiller.epoch_student_attn_weights
            if epoch_weights:
                aggregated_weights = {}
                for batch_dict in epoch_weights:
                    for s_layer, weights in batch_dict.items():
                        if s_layer not in aggregated_weights:
                            aggregated_weights[s_layer] = []
                        # weights can be a list (from FKD_A) or a single float (FKD_DT)
                        if isinstance(weights, list):
                            aggregated_weights[s_layer].extend(weights)
                        else:
                            aggregated_weights[s_layer].append(weights)
                
                # Calculate the mean for each layer
                mean_layer_weights = {
                    layer: np.mean(vals) for layer, vals in aggregated_weights.items()
                }

                if 'per_epoch_student_attn_weights' not in log_data:
                    log_data['per_epoch_student_attn_weights'] = {}
                
                log_data['per_epoch_student_attn_weights'][str(epoch + 1)] = mean_layer_weights
                
                try:
                    with open(save_path, 'w') as f:
                        json.dump(log_data, f, indent=2)
                except IOError as e:
                    log_rank(f"Error writing to fkd_layer_weights.json: {e}")
            
            # Reset for the next epoch
            distiller.epoch_student_attn_weights = []

        if args.save_dir and (epoch + 1) % args.save_interval == 0: #save_interval = 1 then save each epoch
            #eval_interval = 1 then evaluate each epoch
            log_rank("Evaluating before saving model...")
            if "test" in dataset: #evaluate for test, no affect
                eval_loss, eval_accu, eval_precision, eval_recall = evaluate(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
            else:
                eval_loss, eval_accu, eval_precision, eval_recall = evaluate(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
            
            ckpt_name = "epoch{}_step{}_loss{:.4f}".format(epoch + 1, logging_output["global_step"], eval_loss)
            save_dir_path = os.path.join(args.save_dir, ckpt_name)
            
            if dist.get_rank() == 0:
                os.makedirs(save_dir_path, exist_ok=True)
                if not args.only_save_projector:
                    log_rank("Saving tokenizer...")
                    tokenizer.save_pretrained(save_dir_path)
                    log_rank("Saving model...")
                    model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    classifier_path = os.path.join(save_dir_path, "classifier_head.bin")
                    if hasattr(model.module.student_model, 'score'):  # Mistral model
                        log_rank("Saving Mistral classifier head (score)...")
                        torch.save(model.module.student_model.score.state_dict(), classifier_path)
                    elif hasattr(model.module.student_model, 'classifier'):  # BERT model
                        log_rank("Saving BERT classifier head (classifier)...")
                        torch.save(model.module.student_model.classifier.state_dict(), classifier_path)
                    else:
                        log_rank("Warning: Could not identify classifier head structure, no classifier saved.")
                    log_rank("Saving config")
                    model.module.student_model.config.save_pretrained(save_dir_path)
                if hasattr(model.module, "projectors"):
                    log_rank("Saving projector...")
                    torch.save(
                        model.module.projectors.state_dict(), 
                        os.path.join(save_dir_path, "projector.pt")
                    )
                
                model_list.append({"path": save_dir_path, "score": eval_accu + eval_precision + eval_recall})
                model_list = sorted(model_list, key=lambda x: x["score"], reverse=False)
                
                if len(model_list) > args.keep_best_n_checkpoints:
                    removed_model = model_list.pop(0)
                    shutil.rmtree(removed_model["path"])

                log_rank(f"Model has been saved to {save_dir_path}")
            dist.barrier()
            
    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad()
def evaluate(args, tokenizer, student_model, dataset, split, device):
    if dist.get_rank() != 0:
        return None, None, None, None        

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    eval_info = {
        "loss": 0.0,
        "sample_num": 0,
        "correct_samples": 0
    }

    all_preds = []
    all_labels = []
    total_loss = 0
    for input_batch, output_batch in tqdm(dataloader, desc="Processing batches"):

        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]       
        outputs = student_model(
            input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            position_ids=input_batch.get("position_ids", None),
            labels = labels
        )
        logits = outputs.logits
        loss = outputs.loss

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        all_preds.append(preds)
        all_labels.append(labels)
        sample_num = labels.size(0)
        total_loss += loss

        eval_info["sample_num"] += sample_num
        eval_info["correct_samples"] += correct

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = total_loss / len(all_labels)


    eval_info["precision"] = round(precision, 6)
    eval_info["recall"] = round(recall, 6)

    eval_info["loss"] = avg_loss
    eval_info["accuracy"] = (all_preds==all_labels).sum().item() / len(all_preds)
    eval_info["sample_num"] = len(all_preds)
    eval_info["correct_samples"] = (all_preds==all_labels).sum().item()

    for key in eval_info:
        if isinstance(eval_info[key], float):
            eval_info[key] = round(eval_info[key], 6)
    
    print(f"Evaluated: {split} | {eval_info}")

    student_model.train()

    return eval_info["loss"], eval_info["accuracy"], eval_info.get("precision", 0.0), eval_info.get("recall", 0.0)

def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    # save arguments
    if dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    print('user ds_config', ds_config)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * dp_world_size

    log_rank("Initializing a distiller for knowledge distillation...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    # Always use full vocabulary sizes; log them for clarity
    if dist.get_rank() == 0:
        try:
            s_vs = getattr(distiller, 'student_vocab_size', None)
            t_vs = getattr(distiller, 'teacher_vocab_size', None)
            log_rank(f"[Vocab] Student vocab size: {s_vs}")
            log_rank(f"[Vocab] Teacher vocab size: {t_vs}")
        except Exception:
            pass
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size))
        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

        log_rank("Total_iters = {}".format(args.total_iters))
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    optimizer_grouped_parameters = get_optimizer(args, distiller.student_model)
    optimizer_grouped_parameters = distiller.add_optimizer_param_group(optimizer_grouped_parameters)

    lr_scheduler = get_learning_rate_scheduler(args, optimizer_grouped_parameters)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer_grouped_parameters,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
    if args.do_eval:
        if "test" in dataset and len(dataset["test"]) > 0:
            evaluate(args, distiller.student_tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
        else:
            log_rank("No test split found. Skipping evaluation phase.")
        
    
if __name__ == "__main__":
    main()
