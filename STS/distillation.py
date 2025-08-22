def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device):
    log_rank("Start Fine-tuning")
    start_time = time.time()
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    criterion = build_criterion(args)

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
        "pearson": [],
        "spearman": [],
        "micro_step_time": [],
        "step_time": []
    }

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        log_rank(f"Start iterations of epoch {epoch + 1}")
        model.train()
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
                loss_denom=1,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue
            model.backward(loss)
            model.step()
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            num_samples = input_batch["input_ids"].size(0)
            total_samples += num_samples
            total_time += elapsed_time
            step += 1
            logging_output["global_step"] += 1
            logging_output["micro_step_time"].append(elapsed_time)
            logging_output["step_time"].append(elapsed_time)
            if dist.get_rank() == 0:
                data_iter.set_postfix(loss=loss.item())
        # Log per-epoch mean attention weights of student fused layers (I_S)
        if hasattr(model.module, 'epoch_student_attn_weights') and getattr(model.module, 'fkd_info', None) is not None and dist.get_rank() == 0:
            save_path = os.path.join(args.save_dir, "fkd_layer_weights.json")
            # Load or init data
            try:
                with open(save_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}

            # mapped_student_layers: lấy từ file nếu có, hoặc map lại từ teacher_top_k
            mapped_student = data.get('mapped_student_layers', None)
            if mapped_student is None:
                fkd_info = getattr(model.module, 'fkd_info', None)
                if fkd_info is not None:
                    top_idx = fkd_info.get('top_indices', [])
                    bi_scores = fkd_info.get('bi_scores', [])
                    t_layers = len(bi_scores)
                    try:
                        s_layers = getattr(model.module.student_model.config, 'num_hidden_layers', None)
                    except Exception:
                        s_layers = None
                    if s_layers is None:
                        try:
                            s_layers = len(model.module.student_model.base_model.encoder.layer)
                        except Exception:
                            s_layers = None
                    mapped_student = []
                    for l in top_idx:
                        if s_layers is not None:
                            s_l = max(0, min(s_layers - 1, int(((int(l) + 1) * s_layers - 1) // max(1, t_layers))))
                        else:
                            s_l = None
                        mapped_student.append(s_l)
                    data['mapped_student_layers'] = mapped_student
                else:
                    mapped_student = []

            # Gộp attention weights từ distiller.epoch_student_attn_weights
            attn_accum = {str(s_l): [] for s_l in mapped_student}
            for batch_dict in getattr(model.module, 'epoch_student_attn_weights', []):
                for s_l, attn_list in batch_dict.items():
                    if str(s_l) in attn_accum:
                        attn_accum[str(s_l)].extend(attn_list)

            # Tính mean attention weight cho từng student layer
            epoch_attn_means = {}
            for s_l in mapped_student:
                vals = attn_accum[str(s_l)]
                if len(vals) > 0:
                    epoch_attn_means[str(s_l)] = float(sum(vals) / len(vals))
                else:
                    epoch_attn_means[str(s_l)] = None

            if 'per_epoch_student_attn_weights' not in data:
                data['per_epoch_student_attn_weights'] = {}
            data['per_epoch_student_attn_weights'][str(epoch + 1)] = epoch_attn_means
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)
            # Reset attention weights log cho epoch tiếp theo
            model.module.epoch_student_attn_weights = []

        # Validation and checkpointing
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            log_rank("Evaluating before saving model...")
            if "dev" in dataset:
                eval_loss, mse, pearson, spearman = evaluate(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
            elif "test" in dataset:
                eval_loss, mse, pearson, spearman = evaluate(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
            else:
                eval_loss, mse, pearson, spearman = None, None, None, None
            ckpt_name = f"epoch{epoch + 1}_step{logging_output['global_step']}_loss{eval_loss:.4f}_pearson{pearson:.4f}"
            save_dir_path = os.path.join(args.save_dir, ckpt_name)
            if dist.get_rank() == 0:
                os.makedirs(save_dir_path, exist_ok=True)
                if not args.only_save_projector:
                    log_rank("Saving tokenizer...")
                    tokenizer.save_pretrained(save_dir_path)
                    log_rank("Saving model...")
                    model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    log_rank("Saving config")
                    model.module.student_model.config.save_pretrained(save_dir_path)
                if hasattr(model.module, "projectors"):
                    log_rank("Saving projector...")
                    torch.save(
                        model.module.projectors.state_dict(),
                        os.path.join(save_dir_path, "projector.pt")
                    )
                model_list.append({"path": save_dir_path, "score": pearson})
                model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)
                if len(model_list) > args.keep_best_n_checkpoints:
                    removed_model = model_list.pop(-1)
                    shutil.rmtree(removed_model["path"])
                log_rank(f"Model has been saved to {save_dir_path}")
            dist.barrier()
    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600),
        int(total_seconds % 3600 // 60),
        int(total_seconds % 60)
    ))
import time
import os
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
from STS.arguments import get_args
from STS.distiller import Distiller
from STS.data_utils.distill_datasets import DistillDataset
from STS.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from STS.criterions import build_criterion
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

    # ------------------------------------------------------------------
    # FKD PRE-PASS: compute Block Influence (BI) on calibration set using teacher
    # BI_l = 1 - mean(cos(X_l, Y_l)), where X_l is hidden_states[l] and Y_l is hidden_states[l+1]

    if getattr(args, 'criterion', None) in ['fkd', 'fkd_a']:
        distiller = model.module  # unwrap Distiller
        if dist.get_rank() == 0:
            print("[FKD] Starting BI pre-pass on calibration set...")

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

        dist.all_reduce(bi_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_counts, op=dist.ReduceOp.SUM)
        bi_scores = (bi_sums / token_counts.clamp(min=1.0)).to(torch.float32)
        # top-k select
        k = min(getattr(args, 'fkd_k'), bi_scores.numel())
        sorted_vals, sorted_idx = torch.sort(bi_scores, descending=True)
        top_idx = sorted_idx[:k].tolist()
        # store to distiller
        distiller.fkd_info = {
            "bi_scores": bi_scores.detach().cpu().tolist(),
            "top_indices": top_idx,
        }
        if dist.get_rank() == 0:
            printable = [(int(i), float(bi_scores[i].item())) for i in sorted_idx.tolist()]
            print("[FKD] BI scores (desc):", printable)
            print(f"[FKD] Selected top-{k} layers:", top_idx)

            s_layers = None
            if hasattr(distiller.student_model, 'config'):
                s_layers = getattr(distiller.student_model.config, 'num_hidden_layers', None)
            # fallback to common attribute
            if s_layers is None:
                s_layers = len(distiller.student_model.base_model.encoder.layer)

            # Deterministic mapping: s_l = floor(((l+1)*S - 1) / T)
            mapped_student = []
            t_layers = int(bi_scores.numel())
            T = max(1, t_layers)
            S = int(s_layers) if s_layers is not None else None
            for l in top_idx:
                if S is None:
                    mapped_student.append(None)
                    continue
                s_l = int(((int(l) + 1) * S - 1) // T)
                s_l = max(0, min(S - 1, s_l))
                mapped_student.append(s_l)

            save_path = os.path.join(args.save_dir, "fkd_layer_weights.json")
            # compute softmax weights for selected top-k teacher layers
            sel_scores = bi_scores[sorted_idx[:k]]
            topk_weights = torch.softmax(sel_scores, dim=0).detach().cpu().tolist()

            out = {
                "teacher_bi_scores": bi_scores.detach().cpu().tolist(),
                "teacher_top_k": top_idx,
                "teacher_top_k_weights": topk_weights,
                "mapped_student_layers": mapped_student,
                "per_epoch_student_attention_weights": {}
            }
            with open(save_path, 'w') as f:
                json.dump(out, f, indent=2)

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
                data_iter.set_postfix(loss=loss.item())


        # Log per-epoch attention weights from FKD_A student fusion
        if getattr(distiller, 'fkd_info', None) is not None and dist.get_rank() == 0:
            save_path = os.path.join(args.save_dir, "fkd_layer_weights.json")
            try:
                with open(save_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}

            # Get attention weights from the criterion forward pass
            student_attn_weights = getattr(distiller, 'current_student_attn_weights', {})
            
            # Store attention weights for this epoch
            if 'per_epoch_student_attention_weights' not in data:
                data['per_epoch_student_attention_weights'] = {}
            
            # Convert student layer indices to string keys and store their attention weights
            epoch_attn_weights = {}
            for s_layer_idx, attn_weight in student_attn_weights.items():
                epoch_attn_weights[str(s_layer_idx)] = attn_weight
            
            data['per_epoch_student_attention_weights'][str(epoch + 1)] = epoch_attn_weights
            
            try:
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

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

    from scipy.stats import pearsonr, spearmanr
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    sample_num = 0
    import torch.nn.functional as F
    for input_batch, output_batch in tqdm(dataloader, desc="Processing batches"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None),
            output_hidden_states=False,
            return_dict=True,
        )
        # For regression, logits shape: [B, 1] or [B]
        logits = outputs.logits.squeeze(-1)
        loss = F.mse_loss(logits, labels)
        all_preds.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())
        total_loss += loss.item() * labels.size(0)
        sample_num += labels.size(0)

    all_preds = torch.cat(all_preds, dim=0).to(torch.float32)
    all_targets = torch.cat(all_targets, dim=0).to(torch.float32)
    all_preds_np = all_preds.numpy().flatten()
    all_targets_np = all_targets.numpy().flatten()

    # Compute metrics
    mse = ((all_preds_np - all_targets_np) ** 2).mean()
    try:
        pearson = pearsonr(all_preds_np, all_targets_np)[0]
    except Exception:
        pearson = float('nan')
    try:
        spearman = spearmanr(all_preds_np, all_targets_np)[0]
    except Exception:
        spearman = float('nan')

    avg_loss = total_loss / sample_num if sample_num > 0 else float('nan')
    print(f"Evaluated: {split} | loss={avg_loss:.4f} | mse={mse:.4f} | pearson={pearson:.4f} | spearman={spearman:.4f}")
    student_model.train()
    return avg_loss, mse, pearson, spearman

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
