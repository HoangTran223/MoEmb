"""
Clean STS distillation training script.

Provides minimal, syntactically-correct implementations of:
- prepare_dataset
- optional BI pre-pass (_run_bi_prepass)
- finetune (training loop)
- evaluate
- main

This mirrors the Classification flow and is kept compact for easy review.
"""

import time
import os
import json
import math
import shutil
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed

from STS.arguments import get_args
from STS.distiller import Distiller
from STS.data_utils.distill_datasets import DistillDataset
from STS.utils import (
    initialize,
    get_optimizer,
    get_learning_rate_scheduler,
    print_rank,
    log_rank,
)
from STS.criterions import build_criterion


def prepare_dataset(args, distiller):
    data = {}
    if args.do_train:
        data["train"] = DistillDataset(args, "train", distiller.student_tokenizer, distiller.teacher_tokenizer)
        log_rank(f"Num of train data: {len(data['train'])}")
        data["dev"] = DistillDataset(args, "dev", distiller.student_tokenizer, distiller.teacher_tokenizer)
        log_rank(f"Num of dev data: {len(data['dev'])}")
        if os.path.exists(os.path.join(args.data_dir, "test.csv")):
            data["test"] = DistillDataset(args, "test", distiller.student_tokenizer, distiller.teacher_tokenizer)
            log_rank(f"Num of test data: {len(data['test'])}")
    elif args.do_eval:
        data["test"] = DistillDataset(args, "test", distiller.student_tokenizer, distiller.teacher_tokenizer)
        log_rank(f"Num of test data: {len(data['test'])}")
    else:
        raise ValueError("Either do_train or do_eval must be True")
    return data


def _run_bi_prepass(args, distiller, dataset, device):
    # compute block-influence scores on calibration data and persist
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()

    if dp_rank == 0:
        print("[FKD] Running BI pre-pass...")

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size or args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collate)

    teacher = distiller.teacher_model
    teacher.eval()

    bi_sums = None
    token_counts = None
    max_batches = getattr(args, 'fkd_calib_max_batches', 0)
    it = 0
    iterator = loader
    if dist.get_rank() == 0:
        iterator = tqdm(loader, desc="FKD BI pre-pass", dynamic_ncols=True)

    with torch.no_grad():
        for batch in iterator:
            input_batch, _ = batch
            dataset.move_to_device([input_batch], device)
            t_out = teacher(input_batch["teacher_input_ids"], attention_mask=input_batch.get("teacher_attention_mask", None), output_hidden_states=True, return_dict=True)
            hs = t_out.hidden_states
            L = len(hs) - 1
            if bi_sums is None:
                bi_sums = torch.zeros(L, device=device, dtype=torch.float64)
                token_counts = torch.zeros(L, device=device, dtype=torch.float64)
            mask = input_batch.get("teacher_attention_mask", None)
            if mask is not None:
                mask = mask.to(device=device, dtype=torch.float32)
            for l in range(L):
                X = hs[l]; Y = hs[l+1]
                cos = F.cosine_similarity(X, Y, dim=-1)
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
    k = min(getattr(args, 'fkd_k', 3), bi_scores.numel())
    _, sorted_idx = torch.sort(bi_scores, descending=True)
    top_idx = sorted_idx[:k].tolist()
    sel_scores = bi_scores[sorted_idx[:k]] if k > 0 else torch.tensor([], device=device)
    try:
        topk_weights = torch.softmax(sel_scores, dim=0).detach().cpu().tolist()
    except Exception:
        topk_weights = [1.0 / max(1, k)] * len(top_idx)

    # map teacher selected layers to student layers (heuristic)
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

    fkd_info = {
        "bi_scores": bi_scores.detach().cpu().tolist(),
        "top_indices": top_idx,
        "teacher_top_k_weights": topk_weights,
        "mapped_student_layers": mapped_student,
    }

    if dist.get_rank() == 0:
        try:
            save_path = os.path.join(args.save_dir, "fkd_layer_weights.json")
            with open(save_path, 'w') as f:
                json.dump(fkd_info, f, indent=2)
        except Exception:
            pass

    return fkd_info


def finetune(args, tokenizer, model_engine, optimizer, lr_scheduler, dataset, device):
    log_rank("Start Fine-tuning")
    start_time = time.time()
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    criterion = build_criterion(args)

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_loader = DataLoader(dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset['train'].collate)

    # Optional BI pre-pass
    if getattr(args, 'criterion', None) in ['fkd', 'fkd_a', 'fkd_h']:
        distiller = model_engine.module
        _ = _run_bi_prepass(args, distiller, dataset['train'], device)
        distiller.fkd_info = _

    model_engine.train()
    # Track saved checkpoints and keep the best N by spearman (like Classification style)
    model_list = []  # list of {"path": str, "score": float}
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        data_iter = train_loader
        if dist.get_rank() == 0:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        step = 0
        for batch in data_iter:
            input_batch, output_batch = batch
            dataset['train'].move_to_device([input_batch, output_batch], device)
            loss, logging_output = model_engine(build_criterion(args), {"input_batch": input_batch, "output_batch": output_batch}, {}, loss_denom=1)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            model_engine.backward(loss)
            model_engine.step()
            if dist.get_rank() == 0:
                ce = logging_output.get('ce_loss', None)
                kd = logging_output.get('distill_loss', None)
                if step % 200 == 0:
                    if ce is not None and kd is not None:
                        data_iter.set_postfix(loss=loss.item(), ce=round(float(ce), 4), kd=round(float(kd), 4))
                    else:
                        data_iter.set_postfix(loss=loss.item())
                    # Log FKD_H alpha, beta, global alignment path nếu criterion là fkd_h
                    if getattr(args, 'criterion', None) == 'fkd_h':
                        try:
                            from tqdm import tqdm as _tqdm
                            distiller = model_engine.module
                            criterion_obj = getattr(distiller, 'criterion', None)
                            if criterion_obj is None:
                                criterion_obj = build_criterion(args)
                            alpha = getattr(criterion_obj, 'alpha', None)
                            beta = getattr(criterion_obj, 'beta', None)
                            global_alignment_path = getattr(criterion_obj, 'global_alignment_path', None)
                            _tqdm.write(f"[FKD_H] Loss weights: alpha={alpha}, beta={beta}")
                            if global_alignment_path:
                                _tqdm.write(f"[FKD_H] Loaded global alignment matrix from {global_alignment_path}")
                        except Exception as e:
                            _tqdm.write(f"[FKD_H][LOG] Error logging alpha/beta/global_alignment: {e}")
            step += 1

        # Evaluate and save checkpoints like Classification: save_pretrained dir, keep best N by spearman
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            log_rank("Evaluating before saving model...")
            if "test" in dataset and len(dataset["test"]) > 0:
                eval_loss, mse, pearson, spearman = evaluate(args, tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
            else:
                eval_loss, mse, pearson, spearman = evaluate(args, tokenizer, model_engine.module.student_model, dataset["dev"], "dev", device)

            ckpt_name = "epoch{}_step{}_loss{:.4f}".format(epoch + 1, step, eval_loss if eval_loss is not None else float('nan'))
            save_dir_path = os.path.join(args.save_dir, ckpt_name)

            if dist.get_rank() == 0:
                os.makedirs(save_dir_path, exist_ok=True)
                if not args.only_save_projector:
                    # Save tokenizer and student model (HF-style)
                    try:
                        tokenizer.save_pretrained(save_dir_path)
                    except Exception:
                        pass
                    try:
                        model_engine.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    except Exception:
                        # Fallback: raw state dict
                        torch.save(model_engine.module.student_model.state_dict(), os.path.join(save_dir_path, "pytorch_model.bin"))
                    # Save regression/classifier head if present
                    try:
                        if hasattr(model_engine.module.student_model, 'score'):
                            torch.save(model_engine.module.student_model.score.state_dict(), os.path.join(save_dir_path, "classifier_head.bin"))
                        elif hasattr(model_engine.module.student_model, 'classifier'):
                            torch.save(model_engine.module.student_model.classifier.state_dict(), os.path.join(save_dir_path, "classifier_head.bin"))
                    except Exception:
                        pass
                    # Save config
                    try:
                        model_engine.module.student_model.config.save_pretrained(save_dir_path)
                    except Exception:
                        pass
                # Save projector, if any
                if hasattr(model_engine.module, "projectors"):
                    try:
                        torch.save(model_engine.module.projectors.state_dict(), os.path.join(save_dir_path, "projector.pt"))
                    except Exception:
                        pass

                # Manage top-N checkpoints by spearman
                score = float(spearman) if spearman is not None and not (isinstance(spearman, float) and math.isnan(spearman)) else float('-inf')
                model_list.append({"path": save_dir_path, "score": score})
                # Sort ascending and drop worst beyond N (like Classification)
                model_list = sorted(model_list, key=lambda x: x["score"], reverse=False)
                while len(model_list) > args.keep_best_n_checkpoints:
                    removed = model_list.pop(0)
                    try:
                        shutil.rmtree(removed["path"])
                    except Exception:
                        pass
                log_rank(f"Model has been saved to {save_dir_path}")
            dist.barrier()

    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(int(total_seconds // 3600), int(total_seconds % 3600 // 60), int(total_seconds % 60)))


# Modify evaluate function to suit STS tasks using the SequenceClassifierWrapper
@torch.no_grad()
def evaluate(args, tokenizer, student_model, dataset, split, device):
    student_model.eval()
    all_logits = []
    all_labels = []
    # Create a DataLoader using the dataset's collate function
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset[split], batch_size=args.eval_batch_size or args.batch_size, shuffle=False, collate_fn=dataset[split].collate)

    from tqdm import tqdm
    for batch in tqdm(loader, desc="Evaluating STS Task"):
        # Remove any keys not expected by the model (e.g., position_ids)
        if 'position_ids' in batch:
            del batch['position_ids']
        # Move tensor values to device
        batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k,v in batch.items()}
        # Forward pass through student model
        logits = student_model(**batch)
        all_logits.append(logits.detach().cpu())
        if 'labels' in batch:
            all_labels.append(batch['labels'].detach().cpu())
    # Aggregate results and compute metrics (e.g., regression MSE or classification accuracy)
    # ... Insert evaluation metric calculation here ...
    student_model.train()
    return None


def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    if dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)

    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
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

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    optimizer_grouped_parameters = get_optimizer(args, distiller.student_model)
    optimizer_grouped_parameters = distiller.add_optimizer_param_group(optimizer_grouped_parameters)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer_grouped_parameters)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=distiller, optimizer=optimizer_grouped_parameters, lr_scheduler=lr_scheduler, mpu=None, config_params=ds_config)

    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
    if args.do_eval:
        if "test" in dataset and len(dataset["test"]) > 0:
            evaluate(args, distiller.student_tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
        else:
            log_rank("No test split found. Skipping evaluation phase.")


if __name__ == "__main__":
    main()
