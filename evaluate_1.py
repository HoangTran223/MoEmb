import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score
from peft import PeftModel

from SentencePair.data_utils.distill_datasets import DistillDataset


def _inspect_ckpt_dir(ckpt_dir: str):
    if not isinstance(ckpt_dir, (str, os.PathLike)):
        raise ValueError("--ckpt-dir must be a filesystem path to a saved model directory.")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"--ckpt-dir '{ckpt_dir}' is not a directory or does not exist.")
    config_path = os.path.join(ckpt_dir, "config.json")
    adapter_cfg = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(config_path):
        return {"type": "full", "config_path": config_path}
    if os.path.exists(adapter_cfg):
        # Read base hint if available
        base_hint = None
        try:
            with open(adapter_cfg, "r") as f:
                j = json.load(f)
                base_hint = j.get("base_model_name_or_path")
        except Exception:
            base_hint = None
        return {"type": "adapter", "adapter_config": adapter_cfg, "base_hint": base_hint}
    raise FileNotFoundError(
        f"Neither config.json nor adapter_config.json found in '{ckpt_dir}'. Provide a valid checkpoint or adapter directory."
    )


def _dtype_from_str(s: str | None):
    if not s:
        return None
    s = s.lower()
    if s == "fp32":
        return torch.float32
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    return None


def load_model_and_tokenizer(ckpt_dir: str, device: torch.device, base_model_path: str | None = None, model_dtype: str | None = None):
    """
    Load a HF sequence classification model and tokenizer from a checkpoint directory.
    If a separate classifier head file exists, try to load it (best-effort).
    """
    info = _inspect_ckpt_dir(ckpt_dir)
    target_dtype = _dtype_from_str(model_dtype) or torch.float32
    if info["type"] == "full":
        # Use trust_remote_code=True to mirror training-time loaders for custom models (e.g., LLM2Vec/Mistral)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir, config=config, torch_dtype=target_dtype, trust_remote_code=True
        )
        model.to(device)
    else:
        # LoRA adapter path: load base then attach adapter
        base_path = base_model_path or info.get("base_hint")
        if not base_path:
            raise ValueError(
                "LoRA adapter detected but base model path not provided. "
                "Pass --base-model-path or ensure adapter_config.json contains base_model_name_or_path."
            )
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_path, config=config, torch_dtype=target_dtype, trust_remote_code=True
        )
        # Attach adapter
        model = PeftModel.from_pretrained(model, ckpt_dir)
        model.to(device)

    # Optional: load classifier head if available
    classifier_path = os.path.join(ckpt_dir, "classifier_head.bin")
    if os.path.exists(classifier_path):
        try:
            state = torch.load(classifier_path, map_location="cpu")
            if hasattr(model, "score"):
                model.score.load_state_dict(state)
            elif hasattr(model, "classifier"):
                model.classifier.load_state_dict(state)
        except Exception as e:
            print(f"[WARN] Could not load classifier head from {classifier_path}: {e}")

    model.eval()
    return model, tokenizer


def build_dataset(args, tokenizer):
    dataset = DistillDataset(args, args.split, student_tokenizer = tokenizer, teacher_tokenizer=None)
    return dataset


def evaluate(tokenizer, student_model, dataset, split, device, batch_size: int, num_workers: int):
    
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
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
    with torch.no_grad():
        for input_batch, output_batch in tqdm(dataloader, desc="Processing batches", dynamic_ncols=True):
            dataset.move_to_device([input_batch, output_batch], device)
            labels = output_batch["labels"]
            outputs = student_model(
                input_batch["input_ids"],
                attention_mask=input_batch["attention_mask"],
                position_ids=input_batch.get("position_ids", None),
                labels=labels,
            )
            logits = outputs.logits
            loss = outputs.loss

            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            all_preds.append(preds)
            all_labels.append(labels)
            sample_num = labels.size(0)
            total_loss += loss.item() * sample_num

            eval_info["sample_num"] += sample_num
            eval_info["correct_samples"] += correct

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    correct = int((all_preds == all_labels).sum())
    accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0
    avg_loss = float(total_loss) / len(all_labels) if len(all_labels) > 0 else 0.0


    eval_info["precision"] = round(precision, 6)
    eval_info["recall"] = round(recall, 6)

    eval_info["loss"] = round(avg_loss, 6)
    eval_info["accuracy"] = round(accuracy, 6)
    eval_info["sample_num"] = int(len(all_preds))
    eval_info["correct_samples"] = int(correct)

    for key in eval_info:
        if isinstance(eval_info[key], float):
            eval_info[key] = round(eval_info[key], 6)
    
    print(f"Evaluated: {split} | {eval_info}")

    student_model.train()

    return eval_info["loss"], eval_info["accuracy"], eval_info.get("precision", 0.0), eval_info.get("recall", 0.0)

# def evaluate(model, dataset, tokenizer, device, batch_size: int, num_workers: int = 0):
#     dataloader = DataLoader(
#         dataset,
#         shuffle=False,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         collate_fn=dataset.collate,
#     )

#     all_preds = []
#     all_labels = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for input_batch, output_batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
#             # move to device
#             for k in input_batch:
#                 if isinstance(input_batch[k], torch.Tensor):
#                     input_batch[k] = input_batch[k].to(device)
#             for k in output_batch:
#                 if isinstance(output_batch[k], torch.Tensor):
#                     output_batch[k] = output_batch[k].to(device)

#             labels = output_batch["labels"]
#             outputs = model(
#                 input_batch["input_ids"],
#                 attention_mask=input_batch["attention_mask"],
#                 position_ids=input_batch.get("position_ids", None),
#                 labels=labels,
#             )
#             logits = outputs.logits
#             loss = outputs.loss

#             preds = logits.argmax(dim=-1)
#             all_preds.append(preds.detach().cpu())
#             all_labels.append(labels.detach().cpu())
#             total_loss += loss.item() * labels.size(0)

#     all_preds = torch.cat(all_preds, dim=0).numpy()
#     all_labels = torch.cat(all_labels, dim=0).numpy()

#     precision = precision_score(all_labels, all_preds, average="macro")
#     recall = recall_score(all_labels, all_preds, average="macro")
#     accuracy = (all_preds == all_labels).sum() / len(all_labels)
#     avg_loss = total_loss / len(all_labels)

#     results = {
#         "loss": round(float(avg_loss), 6),
#         "accuracy": round(float(accuracy), 6),
#         "precision": round(float(precision), 6),
#         "recall": round(float(recall), 6),
#         "sample_num": int(len(all_labels)),
#         "correct_samples": int((all_preds == all_labels).sum()),
#     }
#     return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a classification checkpoint on a dataset")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Path to the saved checkpoint directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing split CSV files (e.g., test.csv)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="Which split to evaluate")
    # Align with imdb_eaadp.sh style
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--base-model-path", type=str, default=None, help="Base model name or path when --ckpt-dir is a LoRA adapter directory")
    parser.add_argument("--model-dtype", type=str, default=None, choices=["fp32", "fp16", "bf16"], help="Precision to load model (match training)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.ckpt_dir, device, args.base_model_path, args.model_dtype)
    dataset = build_dataset(args, tokenizer)
    results = evaluate(tokenizer, model, dataset, args.split, device, args.eval_batch_size, args.num_workers)
    print("Evaluation:", results)


if __name__ == "__main__":
    main()
