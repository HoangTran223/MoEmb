import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from .eaadp_modules import EAADPConfig, EAADPTrainer, default_collate_fn

# Placeholder dataset (replace with real Patent dataset loader)
class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k,v in enc.items()}
        return item

def build_dataloader(args, tokenizer):
    # Replace with Patent dataset loading
    dummy_texts = ["Example patent abstract about material science."] * args.num_samples
    ds = SimpleTextDataset(dummy_texts, tokenizer, max_len=args.max_len)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate_fn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Pretrained teacher (placeholder for LLM2Vec).")
    parser.add_argument("--student_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase", type=str, choices=["compute_cis","train","infer"], default="train")
    parser.add_argument("--save_dir", type=str, default="eaadp_ckpt")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--max_cis_batches", type=int, default=10)
    parser.add_argument("--attn_lr", type=float, default=1e-4)
    parser.add_argument("--main_lr", type=float, default=2e-5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cis_path", type=str, default="cis_scores.json")
    parser.add_argument("--eaadp_alpha", type=float, default=0.5, help="Weight for CrossEntropy loss in EAADP.")
    parser.add_argument("--eaadp_top_k", type=int, default=3, help="Top-k layers for EAS in EAADP.")
    parser.add_argument("--eaadp_max_cis_batches", type=int, default=10, help="Max batches for CIS computation in EAADP.")
    parser.add_argument("--eaadp_use_batch_cis", action="store_true", help="Whether to use batch-wise CIS computation.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load models & tokenizers
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model_name)
    student_tok = AutoTokenizer.from_pretrained(args.student_model_name)
    # Ensure same tokenizer for simplicity (optional)
    tokenizer = student_tok

    teacher = AutoModel.from_pretrained(args.teacher_model_name).to(device)
    student = AutoModel.from_pretrained(args.student_model_name).to(device)

    dl = build_dataloader(args, tokenizer)

    cfg = EAADPConfig(
        top_k=args.top_k,
        attn_lr=args.attn_lr,
        main_lr=args.main_lr,
        epochs=args.epochs,
        log_every=args.log_every,
        save_dir=args.save_dir,
        fp16=args.fp16,
        max_cis_batches=args.max_cis_batches
    )

    trainer = EAADPTrainer(
        teacher_model=teacher,
        student_model=student,
        teacher_dim=teacher.config.hidden_size,
        student_dim=student.config.hidden_size,
        train_dataloader=dl,
        cfg=cfg,
        device=device
    )

    if args.phase == "compute_cis":
        t_cis, s_cis = trainer.compute_cis()
        with open(args.cis_path, "w") as f:
            json.dump({"teacher_cis": t_cis, "student_cis": s_cis}, f, indent=2)
        print("Saved CIS to", args.cis_path)
        return

    if args.phase == "train":
        if os.path.exists(args.cis_path):
            with open(args.cis_path) as f:
                cis_data = json.load(f)
            t_cis = cis_data["teacher_cis"]
            s_cis = cis_data["student_cis"]
        else:
            print("Computing CIS (no cached file found).")
            t_cis, s_cis = trainer.compute_cis()
            with open(args.cis_path, "w") as f:
                json.dump({"teacher_cis": t_cis, "student_cis": s_cis}, f, indent=2)
        trainer.init_eas(t_cis, s_cis)
        trainer.train()
        return

    if args.phase == "infer":
        # Need eas initialization from saved cis + weights (simplified)
        with open(args.cis_path) as f:
            cis_data = json.load(f)
        t_cis = cis_data["teacher_cis"]
        s_cis = cis_data["student_cis"]
        trainer.init_eas(t_cis, s_cis)
        # Load last checkpoint if exists
        ckpts = [f for f in os.listdir(args.save_dir) if f.startswith("epoch_") and f.endswith(".pt")]
        if ckpts:
            ckpts.sort()
            state = torch.load(os.path.join(args.save_dir, ckpts[-1]), map_location=device)
            trainer.student.load_state_dict(state["student"])
            trainer.dpl.load_state_dict(state["dpl"])
            print(f"Loaded checkpoint {ckpts[-1]}")
        batch = next(iter(dl))
        e_syn = trainer.inference_combine_attention(batch)
        print("Inference combined representation shape:", tuple(e_syn.shape))

if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
from .various_divergence import VariousDivergence
from .eaadp_modules import CISComputer  # tái sử dụng bộ tính CIS

class EAADP(VariousDivergence):
    """
    EAADP Criterion:
    loss = alpha * CrossEntropy(student_logits, labels) + (1-alpha) * CCL(e_syn)
    Trong đó e_syn = LayerNorm( DPL( ∑ w_t,l T_l , ∑ w_s,l S_l ) + student_combined )
    EAS (w_t,l, w_s,l) khởi tạo bằng dummy rồi cập nhật lại bằng CIS batch đầu.
    """
    def __init__(self, args):
        super().__init__(args)
        self.alpha = args.eaadp_alpha
        self.top_k = args.eaadp_top_k
        self.max_cis_batches = args.eaadp_max_cis_batches
        self.use_batch_cis = args.eaadp_use_batch_cis

    def _maybe_update_cis(self, distiller, input_data, batch_idx=0):
        if getattr(distiller, "eaadp_cis_updated", True):
            return
        # Tính CIS nhanh từ 1 hoặc vài batch (student + teacher)
        # Ở đây chỉ dùng batch hiện tại nếu không bật multi-batch.
        with torch.no_grad():
            teacher = distiller.teacher_model
            student = distiller.student_model
            teacher.eval()
            student.eval()
            # giả lập dataloader mini
            mini_list = [{
                "input_ids": input_data["input_ids"],
                "attention_mask": input_data["attention_mask"],
            }]
            # teacher dùng trường teacher_input_ids riêng
            mini_list_teacher = [{
                "input_ids": input_data["teacher_input_ids"],
                "attention_mask": input_data["teacher_attention_mask"],
            }]
            # Student CIS
            s_cis = CISComputer(student, input_data["input_ids"].device, fp16=False).compute(
                dataloader=ListDataLoader(mini_list),
                max_batches=1
            )
            # Teacher CIS
            t_cis = CISComputer(teacher, input_data["input_ids"].device, fp16=False).compute(
                dataloader=ListDataLoader(mini_list_teacher),
                max_batches=1
            )
        # Re-init EAS với CIS thực
        from .eaadp_modules import EnhancedAttentionSelector
        distiller.eas = EnhancedAttentionSelector(
            teacher_num_layers=len(t_cis),
            student_num_layers=len(s_cis),
            teacher_cis=t_cis,
            student_cis=s_cis,
            k=self.top_k
        ).to(input_data["input_ids"].device)
        distiller.eaadp_cis_updated = True

    def forward(
        self,
        distiller,
        input_data,
        output_data,
        logging_output,
        batch_denom,
    ):
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model

        # Student forward
        s_out = student_model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True,
            labels=output_data["labels"]
        )
        ce_loss = s_out.loss
        logits = s_out.logits

        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )

        # Cập nhật CIS lần đầu
        self._maybe_update_cis(distiller, input_data)

        # EAS kết hợp layer
        t_comb, s_comb = distiller.eas(t_out.hidden_states, s_out.hidden_states)

        # DPL -> e_syn
        e_syn = distiller.dpl(
            teacher_repr=t_comb,
            student_repr=s_comb,
            attention_mask=input_data["attention_mask"]
        )

        # TALF (CCL)
        ccl_loss = distiller.talf(e_syn, input_data["attention_mask"])

        loss = self.alpha * ce_loss + (1 - self.alpha) * ccl_loss

        # Logging
        log = {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "ccl_loss": ccl_loss.detach(),
            "accuracy": self.compute_accuracy(logits, output_data["labels"]),
        }
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output


class ListDataLoader:
    """DataLoader đơn giản bọc danh sách dict (phục vụ CIS batch đầu)."""
    def __init__(self, data_list):
        self.data_list = data_list
    def __iter__(self):
        for x in self.data_list:
            yield x
    def __len__(self):
        return len(self.data_list)