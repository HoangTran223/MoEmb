import math
import json
import os
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass

# ----------------------------
# CIS COMPUTER
# ----------------------------
class CISComputer:
    """
    CIS (Contextual Importance Score):
    - Với mỗi layer l (bỏ layer embedding 0), tính cosine(token_hidden, sentence_mean) cho mọi token hợp lệ.
    - Lấy trung bình trên tất cả token (batch-wise streaming).
    - Dùng để xếp hạng mức đóng góp ngữ cảnh của layer.
    """
    def __init__(self, model, device, fp16: bool = True):
        self.model = model
        self.device = device
        self.fp16 = fp16

    @torch.no_grad()
    def compute(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> List[float]:
        self.model.eval()
        n_layers = self._infer_n_layers()
        sums = torch.zeros(n_layers, dtype=torch.float64)
        counts = torch.zeros(n_layers, dtype=torch.float64)
        scaler = torch.cuda.amp.autocast if (self.fp16 and self.device.type == "cuda") else torch.cpu.amp.autocast
        for b_idx, batch in enumerate(dataloader):
            if max_batches and b_idx >= max_batches:
                break
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids","attention_mask","token_type_ids")}
            with (scaler()):
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # tuple(len = n_layers+1 incl. embeddings)
            # skip embeddings layer 0
            layer_tensors = hidden_states[1:]
            # sentence mean from student perspective: use last layer (or could use average)
            with_attn_mask = inputs.get("attention_mask", None)
            for l_idx, h in enumerate(layer_tensors):
                # h: [B, T, H]
                if with_attn_mask is not None:
                    mask = with_attn_mask.unsqueeze(-1)  # [B,T,1]
                    token_sum = (h * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp_min(1)
                    sent_mean = token_sum / denom
                else:
                    sent_mean = h.mean(dim=1)
                # cosine per token
                h_norm = F.normalize(h, dim=-1)
                sent_norm = F.normalize(sent_mean, dim=-1).unsqueeze(1)  # [B,1,H]
                cos = (h_norm * sent_norm).sum(-1)  # [B,T]
                if with_attn_mask is not None:
                    cos = cos * with_attn_mask
                    valid = with_attn_mask.sum()
                else:
                    valid = torch.tensor(h.shape[0]*h.shape[1], device=h.device)
                sums[l_idx] += cos.sum().double().cpu()
                counts[l_idx] += valid.double().cpu()
        cis = (sums / counts.clamp_min(1)).tolist()
        return cis

    def _infer_n_layers(self):
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        # fallback: run one dummy forward later (not ideal)
        raise ValueError("Cannot infer number of layers from model config.")


# ----------------------------
# ENHANCED ATTENTION SELECTION
# ----------------------------
class EnhancedAttentionSelector(nn.Module):
    """
    EAS:
    - Chọn top-k layer của teacher & student theo CIS.
    - Khởi tạo trọng số bằng softmax(CIS chọn).
    - Forward: trả về t_comb (Ht), s_comb (Hs) để DPL hợp nhất.
    """
    def __init__(self,
                 teacher_num_layers: int,
                 student_num_layers: int,
                 teacher_cis: List[float],
                 student_cis: List[float],
                 k: int,
                 init_temp: float = 1.0):
        super().__init__()
        self.teacher_top_idx = self._topk_indices(teacher_cis, k)
        self.student_top_idx = self._topk_indices(student_cis, k)
        t_init = torch.tensor([teacher_cis[i] for i in self.teacher_top_idx], dtype=torch.float32)
        s_init = torch.tensor([student_cis[i] for i in self.student_top_idx], dtype=torch.float32)
        # Normalize via softmax temperature
        self.teacher_weights = nn.Parameter(F.softmax(t_init / init_temp, dim=0))
        self.student_weights = nn.Parameter(F.softmax(s_init / init_temp, dim=0))

    def _topk_indices(self, scores: List[float], k: int) -> List[int]:
        return sorted(torch.topk(torch.tensor(scores), k).indices.tolist())

    def forward(self,
                teacher_hidden_states: Tuple[torch.Tensor],
                student_hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        teacher_hidden_states/student_hidden_states: tuple length L+1 with embeddings first.
        Returns combined token representations: [B,T,H_student] (teacher part projected outside).
        Here we only produce attention-combined hidden (without projection) for subsequent modules.
        """
        # Skip embedding layer 0
        t_layers = teacher_hidden_states[1:]
        s_layers = student_hidden_states[1:]
        # Weighted sum teacher (still teacher dim)
        t_stack = torch.stack([t_layers[i] for i in self.teacher_top_idx], dim=0)  # [k,B,T,Ht]
        s_stack = torch.stack([s_layers[i] for i in self.student_top_idx], dim=0)  # [k,B,T,Hs]
        t_comb = (self.teacher_weights.view(-1,1,1,1) * t_stack).sum(0)  # [B,T,Ht]
        s_comb = (self.student_weights.view(-1,1,1,1) * s_stack).sum(0)  # [B,T,Hs]
        return t_comb, s_comb  # both returned; projection layer fuses

    def get_state(self):
        return {
            "teacher_top_idx": self.teacher_top_idx,
            "student_top_idx": self.student_top_idx,
            "teacher_weights": self.teacher_weights.detach().cpu().tolist(),
            "student_weights": self.student_weights.detach().cpu().tolist()
        }


# ----------------------------
# DYNAMIC PROJECTION LAYER
# ----------------------------
class DynamicProjectionLayer(nn.Module):
    """
    DPL:
    - Linear chiếu Ht -> Hs.
    - Gate = sigmoid( mean(student) - mean(projected_teacher) )
    - Khuếch đại / suy giảm theo từng chiều -> tạo P_dyn.
    - e_syn = LayerNorm(P_dyn + student_repr).
    """
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.base_proj = nn.Parameter(torch.randn(student_dim, teacher_dim) / math.sqrt(teacher_dim))
        self.ln = nn.LayerNorm(student_dim)

    def forward(self, teacher_repr: torch.Tensor, student_repr: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # teacher_repr: [B,T,Ht], student_repr: [B,T,Hs]
        # Ensure dtype alignment (mixed precision safety: teacher/student may be bf16/FP16, param fp32)
        weight = self.base_proj
        if weight.dtype != teacher_repr.dtype:
            # Cast weight to activation dtype (preserve gradient path)
            weight = weight.to(teacher_repr.dtype)
        P = F.linear(teacher_repr, weight)  # [B,T,Hs]
        # Mean pooling (mask aware)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            s_mean = (student_repr * mask).sum(1) / mask.sum(1).clamp_min(1)
            p_mean = (P * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            s_mean = student_repr.mean(1)
            p_mean = P.mean(1)
        diff = s_mean - p_mean  # [B,Hs]
        gate = torch.sigmoid(diff).unsqueeze(1)  # [B,1,Hs]
        # Broadcast gate to scale base projection output adaptively
        P_dyn = P * gate
        e_syn = P_dyn + student_repr
        # LayerNorm expects input and its params to share dtype
        ln_weight_dtype = self.ln.weight.dtype
        if e_syn.dtype != ln_weight_dtype:
            # Strategy: run LN in fp32 for numeric stability, then cast back
            target_dtype = ln_weight_dtype
            out = self.ln(e_syn.to(target_dtype))
            if target_dtype != e_syn.dtype:
                out = out.to(e_syn.dtype)
            e_syn = out
        else:
            e_syn = self.ln(e_syn)
        return e_syn


# ----------------------------
# TALF LOSS
# ----------------------------
class TALFLoss(nn.Module):
    """
    TALF (Task-Agnostic Loss) = 1 - mean_cosine(token, sentence_mean)
    - Thúc đẩy mỗi token vector tiến gần ngữ cảnh toàn câu -> tăng tính nhất quán ngữ nghĩa.
    """
    def __init__(self):
        super().__init__()

    def forward(self, e_syn: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # e_syn: [B,T,H]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            e_context = (e_syn * mask).sum(1) / mask.sum(1).clamp_min(1)
            # token-level cosine
            e_syn_norm = F.normalize(e_syn, dim=-1)
            e_ctx_norm = F.normalize(e_context, dim=-1).unsqueeze(1)
            cos = (e_syn_norm * e_ctx_norm).sum(-1)  # [B,T]
            cos = cos * attention_mask
            loss = 1.0 - (cos.sum() / attention_mask.sum().clamp_min(1))
        else:
            e_context = e_syn.mean(1)
            cos = F.cosine_similarity(e_syn, e_context.unsqueeze(1), dim=-1)
            loss = 1.0 - cos.mean()
        return loss


# ----------------------------
# TRAINER
# ----------------------------
@dataclass
class EAADPConfig:
    top_k: int = 3
    attn_lr: float = 1e-4
    main_lr: float = 2e-5
    epochs: int = 3
    log_every: int = 50
    save_dir: str = "eaadp_ckpt"
    fp16: bool = True
    max_cis_batches: Optional[int] = None

class EAADPTrainer:
    def __init__(self,
                 teacher_model,
                 student_model,
                 teacher_dim: int,
                 student_dim: int,
                 train_dataloader: DataLoader,
                 cfg: EAADPConfig,
                 device: torch.device):
        self.teacher = teacher_model
        self.student = student_model
        self.train_dl = train_dataloader
        self.cfg = cfg
        self.device = device

        # Placeholder: will set after CIS pass
        self.eas: Optional[EnhancedAttentionSelector] = None
        self.dpl = DynamicProjectionLayer(teacher_dim, student_dim).to(device)
        self.talf = TALFLoss().to(device)

    def compute_cis(self) -> Tuple[List[float], List[float]]:
        t_cis = CISComputer(self.teacher, self.device, self.cfg.fp16).compute(self.train_dl, self.cfg.max_cis_batches)
        s_cis = CISComputer(self.student, self.device, self.cfg.fp16).compute(self.train_dl, self.cfg.max_cis_batches)
        return t_cis, s_cis

    def init_eas(self, teacher_cis, student_cis):
        self.eas = EnhancedAttentionSelector(
            teacher_num_layers=len(teacher_cis),
            student_num_layers=len(student_cis),
            teacher_cis=teacher_cis,
            student_cis=student_cis,
            k=self.cfg.top_k
        ).to(self.device)

    def train(self):
        assert self.eas is not None, "Call init_eas after CIS computation first."
        self.teacher.eval()
        self.student.train()
        # Separate optimizers: attention weights & main params
        attn_opt = torch.optim.AdamW([p for p in self.eas.parameters() if p.requires_grad],
                                     lr=self.cfg.attn_lr)
        main_params = list(self.student.parameters()) + list(self.dpl.parameters())
        main_opt = torch.optim.AdamW(main_params, lr=self.cfg.main_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.fp16 and self.device.type == "cuda"))

        global_step = 0
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        for epoch in range(1, self.cfg.epochs + 1):
            for batch in self.train_dl:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids","attention_mask","token_type_ids")}
                with torch.no_grad():
                    t_out = self.teacher(**inputs, output_hidden_states=True, return_dict=True)
                s_out = self.student(**inputs, output_hidden_states=True, return_dict=True)
                with torch.cuda.amp.autocast(enabled=(self.cfg.fp16 and self.device.type == "cuda")):
                    t_comb, s_comb = self.eas(t_out.hidden_states, s_out.hidden_states)
                    e_syn = self.dpl(t_comb, s_comb, inputs.get("attention_mask"))
                    loss = self.talf(e_syn, inputs.get("attention_mask"))
                scaler.scale(loss).backward()
                # Update main
                scaler.step(main_opt)
                # Update attention weights
                attn_opt.step()
                scaler.update()
                main_opt.zero_grad(set_to_none=True)
                attn_opt.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % self.cfg.log_every == 0:
                    print(f"[Epoch {epoch}] Step {global_step} Loss {loss.item():.4f}")

            # Save at epoch end
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int):
        state = {
            "epoch": epoch,
            "eas": self.eas.get_state(),
            "dpl": self.dpl.state_dict(),
            "student": self.student.state_dict()
        }
        path = os.path.join(self.cfg.save_dir, f"epoch_{epoch}.pt")
        torch.save(state, path)
        with open(os.path.join(self.cfg.save_dir, "eas_state.json"), "w") as f:
            json.dump(self.eas.get_state(), f, indent=2)
        print(f"Saved checkpoint: {path}")

    def inference_combine_attention(self, batch):
        self.teacher.eval()
        self.student.eval()
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k,v in batch.items() if k in ("input_ids","attention_mask","token_type_ids")}
            t_out = self.teacher(**inputs, output_hidden_states=True, return_dict=True)
            s_out = self.student(**inputs, output_hidden_states=True, return_dict=True)
            t_comb, s_comb = self.eas(t_out.hidden_states, s_out.hidden_states)
            # Return combined representation after projection (Attn_combined concept)
            e_syn = self.dpl(t_comb, s_comb, inputs.get("attention_mask"))
            return e_syn  # [B,T,Hs]

# Utility: simple collate placeholder
def default_collate_fn(batch):
    # Expect each example already tokenized dict
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([ex[k] for ex in batch], dim=0)
    return out
