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
        self.k = k
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
        t_comb = (self.teacher_weights.view(-1, 1, 1, 1) * t_stack).sum(0)  # [B,T,Ht]
        s_comb = (self.student_weights.view(-1, 1, 1, 1) * s_stack).sum(0)  # [B,T,Hs]
        return t_comb, s_comb  # both returned; projection layer fuses

    def get_state(self):
        return {
            "teacher_top_idx": self.teacher_top_idx,
            "student_top_idx": self.student_top_idx,
            "teacher_weights": self.teacher_weights.detach().cpu().tolist(),
            "student_weights": self.student_weights.detach().cpu().tolist()
        }

    @torch.no_grad()
    def reset_with_cis(self,
                        teacher_cis: List[float],
                        student_cis: List[float],
                        init_temp: float = 1.0):
        """
        Re-initialize top indices and attention weights using provided CIS scores, keeping the module and
        parameter objects the same (so existing optimizers remain valid). The top-k used is the original k.
        """
        k = self.k
        # Re-select indices
        self.teacher_top_idx = self._topk_indices(teacher_cis, k)
        self.student_top_idx = self._topk_indices(student_cis, k)

        # Re-init weights in-place (maintain parameter shapes)
        t_init = torch.tensor([teacher_cis[i] for i in self.teacher_top_idx], dtype=self.teacher_weights.dtype, device=self.teacher_weights.device)
        s_init = torch.tensor([student_cis[i] for i in self.student_top_idx], dtype=self.student_weights.dtype, device=self.student_weights.device)
        t_new = F.softmax(t_init / init_temp, dim=0)
        s_new = F.softmax(s_init / init_temp, dim=0)

        if t_new.shape != self.teacher_weights.data.shape or s_new.shape != self.student_weights.data.shape:
            # If shapes mismatch (unexpected), fall back to copying overlapping elements
            t_copy = torch.zeros_like(self.teacher_weights.data)
            s_copy = torch.zeros_like(self.student_weights.data)
            t_copy[: min(t_copy.numel(), t_new.numel())] = t_new[: min(t_copy.numel(), t_new.numel())]
            s_copy[: min(s_copy.numel(), s_new.numel())] = s_new[: min(s_copy.numel(), s_new.numel())]
            self.teacher_weights.data.copy_(t_copy)
            self.student_weights.data.copy_(s_copy)
        else:
            self.teacher_weights.data.copy_(t_new)
            self.student_weights.data.copy_(s_new)


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
        # Base projection matrix W_p
        self.base_proj = nn.Parameter(torch.randn(student_dim, teacher_dim) / math.sqrt(teacher_dim))
        self.ln = nn.LayerNorm(student_dim)
        # Keep a buffer to store last gate for analysis (optional)
        self.register_buffer("last_gate", torch.zeros(1,1,student_dim))

    def forward(self, teacher_repr: torch.Tensor, student_repr: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # teacher_repr: [B,T,Ht], student_repr: [B,T,Hs]
        # 1. Linear projection using (possibly adapted) weight
        weight = self.base_proj
        act_dtype = teacher_repr.dtype
        if weight.dtype != act_dtype:
            weight = weight.to(act_dtype)
        P = F.linear(teacher_repr, weight)  # [B,T,Hs]

        # 2. Mean pool (mask aware) -> e_t (projected) & e_s
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            s_mean = (student_repr * mask).sum(1) / mask.sum(1).clamp_min(1)
            p_mean = (P * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            s_mean = student_repr.mean(1)
            p_mean = P.mean(1)

        # 3. Approx gradient signal diff = e_s - e_t (no detach to let gradient flow)
        diff = s_mean - p_mean  # [B,Hs]
        gate = torch.sigmoid(diff).unsqueeze(1)  # [B,1,Hs]
        self.last_gate = gate.detach()

        # 4. Adaptive scaling of projected teacher representation (approx W_p(t) * e_t)
        P_dyn = P * gate
        e_syn = P_dyn + student_repr

        # 5. LayerNorm (numerical stability dtype handling)
        ln_dtype = self.ln.weight.dtype
        if e_syn.dtype != ln_dtype:
            out = self.ln(e_syn.to(ln_dtype))
            if ln_dtype != e_syn.dtype:
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