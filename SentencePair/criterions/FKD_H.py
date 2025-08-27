import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FKD_H(nn.Module):
    """
    FKD_H: Hybrid Feature KD with
    - Teacher layer importance (BI top-k, offline pre-pass done in training loop)
    - Token alignment via hybrid scores (contextual + global alignment matrix)
    - Student fusion over mapped student layers using attention

    Total loss: L_total = alpha * CE + beta * (1 - mean_cosine(\tilde{h}_S, \tilde{q}_T))
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Loss weights
        self.alpha = getattr(args, 'fkd_h_alpha', None)
        if self.alpha is None:
            # fallback to previous naming if provided
            self.alpha = getattr(args, 'fkd_final_alpha', 1.0)
        self.beta = getattr(args, 'fkd_h_beta', None)
        if self.beta is None:
            self.beta = getattr(args, 'fkd_final_beta', 1.0)

        # Hybrid mixing coef lambda for global part
        self.lambda_h = getattr(args, 'fkd_h_lambda', None)
        if self.lambda_h is None:
            self.lambda_h = getattr(args, 'fkd_final_lambda', 0.7)

        # How many teacher tokens to aggregate per student token (top-k over alignment probs)
        self.align_topk = getattr(args, 'fkd_h_align_topk', 32)

        # Optional global alignment matrix path
        self.global_alignment_path = getattr(args, 'global_alignment_path', None)
        self._global_align_cpu = None  # torch tensor on CPU
        if self.global_alignment_path and os.path.exists(self.global_alignment_path):
            try:
                import numpy as np
                mat = np.load(self.global_alignment_path)
                if mat.dtype != np.float32:
                    mat = mat.astype(np.float32)
                self._global_align_cpu = torch.from_numpy(mat)
                print(f"[FKD_H] Loaded global alignment matrix from {self.global_alignment_path}")
            except Exception as e:
                print(f"[FKD_H][WARN] Failed to load global alignment: {e}")

        # Optional offline projection init (W: H_S x H_T). If provided, we'll try to load into a Linear.
        self.offline_proj_path = getattr(args, 'offline_projection_path', None)

    # ---------- helpers ----------
    def _slice_global_scores(self, teacher_ids: torch.Tensor, student_ids: torch.Tensor, device, dtype):
        """Slice the global alignment sub-matrix for given token id sequences.
        Returns [B, T, S] tensor on device.
        """
        B, T = teacher_ids.shape
        S = student_ids.shape[1]
        out = torch.zeros(B, T, S, device=device, dtype=dtype)
        GA = self._global_align_cpu
        if GA is None:
            return out
        # Ensure index bounds
        T_vocab, S_vocab = GA.size(0), GA.size(1)
        for b in range(B):
            t_idx = teacher_ids[b].to('cpu')
            s_idx = student_ids[b].to('cpu')
            t_idx = torch.clamp(t_idx, 0, T_vocab - 1)
            s_idx = torch.clamp(s_idx, 0, S_vocab - 1)
            sub = GA.index_select(0, t_idx).index_select(1, s_idx).to(device=device, dtype=dtype)
            out[b, :sub.size(0), :sub.size(1)] = sub
        return out

    def _ensure_wq(self, distiller, in_dim, out_dim, device, dtype):
        """Ensure a projector W_q exists on distiller.projectors mapping teacher->student dims.
        If offline init is provided, try to load weights.
        """
        if not hasattr(distiller, 'projectors'):
            distiller.projectors = nn.ModuleDict()
        if 'W_q' not in distiller.projectors:
            layer = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            distiller.projectors['W_q'] = layer
            # Try to load from offline path (expects a torch .pt with keys 'weight','bias' or full state_dict)
            if self.offline_proj_path and os.path.exists(self.offline_proj_path):
                try:
                    state = torch.load(self.offline_proj_path, map_location='cpu')
                    if 'weight' in state:
                        distiller.projectors['W_q'].load_state_dict(state)
                    else:
                        distiller.projectors['W_q'].load_state_dict(state)
                    print(f"[FKD_H] Loaded offline projection from {self.offline_proj_path}")
                except Exception as e:
                    print(f"[FKD_H][WARN] Failed loading offline projection: {e}")
        distiller.projectors['W_q'] = distiller.projectors['W_q'].to(device=device, dtype=dtype)
        return distiller.projectors['W_q']

    def _teacher_fused(self, teacher_hs, top_indices, top_weights):
        """Fuse teacher hidden states across selected layers.
        teacher_hs: tuple/list of [B,T,H_T] length L+1 (incl. embeddings at 0)
        top_indices: list of teacher layer indices (0-based for transitions between hs[l] and hs[l+1])
        We'll take hs[l+1] as the output of layer l.
        top_weights: list of same length, softmax weights.
        Returns: [B,T,H_T]
        """
        # default to last layer if not provided
        if not top_indices:
            return teacher_hs[-1]
        fused = None
        for idx, w in zip(top_indices, top_weights):
            layer_out = teacher_hs[min(idx + 1, len(teacher_hs) - 1)]  # safe
            fused = layer_out * w if fused is None else fused + layer_out * w
        return fused

    def _student_stack(self, student_hs, mapped_student_layers):
        """Stack student hidden states for mapped layers into [B,S,M,H_S].
        If mapping is empty, fall back to last layer expanded with M=1.
        """
        if not mapped_student_layers:
            last = student_hs[-1]
            return last.unsqueeze(2)  # [B,S,1,H]
        tensors = []
        for s_l in mapped_student_layers:
            if s_l is None:
                s_l = len(student_hs) - 1
            s_l = max(0, min(len(student_hs) - 1, int(s_l) + 1))  # use hs index as output of that layer
            tensors.append(student_hs[s_l])
        stacked = torch.stack(tensors, dim=2)  # [B,S,M,H]
        return stacked

    # ---------- main forward ----------
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        student = distiller.student_model
        teacher = distiller.teacher_model

        # Forward student with hidden states
        s_out = student(
            input_data["input_ids"],
            attention_mask=input_data.get("attention_mask", None),
            output_hidden_states=True,
            return_dict=True,
        )
        s_logits = s_out.logits
        s_hs = s_out.hidden_states  # tuple len Ls+1

        # CE loss
        labels = output_data["labels"]
        ce_loss = F.cross_entropy(s_logits, labels)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = teacher(
                input_data.get("teacher_input_ids", input_data["input_ids"]),
                attention_mask=input_data.get("teacher_attention_mask", input_data.get("attention_mask", None)),
                output_hidden_states=True,
                return_dict=True,
            )
            t_hs = t_out.hidden_states

        device = s_hs[-1].device
        dtype = s_hs[-1].dtype

        # Use BI selection if available
        fkd_info = getattr(distiller, 'fkd_info', {}) or {}
        top_indices = fkd_info.get('top_indices', []) or []
        top_weights = fkd_info.get('teacher_top_k_weights', None)
        if top_weights is None and top_indices:
            # fallback: uniform
            top_weights = [1.0 / max(1, len(top_indices))] * len(top_indices)

        # Fuse teacher layers and project
        t_fused = self._teacher_fused(t_hs, top_indices, top_weights)  # [B,T,H_T]
        H_T = t_fused.size(-1)
        H_S = s_hs[-1].size(-1)
        W_q = self._ensure_wq(distiller, H_T, H_S, device, dtype)
        t_proj = W_q(t_fused)  # [B,T,H_S]

        # Hybrid alignment to build per-student-token teacher representation
        s_last = s_hs[-1]  # [B,S,H_S]
        teacher_ids = input_data.get("teacher_input_ids", input_data["input_ids"])  # [B,T]
        student_ids = input_data["input_ids"]  # [B,S]

        # Contextual score: cosine(s_last[b,j], t_proj[b,i]) for all pairs -> [B,S,T]
        # Efficiently compute via normalized dot product
        s_norm = F.normalize(s_last, p=2, dim=-1)
        t_norm = F.normalize(t_proj, p=2, dim=-1)
        # compute [B,S,T] = (s_norm @ t_norm^T)
        ctx_scores = torch.einsum('bsh,bth->bst', s_norm, t_norm)

        # Global score from matrix (slice on CPU then move)
        glob_scores = self._slice_global_scores(teacher_ids, student_ids, device, dtype)  # [B,T,S]
        glob_scores = glob_scores.transpose(1, 2)  # [B,S,T]

        # Hybrid scores and softmax over teacher tokens (axis=-1 on dim T)
        lam = float(self.lambda_h)
        hybrid = (1.0 - lam) * ctx_scores + lam * glob_scores
        # Mask out padding teacher tokens if mask provided
        t_mask = input_data.get("teacher_attention_mask", None)
        if t_mask is not None:
            t_mask = (t_mask > 0).to(dtype=dtype, device=device)
            hybrid = hybrid.masked_fill(t_mask.unsqueeze(1) == 0, float('-inf'))

        # Top-k on teacher tokens per student token to reduce compute
        k = min(int(self.align_topk), hybrid.size(-1))
        if k < hybrid.size(-1):
            vals, idx = torch.topk(hybrid, k, dim=-1)
            weights = torch.softmax(vals, dim=-1)  # [B,S,k]
            # Gather t_proj along T using idx
            t_exp = t_proj.unsqueeze(1).expand(-1, hybrid.size(1), -1, -1)  # [B,S,T,H]
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, t_proj.size(-1))
            gathered = torch.gather(t_exp, 2, idx_exp)  # [B,S,k,H]
            q_T = (weights.unsqueeze(-1) * gathered).sum(dim=2)  # [B,S,H]
        else:
            weights = torch.softmax(hybrid, dim=-1)  # [B,S,T]
            q_T = torch.einsum('bst,bth->bsh', weights, t_proj)

        # Student fusion over mapped layers with attention against q_T
        mapped_student = fkd_info.get('mapped_student_layers', []) or []
        M_S = self._student_stack(s_hs, mapped_student)  # [B,S,M,H]
        # scores_{b,j,m} = <M_S[b,j,m,:], q_T[b,j,:]> / sqrt(H)
        scores = (M_S * q_T.unsqueeze(2)).sum(dim=-1) / math.sqrt(M_S.size(-1))  # [B,S,M]
        att_w = torch.softmax(scores, dim=-1)  # [B,S,M]
        h_tilde = (att_w.unsqueeze(-1) * M_S).sum(dim=2)  # [B,S,H]

        # Distillation loss: 1 - mean cosine over valid student tokens
        s_mask = input_data.get("attention_mask", None)
        if s_mask is not None:
            s_mask = (s_mask > 0).to(device=device, dtype=dtype)
            cos = F.cosine_similarity(h_tilde, q_T, dim=-1) * s_mask  # [B,S]
            denom = s_mask.sum().clamp(min=1.0)
            distill = 1.0 - (cos.sum() / denom)
        else:
            cos = F.cosine_similarity(h_tilde, q_T, dim=-1)
            distill = 1.0 - cos.mean()

        total = self.alpha * ce_loss + self.beta * distill

        # Optional: collect mean attn weights per student layer for logging
        try:
            if not hasattr(distiller, 'epoch_student_attn_weights'):
                distiller.epoch_student_attn_weights = []
            # Reduce over batch and seq: mean attn per layer m
            layer_means = att_w.mean(dim=(0, 1)).detach().float().cpu().tolist()
            distiller.epoch_student_attn_weights.append({str(i): w for i, w in enumerate(layer_means)})
        except Exception:
            pass

        logging_output["loss"] = float(total.detach().item())
        logging_output["ce_loss"] = float(ce_loss.detach().item())
        logging_output["distill_loss"] = float(distill.detach().item())
        return total, logging_output
