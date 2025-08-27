import torch
import torch.nn as nn
import torch.nn.functional as F


class FKD_A(nn.Module):
    """
    FKD_A: Focused KD with Attentive Student Fusion

    Differences vs FKD:
    - Still uses offline BI to pick top-k teacher layers and compute weights w_i (softmax over BI).
    - Teacher fuse: H_T_fused = sum_i w_i * Pool(H_T_i) [optionally projected t2s].
    - Student fuse: attention over pooled student layers, where query comes from teacher fused vector
      passed through a learnable W_q; keys/values are pooled student layers (and keys may be projected
      via an optional s2s projector if present).
    - Total loss = alpha * CE + beta * (1 - cos(H_S_fused, H_T_fused)) + gamma * InfoNCE(q=H_S_fused, k=H_T_fused).
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.k = getattr(args, "fkd_k", 4)
        self.alpha = getattr(args, "fkd_alpha", 1.0)
        self.beta = getattr(args, "fkd_beta", 1.0)
        self.gamma = getattr(args, "fkd_gamma", 0.1)
        self.ctemp = getattr(args, "fkd_contrastive_temp", 0.07)
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)

        # W_q: learnable projection for teacher fused query into student hidden dim
        # Will be lazily initialized once we know dims from models.
        self.q_proj = None

    # ---------- helpers ----------
    def _ce_loss(self, logits, target):
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        nll = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        if self.label_smoothing > 0:
            smooth = -lprobs.mean(dim=-1).mean()
            return (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        return nll

    def _pooled(self, hs, attn_mask, is_encoder_like=True):
        if is_encoder_like:
            return hs[:, 0, :]
        if attn_mask is None:
            return hs[:, -1, :]
        lengths = attn_mask.long().sum(dim=-1) - 1
        b = torch.arange(hs.size(0), device=hs.device)
        return hs[b, lengths.clamp(min=0), :]

    def _map_teacher_to_student_layers(self, t_idx_list, t_layers, s_layers):
        """
        Deterministic mapping from teacher layer indices (0-based) to student layer indices (0-based).
        Uses formula: s_l = floor(((l+1) * S - 1) / T), clipped to [0, S-1].
        This preserves relative depth and avoids off-by-one issues.
        """
        mapped = []
        T = int(t_layers)
        S = int(s_layers)
        for l in t_idx_list:
            if S is None or T is None or S <= 0 or T <= 0:
                mapped.append(None)
                continue
            # l is 0-based teacher layer index; compute 0-based student index
            s_l = int(((int(l) + 1) * S - 1) // max(1, T))
            s_l = max(0, min(S - 1, s_l))
            mapped.append(s_l)
        return mapped

    def _info_nce(self, q, k):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.matmul(q, k.transpose(0, 1)) / self.ctemp
        targets = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, targets)

    def _init_q_proj_if_needed(self, distiller, t_dim, s_dim):
        # If a projector t2s exists we may already be in student dim; keep W_q generic though
        if self.q_proj is None:
            self.q_proj = nn.Linear(s_dim if t_dim == s_dim else t_dim, s_dim)
            # register as part of the Distiller so optimizer picks it up via projector param group or default
            if not hasattr(distiller, "projectors"):
                distiller.projectors = nn.ModuleDict()
            # ModuleDict does not implement setdefault; only add if missing
            if "W_q" not in distiller.projectors:
                distiller.projectors["W_q"] = self.q_proj

    # ---------- forward ----------
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        student = distiller.student_model
        teacher = distiller.teacher_model
        assert teacher is not None, "FKD_A requires a teacher model."

        s_out = student(
            input_data["input_ids"],
            attention_mask=input_data.get("attention_mask", None),
            output_hidden_states=True,
            return_dict=True,
        )
        logits = s_out.logits
        ce = self._ce_loss(logits, output_data["labels"])  # scalar

        with torch.no_grad():
            teacher.eval()
            t_out = teacher(
                input_data["teacher_input_ids"],
                attention_mask=input_data.get("teacher_attention_mask", None),
                output_hidden_states=True,
                return_dict=True,
            )

        hs_s = s_out.hidden_states
        hs_t = t_out.hidden_states
        t_layers = len(hs_t) - 1
        s_layers = len(hs_s) - 1

        # get BI-based top indices and weights
        fkd_info = getattr(distiller, "fkd_info", None)
        if fkd_info is None:
            k = min(self.k, t_layers)
            idx = torch.linspace(0, t_layers - 1, steps=k).round().long().tolist()
            weights = torch.tensor([1.0 / k] * k, device=logits.device)
        else:
            idx = fkd_info["top_indices"]
            bi_scores = torch.tensor(fkd_info["bi_scores"], device=logits.device, dtype=torch.float32)
            sel_scores = bi_scores[idx]
            weights = torch.softmax(sel_scores, dim=0)

        s_idx = self._map_teacher_to_student_layers(idx, t_layers, s_layers)

        is_encoder_student = hasattr(student.config, "is_encoder_decoder") or student.config.model_type in [
            "bert", "roberta", "distilbert"
        ]
        t_mask = input_data.get("teacher_attention_mask", None)
        s_mask = input_data.get("attention_mask", None)

        # ----- Teacher fuse (in teacher space) -----
        vec_t = 0.0
        for w, l in zip(weights, idx):
            ht = hs_t[l + 1]
            pt = self._pooled(ht, t_mask, is_encoder_like=False)
            vec_t = vec_t + w * pt
        # optional projector t2s -> teacher mapped into student space for loss target
        if hasattr(distiller, "projectors") and "t2s" in distiller.projectors:
            vec_t_proj = distiller.projectors["t2s"](vec_t)
        else:
            vec_t_proj = vec_t

        # ----- Build student K/V from pooled layers -----
        pooled_s = []
        for l in s_idx:
            hs_layer = hs_s[l + 1]
            pooled_s.append(self._pooled(hs_layer, s_mask, is_encoder_like=is_encoder_student))
        # [k, B, H] -> [B, k, H]
        M_S = torch.stack(pooled_s, dim=0).transpose(0, 1)
        # keys and values are same here; allow an optional projector if provided
        K = M_S
        V = M_S
        if hasattr(distiller, "projectors") and "s2s_key" in distiller.projectors:
            K = distiller.projectors["s2s_key"](K)

        s_dim = M_S.size(-1)
        t_dim = vec_t.size(-1)
        # ensure W_q maps from teacher_dim -> student_dim
        self._init_q_proj_if_needed(distiller, t_dim, s_dim)
        # apply W_q on teacher-space fused vector (not on already t2s-projected vec)
        q_proj = distiller.projectors["W_q"](vec_t)  # [B, H]

        # ----- Cross-attention like scores -----
        # scores[b, i] = dot(K[b, i, :], q_proj[b, :])
        scores = torch.einsum("bkh,bh->bk", K, q_proj)
        attn = torch.softmax(scores / (s_dim ** 0.5), dim=1)
        # fuse student
        H_S_fused = torch.einsum("bk,bkh->bh", attn, V)

        # Store attention weights for logging: attn shape [B, K]
        # For each batch, store a dict: {student_layer_idx: [attn_w_b1, attn_w_b2, ...]} for all batch samples
        if not hasattr(distiller, 'epoch_student_attn_weights') or distiller.epoch_student_attn_weights is None:
            distiller.epoch_student_attn_weights = []
        # For each student layer, collect all attention weights in this batch
        attn_np = attn.detach().cpu().to(torch.float32).numpy()  # [B, K]
        batch_dict = {}
        for i, s_l in enumerate(s_idx):
            # collect all B attention weights for this student layer
            batch_dict[int(s_l)] = attn_np[:, i].tolist()
        distiller.epoch_student_attn_weights.append(batch_dict)

        # use teacher fused in student space for matching
        H_T_fused = vec_t_proj

        # ----- losses -----
        cos = F.cosine_similarity(H_S_fused, H_T_fused, dim=-1)
        l_distill = (1.0 - cos).mean()
        l_contrast = self._info_nce(H_S_fused, H_T_fused)
        total = self.alpha * ce + self.beta * l_distill + self.gamma * l_contrast
        return total, logging_output
