import torch
import torch.nn as nn
import torch.nn.functional as F

class FKD_DT(nn.Module):
    """
    Focused Knowledge Distillation with Differentiated Tokenization (FKD_DT).
    Implements the 3-stage distillation process described by the user.
    NOTE: The PCA-MIM stage is complex and requires character-level spans,
    which are assumed to be provided by the data pipeline. This implementation
    uses a simplified alignment as a placeholder.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Stage 1 params (handled in distillation.py pre-pass)
        self.k = getattr(args, "fkd_k", 4)
        
        # Stage 2: PCA-MIM params (placeholders for now)
        self.pca_mim_sigma = getattr(args, "pca_mim_sigma", 5.0)
        self.pca_mim_tau = getattr(args, "pca_mim_tau", 0.1)
        self.pca_mim_iters = getattr(args, "pca_mim_iters", 5)

        # Stage 3 params
        self.alpha = getattr(args, "fkd_alpha", 1.0)
        self.beta = getattr(args, "fkd_beta", 1.0)
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)

    def _ce_loss(self, logits, target):
        # Standard cross-entropy loss with optional label smoothing
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        nll = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        if self.label_smoothing > 0:
            smooth = -lprobs.mean(dim=-1).mean()
            return (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        return nll

    def _map_teacher_to_student_layers(self, t_idx_list, t_layers, s_layers):
        # Maps teacher layer indices to student layer indices based on relative depth
        mapped = []
        T = int(t_layers)
        S = int(s_layers)
        for l in t_idx_list:
            if S is None or T is None or S <= 0 or T <= 0:
                mapped.append(None)
                continue
            s_l = int(((int(l) + 1) * S - 1) // max(1, T))
            s_l = max(0, min(S - 1, s_l))
            mapped.append(s_l)
        return mapped

    def _init_pca_projector_if_needed(self, distiller, t_dim, s_dim, device, dtype):
        # Ensure the projection matrix W_q exists on the right device/dtype
        if 'W_q' not in distiller.projectors:
            distiller.projectors['W_q'] = nn.Linear(t_dim, s_dim, bias=False).to(device=device, dtype=dtype)
        else:
            distiller.projectors['W_q'] = distiller.projectors['W_q'].to(device=device, dtype=dtype)

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        student = distiller.student_model
        teacher = distiller.teacher_model
        assert teacher is not None, "FKD_DT requires a teacher model."

        # 1. Get model outputs
        s_out = student(
            input_data["input_ids"],
            attention_mask=input_data.get("attention_mask", None),
            output_hidden_states=True,
            return_dict=True,
        )
        logits = s_out.logits
        l_ce = self._ce_loss(logits, output_data["labels"])

        with torch.no_grad():
            teacher.eval()
            t_out = teacher(
                input_data["teacher_input_ids"],
                attention_mask=input_data.get("teacher_attention_mask", None),
                output_hidden_states=True,
                return_dict=True,
            )

        hs_s_all = s_out.hidden_states
        hs_t_all = t_out.hidden_states
        t_layers_total = len(hs_t_all) - 1
        s_layers_total = len(hs_s_all) - 1

        # 2. Stage 1: Layer Selection (info from pre-pass)
        fkd_info = getattr(distiller, "fkd_info", None)
        assert fkd_info is not None, "FKD_DT requires a pre-computed fkd_info from the BI pre-pass."
        
        t_idx = fkd_info["top_indices"]
        bi_scores = torch.tensor(fkd_info["bi_scores"], device=logits.device, dtype=torch.float32)
        sel_scores = bi_scores[t_idx]
        teacher_weights = torch.softmax(sel_scores, dim=0)
        
        s_idx = self._map_teacher_to_student_layers(t_idx, t_layers_total, s_layers_total)

        # Fuse teacher hidden states from important layers
        H_T_fused = 0.0
        for i, l in enumerate(t_idx):
            H_T_fused += teacher_weights[i] * hs_t_all[l + 1]

        # 3. Stage 2: PCA-MIM (Simplified Placeholder)
        t_dim = distiller.teacher_hidden_size
        s_dim = distiller.hidden_size
        # Prefer using configured t2s projector if present; else create W_q
        if 't2s' in distiller.projectors:
            proj = distiller.projectors['t2s'].to(device=H_T_fused.device, dtype=H_T_fused.dtype)
        else:
            self._init_pca_projector_if_needed(
                distiller, t_dim, s_dim,
                device=H_T_fused.device, dtype=H_T_fused.dtype
            )
            proj = distiller.projectors['W_q']

        H_T_fused_proj = proj(H_T_fused) # B x T_len x S_dim

        # Simplified alignment: use interpolation as a stand-in for the complex PCA-MIM.
        # This aligns the teacher sequence length to the student's.
        q_tilde_T = F.interpolate(
            H_T_fused_proj.transpose(1, 2), 
            size=hs_s_all[0].shape[1]
        ).transpose(1, 2)

        # 4. Stage 3: Student Training
        # Stack student hidden states from corresponding important layers
        M_S = torch.stack([hs_s_all[l + 1] for l in s_idx], dim=1) # B x K x S_len x S_dim

        # Compute attention scores using aligned teacher representation as query
        q_prime_T = q_tilde_T.unsqueeze(1) # B x 1 x S_len x S_dim
        
        scores = (M_S * q_prime_T).sum(-1) / (s_dim ** 0.5) # B x K x S_len
        attn_w = F.softmax(scores, dim=1) # B x K x S_len
        
        # Fuse student representations based on attention weights
        h_tilde_S = (attn_w.unsqueeze(-1) * M_S).sum(dim=1) # B x S_len x S_dim

        # Distillation Loss (token-level cosine similarity)
        cos_sim = F.cosine_similarity(h_tilde_S, q_tilde_T, dim=-1) # B x S_len
        
        s_mask = input_data.get("attention_mask", None)
        if s_mask is None:
            s_mask = torch.ones_like(cos_sim)
            
        l_distill = 1.0 - (cos_sim * s_mask).sum() / (s_mask.sum().clamp(min=1e-8))

        # Final Objective Function
        total_loss = self.alpha * l_ce + self.beta * l_distill
        
        # Log student attention weights for analysis, similar to FKD_A
        if not hasattr(distiller, 'epoch_student_attn_weights'):
            distiller.epoch_student_attn_weights = []
        
        # Log the mean attention weight for each student layer across the batch
        mean_attn_w = attn_w.mean(dim=[0, 2]).detach().to(torch.float32).cpu().numpy() # K
        batch_dict = {}
        for i, s_l in enumerate(s_idx):
            batch_dict[int(s_l)] = float(mean_attn_w[i])
        distiller.epoch_student_attn_weights.append(batch_dict)

        return total_loss, logging_output
