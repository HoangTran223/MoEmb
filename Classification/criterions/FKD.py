import torch
import torch.nn as nn
import torch.nn.functional as F


class FKD(nn.Module):
    """
    FKD: Focused Knowledge Distillation
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

    def _ce_loss(self, logits, target):
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        if self.label_smoothing > 0:
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        return loss

    
    def _pooled(self, hs, attn_mask, is_encoder_like=True):
        """ Rút gọn chuỗi hidden states thành 1 vector cho mỗi sample trong batch
        - Với encoder-like (e.g., BERT): sử dụng [CLS] token.
        - Với decoder-like (e.g., Mistral): sử dụng hidden state của token cuối cùng ko phải padding trong mỗi sequence
        - hs[batch_size, seq_length, hidden_dim]
        - attn_mask[batch_size, seq_length] \in {0, 1}
        """
        if is_encoder_like:
            return hs[:, 0, :]
        # decoder-like: pick last valid token per sample
        if attn_mask is None:
            return hs[:, -1, :]
        lengths = attn_mask.long().sum(dim=-1) - 1  # last index
        batch_indices = torch.arange(hs.size(0), device=hs.device)
        return hs[batch_indices, lengths.clamp(min=0), :]

    def _map_teacher_to_student_layers(self, t_idx_list, t_layers, s_layers):
        """Ánh xạ index layer của teacher sang student theo depth ratio"""
        mapped = []
        for l in t_idx_list:
            ratio = (l + 1) / float(t_layers)
            s_l = max(0, min(s_layers - 1, int(round(ratio * s_layers)) - 1))
            mapped.append(s_l)
        return mapped

    def _info_nce(self, q, k):
        """InfoNCE with in-batch negatives. q ~ student, k ~ teacher.
        logits = q @ k.T / T, target = diag indices.
        """
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = torch.matmul(q, k.transpose(0, 1)) / self.ctemp
        targets = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, targets)

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        student = distiller.student_model
        teacher = distiller.teacher_model

        outputs = student(
            input_data["input_ids"],
            attention_mask=input_data.get("attention_mask", None),
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        ce = self._ce_loss(logits, output_data["labels"]) 

        with torch.no_grad():
            teacher.eval()
            t_out = teacher(
                input_data["teacher_input_ids"],
                attention_mask=input_data.get("teacher_attention_mask", None),
                output_hidden_states=True,
                return_dict=True,
            )

        hs_s = outputs.hidden_states  
        hs_t = t_out.hidden_states   
        t_layers = len(hs_t) - 1
        s_layers = len(hs_s) - 1

        # get focus indices & weights from distiller (set during pre-pass)
        fkd_info = getattr(distiller, "fkd_info", None)
        idx = fkd_info["top_indices"]
        bi_scores = torch.tensor(fkd_info["bi_scores"], device=logits.device, dtype=torch.float32)
        sel_scores = bi_scores[idx]
        weights = torch.softmax(sel_scores, dim=0)

        s_idx = self._map_teacher_to_student_layers(idx, t_layers, s_layers)

        # fused teacher and student
        is_encoder_student = hasattr(student.config, "is_encoder_decoder") or student.config.model_type in ["bert", "roberta", "distilbert"]
        t_mask = input_data.get("teacher_attention_mask", None)
        s_mask = input_data.get("attention_mask", None)

        # Teacher fused then project to student space
        vec_t = 0.0
        for w, l in zip(weights, idx):
            # Lấy hidden state, pool thành vector
            ht = hs_t[l + 1]  
            pt = self._pooled(ht, t_mask, is_encoder_like=False)
            vec_t = vec_t + w * pt
        if hasattr(distiller, "projectors") and "t2s" in distiller.projectors:
            vec_t = distiller.projectors["t2s"](vec_t)

        vec_s = 0.0
        for w, l in zip(weights, s_idx):
            hs = hs_s[l + 1]
            ps = self._pooled(hs, s_mask, is_encoder_like=is_encoder_student)
            vec_s = vec_s + w * ps

        cos = F.cosine_similarity(vec_s, vec_t, dim=-1)
        l_distill = (1.0 - cos).mean()

        # Student vector gần teacher vector đúng của mình, xa teacher vector của các samples khác
        l_contrast = self._info_nce(vec_s, vec_t)

        total = self.alpha * ce + self.beta * l_distill + self.gamma * l_contrast
        return total, logging_output
