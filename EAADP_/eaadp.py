import torch
import torch.nn.functional as F
from .various_divergence import VariousDivergence
from .eaadp_modules import EnhancedAttentionSelector, DynamicProjectionLayer, TALFLoss


class EAADP(VariousDivergence):
    """
    loss = alpha * CE(student_logits, labels) + (1-alpha) * CCL(e_syn)

    Where:
      - CIS scores identify top-k informative layers for teacher & student.
      - EAS learns weights over those selected layers (initialized from CIS).
      - DPL projects teacher combined representation into student hidden size with a gating mechanism.
      - TALF (CCL) encourages token-context alignment.
    """
    
    def __init__(self, args):
        super().__init__(args)
        # Hyper-parameters
        self.alpha = args.eaadp_alpha
        self.top_k = args.eaadp_top_k
        self.use_attn_for_cis = getattr(args, "eaadp_cis_use_attn", False)

        # CIS accumulation state
        self._t_sums = None
        self._s_sums = None
        self._t_counts = 0.0
        self._s_counts = 0.0
        self._cis_steps = 0


    def _accumulate_cis_batch(self, hs_tuple, attn_mask, is_teacher: bool, attn_tuple=None):
        # hs_tuple: tuple(length L+1); bỏ 0 vì là embedding
        layer_list = hs_tuple[1:]
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1)  # [B,T,1]

        B = layer_list[0].size(0)
        T = layer_list[0].size(1)
        L = len(layer_list)
        device = layer_list[0].device

        # Init buffers
        if is_teacher and self._t_sums is None:
            self._t_sums = torch.zeros(L, dtype=torch.float64, device=device)
        if (not is_teacher) and self._s_sums is None:
            self._s_sums = torch.zeros(L, dtype=torch.float64, device=device)
        
        for l, h in enumerate(layer_list):
            if self.use_attn_for_cis and attn_tuple is not None:
                # Aggregate attention probs -> contextualized representation proxy
                attn = attn_tuple[l]  
                attn_mean = attn.mean(1)  # [B,T,T]
                # context aggregation via attention to current layer tokens
                context_tokens = attn_mean.matmul(h)  # [B,T,H]
                rep = context_tokens
            else:
                rep = h

            if attn_mask is not None:
                token_sum = (rep * mask).sum(1)
                denom = mask.sum(1).clamp_min(1)
                sent_mean = token_sum / denom
            else:
                sent_mean = rep.mean(1)

            rep_norm = F.normalize(rep, dim=-1)
            sent_norm = F.normalize(sent_mean, dim=-1).unsqueeze(1)
            cos = (rep_norm * sent_norm).sum(-1)

            if attn_mask is not None:
                cos = cos * attn_mask
                valid = attn_mask.sum().double()
            else:
                valid = torch.tensor(B * T, device=device, dtype=torch.float64)
            
            total_cos = cos.sum().double()
            if is_teacher:
                self._t_sums[l] += total_cos
                self._t_counts += valid
            else:
                self._s_sums[l] += total_cos
                self._s_counts += valid


    def _finalize_cis(self, device):
        t_cis = (self._t_sums / self._t_counts.clamp_min(1)).detach().cpu().tolist()
        s_cis = (self._s_sums / self._s_counts.clamp_min(1)).detach().cpu().tolist()
        return t_cis, s_cis


    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model

        # Student forward (with labels -> CE)
        s_out = student_model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True,
            output_attentions=self.use_attn_for_cis,
            labels=output_data["labels"],
        )
        ce_loss = s_out.loss
        logits = s_out.logits

        with torch.no_grad():
            teacher_model.eval()
            t_out = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True,
                output_attentions=self.use_attn_for_cis,
                return_dict=True,
            )

        # After initialization, perform EAS + DPL + TALF
        t_comb, s_comb = distiller.eas(t_out.hidden_states, s_out.hidden_states)
        e_syn = distiller.dpl(
            teacher_repr=t_comb,
            student_repr=s_comb,
            attention_mask=input_data["attention_mask"],
        )
        ccl_loss = distiller.talf(e_syn, input_data["attention_mask"])
        loss = self.alpha * ce_loss + (1 - self.alpha) * ccl_loss

        log = {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "ccl_loss": ccl_loss.detach(),
            "accuracy": self.compute_accuracy(logits, output_data["labels"]),
        }
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output
