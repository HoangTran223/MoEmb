import torch
from .various_divergence import VariousDivergence

class DualSpaceKDWithCMA(VariousDivergence):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        # Ý tưởng tổng quát:
        # 1. Tính loss CE chuẩn trên logits của student.
        # 2. Lấy hidden layer cuối (token [CLS]) của teacher & student.
        # 3. Dựng attention chéo (CMA) thông qua query từ student embedding, key từ teacher embedding.
        # 4. Chiếu hai chiều:
        #    - Teacher -> Student (t2s): lấy align softmax * value (teacher mapped sang student qua projector t2s)
        #    - Student -> Teacher (s2t): align^T softmax * value (student mapped sang teacher qua projector s2t)
        # 5. Tính các loss: CE(t2s_logits), KD(student_logits, t2s_logits), KL(s2t_logits, teacher_logits)
        # 6. Tổng hợp với kd_rate.
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        
        # Cross-entropy loss với ground-truth
        loss = self.compute_cross_entropy_loss(outputs.logits, output_data["labels"])[0]
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # KD hai không gian + attention chéo
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        print("dskd_cma_loss:", kd_loss)
        # Trộn CE và KD
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Độ chính xác
        accuracy = self.compute_accuracy(logits, output_data["labels"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output
    
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # Nhãn ground-truth
        target = output_data["labels"]
        
        # Lấy [CLS] representation (hàng 0)
        hiddens = outputs.hidden_states[-1][:, 0, :]
        teacher_hiddens = teacher_outputs.hidden_states[-1][:, 0, :]

        # Lấy bảng embedding student
        if hasattr(distiller.student_model, "get_input_embeddings"):
            stu_embed_tokens = distiller.student_model.get_input_embeddings()
        elif hasattr(distiller.student_model, "bert") and hasattr(distiller.student_model.bert, "embeddings"):
            stu_embed_tokens = distiller.student_model.bert.embeddings.word_embeddings
        elif hasattr(distiller.student_model, "model") and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError("Unsupported student model architecture for embedding extraction")

        # Lấy embedding teacher
        teacher_model = distiller.teacher_model
        if hasattr(teacher_model, "get_input_embeddings"):
            tea_embed_tokens = teacher_model.get_input_embeddings()
        elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.embed_tokens
        elif hasattr(teacher_model, "bert") and hasattr(teacher_model.bert, "embeddings"):
            tea_embed_tokens = teacher_model.bert.embeddings.word_embeddings
        else:
            raise NotImplementedError("Unsupported teacher model architecture for embedding extraction")

        # Lấy embedding token đầu tiên (giả sử là [CLS] / BOS)
        stu_input_embeds = stu_embed_tokens(input_data["input_ids"][:, 0]).detach()
        tea_input_embeds = tea_embed_tokens(input_data["teacher_input_ids"][:, 0]).detach()

        # Chuẩn hoá teacher
        norm_tea_input_embeds = tea_input_embeds / tea_input_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        # Projectors (định nghĩa trong Distiller.projectors)
        stu_q_hiddens = distiller.projectors["query"](stu_input_embeds).float()      # Q: student side
        tea_k_hiddens = norm_tea_input_embeds.float()                                # K: teacher side

        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()                 # V_s (đưa sang teacher)
        tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens).float()    # V_t (đưa sang student)

        # Ma trận canh chỉnh (alignment scores)
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / (hiddens.shape[-1] ** 0.5)  # scale

        # Teacher → Student
        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)

        if hasattr(distiller.student_model, "classifier"):
            t2s_logits = distiller.student_model.classifier(t2s_hiddens)
        elif hasattr(distiller.student_model, "score"):
            t2s_logits = distiller.student_model.score(t2s_hiddens)
        else:
            raise AttributeError("Student model has neither 'classifier' nor 'score' attribute")

        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_kd_loss = self.dist_func(outputs.logits, t2s_logits.detach(), target, reduction="mean")

        # Student → Teacher
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)

        if hasattr(distiller.teacher_model, "classifier"):
            s2t_logits = distiller.teacher_model.classifier(s2t_hiddens)
        elif hasattr(distiller.teacher_model, "score"):
            s2t_logits = distiller.teacher_model.score(s2t_hiddens)
        else:
            raise AttributeError("Teacher model has neither 'classifier' nor 'score' attribute")

        s2t_kd_loss = self.compute_forward_kl_divergence(s2t_logits, teacher_outputs.logits, target, reduction="mean")

        # Tổng KD loss (cộng các thành phần)
        kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss

        # Accuracy phụ trợ
        t2s_acc = (t2s_logits.argmax(-1) == target).float().mean()
        s2t_acc = (s2t_logits.argmax(-1) == target).float().mean()

        # Ghi log
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_acc"] = t2s_acc
        log["s2t_acc"] = s2t_acc
        log["kd_loss"] = kd_loss

        return kd_loss, log
