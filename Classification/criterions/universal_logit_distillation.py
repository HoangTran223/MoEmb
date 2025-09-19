import os
import torch
import torch.nn as nn
from .cross_entropy_loss import CrossEntropyLoss

class UniversalLogitDistillation(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        # Student forward pass
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        
        # Compute cross-entropy loss with ground-truth labels
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["labels"]
        )[0]

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute distillation loss
        kd_loss, log = self.compute_universal_logit_distillation_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("uld_loss:", kd_loss)
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(
            logits, output_data["labels"]
        )
        log["accuracy"] = accuracy

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss , logging_output

    def compute_universal_logit_distillation_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        student_logits = outputs.logits  # [batch_size, num_labels]
        # Teacher backbone doesn't output logits; build teacher logits via classifier head
        last_hidden = teacher_outputs.hidden_states[-1]
        # Mistral (decoder-only) -> take last token representation
        teacher_pooled = last_hidden[:, -1, :]
        device = teacher_pooled.device

        # reuse cached classifier if available
        teacher_classifier = getattr(distiller, 'teacher_classifier', None)
        if teacher_classifier is None:
            ckpt_dir = getattr(distiller.args, 'teacher_model_path', None)
            if ckpt_dir:
                clf_path = os.path.join(ckpt_dir, 'classifier.pt')
                if os.path.exists(clf_path):
                    loaded = torch.load(clf_path, map_location=device)
                    if isinstance(loaded, nn.Module):
                        teacher_classifier = loaded
                    else:
                        in_features = teacher_pooled.size(-1)
                        out_features = student_logits.size(-1)
                        head = nn.Linear(in_features, out_features, device=device, dtype=teacher_pooled.dtype)
                        head.load_state_dict(loaded)
                        teacher_classifier = head
                    distiller.teacher_classifier = teacher_classifier.to(device).eval()
        if teacher_classifier is None:
            raise RuntimeError("Teacher classifier head not found. Ensure classifier.pt exists in teacher_model_path.")

        with torch.no_grad():
            teacher_logits = teacher_classifier(teacher_pooled)

    # Handle potential mismatch in number of classes (should match num_labels)
        vocab_size_gap = student_logits.shape[-1] - teacher_logits.shape[-1]
        if vocab_size_gap > 0:
            # Pad teacher logits with zeros if student has more classes
            teacher_logits = torch.cat(
                [teacher_logits, torch.zeros_like(student_logits[:, :vocab_size_gap])], 
                dim=-1
            )
        elif vocab_size_gap < 0:
            # Pad student logits with zeros if teacher has more classes
            student_logits = torch.cat(
                [student_logits, torch.zeros_like(teacher_logits[:, :(-vocab_size_gap)])], 
                dim=-1
            )

        # Compute softened probabilities
        student_probs = torch.softmax(student_logits, dim=-1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, dim=-1, dtype=torch.float32)

        # Universal Logit Distillation loss: absolute difference between sorted probabilities
        sorted_student_probs = student_probs.sort(dim=-1, descending=True).values
        sorted_teacher_probs = teacher_probs.sort(dim=-1, descending=True).values
        
        # Compute loss as mean absolute difference across the batch
        uld_loss = (sorted_student_probs - sorted_teacher_probs).abs().mean()
        log["uld_loss"] = uld_loss
        
        return uld_loss, log
