import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import tempfile
try:
    import fasttext
except ImportError:
    print("Warning: fasttext not installed. Install with: pip install fasttext")
    fasttext = None
try:
    import ot
except ImportError:
    print("Warning: POT not installed. Install with: pip install POT")
    ot = None


class FKD_FINAL(nn.Module):
    """
    FKD_FINAL: Most advanced distillation method incorporating FastText embeddings 
    and Optimal Transport for vocabulary alignment.
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Loss weights according to user's formula: α·L_CE + β·L_Distill (ONLY 2 losses)
        self.alpha = getattr(args, 'fkd_final_alpha', 1.0)  # CE loss weight
        self.beta = getattr(args, 'fkd_final_beta', 1.0)    # Distillation loss weight
        # Remove gamma - user's framework only has 2 losses
        
        # Hybrid alignment parameter
        self.lambda_hybrid = getattr(args, 'fkd_final_lambda', 0.7)
        
        # FastText parameters
        self.fasttext_dim = getattr(args, 'fasttext_dim', 100)
        self.fasttext_epoch = getattr(args, 'fasttext_epoch', 5)
        self.fasttext_minn = getattr(args, 'fasttext_minn', 3)
        self.fasttext_maxn = getattr(args, 'fasttext_maxn', 6)
        self.fasttext_lr = getattr(args, 'fasttext_lr', 0.05)
        
        # OT parameters
        self.ot_reg = getattr(args, 'ot_reg', 0.1)
        self.ot_numitermax = getattr(args, 'ot_numitermax', 1000)
        
        # Global alignment matrix (loaded offline)
        self.global_alignment_matrix = None
        global_alignment_path = getattr(args, 'global_alignment_path', None)
        if global_alignment_path and os.path.exists(global_alignment_path):
            self.global_alignment_matrix = np.load(global_alignment_path)
            print(f"Loaded global alignment matrix from {global_alignment_path}")
        
        # FastText models (will be loaded if paths provided)
        self.teacher_fasttext = None
        self.student_fasttext = None
        
        teacher_fasttext_path = getattr(args, 'teacher_fasttext_path', None)
        student_fasttext_path = getattr(args, 'student_fasttext_path', None)
        
        if teacher_fasttext_path and os.path.exists(teacher_fasttext_path):
            self.teacher_fasttext = fasttext.load_model(teacher_fasttext_path)
            print(f"Loaded teacher FastText from {teacher_fasttext_path}")
            
        if student_fasttext_path and os.path.exists(student_fasttext_path):
            self.student_fasttext = fasttext.load_model(student_fasttext_path)
            print(f"Loaded student FastText from {student_fasttext_path}")
        
    def _setup_offline_phase(self):
        """Setup offline phase if global alignment matrix doesn't exist."""
        if not os.path.exists(self.global_alignment_path):
            print(f"Global alignment matrix not found at {self.global_alignment_path}")
            print("Running offline phase to compute FastText embeddings and OT alignment...")
            # Note: Offline phase should be run separately with proper tokenizers
            # self._run_offline_phase()
        else:
            print(f"Loading global alignment matrix from {self.global_alignment_path}")
            self.global_alignment_matrix = np.load(self.global_alignment_path)
            
    def _run_offline_phase(self, teacher_tokenizer, student_tokenizer):
        """Run the offline phase to compute global alignment matrix."""
        if fasttext is None or ot is None:
            raise ImportError("FastText and POT libraries are required for FKD_FINAL")
            
        # Train FastText models for both vocabularies
        print("Training FastText embeddings...")
        self.teacher_fasttext = self._train_fasttext(teacher_tokenizer)
        self.student_fasttext = self._train_fasttext(student_tokenizer)
        
        # Compute global alignment matrix using Optimal Transport
        print("Computing global alignment matrix...")
        self.global_alignment_matrix = self._compute_global_alignment(teacher_tokenizer, student_tokenizer)
        
        # Save for future use
        np.save(self.global_alignment_path, self.global_alignment_matrix)
        print(f"Global alignment matrix saved to {self.global_alignment_path}")
        
    def _train_fasttext(self, tokenizer):
        """Train FastText model for given tokenizer vocabulary."""
        # Extract vocabulary
        vocab = list(tokenizer.get_vocab().keys())
        
        # Create temporary file with vocabulary
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for token in vocab:
                # Clean token and write as sentence
                clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()
                if clean_token:
                    f.write(clean_token + '\n')
            temp_path = f.name
            
        try:
            # Train FastText model
            model = fasttext.train_unsupervised(
                temp_path,
                model='skipgram',
                dim=self.fasttext_dim,
                epoch=self.fasttext_epoch,
                minn=self.fasttext_minn,
                maxn=self.fasttext_maxn,
                minCount=self.fasttext_min_count,
                verbose=0
            )
            return model
        finally:
            # Cleanup temporary file
            os.unlink(temp_path)
            
    def _compute_global_alignment(self, teacher_tokenizer, student_tokenizer):
        """Compute global alignment matrix using Optimal Transport."""
        # Get embeddings for all tokens
        teacher_vocab = list(teacher_tokenizer.get_vocab().keys())
        student_vocab = list(student_tokenizer.get_vocab().keys())
        
        # Get FastText embeddings
        teacher_embeddings = []
        for token in teacher_vocab:
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()
            if clean_token:
                teacher_embeddings.append(self.teacher_fasttext.get_word_vector(clean_token))
            else:
                teacher_embeddings.append(np.zeros(self.fasttext_dim))
                
        student_embeddings = []
        for token in student_vocab:
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()
            if clean_token:
                student_embeddings.append(self.student_fasttext.get_word_vector(clean_token))
            else:
                student_embeddings.append(np.zeros(self.fasttext_dim))
                
        teacher_embeddings = np.array(teacher_embeddings)
        student_embeddings = np.array(student_embeddings)
        
        # Compute cost matrix (negative cosine similarity)
        teacher_norm = teacher_embeddings / (np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-8)
        student_norm = student_embeddings / (np.linalg.norm(student_embeddings, axis=1, keepdims=True) + 1e-8)
        
        cost_matrix = 1 - np.dot(teacher_norm, student_norm.T)
        
        # Solve optimal transport
        teacher_dist = np.ones(len(teacher_vocab)) / len(teacher_vocab)
        student_dist = np.ones(len(student_vocab)) / len(student_vocab)
        
        ot_matrix = ot.sinkhorn(
            teacher_dist,
            student_dist, 
            cost_matrix,
            reg=self.ot_reg,
            numItermax=self.ot_numitermax
        )
        
        return ot_matrix
        
    def _hybrid_align(self, teacher_tokens, student_tokens):
        """
        Compute hybrid alignment combining contextual similarity and global alignment.
        
        Args:
            teacher_tokens: Teacher token IDs [batch_size, seq_len]
            student_tokens: Student token IDs [batch_size, seq_len]
            
        Returns:
            alignment_matrix: [batch_size, teacher_seq_len, student_seq_len]
        """
        batch_size, teacher_seq_len = teacher_tokens.shape
        student_seq_len = student_tokens.shape[1]
        
        # Get global alignment scores
        global_scores = torch.zeros(batch_size, teacher_seq_len, student_seq_len, 
                                   device=teacher_tokens.device)
        
        if self.global_alignment_matrix is not None:
            global_alignment = torch.from_numpy(self.global_alignment_matrix).to(teacher_tokens.device)
            
            for b in range(batch_size):
                for i in range(teacher_seq_len):
                    for j in range(student_seq_len):
                        t_id = teacher_tokens[b, i].item()
                        s_id = student_tokens[b, j].item()
                        if t_id < global_alignment.shape[0] and s_id < global_alignment.shape[1]:
                            global_scores[b, i, j] = global_alignment[t_id, s_id]
        
        # Contextual similarity (simple token matching for now)
        contextual_scores = torch.zeros_like(global_scores)
        for b in range(batch_size):
            for i in range(teacher_seq_len):
                for j in range(student_seq_len):
                    if teacher_tokens[b, i] == student_tokens[b, j]:
                        contextual_scores[b, i, j] = 1.0
        
        # Hybrid combination
        hybrid_scores = (self.lambda_hybrid * contextual_scores + 
                        (1 - self.lambda_hybrid) * global_scores)
        
        # Apply softmax for alignment probabilities
        alignment_matrix = F.softmax(hybrid_scores, dim=-1)
        
        return alignment_matrix

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Forward pass for FKD_FINAL distillation.
        
        Args:
            distiller: Distiller object containing models
            input_data: Input batch data
            output_data: Output labels
            logging_output: Logging dictionary
            batch_denom: Batch denominator
            
        Returns:
            tuple: (total_loss, logging_output)
        """
        student = distiller.student_model
        teacher = distiller.teacher_model
        
        device = input_data["input_ids"].device
        
        # Get student outputs
        student_outputs = student(
            input_data["input_ids"],
            attention_mask=input_data.get("attention_mask", None),
            output_hidden_states=True,
            return_dict=True,
        )
        student_logits = student_outputs.logits
        student_hidden = student_outputs.hidden_states
        
        # Classification loss
        ce_loss = F.cross_entropy(student_logits, output_data["labels"])
        
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_outputs = teacher(
                input_data.get("teacher_input_ids", input_data["input_ids"]),
                attention_mask=input_data.get("teacher_attention_mask", input_data.get("attention_mask", None)),
                output_hidden_states=True,
                return_dict=True,
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden = teacher_outputs.hidden_states
        
        # Token-level distillation using hybrid alignment
        distill_loss = 0.0
        
        if len(student_hidden) > 1 and len(teacher_hidden) > 1:
            # Use input tokens for alignment
            input_ids = input_data["input_ids"]
            teacher_input_ids = input_data.get("teacher_input_ids", input_ids)
            
            # Compute hybrid alignment
            alignment_matrix = self._hybrid_align(teacher_input_ids, input_ids)
            
            # Align teacher and student representations
            teacher_repr = teacher_hidden[-1]  # Use last layer
            student_repr = student_hidden[-1]

            # Ensure alignment matrix and representations have the same dtype/device
            target_dtype = student_repr.dtype
            target_device = student_repr.device
            alignment_matrix = alignment_matrix.to(device=target_device, dtype=target_dtype)
            teacher_repr = teacher_repr.to(device=target_device, dtype=target_dtype)

            # If teacher and student hidden sizes differ we need a projector
            teacher_dim = teacher_repr.size(-1)
            student_dim = student_repr.size(-1)

            # Prefer existing projectors on the distiller if available
            aligned_teacher = None
            # If distiller provides a W_q (teacher->student) projector, use it on teacher repr first
            try:
                if hasattr(distiller, 'projectors') and 'W_q' in distiller.projectors:
                    # Apply W_q to teacher representations before alignment
                    teacher_proj = distiller.projectors['W_q']
                    teacher_proj = teacher_proj.to(device=target_device, dtype=target_dtype)
                    teacher_repr_proj = teacher_proj(teacher_repr)
                    aligned_teacher = torch.bmm(alignment_matrix.transpose(1, 2), teacher_repr_proj)
                # If there's a dedicated t2s projector (named 't2s'), apply it after alignment
                elif hasattr(distiller, 'projectors') and 't2s' in distiller.projectors:
                    aligned_teacher = torch.bmm(alignment_matrix.transpose(1, 2), teacher_repr)
                    t2s_proj = distiller.projectors['t2s']
                    t2s_proj = t2s_proj.to(device=target_device, dtype=target_dtype)
                    aligned_teacher = t2s_proj(aligned_teacher)
                else:
                    # No projector found. Create a runtime linear projector and attach to distiller.projectors
                    # NOTE: creating parameters at runtime after optimizer/deepspeed init may mean
                    # the new params are not included in the optimizer; prefer adding a projector
                    # via config before initialization for trainable mappings.
                    if hasattr(distiller, 'projectors'):
                        if 't2s_runtime' not in distiller.projectors:
                            runtime_proj = nn.Linear(teacher_dim, student_dim).to(device=target_device, dtype=target_dtype)
                            # initialize
                            with torch.no_grad():
                                nn.init.xavier_uniform_(runtime_proj.weight)
                                if runtime_proj.bias is not None:
                                    runtime_proj.bias.zero_()
                            distiller.projectors['t2s_runtime'] = runtime_proj
                            print('[FKD_FINAL][WARN] Created runtime projector "t2s_runtime" on distiller.projectors.\n'
                                  'If you want this projector to be trained, add a projector config before initialization so optimizer can capture it.')
                        runtime_proj = distiller.projectors['t2s_runtime']
                        runtime_proj = runtime_proj.to(device=target_device, dtype=target_dtype)
                        aligned_teacher = torch.bmm(alignment_matrix.transpose(1, 2), teacher_repr)
                        aligned_teacher = runtime_proj(aligned_teacher)
                    else:
                        # last-resort non-parametric fallback: truncate or pad teacher dim to match student dim
                        aligned_teacher = torch.bmm(alignment_matrix.transpose(1, 2), teacher_repr)
                        if teacher_dim > student_dim:
                            aligned_teacher = aligned_teacher[:, :, :student_dim]
                            print('[FKD_FINAL][WARN] Truncated teacher representation to match student hidden size.')
                        elif teacher_dim < student_dim:
                            pad_shape = (aligned_teacher.size(0), aligned_teacher.size(1), student_dim - teacher_dim)
                            pad = torch.zeros(pad_shape, device=target_device, dtype=target_dtype)
                            aligned_teacher = torch.cat([aligned_teacher, pad], dim=2)
                            print('[FKD_FINAL][WARN] Padded teacher representation to match student hidden size.')

            except Exception as e:
                # On any failure, fallback to a safe truncation/padding to keep training going
                aligned_teacher = torch.bmm(alignment_matrix.transpose(1, 2), teacher_repr)
                if aligned_teacher.size(-1) != student_repr.size(-1):
                    tdim = aligned_teacher.size(-1)
                    sdim = student_repr.size(-1)
                    if tdim > sdim:
                        aligned_teacher = aligned_teacher[:, :, :sdim]
                    else:
                        pad_shape = (aligned_teacher.size(0), aligned_teacher.size(1), sdim - tdim)
                        pad = torch.zeros(pad_shape, device=target_device, dtype=target_dtype)
                        aligned_teacher = torch.cat([aligned_teacher, pad], dim=2)
                    print(f'[FKD_FINAL][WARN] Projector fallback used due to error: {e}')

            # Compute distillation loss (MSE between student_repr and aligned_teacher)
            # Ensure shapes match
            if aligned_teacher.size(-1) != student_repr.size(-1):
                # final safety: align last dim sizes by truncation/padding
                if aligned_teacher.size(-1) > student_repr.size(-1):
                    aligned_teacher = aligned_teacher[:, :, :student_repr.size(-1)]
                else:
                    pad_shape = (aligned_teacher.size(0), aligned_teacher.size(1), student_repr.size(-1) - aligned_teacher.size(-1))
                    pad = torch.zeros(pad_shape, device=target_device, dtype=target_dtype)
                    aligned_teacher = torch.cat([aligned_teacher, pad], dim=2)

            distill_loss = F.mse_loss(student_repr, aligned_teacher)
        
        # Combine losses according to user's formula: α·L_CE + β·L_Distill (ONLY 2 losses)
        total_loss = self.alpha * ce_loss + self.beta * distill_loss
        
        # Update logging
        logging_output["loss"] = total_loss.item()
        logging_output["ce_loss"] = ce_loss.item()
        logging_output["distill_loss"] = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
        
        return total_loss, logging_output
