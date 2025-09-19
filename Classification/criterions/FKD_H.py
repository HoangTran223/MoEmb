import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import torch.distributed as dist
except ImportError:
    dist = None


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

        self.alpha = getattr(args, 'fkd_h_alpha', None)
        if self.alpha is None:
            self.alpha = getattr(args, 'fkd_final_alpha', 1.0)
        self.beta = getattr(args, 'fkd_h_beta', None)
        if self.beta is None:
            self.beta = getattr(args, 'fkd_final_beta', 1.0)
        print(f"[FKD_H] Loss weights: alpha={self.alpha}, beta={self.beta}")

        # Hybrid mixing coef lambda for global part
        self.lambda_h = getattr(args, 'fkd_h_lambda', None)
        if self.lambda_h is None:
            self.lambda_h = getattr(args, 'fkd_final_lambda', 0.5)

        # Required global alignment matrix path
        self.global_alignment_path = getattr(args, 'global_alignment_path', None)
        if not self.global_alignment_path:
            raise ValueError("[FKD_H] 'global_alignment_path' is required but not provided.")
        if not os.path.exists(self.global_alignment_path):
            raise FileNotFoundError(f"[FKD_H] Global alignment file not found: {self.global_alignment_path}")
        # Prefer memory-mapped or compressed formats to avoid loading the full matrix into RAM
        self._global_align_cpu = None   # Optional fallback (dense torch tensor on CPU)
        self._global_align_np = None    # Primary: memmap-backed array for .npy
        self._global_align_npz = None   # For compressed .npz (uint8/topk)
        self._global_align_meta = None  # dict with format/shape and extras
        try:
            if self.global_alignment_path.endswith('.npz'):
                npz = np.load(self.global_alignment_path, allow_pickle=True)
                # Extract metadata
                if 'format' in npz.files:
                    fmt_raw = npz['format']
                    fmt = str(fmt_raw.item() if hasattr(fmt_raw, 'item') else fmt_raw)
                else:
                    fmt = 'unknown'
                if 'shape' in npz.files:
                    shape_raw = npz['shape']
                    try:
                        shape = tuple(int(x) for x in shape_raw.tolist())
                    except Exception:
                        shape = tuple(int(x) for x in shape_raw)
                else:
                    shape = None
                meta = {'format': fmt, 'shape': shape}

                if fmt == 'uint8':
                    data = npz['data']
                    scales = npz['scales']
                    self._global_align_npz = {'data': data, 'scales': scales}
                    mem_mb = (data.nbytes + scales.nbytes) / (1024**2)
                    print(f"[FKD_H] Loaded compressed alignment uint8 npz with shape {shape}, approx RAM once loaded: {mem_mb:.1f} MB")
                elif fmt == 'topk':
                    inds = npz['indices']
                    vals = npz['values']
                    k = int(npz['k'].item()) if 'k' in npz.files and hasattr(npz['k'], 'item') else (int(npz['k']) if 'k' in npz.files else vals.shape[1])
                    self._global_align_npz = {'indices': inds, 'values': vals, 'k': k}
                    mem_mb = (inds.nbytes + vals.nbytes) / (1024**2)
                    print(f"[FKD_H] Loaded sparse alignment topk npz with shape {shape}, K={k}, approx RAM: {mem_mb:.1f} MB")
                else:
                    raise RuntimeError(f"Unknown .npz alignment format: {fmt}")
                self._global_align_meta = meta
            else:
                # Use memmap for dense .npy to slice on-demand
                mat = np.load(self.global_alignment_path, mmap_mode='r')
                print(f"[FKD_H] Loaded global alignment matrix: {mat.shape} ({mat.dtype}) [memmap]")
                self._global_align_np = mat
                bytes_per_elem = np.dtype(mat.dtype).itemsize
                disk_size_gb = (mat.shape[0] * mat.shape[1] * bytes_per_elem) / (1024**3)
                print(f"[FKD_H] Global alignment on-disk size: {disk_size_gb:.2f} GB; using slice-on-demand")
        except Exception as e:
            raise RuntimeError(
                f"[FKD_H][ERROR] Failed to load global alignment from {self.global_alignment_path}: {e}.\n"
                f"Please generate it first using create_global_alignment_from_embeddings.py."
            )

        # Offline projection path (W_q).
        self.offline_proj_path = getattr(args, 'offline_projection_path', None)
        if not self.offline_proj_path:
            base_dir = os.path.dirname(self.global_alignment_path) or os.getcwd()
            self.offline_proj_path = os.path.join(base_dir, 'W_q.pt')
            # reflect back to args so others can log/use it
            try:
                setattr(self.args, 'offline_projection_path', self.offline_proj_path)
            except Exception:
                pass
            print(f"[FKD_H] 'offline_projection_path' not provided. Defaulting to {self.offline_proj_path}")

    # ---------- helpers ----------
    def _find_token_overlaps(self,
                              teacher_tokenizer,
                              student_tokenizer,
                              teacher_ids,
                              student_ids,
                              teacher_texts=None,
                              student_texts=None,
                              teacher_attention_mask=None,
                              student_attention_mask=None):
        """
        Strict span-based overlap using offset_mapping from both tokenizers.
        Supports BOTH:
          - single-sentence inputs (Banking77, IMDB, Patent single text): each entry is a string
          - sentence-pair inputs: each entry is a (premise, hypothesis) tuple

        Returns a list of length B, each element is a list of (t_idx, s_idx) pairs
        such that the teacher token span intersects the student token span in character space.
        If no strict overlaps are found, falls back to a safe zip-by-order over valid tokens.
        """
        if teacher_texts is None or student_texts is None:
            raise ValueError("[FKD_H] teacher_texts and student_texts are required for span-based overlap.")

        B = teacher_ids.size(0)
        overlaps = []

        for b in range(B):
            # fetch raw texts; handle single-sentence or sentence-pair
            t_text = teacher_texts[b]
            s_text = student_texts[b]
            # normalize types
            t_is_pair = isinstance(t_text, (tuple, list)) and len(t_text) == 2
            s_is_pair = isinstance(s_text, (tuple, list)) and len(s_text) == 2

            if t_is_pair and s_is_pair:
                t_prem, t_hypo = t_text
                s_prem, s_hypo = s_text
                t_enc = teacher_tokenizer(
                    t_prem, t_hypo,
                    add_special_tokens=True,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding=False,
                    return_offsets_mapping=True,
                )
                s_enc = student_tokenizer(
                    s_prem, s_hypo,
                    add_special_tokens=True,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding=False,
                    return_offsets_mapping=True,
                )
            else:
                # Treat as single sentence inputs
                if isinstance(t_text, (tuple, list)):
                    # if tuple accidentally provided, concatenate with a space
                    t_text = " ".join([str(x) for x in t_text])
                if isinstance(s_text, (tuple, list)):
                    s_text = " ".join([str(x) for x in s_text])
                t_enc = teacher_tokenizer(
                    str(t_text),
                    add_special_tokens=True,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding=False,
                    return_offsets_mapping=True,
                )
                s_enc = student_tokenizer(
                    str(s_text),
                    add_special_tokens=True,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding=False,
                    return_offsets_mapping=True,
                )

            # Determine valid token positions (exclude specials) using sequence_ids
            t_seq_ids = t_enc.sequence_ids()
            s_seq_ids = s_enc.sequence_ids()
            t_offsets = t_enc["offset_mapping"]
            s_offsets = s_enc["offset_mapping"]

            # Limit by provided attention masks to ensure index alignment with input ids
            if teacher_attention_mask is not None:
                t_max = int(teacher_attention_mask[b].sum().item())
            else:
                t_max = len(t_seq_ids)
            if student_attention_mask is not None:
                s_max = int(student_attention_mask[b].sum().item())
            else:
                s_max = len(s_seq_ids)

            def collect_valid(seq_ids, offsets, max_len):
                spans = []  # list of (start, end, idx)
                for i in range(min(len(seq_ids), max_len)):
                    sid = seq_ids[i]
                    off = offsets[i]
                    # skip special tokens and padding (offsets like (0,0))
                    if sid is None:
                        continue
                    if off is None:
                        continue
                    start, end = off
                    if start is None or end is None:
                        continue
                    if end <= start:
                        continue
                    spans.append((start, end, i))
                return spans

            t_spans = collect_valid(t_seq_ids, t_offsets, t_max)
            s_spans = collect_valid(s_seq_ids, s_offsets, s_max)

            # Build overlaps by span intersection
            batch_overlaps = []
            ti = 0
            si = 0
            # two-pointer sweep by character start positions
            t_spans_sorted = sorted(t_spans, key=lambda x: (x[0], x[1]))
            s_spans_sorted = sorted(s_spans, key=lambda x: (x[0], x[1]))
            while ti < len(t_spans_sorted) and si < len(s_spans_sorted):
                t_start, t_end, t_idx = t_spans_sorted[ti]
                s_start, s_end, s_idx = s_spans_sorted[si]
                # intersection length > 0?
                inter = min(t_end, s_end) - max(t_start, s_start)
                if inter > 0:
                    batch_overlaps.append((t_idx, s_idx))
                    # advance the one that ends first to find more matches
                    if t_end <= s_end:
                        ti += 1
                    else:
                        si += 1
                else:
                    # advance the earlier-starting token
                    if t_start < s_start:
                        ti += 1
                    else:
                        si += 1

            if not batch_overlaps:
                # Safe fallback: zip by order over valid (non-special) tokens
                N = min(len(t_spans_sorted), len(s_spans_sorted))
                for i in range(N):
                    batch_overlaps.append((t_spans_sorted[i][2], s_spans_sorted[i][2]))

            overlaps.append(batch_overlaps)

        return overlaps

    def _compute_hybrid_alignment(self, s_last, t_proj, teacher_ids, student_ids, teacher_tokenizer, student_tokenizer, overlaps, device, dtype):
        """Sparse hybrid alignment: compute scores only on overlapping (t_idx, s_idx) pairs.

        Returns q_T with shape [B, S, H_S].
        """
        B, S, H_S = s_last.shape
        q_T = torch.zeros_like(s_last)

        # Normalize once
        s_norm = F.normalize(s_last, p=2, dim=-1)
        t_norm = F.normalize(t_proj, p=2, dim=-1)
        lam = float(self.lambda_h)

        for b in range(B):
            batch_overlaps = overlaps[b]
            if not batch_overlaps:
                # if nothing, keep zeros (or copy last teacher state if desired)
                continue

            # group by student index
            s_to_t = {}
            for t_i, s_i in batch_overlaps:
                if 0 <= s_i < S:
                    s_to_t.setdefault(s_i, []).append(t_i)

            if not s_to_t:
                continue

            # cache teacher ids row for global pairs
            t_ids_row = teacher_ids[b]
            s_ids_row = student_ids[b]

            for s_i, t_list in s_to_t.items():
                if len(t_list) == 0:
                    continue
                t_idx = torch.tensor(t_list, device=device, dtype=torch.long)

                # contextual scores for this s_i against selected t's: [K]
                ctx = torch.mv(t_norm[b, t_idx], s_norm[b, s_i])

                # global scores for these (t_ids, s_id): [K]
                gvec = self._global_scores_for_pairs(t_ids_row[t_idx].detach().cpu().numpy(), int(s_ids_row[s_i].item()))
                gvec = torch.from_numpy(gvec).to(device=device, dtype=ctx.dtype)

                scores = (1.0 - lam) * ctx + lam * gvec
                w = torch.softmax(scores, dim=-1)
                q_T[b, s_i] = torch.matmul(w, t_proj[b, t_idx])

        return q_T

    def _global_scores_for_pairs(self, teacher_ids_vec_np: np.ndarray, student_id: int) -> np.ndarray:
        """Return global alignment scores for pairs (teacher_ids_vec, student_id).
        Output shape: [len(teacher_ids_vec)], dtype float16 as numpy.
        """
        GA_np = getattr(self, '_global_align_np', None)
        GA_npz = getattr(self, '_global_align_npz', None)
        if GA_np is not None:
            # memmap dense: advanced indexing rows, single column
            col = np.asarray([student_id], dtype=np.int64)
            sub = GA_np[np.asarray(teacher_ids_vec_np, dtype=np.int64)[:, None], col]  # [K,1]
            return sub.astype(np.float16, copy=False).ravel()
        elif GA_npz is not None:
            fmt = (self._global_align_meta or {}).get('format', 'unknown')
            if fmt == 'uint8':
                data = GA_npz['data']
                scales = GA_npz['scales'].astype(np.float32, copy=False)
                rows = np.asarray(teacher_ids_vec_np, dtype=np.int64)
                u8 = data[rows, student_id]
                sc = scales[rows]
                sub = (u8.astype(np.float32) / 255.0) * sc
                return sub.astype(np.float16, copy=False)
            elif fmt == 'topk':
                inds = GA_npz['indices']  # [V, K]
                vals = GA_npz['values']   # [V, K] (fp16)
                rows = np.asarray(teacher_ids_vec_np, dtype=np.int64)
                out = np.zeros((rows.shape[0],), dtype=np.float16)
                for i, r in enumerate(rows):
                    row_inds = inds[r]
                    row_vals = vals[r]
                    # find position of student_id in row_inds (K is small)
                    # vectorized equality then any
                    match = (row_inds == student_id)
                    if match.any():
                        pos = int(np.argmax(match))
                        out[i] = row_vals[pos]
                return out
            else:
                raise RuntimeError(f"[FKD_H] Unknown compressed alignment format: {fmt}")
        else:
            # No GA available; return zeros
            return np.zeros((teacher_ids_vec_np.shape[0],), dtype=np.float16)
    def _slice_global_scores(self, teacher_ids: torch.Tensor, student_ids: torch.Tensor, device, dtype):
        """Slice the global alignment sub-matrix for given token id sequences.
        Returns [B, T, S] tensor on device.
        
        Memory-optimized version that processes batches to avoid OOM and supports .npy memmap and .npz compressed formats.
        """
        B, T = teacher_ids.shape
        S = student_ids.shape[1]

        # Prefer numpy memmap or compressed if available
        GA_np = getattr(self, '_global_align_np', None)
        GA_npz = getattr(self, '_global_align_npz', None)
        GA = self._global_align_cpu
        if GA_np is None and GA_npz is None and GA is None:
            return torch.full((B, T, S), 1.0 / max(1, S), device=device, dtype=dtype)

        # Ensure index bounds - clamp to vocab dimensions
        if GA_np is not None:
            T_vocab, S_vocab = GA_np.shape[0], GA_np.shape[1]
        elif GA_npz is not None:
            meta = self._global_align_meta or {}
            shape = meta.get('shape')
            if shape is None:
                raise RuntimeError("[FKD_H] Compressed alignment missing shape metadata")
            T_vocab, S_vocab = int(shape[0]), int(shape[1])
        else:
            T_vocab, S_vocab = GA.size(0), GA.size(1)
        teacher_ids_clamped = torch.clamp(teacher_ids.cpu(), 0, T_vocab - 1).to(torch.long)
        student_ids_clamped = torch.clamp(student_ids.cpu(), 0, S_vocab - 1).to(torch.long)

        # Prepare output on CPU first (fp16), move to device at the end
        out = torch.zeros(B, T, S, dtype=torch.float16)

        # Define a provider to fetch small blocks without materializing the full matrix on GPU
        if GA_np is not None:
            def get_block(t_idx_np, s_idx_np):
                sub_np = GA_np[t_idx_np][:, s_idx_np]
                return torch.from_numpy(sub_np.astype(np.float16, copy=False))
        elif GA_npz is not None:
            fmt = (self._global_align_meta or {}).get('format', 'unknown')
            if fmt == 'uint8':
                data = GA_npz['data']
                scales = GA_npz['scales'].astype(np.float32, copy=False)
                def get_block(t_idx_np, s_idx_np):
                    sub_u8 = data[t_idx_np][:, s_idx_np]
                    sub = (sub_u8.astype(np.float32) / 255.0) * scales[t_idx_np][:, None]
                    return torch.from_numpy(sub.astype(np.float16))
            elif fmt == 'topk':
                inds = GA_npz['indices']
                vals = GA_npz['values']  # float16
                def get_block(t_idx_np, s_idx_np):
                    t_list = t_idx_np.tolist()
                    s_list = s_idx_np.tolist()
                    s_pos = {sid: j for j, sid in enumerate(s_list)}
                    block = np.zeros((len(t_list), len(s_list)), dtype=np.float16)
                    for row_i, tv in enumerate(t_list):
                        row_inds = inds[tv]
                        row_vals = vals[tv]
                        for k_i in range(row_vals.shape[0]):
                            col = int(row_inds[k_i])
                            pos = s_pos.get(col)
                            if pos is not None:
                                block[row_i, pos] = row_vals[k_i]
                    return torch.from_numpy(block)
            else:
                raise RuntimeError(f"[FKD_H] Unknown compressed alignment format: {fmt}")
        else:
            def get_block(t_idx_np, s_idx_np):
                sub = GA[torch.from_numpy(t_idx_np).long()][:, torch.from_numpy(s_idx_np).long()]
                return sub.to(torch.float16)

        # Process in smaller chunks to avoid OOM
        max_batch_size = 2
        for batch_start in range(0, B, max_batch_size):
            batch_end = min(batch_start + max_batch_size, B)
            batch_size = batch_end - batch_start
            t_ids_batch = teacher_ids_clamped[batch_start:batch_end]  # [b, T]
            s_ids_batch = student_ids_clamped[batch_start:batch_end]  # [b, S]
            for i in range(batch_size):
                rows = t_ids_batch[i].numpy()
                cols = s_ids_batch[i].numpy()
                submat = get_block(rows, cols)
                out[batch_start + i] = submat if submat.dtype == torch.float16 else submat.to(torch.float16)

        # Cast to requested dtype/device
        if dtype == torch.bfloat16:
            return out.bfloat16().to(device)
        elif dtype == torch.float16:
            return out.to(device)
        else:
            return out.float().to(device)

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
            
            # Cache key for projection loading to avoid repeated file I/O
            cache_key = f"wq_loaded_{self.offline_proj_path}"
            if not hasattr(self, cache_key):
                setattr(self, cache_key, False)
            
            # Try to load from offline path (only once per session)
            if self.offline_proj_path and not getattr(self, cache_key):
                if os.path.exists(self.offline_proj_path):
                    try:
                        state = torch.load(self.offline_proj_path, map_location='cpu')
                        # Handle different formats
                        if 'weight' in state and 'bias' in state:
                            # Standard state_dict format
                            distiller.projectors['W_q'].load_state_dict(state)
                        elif 'meta' in state:
                            # Our custom format with metadata
                            proj_state = {k: v for k, v in state.items() if k != 'meta'}
                            distiller.projectors['W_q'].load_state_dict(proj_state)
                        else:
                            # Direct weight matrix
                            distiller.projectors['W_q'].weight.data = state['weight'] if 'weight' in state else state
                            if 'bias' in state and distiller.projectors['W_q'].bias is not None:
                                distiller.projectors['W_q'].bias.data = state['bias']
                        print(f"[FKD_H] Loaded offline projection from {self.offline_proj_path}")
                        setattr(self, cache_key, True)  # Mark as loaded
                    except Exception as e:
                        print(f"[FKD_H][WARN] Failed loading offline projection: {e}")
                else:
                    # Save the freshly-initialized weights so the path exists for future runs
                    try:
                        import torch.distributed as dist
                        torch.save(distiller.projectors['W_q'].state_dict(), self.offline_proj_path)
                        if dist.is_available() and dist.is_initialized():
                            if dist.get_rank() == 0:
                                print(f"[FKD_H] Saved initial W_q to {self.offline_proj_path}")
                        else:
                            print(f"[FKD_H] Saved initial W_q to {self.offline_proj_path}")
                    except Exception as e:
                        print(f"[FKD_H][WARN] Could not save initial W_q to {self.offline_proj_path}: {e}")
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
        mapped_student_layers là index thực của student_hs (không cộng +1, không lấy embedding).
        Nếu mapping rỗng, trả về last layer [B,S,1,H].
        """
        last = student_hs[-1]
        if not mapped_student_layers or len(mapped_student_layers) == 0:
            return last.unsqueeze(2)  # [B, S, 1, H]
        tensors = []
        num_layers = len(student_hs)
        for idx in mapped_student_layers:
            try:
                idx = int(idx)
            except Exception:
                continue
            # Bỏ qua embedding (idx=0), chỉ lấy hidden state thực sự
            if idx < 1 or idx >= num_layers:
                continue
            t = student_hs[idx]
            assert t.dim() == 3, f"student_hs[{idx}] shape: {t.shape}, expected [B,S,H]"
            tensors.append(t)
        if len(tensors) == 0:
            # Nếu mapping không hợp lệ, fallback về last layer
            return last.unsqueeze(2)
        # Đảm bảo mọi tensor đều cùng shape
        h_sizes = [t.shape[-1] for t in tensors]
        assert all(h == h_sizes[0] for h in h_sizes), f"All student hidden states must have same hidden size, got: {h_sizes}"
        stacked = torch.stack(tensors, dim=2)  # [B, S, M, H]
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

        # Token id tensors
        teacher_ids = input_data.get("teacher_input_ids", input_data["input_ids"])  # [B,T]
        student_ids = input_data["input_ids"]  # [B,S]

        # Find token overlaps using strict span policy
        teacher_tokenizer = distiller.teacher_tokenizers
        student_tokenizer = distiller.student_tokenizer
        overlaps = input_data.get("overlaps")
        if overlaps is None:
            overlaps = self._find_token_overlaps(
                teacher_tokenizer,
                student_tokenizer,
                teacher_ids,
                student_ids,
                teacher_texts=input_data.get("text", None),
                student_texts=input_data.get("text", None),
                teacher_attention_mask=input_data.get("teacher_attention_mask", None),
                student_attention_mask=input_data.get("attention_mask", None),
            )

        # Student fusion over mapped layers with attention; compute per-layer Aj,i,k using H_S^{(k)}[j]
        mapped_student = fkd_info.get('mapped_student_layers', []) or []
        M_S = self._student_stack(s_hs, mapped_student)  # [B,S,M,H]
        B, S, M, Hs = M_S.shape
        # Compute per-layer teacher aggregates q_T_m using spans and contextual similarities from each student layer k
        q_T_list = []
        for m in range(M):
            s_layer = M_S[:, :, m, :]  # [B,S,H]
            q_T_m = self._compute_hybrid_alignment(
                s_layer, t_proj, teacher_ids, student_ids,
                teacher_tokenizer,
                student_tokenizer,
                overlaps, device, dtype
            )  # [B,S,H]
            q_T_list.append(q_T_m)
        # Stack to [B,S,M,H]
        if M == 1:
            Q_T = q_T_list[0].unsqueeze(2)
        else:
            Q_T = torch.stack(q_T_list, dim=2)

        # Attention scores per layer using similarity between M_S[:,:,m,:] and corresponding Q_T[:,:,m,:]
        # scores[b,s,m] = dot(M_S[b,s,m], Q_T[b,s,m]) / sqrt(H)
        scores = (M_S * Q_T).sum(dim=-1) / math.sqrt(Hs)  # [B,S,M]
        att_w = torch.softmax(scores, dim=-1)  # [B,S,M]
        # Fuse student and teacher sides with same attention
        h_tilde = torch.einsum('bsm,bsmh->bsh', att_w, M_S)  # [B,S,H]
        q_tilde = torch.einsum('bsm,bsmh->bsh', att_w, Q_T)  # [B,S,H]

        # Distillation loss: 1 - mean cosine over valid student tokens between fused reps
        s_mask = input_data.get("attention_mask", None)
        if s_mask is not None:
            s_mask = (s_mask > 0).to(device=device, dtype=dtype)
            cos = F.cosine_similarity(h_tilde, q_tilde, dim=-1) * s_mask  # [B,S]
            denom = s_mask.sum().clamp(min=1.0)
            distill = 1.0 - (cos.sum() / denom)
        else:
            cos = F.cosine_similarity(h_tilde, q_tilde, dim=-1)
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
