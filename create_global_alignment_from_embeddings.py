#!/usr/bin/env python3
"""
Create global alignment matrix and an initial teacher->student projection W
directly from model embedding layers (no FastText), matching the FKD_H spec:

1) Load teacher and student tokenizers and models to get input embedding
   matrices E_T [|V_T|, H_T] and E_S [|V_S|, H_S].
2) Find overlap tokens by exact string match; extract E_T^overlap, E_S^overlap.
3) Whiten E_T^overlap and E_S^overlap; solve ridge-regularized regression to
   learn W (H_S x H_T) that maps teacher space -> student space.
4) Compute projected teacher embeddings for all tokens: E_T_proj = E_T @ W^T.
5) Build cost matrix C = 1 - cosine(E_T_proj, E_S) with shape (|V_T|, |V_S|).
6) Solve OT (Sinkhorn) with uniform marginals to get M_global [|V_T|, |V_S|].
7) Save M_global as .npy and W as a torch state_dict (.pt) suitable for nn.Linear.

Note: This can be memory intensive for large models. Use --teacher-vocab-max or
--student-vocab-max to limit vocab sizes for quick tests.
"""

import os
import json
import argparse
import math
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

try:
    import ot  # POT
except Exception as e:
    raise RuntimeError("POT (python Optimal Transport) is required: pip install POT") from e

from peft import PeftModel


def _load_embeddings(model_name: str, adapter_path: str = None, torch_dtype: torch.dtype = torch.float16) -> Tuple[np.ndarray, List[str]]:
    """Load input embedding matrix and corresponding vocab tokens.
    If adapter_path is provided, load and merge adapters.
    Returns (embeddings [V, H] float32, vocab_tokens list[str]).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    if adapter_path and os.path.exists(adapter_path):
        print(f"[FKD_H][Offline] Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    with torch.no_grad():
        emb = model.get_input_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    # tokenizer.get_vocab() returns token->id dict; invert to id->token
    vocab_dict = tokenizer.get_vocab()
    id_to_token = [None] * len(vocab_dict)
    for tok, idx in vocab_dict.items():
        if idx < len(id_to_token):
            id_to_token[idx] = tok
    # Some tokenizers might have holes; fill with placeholder
    for i, v in enumerate(id_to_token):
        if v is None:
            id_to_token[i] = f"<unk_{i}>"
    return emb, id_to_token


def _whiten(X: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whiten rows of X (n x d) with GPU acceleration when available.

    Returns (X_hat, mean, inv_sqrt_cov) in float32 for numerical stability.
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        mean = Xt.mean(dim=0, keepdim=True)
        Xc = Xt - mean
        n = max(1, Xc.shape[0] - 1)
        cov = (Xc.transpose(0, 1) @ Xc) / float(n)
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=eps)
        inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.transpose(0, 1)
        X_hat = Xc @ inv_sqrt
        return (
            X_hat.detach().cpu().numpy().astype(np.float32, copy=False),
            mean.detach().cpu().numpy().astype(np.float32, copy=False),
            inv_sqrt.detach().cpu().numpy().astype(np.float32, copy=False),
        )
    except Exception:
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
        s, U = np.linalg.eigh(cov)
        s_clamped = np.clip(s, a_min=eps, a_max=None)
        inv_sqrt = (U @ np.diag(1.0 / np.sqrt(s_clamped)) @ U.T).astype(np.float32)
        X_hat = (Xc @ inv_sqrt).astype(np.float32)
        return X_hat, X.mean(axis=0, keepdims=True).astype(np.float32), inv_sqrt


def _ridge_t2s(X_t: np.ndarray, Y_s: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Solve ridge regression Y_s ~= X_t @ W^T for W in R^{H_S x H_T}.
    Uses GPU when available; returns float32 W (H_S x H_T).
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Xt = torch.as_tensor(X_t, dtype=torch.float32, device=device)
        Ys = torch.as_tensor(Y_s, dtype=torch.float32, device=device)
        XtX = Xt.transpose(0, 1) @ Xt  # [d_t, d_t]
        d_t = XtX.shape[0]
        A = XtX + lam * torch.eye(d_t, dtype=torch.float32, device=device)
        XtY = Xt.transpose(0, 1) @ Ys  # [d_t, d_s]
        W_T = torch.linalg.solve(A, XtY)
        W = W_T.transpose(0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
        return W
    except Exception:
        XtX = X_t.T @ X_t
        H_T = XtX.shape[0]
        reg = lam * np.eye(H_T, dtype=np.float32)
        A = XtX.astype(np.float32) + reg
        XtY = (X_t.T @ Y_s).astype(np.float32)
        W_T = np.linalg.solve(A, XtY)
        W = W_T.T.astype(np.float32)
        return W


def _cosine_matrix(A: np.ndarray, B: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Compute cosine similarity matrix with batching.

    Optimized: use PyTorch on GPU (float32 compute) when available, fallback to NumPy.
    Returns float16 matrix for compactness; caller can cast.
    """
    n, m = A.shape[0], B.shape[0]
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            At = torch.as_tensor(A, dtype=torch.float32, device=device)
            Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
            An = At / (At.norm(dim=1, keepdim=True) + 1e-8)
            Bn = Bt / (Bt.norm(dim=1, keepdim=True) + 1e-8)

            result = np.zeros((n, m), dtype=np.float16)
            row_bs = max(1, batch)
            col_bs = 8192 if m >= 8192 else m
            total_blocks = ((n - 1) // row_bs + 1) * ((m - 1) // col_bs + 1)
            pbar = tqdm(total=total_blocks, desc="Computing cosine similarity (GPU)", unit="blk")
            for i in range(0, n, row_bs):
                ie = min(i + row_bs, n)
                block = An[i:ie]
                for j in range(0, m, col_bs):
                    je = min(j + col_bs, m)
                    sim_block = block @ Bn[j:je].transpose(0, 1)
                    result[i:ie, j:je] = sim_block.to('cpu', dtype=torch.float16).numpy()
                    pbar.update(1)
            pbar.close()
            return result
    except Exception:
        pass

    # CPU fallback (NumPy)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    result = np.zeros((n, m), dtype=np.float16)
    pbar = tqdm(range(0, n, batch), desc="Computing cosine similarity (CPU)", unit="blk")
    for i in pbar:
        ie = min(i + batch, n)
        result[i:ie] = (A_norm[i:ie] @ B_norm.T).astype(np.float16)
    pbar.close()
    return result

def _project_embeddings(teacher_emb: np.ndarray, W: np.ndarray, row_bs: int = 8192) -> np.ndarray:
    """Project teacher embeddings using W.T with GPU acceleration when available.

    Computes: teacher_proj = teacher_emb @ W.T
    - teacher_emb: [V_T, d_t] float16/float32 numpy
    - W: [d_s, d_t] float16/float32 numpy (will upcast to float32 for compute)
    Returns: [V_T, d_s] float16 numpy
    """
    V_T, d_t = teacher_emb.shape
    d_s, d_t_w = W.shape
    assert d_t == d_t_w, "Dimension mismatch between teacher_emb and W"

    result = np.empty((V_T, d_s), dtype=np.float16)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        W_tT = torch.as_tensor(W, dtype=torch.float32, device=device).transpose(0, 1)
        with torch.no_grad():
            for i in tqdm(range(0, V_T, row_bs), desc="Projecting teacher embeddings", unit="blk"):
                ie = min(i + row_bs, V_T)
                block = torch.as_tensor(teacher_emb[i:ie], dtype=torch.float32, device=device)
                out = block @ W_tT
                result[i:ie] = out.to('cpu', dtype=torch.float16).numpy()
        return result
    except Exception:
        W_tT_np = W.astype(np.float32).T
        for i in tqdm(range(0, V_T, row_bs), desc="Projecting teacher embeddings (CPU)", unit="blk"):
            ie = min(i + row_bs, V_T)
            out = teacher_emb[i:ie].astype(np.float32) @ W_tT_np
            result[i:ie] = out.astype(np.float16)
        return result

def _sinkhorn(C: np.ndarray, reg: float = 0.1, max_iter: int = 1000, tol: float = 1e-7) -> np.ndarray:
    """Sinkhorn algorithm for optimal transport with POT when available, else numpy fallback."""
    n, m = C.shape
    C = np.nan_to_num(C, nan=1.0, posinf=2.0, neginf=0.0).astype(np.float32, copy=False)
    a = np.ones(n, dtype=np.float32) / float(n)
    b = np.ones(m, dtype=np.float32) / float(m)
    if 'ot' in globals() and ot is not None:
        try:
            P = ot.bregman.sinkhorn(a, b, C, reg, numItermax=max_iter, stopThr=tol, verbose=False)
            return P.astype(np.float16)
        except Exception:
            pass
    # Numpy fallback
    K = np.exp(-C / reg).astype(np.float32)
    u = np.ones(n, dtype=np.float32) / float(n)
    b_f = np.ones(m, dtype=np.float32) / float(m)
    for _ in range(max_iter):
        denom = (K.T @ u) + 1e-12
        v = b_f / denom
        u = 1.0 / (K @ v + 1e-12)
        u = np.clip(u, 1e-20, 1e20)
        v = np.clip(v, 1e-20, 1e20)
        # optional early stop omitted for simplicity
    v = b_f / (K.T @ u + 1e-12)
    P = np.diag(u) @ K @ np.diag(v)
    return P.astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description="Global alignment and W from embeddings (no FastText)")
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--teacher-adapter-path", default=None, help="Optional path to teacher adapter (LoRA) to merge")
    ap.add_argument("--student-adapter-path", default=None, help="Optional path to student adapter (LoRA) to merge")
    ap.add_argument("--output-path", required=True, help="Path to save global alignment (.npy or .npz)")
    ap.add_argument("--save-projection-path", required=True, help="Path to save W_q state_dict .pt")
    ap.add_argument("--ridge-lambda", type=float, default=1e-3)
    ap.add_argument("--teacher-vocab-max", type=int, default=None, help="Optional cap on teacher vocab size")
    ap.add_argument("--student-vocab-max", type=int, default=None, help="Optional cap on student vocab size")
    # Parity with DSKD: sinkhorn reg and compressed save formats
    ap.add_argument("--sinkhorn-reg", type=float, default=0.1, help="Sinkhorn regularization parameter")
    ap.add_argument("--save-format", type=str, default="npy", choices=["npy", "fp16", "uint8", "topk"],
                    help="How to store the alignment matrix: raw npy (float32), fp16 npy, quantized uint8 .npz, or top-k sparse .npz")
    ap.add_argument("--topk", type=int, default=64,
                    help="Top-K per row for 'topk' save-format. Set 0 or negative to use all columns (maximum K).")
    ap.add_argument("--quantize-bits", type=int, default=8,
                    help="Bits for quantization when save-format is 'uint8' (currently only 8 supported)")
    args = ap.parse_args()

    print("[FKD_H][Offline] Loading embeddings...")
    E_T, toks_T = _load_embeddings(args.teacher_model, args.teacher_adapter_path)
    E_S, toks_S = _load_embeddings(args.student_model, args.student_adapter_path)

    if args.teacher_vocab_max is not None:
        E_T = E_T[: args.teacher_vocab_max]
        toks_T = toks_T[: args.teacher_vocab_max]
    if args.student_vocab_max is not None:
        E_S = E_S[: args.student_vocab_max]
        toks_S = toks_S[: args.student_vocab_max]

    print(f"Teacher emb: {E_T.shape}, Student emb: {E_S.shape}")

    # Build token->id maps for overlap by exact string
    t2id: Dict[str, int] = {t: i for i, t in enumerate(toks_T)}
    s2id: Dict[str, int] = {t: i for i, t in enumerate(toks_S)}
    overlap_tokens = [t for t in toks_T if t in s2id]
    print(f"[FKD_H][Offline] Overlap tokens: {len(overlap_tokens)} |V_T|={len(toks_T)} |V_S|={len(toks_S)}")

    if len(overlap_tokens) < 10:
        print("[WARN] Very small overlap; W may be poor. Proceeding anyway.")

    # Gather overlap matrices
    idx_T = np.array([t2id[t] for t in overlap_tokens], dtype=np.int32)
    idx_S = np.array([s2id[t] for t in overlap_tokens], dtype=np.int32)
    E_To = E_T[idx_T]
    E_So = E_S[idx_S]

    # Whiten both sides (use overlap rows on both sides)
    E_To_w, mu_T, invsqrt_T = _whiten(E_To)
    E_So_w, mu_S, invsqrt_S = _whiten(E_So)

    # Sanity check shapes match on n (overlap count)
    if E_To_w.shape[0] != E_So_w.shape[0]:
        raise RuntimeError(
            f"[FKD_H][Offline] Overlap row mismatch: teacher {E_To_w.shape} vs student {E_So_w.shape}"
        )

    # Learn ridge W (H_S x H_T)
    print("[FKD_H][Offline] Solving for W (ridge)...")
    W = _ridge_t2s(E_To_w, E_So_w, lam=args.ridge_lambda)

    # Save W as state_dict compatible with nn.Linear(H_T -> H_S)
    os.makedirs(os.path.dirname(args.save_projection_path), exist_ok=True)
    torch.save({
        "weight": torch.from_numpy(W),
        "bias": torch.zeros((W.shape[0],), dtype=torch.float32),
        "meta": {
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "ridge_lambda": args.ridge_lambda,
            "overlap": len(overlap_tokens),
            "V_T": len(toks_T),
            "V_S": len(toks_S),
        }
    }, args.save_projection_path)
    print(f"[FKD_H][Offline] Saved W to {args.save_projection_path} with shape {W.shape}")

    # Compute projected teacher embeddings for all vocabulary tokens (batched, GPU-accelerated when available)
    E_T_proj = _project_embeddings(E_T, W)

    # Build cosine similarity matrix and OT plan
    print("[FKD_H][Offline] Building cosine matrix and solving OT...")
    cos_TS = _cosine_matrix(E_T_proj.astype(np.float32), E_S.astype(np.float32))  # (|V_T|, |V_S|)
    cost = (1.0 - cos_TS).astype(np.float32)
    reg = float(getattr(args, "sinkhorn_reg", 0.1))
    P = _sinkhorn(cost, reg)
    # Save in requested format (parity with DSKD)
    n_rows, n_cols = P.shape
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_fmt = getattr(args, "save_format", "npy")
    if save_fmt == "npy":
        out_path = args.output_path if args.output_path.endswith('.npy') else args.output_path + '.npy'
        print(f"[FKD_H][Offline] Saving float32 npy to {out_path}")
        np.save(out_path, P.astype(np.float32, copy=False))
    elif save_fmt == "fp16":
        out_path = args.output_path if args.output_path.endswith('.npy') else args.output_path + '.npy'
        print(f"[FKD_H][Offline] Saving float16 npy to {out_path}")
        np.save(out_path, P.astype(np.float16, copy=False))
    elif save_fmt == "uint8":
        assert int(getattr(args, 'quantize_bits', 8)) == 8, "Only 8-bit quantization supported"
        print("[FKD_H][Offline] Quantizing rows to uint8 with per-row scales ...")
        row_max = P.max(axis=1).astype(np.float16)
        safe_scale = np.where(row_max > 1e-12, row_max, 1.0).astype(np.float16)
        data_u8 = (P / safe_scale[:, None] * 255.0).clip(0, 255).round().astype(np.uint8)
        out_path = args.output_path if args.output_path.endswith('.npz') else args.output_path + '.npz'
        np.savez(out_path,
                 format=np.array(["uint8"], dtype=object),
                 data=data_u8,
                 scales=safe_scale.astype(np.float16),
                 shape=np.array([n_rows, n_cols], dtype=np.int32))
        del data_u8
        print(f"[FKD_H][Offline] Saved uint8 + scales to {out_path}")
    elif save_fmt == "topk":
        K_req = int(getattr(args, 'topk', 64))
        K = n_cols if K_req <= 0 else min(K_req, n_cols)
        print(f"[FKD_H][Offline] Extracting top-{K} per row (of {n_cols}) ...")
        inds = np.empty((n_rows, K), dtype=np.int32)
        vals = np.empty((n_rows, K), dtype=np.float16)
        batch = 2048
        for i in range(0, n_rows, batch):
            ie = min(i + batch, n_rows)
            block = P[i:ie]
            part = np.argpartition(block, -K, axis=1)[:, -K:]
            v = np.take_along_axis(block, part, axis=1)
            order = np.argsort(-v, axis=1)
            part_sorted = np.take_along_axis(part, order, axis=1)
            v_sorted = np.take_along_axis(v, order, axis=1)
            inds[i:ie] = part_sorted.astype(np.int32)
            vals[i:ie] = v_sorted.astype(np.float16)
        out_path = args.output_path if args.output_path.endswith('.npz') else args.output_path + '.npz'
        np.savez(out_path,
                 format=np.array(["topk"], dtype=object),
                 indices=inds,
                 values=vals,
                 k=np.array([K], dtype=np.int32),
                 shape=np.array([n_rows, n_cols], dtype=np.int32))
        del inds, vals
        print(f"[FKD_H][Offline] Saved top-{K} sparse .npz to {out_path}")
    else:
        raise ValueError(f"Unknown save-format: {save_fmt}")

    print(f"[FKD_H][Offline] Saved global alignment with shape {P.shape}")


if __name__ == "__main__":
    main()
