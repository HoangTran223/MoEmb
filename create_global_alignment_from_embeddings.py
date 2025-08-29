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
    """Whiten rows of X (n x d). Returns (X_hat, mean, inv_sqrt_cov)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    # Eigendecomposition
    s, U = np.linalg.eigh(cov)
    s_clamped = np.clip(s, a_min=eps, a_max=None)
    inv_sqrt = (U @ np.diag(1.0 / np.sqrt(s_clamped)) @ U.T).astype(np.float32)
    X_hat = (Xc @ inv_sqrt).astype(np.float32)
    return X_hat, X.mean(axis=0, keepdims=True).astype(np.float32), inv_sqrt


def _ridge_t2s(X_t: np.ndarray, Y_s: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Solve ridge regression Y_s ~= X_t @ W^T for W in R^{H_S x H_T}.
    Closed form: W^T = (X^T X + lam I)^{-1} X^T Y
    """
    # shapes: X (n, H_T), Y (n, H_S)
    XtX = X_t.T @ X_t  # (H_T, H_T)
    H_T = XtX.shape[0]
    reg = lam * np.eye(H_T, dtype=np.float32)
    A = XtX + reg
    XtY = X_t.T @ Y_s  # (H_T, H_S)
    # Solve A @ W_T = XtY for W_T
    W_T = np.linalg.solve(A, XtY)  # (H_T, H_S)
    W = W_T.T.astype(np.float32)   # (H_S, H_T)
    return W


def _cosine_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8, batch: int = 4096) -> np.ndarray:
    """Compute cosine similarity between rows of A (n x d) and B (m x d) with batching.
    Returns (n x m) float32.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    n, m = A.shape[0], B.shape[0]
    out = np.empty((n, m), dtype=np.float32)
    bs = max(1, batch)
    for i in range(0, n, bs):
        j = min(n, i + bs)
        out[i:j] = A_norm[i:j] @ B_norm.T
    return out


def main():
    ap = argparse.ArgumentParser(description="Global alignment and W from embeddings (no FastText)")
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--teacher-adapter-path", default=None, help="Optional path to teacher adapter (LoRA) to merge")
    ap.add_argument("--student-adapter-path", default=None, help="Optional path to student adapter (LoRA) to merge")
    ap.add_argument("--output-path", required=True, help="Path to save global alignment .npy (shape |V_T| x |V_S|)")
    ap.add_argument("--save-projection-path", required=True, help="Path to save W_q state_dict .pt")
    ap.add_argument("--ridge-lambda", type=float, default=1e-3)
    ap.add_argument("--teacher-vocab-max", type=int, default=None, help="Optional cap on teacher vocab size")
    ap.add_argument("--student-vocab-max", type=int, default=None, help="Optional cap on student vocab size")
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
    idx_T = np.array([t2id[t] for t in overlap_tokens], dtype=np.int64)
    idx_S = np.array([s2id[t] for t in overlap_tokens], dtype=np.int64)
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

    # Compute projected teacher embeddings for all vocabulary tokens
    E_T_proj = E_T @ W.T  # (|V_T|, H_S)

    # Build cosine similarity matrix and OT plan
    print("[FKD_H][Offline] Building cosine matrix and solving OT...")
    cos_TS = _cosine_matrix(E_T_proj.astype(np.float32), E_S.astype(np.float32))  # (|V_T|, |V_S|)
    cost = (1.0 - cos_TS).astype(np.float32)
    a = np.ones(cost.shape[0], dtype=np.float64) / float(cost.shape[0])
    b = np.ones(cost.shape[1], dtype=np.float64) / float(cost.shape[1])
    P = ot.sinkhorn(a, b, cost, reg=0.1, numItermax=2000)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output_path, P.astype(np.float32))
    print(f"[FKD_H][Offline] Saved global alignment to {args.output_path} with shape {P.shape}")


if __name__ == "__main__":
    main()
