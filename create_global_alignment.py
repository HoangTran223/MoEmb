#!/usr/bin/env python3
"""
Offline script to generate global alignment matrix for FKD_FINAL method.
This script should be run before training with FKD_FINAL.
"""

import os
import sys
import argparse
import tempfile
import numpy as np
from transformers import AutoTokenizer

try:
    import fasttext
except ImportError:
    print("Error: fasttext not installed. Install with: pip install fasttext")
    sys.exit(1)

try:
    import ot
except ImportError:
    print("Error: POT not installed. Install with: pip install POT")
    sys.exit(1)


def train_fasttext(tokenizer, fasttext_dim=500, fasttext_epoch=10, 
                   fasttext_minn=3, fasttext_maxn=6, fasttext_min_count=1):
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
        print(f"Training FastText model for vocabulary size: {len(vocab)}")
        model = fasttext.train_unsupervised(
            temp_path,
            model='skipgram',
            dim=fasttext_dim,
            epoch=fasttext_epoch,
            minn=fasttext_minn,
            maxn=fasttext_maxn,
            minCount=fasttext_min_count,
            verbose=0
        )
        return model
    finally:
        # Cleanup temporary file
        os.unlink(temp_path)


def compute_global_alignment(teacher_tokenizer, student_tokenizer, 
                           teacher_fasttext, student_fasttext,
                           ot_reg=0.1, ot_numitermax=5000):
    """Compute global alignment matrix using Optimal Transport."""
    # Get embeddings for all tokens
    teacher_vocab = list(teacher_tokenizer.get_vocab().keys())
    student_vocab = list(student_tokenizer.get_vocab().keys())
    
    print(f"Teacher vocabulary size: {len(teacher_vocab)}")
    print(f"Student vocabulary size: {len(student_vocab)}")
    
    # Get FastText embeddings
    teacher_embeddings = []
    for token in teacher_vocab:
        clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()
        if clean_token:
            teacher_embeddings.append(teacher_fasttext.get_word_vector(clean_token))
        else:
            teacher_embeddings.append(np.zeros(teacher_fasttext.get_dimension()))
            
    student_embeddings = []
    for token in student_vocab:
        clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()
        if clean_token:
            student_embeddings.append(student_fasttext.get_word_vector(clean_token))
        else:
            student_embeddings.append(np.zeros(student_fasttext.get_dimension()))
            
    teacher_embeddings = np.array(teacher_embeddings)
    student_embeddings = np.array(student_embeddings)
    
    print(f"Teacher embeddings shape: {teacher_embeddings.shape}")
    print(f"Student embeddings shape: {student_embeddings.shape}")
    
    # Compute cost matrix (negative cosine similarity)
    teacher_norm = teacher_embeddings / (np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-8)
    student_norm = student_embeddings / (np.linalg.norm(student_embeddings, axis=1, keepdims=True) + 1e-8)
    
    cost_matrix = 1 - np.dot(teacher_norm, student_norm.T)
    
    print(f"Cost matrix shape: {cost_matrix.shape}")
    
    # Solve optimal transport
    teacher_dist = np.ones(len(teacher_vocab)) / len(teacher_vocab)
    student_dist = np.ones(len(student_vocab)) / len(student_vocab)
    
    print("Computing optimal transport alignment...")
    ot_matrix = ot.sinkhorn(
        teacher_dist,
        student_dist, 
        cost_matrix,
        reg=ot_reg,
        numItermax=ot_numitermax
    )
    
    print(f"OT matrix shape: {ot_matrix.shape}")
    return ot_matrix


def main():
    parser = argparse.ArgumentParser(description="Generate global alignment matrix for FKD_FINAL")
    parser.add_argument("--teacher-model", type=str, required=True, 
                       help="Path to teacher model or model name")
    parser.add_argument("--teacher-tokenizer", type=str, default=None,
                       help="(Optional) explicit tokenizer repo/path for teacher (use when teacher arg points to adapter)")
    parser.add_argument("--student-model", type=str, required=True,
                       help="Path to student model or model name")
    parser.add_argument("--student-tokenizer", type=str, default=None,
                       help="(Optional) explicit tokenizer repo/path for student")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save the global alignment matrix")
    parser.add_argument("--fasttext-dim", type=int, default=1000,
                       help="Dimension of FastText embeddings")
    parser.add_argument("--fasttext-epoch", type=int, default=10,
                       help="Number of epochs for FastText training")
    parser.add_argument("--fasttext-minn", type=int, default=3,
                       help="Minimum character n-gram size")
    parser.add_argument("--fasttext-maxn", type=int, default=10,
                       help="Maximum character n-gram size")
    parser.add_argument("--fasttext-min-count", type=int, default=1,
                       help="Minimum count for FastText training")
    parser.add_argument("--ot-reg", type=float, default=0.05,
                       help="Regularization parameter for optimal transport")
    parser.add_argument("--ot-numitermax", type=int, default=5000,
                       help="Maximum number of iterations for OT solver")
    parser.add_argument("--teacher-fasttext-path", type=str, default=None,
                       help="Optional path to save teacher FastText model (.bin)")
    parser.add_argument("--student-fasttext-path", type=str, default=None,
                       help="Optional path to save student FastText model (.bin)")
    
    args = parser.parse_args()
    
    # Load tokenizers with robust fallback for adapter names
    print("Loading tokenizers...")
    # Teacher tokenizer: try explicit override first
    def try_load_tokenizer(name):
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception as e:
            raise e

    # Teacher
    teacher_tokenizer = None
    teacher_try_names = []
    if args.teacher_tokenizer:
        teacher_try_names.append(args.teacher_tokenizer)
    teacher_try_names.append(args.teacher_model)
    # If teacher_model looks like an adapter name with extra suffixes, try progressively shorter hyphen-truncated names
    if '-' in args.teacher_model and args.teacher_model not in teacher_try_names:
        parts = args.teacher_model.split('-')
        for i in range(len(parts) - 1, 0, -1):
            candidate = '-'.join(parts[:i])
            if candidate not in teacher_try_names:
                teacher_try_names.append(candidate)

    last_err = None
    for name in teacher_try_names:
        try:
            teacher_tokenizer = AutoTokenizer.from_pretrained(name)
            print(f"Loaded teacher tokenizer from: {name}")
            break
        except Exception as e:
            last_err = e
            print(f"Failed to load tokenizer '{name}': {e}")
            continue
    if teacher_tokenizer is None:
        raise RuntimeError("Failed to load teacher tokenizer. Provide a valid tokenizer repo/path via --teacher-tokenizer or a model name with tokenizer/config.json.")

    # Student
    if args.student_tokenizer:
        try:
            student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer)
            print(f"Loaded student tokenizer from: {args.student_tokenizer}")
        except Exception as e:
            print(f"Failed to load student tokenizer override '{args.student_tokenizer}': {e}. Falling back to --student-model")
            student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    else:
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    
    # Train FastText models
    print("Training FastText models...")
    teacher_fasttext = train_fasttext(
        teacher_tokenizer,
        fasttext_dim=args.fasttext_dim,
        fasttext_epoch=args.fasttext_epoch,
        fasttext_minn=args.fasttext_minn,
        fasttext_maxn=args.fasttext_maxn,
        fasttext_min_count=args.fasttext_min_count
    )
    if args.teacher_fasttext_path:
        os.makedirs(os.path.dirname(args.teacher_fasttext_path), exist_ok=True)
        print(f"Saving teacher FastText model to {args.teacher_fasttext_path}")
        teacher_fasttext.save_model(args.teacher_fasttext_path)
    
    student_fasttext = train_fasttext(
        student_tokenizer,
        fasttext_dim=args.fasttext_dim,
        fasttext_epoch=args.fasttext_epoch,
        fasttext_minn=args.fasttext_minn,
        fasttext_maxn=args.fasttext_maxn,
        fasttext_min_count=args.fasttext_min_count
    )
    if args.student_fasttext_path:
        os.makedirs(os.path.dirname(args.student_fasttext_path), exist_ok=True)
        print(f"Saving student FastText model to {args.student_fasttext_path}")
        student_fasttext.save_model(args.student_fasttext_path)
    
    # Compute global alignment matrix
    print("Computing global alignment matrix...")
    global_alignment_matrix = compute_global_alignment(
        teacher_tokenizer,
        student_tokenizer,
        teacher_fasttext,
        student_fasttext,
        ot_reg=args.ot_reg,
        ot_numitermax=args.ot_numitermax
    )
    
    # Save alignment matrix
    print(f"Saving global alignment matrix to {args.output_path}")
    np.save(args.output_path, global_alignment_matrix)
    
    print("Done!")


if __name__ == "__main__":
    main()
