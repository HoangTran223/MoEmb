#!/usr/bin/env python3
"""
Generate Global Alignment Matrix for FKD_FINAL using FastText + Optimal Transport

This script implements the offline phase of FKD_FINAL:
1. Extract vocabularies from teacher and student tokenizers
2. Train FastText models on both vocabularies  
3. Compute alignment matrix using Optimal Transport
4. Save alignment matrix and FastText models for use during training

Based on the theoretical framework provided by the user.
"""

import argparse
import os
import numpy as np
import fasttext
import ot
from transformers import AutoTokenizer
import tempfile
import shutil


def extract_vocabulary(tokenizer_path, output_file):
    """Extract vocabulary from tokenizer and save to text file."""
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    vocab = tokenizer.get_vocab()
    vocab_list = [token for token, _ in sorted(vocab.items(), key=lambda x: x[1])]
    
    print(f"Extracted vocabulary with {len(vocab_list)} tokens")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in vocab_list:
            # Handle special tokens and ensure each token is on its own line
            clean_token = token.replace('\n', '\\n').replace('\r', '\\r')
            f.write(f"{clean_token}\n")
    
    return vocab_list


def train_fasttext_model(vocab_file, output_path, dim=100, epoch=5, minn=3, maxn=6):
    """Train FastText model on vocabulary."""
    print(f"Training FastText model: dim={dim}, epoch={epoch}, minn={minn}, maxn={maxn}")
    
    # Train FastText model
    model = fasttext.train_unsupervised(
        vocab_file,
        model='cbow',  # or 'skipgram'
        dim=dim,
        epoch=epoch,
        minn=minn,
        maxn=maxn,
        lr=0.05,
        thread=4
    )
    
    # Save model
    model.save_model(output_path)
    print(f"FastText model saved to {output_path}")
    
    return model


def compute_cost_matrix(teacher_fasttext, student_fasttext, teacher_vocab, student_vocab):
    """Compute cost matrix between teacher and student vocabularies using FastText embeddings."""
    print("Computing cost matrix using FastText embeddings...")
    
    teacher_embeddings = []
    student_embeddings = []
    
    # Get embeddings for teacher vocabulary
    for token in teacher_vocab:
        try:
            emb = teacher_fasttext.get_word_vector(token)
            teacher_embeddings.append(emb)
        except:
            # If token not found, use zero vector
            emb = np.zeros(teacher_fasttext.get_dimension())
            teacher_embeddings.append(emb)
    
    # Get embeddings for student vocabulary  
    for token in student_vocab:
        try:
            emb = student_fasttext.get_word_vector(token)
            student_embeddings.append(emb)
        except:
            # If token not found, use zero vector
            emb = np.zeros(student_fasttext.get_dimension())
            student_embeddings.append(emb)
    
    teacher_embeddings = np.array(teacher_embeddings)
    student_embeddings = np.array(student_embeddings)
    
    print(f"Teacher embeddings shape: {teacher_embeddings.shape}")
    print(f"Student embeddings shape: {student_embeddings.shape}")
    
    # Compute cosine distance matrix
    # Normalize embeddings
    teacher_norm = teacher_embeddings / np.linalg.norm(teacher_embeddings, axis=1, keepdims=True)
    student_norm = student_embeddings / np.linalg.norm(student_embeddings, axis=1, keepdims=True)
    
    # Cosine similarity matrix
    similarity_matrix = np.dot(teacher_norm, student_norm.T)
    
    # Convert to distance (cost) matrix
    cost_matrix = 1.0 - similarity_matrix
    
    # Ensure non-negative costs
    cost_matrix = np.maximum(cost_matrix, 0.0)
    
    return cost_matrix


def compute_optimal_transport(cost_matrix, reg=0.1, numitermax=1000):
    """Compute optimal transport alignment matrix."""
    print(f"Computing optimal transport alignment with reg={reg}, max_iters={numitermax}")
    
    # Uniform distributions
    teacher_dist = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
    student_dist = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]
    
    # Compute optimal transport matrix using Sinkhorn algorithm
    alignment_matrix = ot.sinkhorn(
        teacher_dist,
        student_dist,
        cost_matrix,
        reg=reg,
        numItermax=numitermax
    )
    
    print(f"Optimal transport matrix shape: {alignment_matrix.shape}")
    print(f"Matrix sum: {alignment_matrix.sum():.6f} (should be close to 1.0)")
    
    return alignment_matrix


def main():
    parser = argparse.ArgumentParser(description="Generate global alignment matrix for FKD_FINAL")
    parser.add_argument("--teacher-vocab-path", type=str, required=True,
                       help="Path to teacher model/tokenizer")
    parser.add_argument("--student-vocab-path", type=str, required=True,
                       help="Path to student model/tokenizer")
    parser.add_argument("--fasttext-dim", type=int, default=100,
                       help="FastText embedding dimension")
    parser.add_argument("--fasttext-epoch", type=int, default=5,
                       help="FastText training epochs")
    parser.add_argument("--fasttext-minn", type=int, default=3,
                       help="FastText min character n-gram")
    parser.add_argument("--fasttext-maxn", type=int, default=6,
                       help="FastText max character n-gram")
    parser.add_argument("--ot-reg", type=float, default=0.1,
                       help="Optimal transport regularization")
    parser.add_argument("--ot-numitermax", type=int, default=1000,
                       help="Optimal transport max iterations")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save global alignment matrix")
    parser.add_argument("--teacher-fasttext-path", type=str, required=True,
                       help="Path to save teacher FastText model")
    parser.add_argument("--student-fasttext-path", type=str, required=True,
                       help="Path to save student FastText model")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.teacher_fasttext_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.student_fasttext_path), exist_ok=True)
    
    # Create temporary directory for vocabulary files
    with tempfile.TemporaryDirectory() as temp_dir:
        teacher_vocab_file = os.path.join(temp_dir, "teacher_vocab.txt")
        student_vocab_file = os.path.join(temp_dir, "student_vocab.txt")
        
        # Extract vocabularies
        print("=== Step 1: Extracting vocabularies ===")
        teacher_vocab = extract_vocabulary(args.teacher_vocab_path, teacher_vocab_file)
        student_vocab = extract_vocabulary(args.student_vocab_path, student_vocab_file)
        
        # Train FastText models
        print("=== Step 2: Training FastText models ===")
        teacher_fasttext = train_fasttext_model(
            teacher_vocab_file,
            args.teacher_fasttext_path,
            dim=args.fasttext_dim,
            epoch=args.fasttext_epoch,
            minn=args.fasttext_minn,
            maxn=args.fasttext_maxn
        )
        
        student_fasttext = train_fasttext_model(
            student_vocab_file,
            args.student_fasttext_path,
            dim=args.fasttext_dim,
            epoch=args.fasttext_epoch,
            minn=args.fasttext_minn,
            maxn=args.fasttext_maxn
        )
        
        # Compute cost matrix
        print("=== Step 3: Computing cost matrix ===")
        cost_matrix = compute_cost_matrix(
            teacher_fasttext, student_fasttext,
            teacher_vocab, student_vocab
        )
        
        # Compute optimal transport alignment
        print("=== Step 4: Computing optimal transport alignment ===")
        alignment_matrix = compute_optimal_transport(
            cost_matrix,
            reg=args.ot_reg,
            numitermax=args.ot_numitermax
        )
        
        # Save alignment matrix
        print("=== Step 5: Saving alignment matrix ===")
        np.save(args.output_path, alignment_matrix)
        print(f"Global alignment matrix saved to {args.output_path}")
        
        print("=== Generation completed successfully! ===")
        print(f"Teacher vocab size: {len(teacher_vocab)}")
        print(f"Student vocab size: {len(student_vocab)}")
        print(f"Alignment matrix shape: {alignment_matrix.shape}")
        print(f"Teacher FastText model: {args.teacher_fasttext_path}")
        print(f"Student FastText model: {args.student_fasttext_path}")
        print(f"Global alignment matrix: {args.output_path}")


if __name__ == "__main__":
    main()
