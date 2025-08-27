# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import deepspeed
import numpy as np


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-path', type=str, help='model path')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--model-type", type=str, default="gpt2")
    group.add_argument("--teacher-model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--teacher-model-path", type=str)
    group.add_argument("--teacher-model-fp16", action="store_true")
    group.add_argument("--model-parallel", action="store_true")
    group.add_argument("--model-parallel-size", type=int, default=None)
    group.add_argument("--no-value", action="store_true")
    group.add_argument("--dropout-path-rate", type=float, default=None)
    group.add_argument("--fp32", action="store_true")
    group.add_argument("--model-dtype", type=str, default="fp16")
    # Add num_labels for classification tasks
    group.add_argument("--num-labels", type=int, default=2, help="Number of labels for classification tasks")
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--task", type=str, default=None)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save-dir', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument('--save-interval', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--save-additional-suffix", type=str, default="")
    group.add_argument("--save-rollout", action="store_true")
    group.add_argument("--eb-sample-times", type=int, default=3)
    group.add_argument("--keep-best-n-checkpoints", type=int, default=3)
    group.add_argument("--criterion", type=str, default="cross_entropy")
    group.add_argument("--eval-tqdm", action="store_true")
    group.add_argument("--report-logits", action="store_true")
    group.add_argument("--only-save-projector", action="store_true")
    group.add_argument("--debug", action="store_true")
    # Thêm các tham số mới cho evaluation
    group.add_argument("--checkpoint-path", type=str, default=None,
                     help='Đường dẫn đến checkpoint cần đánh giá')
    group.add_argument("--overwrite-results", action="store_true",
                     help='Ghi đè kết quả evaluation nếu đã tồn tại')
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--gen-num", type=int, default=-1)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--min-prompt-length", type=int, default=128)
    group.add_argument("--json-data", action="store_true")
    group.add_argument("--bin-data", action="store_true")
    group.add_argument("--txt-data", action="store_true")
    
    group.add_argument("--prompt-data-dir", type=str)
    group.add_argument("--pretrain-data-dir", type=str)
    group.add_argument("--eval-ppl", action="store_true")
    group.add_argument("--eval-rw", action="store_true")
    group.add_argument("--eval-gen", action="store_true")
    
    group.add_argument("--only-prompt", action="store_true")
    return parser


def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--total-iters', type=int, default=None,
                       help='total number of iterations')
    group.add_argument('--train-iters-per-epoch', type=int, default=-1,
                       help='total number of iterations per epoch')
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--seed-ppo", type=int, default=42)
    group.add_argument("--seed-lm", type=int, default=7)
    group.add_argument('--num-epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    group.add_argument('--training-epochs', type=int, default=10000)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--attn-dtype", default=None)
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument("--lr-min", type=float, default=0.0000001)
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')
    group.add_argument("--kd-rate", type=float, default=0.5)
    group.add_argument("--kd-temperature", type=float, default=1.0)
    group.add_argument("--kd-objective", type=str, default="forward_kl")
    group.add_argument("--teacher-temperature", type=float, default=1.0)
    group.add_argument("--label-smoothing", type=float, default=0.0)
    group.add_argument("--adaptive-kl-alpha", type=float, default=0.5)
    group.add_argument("--skew-lambda", type=float, default=0.1)

    # FKD hyperparameters
    group.add_argument("--fkd-k", type=int, default=4, help="Top-k teacher layers to focus on")
    group.add_argument("--fkd-alpha", type=float, default=1.0, help="Weight for CE loss")
    group.add_argument("--fkd-beta", type=float, default=1.0, help="Weight for distillation loss (e.g., 1-cos)")
    group.add_argument("--fkd-gamma", type=float, default=0.1, help="Weight for contrastive loss (used in FKD_A)")
    group.add_argument("--fkd-contrastive-temp", type=float, default=0.07, help="Temperature for InfoNCE (used in FKD_A)")
    group.add_argument("--fkd-calib-max-batches", type=int, default=0, help="Limit number of batches for BI pre-pass (0 = all)")

    # FKD_DT removed

    # FKD_FINAL / FKD_H specific hyperparameters
    group.add_argument("--fkd-final-lambda", type=float, default=0.7, help="Weight for contextual score in hybrid alignment")
    group.add_argument("--fasttext-min-count", type=int, default=1, help="Minimum count for FastText training")
    group.add_argument("--fasttext-dim", type=int, default=100, help="Dimension of FastText embeddings")
    group.add_argument("--fasttext-epoch", type=int, default=5, help="Number of epochs for FastText training")
    group.add_argument("--fasttext-minn", type=int, default=3, help="Minimum character n-gram size for FastText")
    group.add_argument("--fasttext-maxn", type=int, default=6, help="Maximum character n-gram size for FastText")
    group.add_argument("--ot-reg", type=float, default=0.1, help="Regularization parameter for optimal transport")
    group.add_argument("--ot-numitermax", type=int, default=1000, help="Maximum number of iterations for OT solver")
    group.add_argument("--global-alignment-path", type=str, default="global_alignment.npy", help="Path to save/load global alignment matrix")
    group.add_argument("--teacher-fasttext-path", type=str, default=None, help="Path to pre-trained teacher FastText model")
    group.add_argument("--student-fasttext-path", type=str, default=None, help="Path to pre-trained student FastText model")
    group.add_argument("--fkd-final-alpha", type=float, default=1.0, help="Weight for CE loss in FKD_FINAL")
    group.add_argument("--fkd-final-beta", type=float, default=1.0, help="Weight for distillation loss in FKD_FINAL")
    # FKD_H naming (aliases fall back to *_final above when unset)
    group.add_argument("--fkd-h-alpha", type=float, default=None, help="Weight for CE loss in FKD_H")
    group.add_argument("--fkd-h-beta", type=float, default=None, help="Weight for distillation loss in FKD_H")
    group.add_argument("--fkd-h-lambda", type=float, default=None, help="Hybrid lambda for global scores in FKD_H")
    group.add_argument("--fkd-h-align-topk", type=int, default=32, help="Top-k teacher tokens per student token in FKD_H")
    group.add_argument("--offline-projection-path", type=str, default=None, help="Optional path to a teacher->student projection state_dict for initializing W_q in FKD_H")

    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--num-epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument("--scheduler-name", type=str, default="constant_trm")


    # EAADP args removed per FKD method focus
    return parser


def add_gen_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)
    group.add_argument("--eval-gen-repeat-times", type=int, default=3)
    
    return parser


def add_peft_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--peft", type=str, default=None)
    group.add_argument("--peft-lora-r", type=int, default=16)
    group.add_argument("--peft-lora-alpha", type=int, default=64)
    group.add_argument("--peft-lora-dropout", type=float, default=0.1)
    group.add_argument("--peft-name", type=str, default=None)
    group.add_argument("--peft-path", type=str, default=None)
    group.add_argument("--teacher-peft-name", type=str, default=None)
    group.add_argument("--teacher-peft-path", type=str, default=None)
    return parser

def add_distiller_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("distiller", "distiller configurations")
    group.add_argument("--projector-config-path", type=str, default=None,
                       help='path to projector_config.json')
    group.add_argument("--projector-path", type=str, default=None,
                       help='path to pretrained projector')
    group.add_argument("--projector-lr", type=float, default=0.001,
                       help='learning rate only for projection')
    group.add_argument("--pretrained-projector", type=str, default=None,
                       help='pretrained projector name')
    group.add_argument("--pretrained-projector-lr", type=float, default=0.001,
                       help='learning rate only for pretrained projector')
    group.add_argument("--vocab-alignment-path", type=str, default=None,
                       help='path for the vocab alignment file')
    group.add_argument("--teacher-to-student-token-mapping", type=str, default=None,
                       help='path for the vocab alignment file (token, teacher-to-student)')
    group.add_argument("--teacher-to-student-id-mapping", type=str, default=None,
                       help='path for the vocab alignment file (id, teacher-to-student)')
    group.add_argument("--student-to-teacher-token-mapping", type=str, default=None,
                       help='path for the vocab alignment file (token, student-to-teacher)')
    group.add_argument("--student-to-teacher-id-mapping", type=str, default=None,
                       help='path for the vocab alignment file (id, student-to-teacher)')
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_gen_args(parser)
    parser = add_peft_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    parser = add_distiller_args(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.n_gpu = args.n_gpu * args.n_nodes
        
    return args
        
    # args.n_gpu = args.n_gpu * args.n_nodes
        
    # return args
