#!/usr/bin/env bash

# Simple evaluation launcher for IMDb classification checkpoints.
# Replace variables below as needed.

# Required paths
CKPT_DIR="/mnt/hungpv/projects/MoEmb/outputs/bert/eaadp/imdb$/criterion=eaadp__forward_kl__teacher=0.5__kd^temp=2.0__tea^temp=2.0__epoch=10__bsz=4x2x1=8__lr=1e-05/epoch10_stepXXXX_loss0.XXXX"
DATA_DIR="/mnt/hungpv/projects/MoEmb/dataset/data/imdb"

# Eval options
SPLIT="test"           # train | dev | test
EVAL_BATCH_SIZE=4
NUM_WORKERS=0
MAX_LENGTH=128
DEVICE="cuda"          # cuda | cpu

BASE_PATH="/mnt/hungpv/projects/MoEmb"
export PYTHONPATH="$BASE_PATH"

CMD="python ${BASE_PATH}/evaluate.py \
  --ckpt-dir ${CKPT_DIR} \
  --data-dir ${DATA_DIR} \
  --split ${SPLIT} \
  --eval-batch-size ${EVAL_BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --max-length ${MAX_LENGTH} \
  --device ${DEVICE}"

echo "$CMD"
eval $CMD
