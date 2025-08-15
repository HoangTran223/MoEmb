#!/usr/bin/env bash

# Required paths
CKPT_DIR="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/imdb/epoch10_step3910_loss0.0298"
DATA_DIR="/mnt/hungpv/projects/MoEmb/dataset/data/imdb"
# If CKPT_DIR is a LoRA adapter, set the base model path
# BASE_MODEL_PATH="/mnt/hungpv/projects/MoEmb/model_hub/llm2vec"

# Eval options
SPLIT="dev"            # train | dev | test
EVAL_BATCH_SIZE=1
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

# --base-model-path ${BASE_MODEL_PATH}"