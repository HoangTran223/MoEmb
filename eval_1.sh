#!/usr/bin/env bash

# Required paths
CKPT_DIR="/mnt/hungpv/projects/MoEmb/outputs/llm2vec/sft/anli_r2/criterion=cross_entropy__lora-rank=256-alpha=32-dropout=0.1-bf16__epoch=5__bsz=2x2x1=4__lr=0.00001/epoch4_step90920_loss0.3887"
DATA_DIR="/mnt/hungpv/projects/MoEmb/dataset/data/anli_r2"
# If CKPT_DIR is a LoRA adapter, set the base model path
BASE_MODEL_PATH="/mnt/hungpv/projects/MoEmb/model_hub/llm2vec"

# Eval options
SPLIT="test"            # train | dev | test
EVAL_BATCH_SIZE=2
NUM_WORKERS=0
MAX_LENGTH=512
DEVICE="cuda"          # cuda | cpu
MODEL_DTYPE="bf16"     # bf16 | fp16 | fp32

BASE_PATH="/mnt/hungpv/projects/MoEmb"
export PYTHONPATH="$BASE_PATH"
export TOKENIZERS_PARALLELISM=false

CMD="python ${BASE_PATH}/evaluate_1.py \
  --ckpt-dir ${CKPT_DIR} \
  --data-dir ${DATA_DIR} \
  --split ${SPLIT} \
  --eval-batch-size ${EVAL_BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --max-length ${MAX_LENGTH} \
  --device ${DEVICE} \
  --model-dtype ${MODEL_DTYPE} \
  --base-model-path ${BASE_MODEL_PATH}"

echo "$CMD"
eval $CMD 2>&1 | tee "${CKPT_DIR}/eval_${SPLIT}.log"