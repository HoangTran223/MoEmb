#! /bin/bash
GPUS=(0)
export HF_TOKEN=${HF_TOKEN}

export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=50$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/mnt/hungpv/projects/MoEmb"
CKPT_NAME="bert"
CKPT_PATH="/mnt/hungpv/projects/MoEmb/model_hub/bert"

# Teacher (LLM2Vec Mistral with adapters merged/loaded by Distiller)
TEACHER_MODEL_PATH="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/banking77/epoch2_step1668_loss0.1647"
NUM_LABELS=77
TASK="fkd_h"

# Data/Train
BATCH_SIZE=4
LR=0.00002
GRAD_ACC=1
EVAL_BATCH_SIZE=4
EPOCH=12
MAX_LENGTH=512

# FKD_H hyperparameters
FKD_H_ALPHA=0.2      # CE loss weight
FKD_H_BETA=1.0       # Distillation loss weight
FKD_H_LAMBDA=0.3     # Global-vs-contextual mix (lambda)
FKD_H_ALIGN_TOPK=32  # Top-k teacher tokens per student token
FKD_K=4              # top-k teacher layers to focus on
FKD_CALIB_MAX_BATCHES=0  # Max batches for BI calibration (0 = all)

# Global alignment (OT) from embedding layers (no FastText)
OT_REG=0.1
OT_NUMITERMAX=1000

GLOBAL_DIR="${SAVE_DIR}"
mkdir -p "${GLOBAL_DIR}"
GLOBAL_ALIGNMENT_PATH="${GLOBAL_DIR}/global_alignment_mistral_bert.npy"
TEACHER_FASTTEXT_PATH="${GLOBAL_DIR}/teacher_fasttext.bin"
STUDENT_FASTTEXT_PATH="${GLOBAL_DIR}/student_fasttext.bin"

# Offline projection (teacher->student) produced by offline step
OFFLINE_PROJ_PATH="${GLOBAL_DIR}/W_q.pt"

PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001

DATA="banking77"
CRITERION="fkd_h"
SETTING=criterion=${CRITERION}__alpha=${FKD_H_ALPHA}__beta=${FKD_H_BETA}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${DATA}/${SETTING}/${MAX_LENGTH}"
SAVE_DIR="${SAVE_PATH}"
SAVE_BEST_N_CKPTS=1
SEED=10

mkdir -p "${SAVE_DIR}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"

# Teacher adapter
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"

OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"

OPTS+=" --gradient-checkpointing"
OPTS+=" --data-dir ${BASE_PATH}/dataset/data/banking77/"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --num-labels ${NUM_LABELS}"
OPTS+=" --task ${TASK}"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
OPTS+=" --do-train"
OPTS+=" --do-eval"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_DIR}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"

# FKD_H-specific
OPTS+=" --fkd-h-alpha ${FKD_H_ALPHA}"
OPTS+=" --fkd-h-beta ${FKD_H_BETA}"
OPTS+=" --fkd-h-lambda ${FKD_H_LAMBDA}"
OPTS+=" --fkd-h-align-topk ${FKD_H_ALIGN_TOPK}"
OPTS+=" --fkd-k ${FKD_K}"
OPTS+=" --fkd-calib-max-batches ${FKD_CALIB_MAX_BATCHES}"
OPTS+=" --global-alignment-path ${GLOBAL_ALIGNMENT_PATH}"

# Always pass projection path (script will generate if missing)
OPTS+=" --offline-projection-path ${OFFLINE_PROJ_PATH}"

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_test.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/Classification/distillation.py ${OPTS}"
echo "${CMD}"

# Auto-generate global alignment matrix and W from embeddings if missing
if [ ! -f "${GLOBAL_ALIGNMENT_PATH}" ] || [ ! -f "${OFFLINE_PROJ_PATH}" ]; then
  echo "[FKD_H] Generating global alignment and W from embedding layers (no FastText)..."
  mkdir -p "${GLOBAL_DIR}"
  python ${BASE_PATH}/create_global_alignment_from_embeddings.py \
    --teacher-model McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp \
    --student-model bert-base-uncased \
    --teacher-adapter-path ${TEACHER_MODEL_PATH} \
    --output-path ${GLOBAL_ALIGNMENT_PATH} \
    --save-projection-path ${OFFLINE_PROJ_PATH} \
    --ridge-lambda 1e-3 \
    --teacher-vocab-max 64000 \
    --student-vocab-max 30522
fi

# Run training
${CMD} 2>&1 | tee ${SAVE_DIR}/train.log
