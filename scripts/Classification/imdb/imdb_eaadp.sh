#! /bin/bash
GPUS=(1)
export HF_TOKEN=

export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=40$(($RANDOM%90+10))
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

# Adapter (được Distiller load qua --teacher-model-path như adapter bổ sung)
TEACHER_MODEL_PATH="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/imdb/epoch10_step3910_loss0.0298"
NUM_LABELS=2
TASK="eaadp"

BATCH_SIZE=2
LR=0.00001
GRAD_ACC=2
EVAL_BATCH_SIZE=4
EPOCH=15
KD_RATE=0.3
KD_TEMP=2.0
TEA_TEMP=2.0
MAX_LENGTH=512

# EAADP specific hyperparameters
EAADP_TOP_K=3               # top-k layers for teacher & student
EAADP_ATTN_LR=0.0001         # learning rate for attention weights (EAS)
EAADP_MAIN_LR=0.0001         # learning rate for student + DPL + TALF (can differ from --lr scheduler)
# 2 tham số lr kia siêu ảnh hưởng đến kq. Để 0.00001 thì chỉ được 0.5
EAADP_ALPHA=0.1          # weight for CE vs CCL: loss = alpha*CE + (1-alpha)*CCL
EAADP_MAX_CIS_BATCHES=0     # 0 = only first batch; >0 accumulate that many batches then finalize CIS
USE_ATTENTION_CIS=1         # 1 = compute CIS using attention probs
USE_BATCH_CIS=0             # 1 = enable --eaadp-use-batch-cis flag

PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001
PRECISION="bf16"

DATA="imdb"
CRITERION="eaadp"
KD_OBJ="forward_kl"
CONFIG="${KD_OBJ}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${KD_RATE}__kd^temp=${KD_TEMP}__tea^temp=${TEA_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${DATA}/${SETTING}"
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
# Use dtype flag to control precision (fp16/bf16/fp32)
OPTS+=" --model-dtype ${PRECISION}"

OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"

OPTS+=" --gradient-checkpointing"
OPTS+=" --data-dir ${BASE_PATH}/dataset/data/imdb/"
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
# KD-related flags are unused by EAADP; omit to avoid confusion
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

# add
OPTS+=" --eaadp-top-k ${EAADP_TOP_K}"
OPTS+=" --eaadp-attn-lr ${EAADP_ATTN_LR}"
OPTS+=" --eaadp-main-lr ${EAADP_MAIN_LR}"
OPTS+=" --eaadp-alpha ${EAADP_ALPHA}"
# CIS is computed via a full pre-pass; no batch-limit flag needed
if [[ ${USE_ATTENTION_CIS} -eq 1 ]]; then
    OPTS+=" --eaadp-cis-use-attn"
fi
# Batch-CIS mode is deprecated in favor of full pre-pass

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_test.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/Classification/distillation.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee ${SAVE_DIR}/train.log