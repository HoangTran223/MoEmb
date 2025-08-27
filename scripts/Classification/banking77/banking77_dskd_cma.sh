#! /bin/bash
GPUS=(1)
export HF_TOKEN=

export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
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
TEACHER_MODEL_PATH="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/banking77"

NUM_LABELS=77
TASK="dual_space_kd_with_cross_model_attention"

BATCH_SIZE=4
LR=0.00001
GRAD_ACC=2
EVAL_BATCH_SIZE=4
EPOCH=10
KD_RATE=0.1
KD_TEMP=2.0
TEA_TEMP=2.0
MAX_LENGTH=128

PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001
PRECISION="bf16"

CRITERION="dual_space_kd_with_cross_model_attention"
KD_OBJ="forward_kl"
CONFIG="${KD_OBJ}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${KD_RATE}__kd^temp=${KD_TEMP}__tea^temp=${TEA_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}

DATA="banking77"
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${DATA}$/${SETTING}"

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
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --teacher-temperature ${TEA_TEMP}"
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