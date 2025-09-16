#! /bin/bash
GPUS=(1)
export HF_TOKEN=

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

# Adapter (được Distiller load qua --teacher-model-path như adapter bổ sung)
TEACHER_MODEL_PATH="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/control/epoch3_step5040_loss0.3079"
NUM_LABELS=3
TASK="fkd"

BATCH_SIZE=4
LR=0.00001
GRAD_ACC=2
EVAL_BATCH_SIZE=8
EPOCH=10
KD_RATE=0.2
KD_TEMP=2.0
TEA_TEMP=2.0
MAX_LENGTH=512
# FKD specific hyperparameters
FKD_K=5               # top-k teacher layers to focus on
FKD_ALPHA=0.5              # weight for CE loss
FKD_BETA=1               # weight for distillation loss (1 - cosine)
FKD_GAMMA=1              # weight for contrastive InfoNCE loss
FKD_CONTRASTIVE_TEMP=0.1   # temperature for InfoNCE
FKD_CALIB_MAX_BATCHES=0     # 0 = use full calibration pass; >0 to limit pre-pass batches

PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001
PRECISION="bf16"

DATA="control"
CRITERION="fkd"
SETTING=criterion=${CRITERION}__k=${FKD_K}__alpha=${FKD_ALPHA}__beta=${FKD_BETA}__gamma=${FKD_GAMMA}__ct=${FKD_CONTRASTIVE_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
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
OPTS+=" --data-dir ${BASE_PATH}/dataset/data/control/"
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

# FKD options
OPTS+=" --fkd-k ${FKD_K}"
OPTS+=" --fkd-alpha ${FKD_ALPHA}"
OPTS+=" --fkd-beta ${FKD_BETA}"
OPTS+=" --fkd-gamma ${FKD_GAMMA}"
OPTS+=" --fkd-contrastive-temp ${FKD_CONTRASTIVE_TEMP}"
OPTS+=" --fkd-calib-max-batches ${FKD_CALIB_MAX_BATCHES}"

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_test.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/SentencePair/distillation.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee ${SAVE_DIR}/train.log