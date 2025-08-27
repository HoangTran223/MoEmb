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

# Teacher model path (Mistral-7B LLM2Vec with LoRA adapters)
TEACHER_MODEL_PATH="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/scitail/epoch1_step1444_loss0.1227"
NUM_LABELS=2
TASK="fkd_final" # Changed from fkd_a


BATCH_SIZE=4
LR=0.00002
GRAD_ACC=1
EVAL_BATCH_SIZE=4
EPOCH=12
KD_RATE=0.2
KD_TEMP=2.0
TEA_TEMP=2.0
MAX_LENGTH=512

FKD_K=8 
# FKD_FINAL specific hyperparameters - Following user's formula: α·L_CE + β·L_Distill
FKD_FINAL_ALPHA=0.2     # weight for CE loss 
FKD_FINAL_BETA=2         # weight for distillation loss
FKD_FINAL_LAMBDA=0.3      # weight for contextual score in hybrid alignment

# FastText parameters
FASTTEXT_DIM=500
FASTTEXT_EPOCH=10
FASTTEXT_MINN=3
FASTTEXT_MAXN=10

# Optimal Transport parameters  
OT_REG=0.05
OT_NUMITERMAX=2000

# Paths for pre-computed components (will be generated if not exist)
GLOBAL_ALIGNMENT_PATH="${BASE_PATH}/outputs/fkd_final/global_alignment_mistral_bert.npy"
TEACHER_FASTTEXT_PATH="${BASE_PATH}/outputs/fkd_final/teacher_fasttext.bin"
STUDENT_FASTTEXT_PATH="${BASE_PATH}/outputs/fkd_final/student_fasttext.bin"

PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001
PRECISION="bf16"

DATA="scitail"
CRITERION="fkd_final" 
SETTING=criterion=${CRITERION}__alpha=${FKD_FINAL_ALPHA}__beta=${FKD_FINAL_BETA}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
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
OPTS+=" --data-dir ${BASE_PATH}/dataset/data/scitail/"
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

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_test.json"

# FKD_FINAL specific options
OPTS+=" --fkd-final-alpha ${FKD_FINAL_ALPHA}"
OPTS+=" --fkd-final-beta ${FKD_FINAL_BETA}"
OPTS+=" --fkd-final-lambda ${FKD_FINAL_LAMBDA}"
OPTS+=" --fasttext-dim ${FASTTEXT_DIM}"
OPTS+=" --fasttext-epoch ${FASTTEXT_EPOCH}"
OPTS+=" --fasttext-minn ${FASTTEXT_MINN}"
OPTS+=" --fasttext-maxn ${FASTTEXT_MAXN}"
OPTS+=" --ot-reg ${OT_REG}"
OPTS+=" --ot-numitermax ${OT_NUMITERMAX}"
OPTS+=" --global-alignment-path ${GLOBAL_ALIGNMENT_PATH}"
OPTS+=" --teacher-fasttext-path ${TEACHER_FASTTEXT_PATH}"
OPTS+=" --student-fasttext-path ${STUDENT_FASTTEXT_PATH}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/SentencePair/distillation.py ${OPTS}"
echo "${CMD}"

# Check if global alignment matrix exists, if not, generate it using create_global_alignment.py
mkdir -p "${BASE_PATH}/outputs/fkd_final"
if [ ! -f "${GLOBAL_ALIGNMENT_PATH}" ]; then
    echo "Global alignment matrix not found. Generating using FastText + Optimal Transport..."
    python /mnt/hungpv/projects/MoEmb/create_global_alignment.py \
        --teacher-model McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp \
        --student-model bert-base-uncased \
        --output-path ${GLOBAL_ALIGNMENT_PATH} \
        --fasttext-dim ${FASTTEXT_DIM} \
        --fasttext-epoch ${FASTTEXT_EPOCH} \
        --fasttext-minn ${FASTTEXT_MINN} \
        --fasttext-maxn ${FASTTEXT_MAXN} \
        --ot-reg ${OT_REG} \
        --ot-numitermax ${OT_NUMITERMAX} \
        --teacher-fasttext-path ${TEACHER_FASTTEXT_PATH} \
        --student-fasttext-path ${STUDENT_FASTTEXT_PATH}
fi

# Run the training and log
${CMD} 2>&1 | tee ${SAVE_DIR}/train.log
