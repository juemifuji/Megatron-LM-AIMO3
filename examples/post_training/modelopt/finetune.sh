#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

export PYTHONPATH=/llm-align/wenliang/wenliang/AIMO3/Megatron-LM:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# Set up cache dir for HF to avoid out of space error
# export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

# Extra arguments of this script
MLM_DEFAULT_ARGS=" \
    --distributed-timeout-minutes 30 \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
    --calculate-per-token-loss \
    --finetune \
"


if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_MODEL_CKPT}
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_CKPT})!\n"
fi

if [ -z ${MLM_DATA_ARGS} ]; then
    MLM_DATA_ARGS=" \
        --train-samples 50000 \
        --lr-decay-samples 50000 \
        --lr-warmup-samples 0 \
	--train-data-path /llm-align/wenliang/wenliang/AIMO3/data/aimo3-nvidia-with-tool-2w-260203.jsonl \
        --valid-data-path /llm-align/wenliang/wenliang/AIMO3/data/aimo3-hard-sample-260113-5w-valid.jsonl \
        --test-data-path  /llm-align/wenliang/wenliang/AIMO3/data/aimo3-hard-sample-260113-5w-test.jsonl \
	--dataloader-type cyclic \
    "
fi

if [ -z ${MLM_TRAIN_ARGS} ]; then
    MLM_TRAIN_ARGS=" \
        --eod-mask-loss \
        --micro-batch-size 1 \
	--global-batch-size 32 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
    "
fi

if [ -z ${MLM_OPTIM_ARGS} ]; then
    MLM_OPTIM_ARGS=" \
        --lr 1.0e-5 \
        --min-lr 5.0e-6 \
        --lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.008 \
        --use-distributed-optimizer \
    "
fi

if [ -z ${MLM_EVAL_ARGS} ]; then
    MLM_EVAL_ARGS=" \
        --eval-iters 1 \
        --eval-interval 100 \
        --save-interval 100 \
        --log-interval 1 \
    "
fi

export HF_TOKEN=${HF_TOKEN}

${LAUNCH_SCRIPT} ${SCRIPT_DIR}/finetune.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load /llm-align/wenliang/wenliang/AIMO3/ckpts/gpt-oss-120b-modelopt-ft \
    --save /llm-align/wenliang/wenliang/AIMO3/ckpts/gpt-oss-120b-modelopt-ft-save-260203 \
    ${MLM_DATA_ARGS} \
    ${MLM_OPTIM_ARGS} \
    ${MLM_TRAIN_ARGS} \
    ${MLM_EVAL_ARGS} \
    ${MLM_RESUME_ARGS} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
