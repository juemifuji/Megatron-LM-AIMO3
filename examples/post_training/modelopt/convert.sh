#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
export MLM_MODEL_SAVE=/path/gpt-oss-120b-modelopt-ft
export MLM_MODEL_CKPT=openai/gpt-oss-120b
export TOKENIZER_MODEL=openai/gpt-oss-120b
source "${SCRIPT_DIR}/conf/arguments.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:256

# Default arguments of this script
MLM_DEFAULT_ARGS="
    --distributed-timeout-minutes 60 \
    --finetune \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
"

if [ -z ${HF_TOKEN} ]; then
    printf "${MLM_WARNING} Variable ${PURPLE}HF_TOKEN${WHITE} is not set! HF snapshot download may fail!\n"
fi

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}_mlm
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_SAVE})!\n"
fi

if [ ! -d ${MLM_MODEL_SAVE} ]; then
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_model.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size ${ETP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --pretrained-model-path ${HF_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
	--use-cpu-initialization \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
else
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_model.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size ${ETP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --load ${MLM_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
fi
