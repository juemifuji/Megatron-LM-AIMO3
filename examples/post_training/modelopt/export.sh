#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Default arguments of this script
MLM_DEFAULT_ARGS="--finetune --auto-detect-ckpt-format --export-te-mcore-model --use-cpu-initialization"

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=${1}
fi

if [ -z ${HF_TOKEN} ]; then
    printf "${MLM_WARNING} Variable ${PURPLE}HF_TOKEN${WHITE} is not set! Pretrained config download may fail!\n"
fi

if [ -z ${EXPORT_DIR} ]; then
    EXPORT_DIR=${MLM_WORK_DIR}/${MLM_MODEL_CFG}_export
    printf "${MLM_WARNING} Variable ${PURPLE}EXPORT_DIR${WHITE} is not set (default: ${EXPORT_DIR})!\n"
fi

if [ "${TP}" != "1" ]; then
    TP=1
    printf "${MLM_WARNING} Variable ${PURPLE}TP${WHITE} is forced to be 1 during export!!\n"
fi

export MLM_EXTRA_ARGS=" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /llm-align/open_models/modelscope/hub/gpt-oss/gpt-oss-120b \
  --seq-length 61440 \
  --context-parallel-size 1 \
  --attention-backend fused \
  --no-load-optim \
  --no-load-rng \
  --cross-entropy-loss-fusion \
  --no-save-optim \
  --use-distributed-optimizer \
  --num-workers 0 \
  --optimizer-cpu-offload \
  --use-precision-aware-optimizer \
  --optimizer-offload-fraction 1.0 \
  --decoder-first-pipeline-num-layers 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
"


${LAUNCH_SCRIPT} ${SCRIPT_DIR}/export.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MLM_MODEL_CKPT} \
    --pretrained-model-name ${HF_MODEL_CKPT} \
    --export-dir ${EXPORT_DIR} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
