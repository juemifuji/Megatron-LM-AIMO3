export UB_SKIPMC=1
export PYTHONPATH=/path/Megatron-LM-AIMO3:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLM_MODEL_CFG=conf/openai/gpt-oss-120b.sh
export HF_MODEL_CKPT=openai/gpt-oss-120b

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

source base_init.sh 
source /etc/profile


export MLM_EXTRA_ARGS=" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model openai/gpt-oss-120b \
  --seq-length 61440 \
  --sequence-parallel \
  --context-parallel-size 1 \
  --attention-backend fused \
  --no-load-optim \
  --no-load-rng \
  --cross-entropy-loss-fusion \
  --no-save-optim \
  --tp-comm-overlap \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --use-distributed-optimizer \
  --num-workers 0 \
  --optimizer-cpu-offload \
  --use-precision-aware-optimizer \
  --optimizer-offload-fraction 1.0 \
  --decoder-first-pipeline-num-layers 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
"

export TOKENIZER_MODEL=openai/gpt-oss-120b

# 并行：8 GPU 整除
export TP=8
export PP=8
export EP=8
export ETP=1
export CP=1   # 你的脚本里没传 CP，就保持 1

# 训练 ckpt 路径（finetune.sh 会用 --load/--save）
# auto-detect-ckpt-format 打开后，--load 可以是 HF id 或本地 HF snapshot 路径（看你 conf/arguments.sh 的实现）
export MLM_MODEL_CKPT=$HF_MODEL_CKPT
export MLM_MODEL_SAVE=/path/gpt-oss-120b-modelopt-ft

# 多机 torchrun（如果 arguments.sh 里已经定义了 LAUNCH_SCRIPT，你也可以不设）
export NNODES=8
export GPUS_PER_NODE=8
export NODE_RANK=$1
export MASTER_ADDR=10.178.168.79
export MASTER_PORT=29500
export NSYS="nsys profile --trace=cuda,nvtx --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
--capture-range=cudaProfilerApi --capture-range-end=repeat[:2] --python-backtrace=cuda \
--cuda-memory-usage=true --cudabacktrace=memory"
export LAUNCH_SCRIPT="$NSYS torchrun --nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"

sh finetune.sh openai/gpt-oss-120b
