## About
Our codebase is derived from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). The key difference from the original repository is that we modified `megatron/post_training/model_builder.py` by forcing `use_arbitrary_attention_mask = False`. This change enables fused attention when training GPT-OSS-120B, allowing us to support a 64K context length without enabling Context Parallelism (CP). The training was conducted on a cluster of 64 NVIDIA A800 GPUs.

## Quick Start
To get started quickly, we strongly recommend using the NVIDIA Docker environment nvcr.io/nvidia/pytorch:25.12-py3, which eliminates the need for additional software installation.

## SFT
> To train GPT-OSS-120B, the model must first be converted into the Megatron format. Navigate to `examples/post_training/modelopt/` and execute:
>
> ```bash
> sh convert_gpt_oss.sh $i
> ```
>
> where `$i` denotes the node index. Since the conversion requires 8 * A800 nodes, `$i` should range from **0 to 7**.
>
> After the conversion is completed, launch training with:
>
> ```bash
> sh run.sh $i
> ```
>
> This configuration supports context lengths of up to **64K** without enabling Context Parallelism (CP). For training with longer contexts, we recommend enabling CP to ensure better scalability and memory efficiency.

## Megatron-> HF
> After training is complete, the model can be converted back to the Hugging Face (HF) format. Navigate to `examples/post_training/modelopt/` and run:
>
> ```bash
> TP=8 PP=8 \
> HF_MODEL_CKPT=/path/gpt-oss-120b \
> MLM_MODEL_CKPT=/path/gpt-oss-120b-modelopt-ft \
> EXPORT_DIR=./tmp \
> ./export.sh openai/gpt-oss-120b
> ```
>
> This script exports the fine-tuned Megatron checkpoint to a Hugging Face–compatible format for downstream usage such as inference or further fine-tuning.
