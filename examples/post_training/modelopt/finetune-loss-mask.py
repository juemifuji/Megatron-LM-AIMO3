# finetune.py
# -*- coding: utf-8 -*-
"""
Supervised Finetuning GPT.
包含：
- transcript 分段 mask 的处理
- 修正 PAD 逻辑
- 仍保留你原有的训练框架与 Megatron 相关调用
"""

import itertools
import json
import os
import sys
from functools import partial
from typing import Any, Dict, Optional, List, Tuple

import jsonlines

# ---------------------------------------------------------------------------
# 确保项目路径正确（与你原来的写法一致）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import datasets
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.loss_func import loss_func
from megatron.post_training.model_builder import modelopt_gpt_mamba_builder
from megatron.post_training.non_loss_data_func import report_draft_acceptance_length
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    print_rank_0,
)
from model_provider import model_provider

# ---------------------------------------------------------------------------
# 你原来的特殊模板替换逻辑
REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)

# ---------------------------------------------------------------------------
# transcript parsing helper -----------------------------------------------
# 这些 marker 必须按你在数据里实际使用的保持一致

SYSTEM_START = "<|start|>system<|message|>"
USER_START   = "<|start|>user<|message|>"
ASSIST_ANALYSIS_START = "<|start|>assistant<|channel|>analysis<|message|>"
ASSIST_FINAL_START    = "<|start|>assistant<|channel|>final<|message|>"

PY_CODE_START = "<|start|>assistant to=python<|channel|>analysis code<|message|>"
PY_OUT_START  = "<|call|><|start|>python to=assistant<|channel|>analysis<|message|>"

END = "<|end|>"

# 作为“切割符”的全集：这些串出现的位置都要切出来，且切割符本身 mask=1
ALL_MARKERS = [
    SYSTEM_START,
    USER_START,
    ASSIST_ANALYSIS_START,
    ASSIST_FINAL_START,
    PY_CODE_START,
    PY_OUT_START,
    END,
]
# 为避免短 marker 抢匹配，按长度从长到短匹配
ALL_MARKERS_SORTED = sorted(ALL_MARKERS, key=len, reverse=True)


def split_transcript_for_loss_mask(text: str) -> List[Tuple[str, int]]:
    """
    将完整 transcript 文本切成多段，每段返回 (text, mask_flag)
    mask_flag == 1: 参与 loss（不 mask）
    mask_flag == 0: mask 掉

    严格按规则：
    - 所有切割符（start markers + <|end|>）mask=1
    - system/user/py_out 的“内容区”mask=0
    - assistant analysis/final/py_code 的“内容区”mask=1
    """
    i = 0
    n = len(text)
    segs: List[Tuple[str, int]] = []

    # 当前处于哪个区间（决定普通内容 chunk 的 mask）
    # None / "system" / "user" / "assistant_analysis" / "assistant_final" / "py_code" / "py_out"
    mode: Optional[str] = None

    def content_mask_for_mode(m: Optional[str]) -> int:
        if m in ("assistant_analysis", "assistant_final", "py_code"):
            return 1
        if m in ("system", "user", "py_out"):
            return 0
        # 不在任何块内的“游离文本”，按 0 处理（更安全）
        return 0

    def match_marker_at(pos: int) -> Optional[str]:
        for mk in ALL_MARKERS_SORTED:
            if text.startswith(mk, pos):
                return mk
        return None

    def find_next_marker_pos(pos: int) -> int:
        """返回从 pos 开始，下一个 marker 的最早位置；找不到则返回 n。"""
        nxt = n
        for mk in ALL_MARKERS:
            j = text.find(mk, pos)
            if j != -1 and j < nxt:
                nxt = j
        return nxt

    while i < n:
        mk = match_marker_at(i)
        if mk is not None:
            # 切割符本身：mask=1
            segs.append((mk, 1))
            i += len(mk)

            # 进入/切换 mode（由 marker 决定）
            if mk == SYSTEM_START:
                mode = "system"
            elif mk == USER_START:
                mode = "user"
            elif mk == ASSIST_ANALYSIS_START:
                mode = "assistant_analysis"
            elif mk == ASSIST_FINAL_START:
                mode = "assistant_final"
            elif mk == PY_CODE_START:
                mode = "py_code"
            elif mk == PY_OUT_START:
                mode = "py_out"
            elif mk == END:
                # END 结束当前块
                mode = None
            continue

        # 普通内容：吃到下一个 marker 前
        j = find_next_marker_pos(i)
        chunk = text[i:j]
        if chunk:
            segs.append((chunk, content_mask_for_mode(mode)))
        i = j

    # 合并相邻同 mask 的段（减少 tokenizer 调用次数）
    merged: List[Tuple[str, int]] = []
    for s, m in segs:
        if not s:
            continue
        if merged and merged[-1][1] == m:
            merged[-1] = (merged[-1][0] + s, m)
        else:
            merged.append((s, m))
    return merged

# ---------------------------------------------------------------------------


def add_finetune_args(parser):
    """Add additional arguments for finetune."""
    group = parser.add_argument_group(title='Finetune')
    group.add_argument(
        "--offline-distillation-data",
        type=str,
        help="Path to the offline dataset directory with base model features."
    )

    add_modelopt_args(parser)
    return parser


def get_eos_id():
    """Return the eos token id.

    We insert eos_token between two samples during packing. However, if the eos_token is used in message or after turns,
    we need to replace it with some other special tokens that do not appear in message."""
    tokenizer = get_tokenizer()
    hf_tokenizer = tokenizer._tokenizer

    if hf_tokenizer.eos_token == "<|eot_id|>":
        return 128001
    if hf_tokenizer.eos_token == "<|eot|>":
        return 200001
    if hf_tokenizer.eos_token == "<|im_end|>":
        return 151643
    if hf_tokenizer.eos_token == "<|return|>":
        return 199999

    return hf_tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Dataset implementations
# ---------------------------------------------------------------------------

class OfflineDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, num_samples):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.file_paths = []

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isfile(item_path):
                self.file_paths.append(item_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = idx % len(self.file_paths)
        file_path = self.file_paths[idx]
        sample = torch.load(file_path)
        return sample


class SFTDataset(torch.utils.data.Dataset):

    hf_dataset_to_kwargs = {
        "Open-Orca/OpenOrca": {"split": "train"},
        "Open-Orca/SlimOrca": {"split": "train"},
        "nvidia/Daring-Anteater": {"split": "train"},
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": {"split": "train"},
        "HuggingFaceH4/ultrachat_200k": {"split": "train_sft"},
    }

    hf_dataset_to_conversation = {
        "Open-Orca/OpenOrca": lambda data: SFTDataset._to_conversation(
            data["question"], data["response"]
        ),
        "Open-Orca/SlimOrca": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "nvidia/Daring-Anteater": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": lambda data: SFTDataset._sharegpt_to_openai_conversations(
            data
        ),
    }

    hf_dataset_to_prompt_template = {
        "Open-Orca/OpenOrca": "{{ messages['question'] + ' ' + messages['response'] + ' ' }}",
    }

    def __init__(
        self,
        num_packed_samples: int,
        data_path: Optional[str],
        tokenizer: transformers.PreTrainedTokenizerBase,
        seq_length: int,
        hf_dataset: Optional[str] = None,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        """A simple dataset implementation for supervised fine-tuning.

        The raw data is processed and packed to an indexed dataset on the fly.
        """
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

        self.num_packed_samples = num_packed_samples
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.hf_dataset = hf_dataset
        self.data_transformation = lambda data: data
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.indexed_dataset = []
        self._raw_sample_index = 0

        # [WAR]: For DeepSeek-V3/R1 tokenizer, we modify the chat_template such that the <think>
        # tokens are preserved for supervised learning.
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

        if data_path is not None:
            if data_path.endswith(".json"):
                self._raw_samples = json.load(open(data_path))
            elif data_path.endswith(".jsonl"):
                with jsonlines.open(data_path, mode='r') as reader:
                    self._raw_samples = [obj for obj in reader]
            else:
                raise ValueError("data_path must be json or jsonl")
        elif self.hf_dataset is not None:
            hf_dataset_kwargs = SFTDataset.hf_dataset_to_kwargs.get(
                self.hf_dataset, {"split": "train"}
            )
            self._raw_samples = datasets.load_dataset(
                self.hf_dataset,
                token=os.environ.get("HF_TOKEN", None),
                **hf_dataset_kwargs
            )
            self._raw_samples = self._raw_samples.shard(
                num_shards=self.num_shards, index=shard_index
            )

            print(
                "Rank {:3}/{:3} creates SFT data shard {:3}/{:3} with {:10} raw samples".format(
                    torch.distributed.get_rank(),
                    torch.distributed.get_world_size(),
                    self.shard_index,
                    self.num_shards,
                    len(self._raw_samples),
                ),
                flush=True,
            )
        else:
            raise ValueError("Either hf_dataset or data_path must be provided!")

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SFTDataset.hf_dataset_to_prompt_template
        elif self.hf_dataset is not None:
            self.data_transformation = SFTDataset.hf_dataset_to_conversation.get(
                self.hf_dataset, lambda data: data
            )

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

    def __len__(self):
        return self.num_packed_samples

    def __getitem__(self, idx):
        """Get the idx packed data."""
        idx = idx // self.num_shards

        while idx >= len(self.indexed_dataset):
            packed_samples = self._process_and_pack_example()
            if packed_samples is None:
                break
            else:
                self.indexed_dataset.append(packed_samples)
            if len(self.indexed_dataset) % 100 == 0:
                print(
                    "Rank {:3}/{:3} requests {:10}/{:10} packed SFT sample".format(
                        torch.distributed.get_rank(),
                        torch.distributed.get_world_size(),
                        idx,
                        len(self.indexed_dataset),
                    ),
                    flush=True,
                )

        idx = idx % len(self.indexed_dataset)
        torch_sample = {}
        for key, val in self.indexed_dataset[idx].items():
            torch_sample[key] = torch.LongTensor(val)
        return torch_sample

    # ------------------------------------------------------------------
    # 单样本处理与 packing
    # ------------------------------------------------------------------

    def _process_and_pack_example(self):
        """
        单样本处理：
        - 如果样本 token_count < required_packed_tokens，则 PAD 到固定长度
        - loss_mask 对 PAD 部分设为 0
        - 不再对多个 sample 进行拼接
        """
        required_packed_tokens = self.seq_length + 1

        if self._raw_sample_index >= len(self._raw_samples):
            return None

        raw_sample = self._raw_samples[self._raw_sample_index]
        self._raw_sample_index += 1

        processed = self._process_example(raw_sample)
        if processed is None:
            return None

        input_ids = processed["input_ids"]
        loss_mask = processed["loss_mask"]
        token_count = processed["token_count"]

        if token_count >= required_packed_tokens:
            packed = {
                "input_ids": input_ids[:required_packed_tokens],
                "loss_mask": loss_mask[:required_packed_tokens],
                "token_count": min(token_count, required_packed_tokens),
            }
            return packed

        pad_len = required_packed_tokens - token_count

        # ---------------------------
        # 修正 PAD 逻辑：不要用 -100 填充 input_ids
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        padded_input_ids = input_ids + [IGNORE_TOKEN_ID] * pad_len
        padded_loss_mask = loss_mask + [0] * pad_len
        # ---------------------------

        packed = {
            "input_ids": padded_input_ids,
            "loss_mask": padded_loss_mask,
            "token_count": token_count,
        }
        return packed

    # ------------------------------------------------------------------
    # _process_example：基于“切割符 + 区间规则”的 mask 逻辑
    # ------------------------------------------------------------------

    def _process_example(self, example: Dict[str, Any]):
        """
        Apply the chat template and compute the answer-only loss mask.
        修改为基于 transcript 切割 + 区间赋 mask 的逻辑。
        """
        if not isinstance(example, Dict):
            raise ValueError("The sample must be a Dict but got {}".format(type(example)))

        example = self.data_transformation(example)

        # ---------- 获取完整文本 ----------
        text = example["messages"][1]["content"]

        if not text:
            return None

        segments = split_transcript_for_loss_mask(text)

        input_ids: List[int] = []
        loss_mask: List[int] = []
        for seg_text, mask_flag in segments:
            ids = self.tokenizer(seg_text, add_special_tokens=False)["input_ids"]
            input_ids.extend(ids)
            loss_mask.extend([mask_flag] * len(ids))

        # ---------- 加 eos ----------
        eos = get_eos_id()
        input_ids.append(eos)
        loss_mask.append(0)  # eos 不计算 loss

        # ---------- truncate if needed ----------
        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            loss_mask = loss_mask[: self.seq_length]

        processed_example = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "token_count": len(input_ids),
        }
        return processed_example

    # ------------------------------------------------------------------
    # 一些辅助转换函数
    # ------------------------------------------------------------------

    @classmethod
    def _to_conversation(cls, question, response):
        msg_question = {"role": "user", "content": question}
        msg_response = {"role": "assistant", "content": response}
        return {"conversations": [msg_question, msg_response]}

    @classmethod
    def _sharegpt_to_openai_conversations(cls, data):
        role_mapping = {
            "user": "user",
            "User": "user",
            "human": "user",
            "assistant": "assistant",
            "Assistant": "assistant",
            "gpt": "assistant",
            "system": "system",
            "System": "system",
        }
        processed_data = {"conversations": []}
        for msg in data["conversations"]:
            role = role_mapping[msg["from"]]
            content = msg["value"]
            processed_data["conversations"].append({"role": role, "content": content})
        return processed_data

    @classmethod
    def _special_to_openai_conversations(cls, data):
        processed_data = {"conversations": data["input"]["messages"]}
        return processed_data


# ---------------------------------------------------------------------------
# Data provider helper (保持你原有逻辑）
# ---------------------------------------------------------------------------

def train_valid_test_sft_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets."""
    print_rank_0("> building train, validation, and test SFT datasets ...")
    args = get_args()
    tokenizer = get_tokenizer()

    if not isinstance(tokenizer._tokenizer, transformers.PreTrainedTokenizerBase):
        raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

    if args.micro_batch_size > 1:
        raise ValueError("SFTDataloader only supports micro_batch_size=1.")

    if args.export_offline_model:
        train_ds = OfflineDataset(
            os.path.join(args.offline_distillation_data, "train"),
            train_val_test_num_samples[0]
        )
        valid_ds = OfflineDataset(
            os.path.join(args.offline_distillation_data, "valid"),
            train_val_test_num_samples[1]
        )
        test_ds = OfflineDataset(
            os.path.join(args.offline_distillation_data, "test"),
            train_val_test_num_samples[2]
        )

        print_rank_0("> finished creating offline SFT datasets ...")
    else:
        kwargs = {
            "tokenizer": tokenizer._tokenizer,
            "seq_length": args.seq_length,
            # Optional kwargs
            "hf_dataset": args.finetune_hf_dataset,
            "num_shards": mpu.get_expert_data_parallel_world_size(),
            "shard_index": mpu.get_expert_data_parallel_rank(),
        }

        data_path = [
            args.train_data_path[0] if args.train_data_path else None,
            args.valid_data_path[0] if args.valid_data_path else None,
            args.test_data_path[0] if args.test_data_path else None,
        ]

        print("==================================train_val_test_num_samples============",
              train_val_test_num_samples)
        train_ds = SFTDataset(train_val_test_num_samples[0], data_path[0], **kwargs)
        valid_ds = SFTDataset(train_val_test_num_samples[1], data_path[1], **kwargs)
        test_ds = SFTDataset(train_val_test_num_samples[2], data_path[2], **kwargs)

        print_rank_0("> finished creating SFT datasets ...")

    return train_ds, valid_ds, test_ds


# ---------------------------------------------------------------------------
# Batch / forward 的逻辑保持不变，主要改了输入 mask
# ---------------------------------------------------------------------------

def get_batch(data_iterator):
    """Generate a batch.

    For OfflineDataset, the aux_hidden_states and final hidden_states from the
    base model are loaded for offline speculative model training."""
    # TODO: this is pretty hacky
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    # Broadcast data since only TP rank-0 has the data_iterator.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    if not args.export_offline_model:
        keys = ["input_ids", "loss_mask"]
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    else:
        keys = ["input_ids"]
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
        data_b["loss_mask"] = torch.ones_like(data_b["input_ids"])
        data_b["loss_mask"][data_b["loss_mask"] == get_eos_id()] = 0
        data_b["loss_mask"] = torch.cat(
            [data_b["loss_mask"], torch.zeros(1, 1).to(torch.cuda.current_device())], dim=-1
        )

        keys = ["aux_hidden_states", "hidden_states"]
        datatype = torch.bfloat16
        feature_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack the data received.
    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, 0: 0 + args.seq_length].contiguous()
    labels = tokens_[:, 1: 1 + args.seq_length].contiguous()
    answer_only_loss_mask = data_b["loss_mask"][:, 1: 1 + args.seq_length].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        get_eos_id(),
        get_eos_id(),
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        False
    )
    loss_mask = loss_mask * answer_only_loss_mask.to(dtype=loss_mask.dtype)

    labels = labels.contiguous()
    loss_mask = loss_mask.contiguous()

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    if args.export_offline_model:
        batch["aux_hidden_states"] = feature_b["aux_hidden_states"].transpose(0, 1)[:args.seq_length]
        batch["hidden_states"] = feature_b["hidden_states"].transpose(0, 1)[:args.seq_length]

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch["tokens"], batch["labels"], batch["loss_mask"], batch["attention_mask"], batch["position_ids"]


def non_loss_data_func(model: GPTModel):
    """Callback to compute the acceptance length."""
    args = get_args()
    if not args.export_offline_model:
        try:
            report_draft_acceptance_length(model)
        except Exception as e:
            print(e)


def forward_step(data_iterator, model: GPTModel):
    """Forward training step."""
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    if args.export_offline_model:
        # 这里之前你的代码里有 aux_hidden_states, hidden_states
        # 维持原样（如果有），否则略
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            # 假如 export_offline_model 时需要使用的参数：
            # aux_hidden_states=aux_hidden_states,
            # hidden_states=hidden_states,
        )
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask, model=model)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pretrain(
        train_valid_test_sft_datasets_provider,
        partial(model_provider, modelopt_gpt_mamba_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_finetune_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        non_loss_data_func=non_loss_data_func,
    )
