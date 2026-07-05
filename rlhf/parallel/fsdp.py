"""FSDP wrapping helpers for sharding a HuggingFace model's parameters + optimizer
state across data-parallel ranks.

FSDP is the memory lever for post-training: with the reference model, policy,
and (for PPO) a value head all resident, a 7B policy will not fit on one 24 GB
GPU. Sharding parameters, gradients, and optimizer state across N ranks cuts
per-GPU state by ~N, which is what ``experiments/gsm8k_grpo.py`` measures (single vs
DDP vs FSDP).

We wrap per-transformer-block so all-gather/reduce-scatter overlaps compute.
"""
import functools

import torch


def wrap_fsdp(model, transformer_layer_cls=None, mixed_precision=True):
    """Wrap ``model`` in FSDP with sensible post-training defaults.

    Args:
        model: an nn.Module already moved to the local CUDA device.
        transformer_layer_cls: the block class to shard on (e.g. GPT2Block).
            If None, FSDP shards at the top level (coarser, still valid).
        mixed_precision: run params/reduce in bf16, keep fp32 master where useful.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    auto_wrap_policy = None
    if transformer_layer_cls is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )

    mp = None
    if mixed_precision and torch.cuda.is_available():
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
        mixed_precision=mp,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
    )


def gpt2_block_cls():
    """Return the GPT2Block class for the auto-wrap policy (lazy import)."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    return GPT2Block


def qwen_block_cls():
    """Return the Qwen2 decoder-block class for the auto-wrap policy (lazy import)."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    return Qwen2DecoderLayer


def peak_memory_mb():
    """Peak allocated CUDA memory on the current device, in MB (0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def unwrap_model(model):
    """Return the underlying HF module inside an FSDP/DDP wrapper (or the model itself)."""
    return model.module if hasattr(model, "module") else model


def gather_full_state_dict(model):
    """Return a full (unsharded) CPU ``state_dict`` on rank 0, ``{}`` on other ranks.

    Under FSDP the parameters are sharded, so saving requires reassembling the whole
    model on one rank. ``FULL_STATE_DICT`` with ``rank0_only`` + CPU offload does that
    without every rank materializing the full model (which would defeat the point of
    sharding). This is a **collective** — every rank must call it, in lockstep — but
    only rank 0 gets populated data. For DDP / single-GPU it's just the CPU state_dict
    of the underlying module.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}
