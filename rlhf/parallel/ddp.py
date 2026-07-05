"""DistributedDataParallel wrapping — the *replicated* data-parallel strategy.

DDP is the counterpoint to FSDP in the scaling benchmark: it keeps a full copy of the
model, gradients, and optimizer state on **every** GPU and only all-reduces gradients on
backward. That makes it the throughput-friendly choice, but per-GPU memory does NOT shrink
with more GPUs — so it caps the model size at what one device can hold. FSDP (ZeRO-3)
trades some throughput to shard that state ~1/N. Benchmarking both on the same GRPO
workload is the systems story.
"""
import torch


def wrap_ddp(model, local_rank):
    """Wrap a CUDA-resident ``model`` in DistributedDataParallel.

    Args:
        model: an nn.Module already moved to ``cuda:local_rank``.
        local_rank: the local device index this rank owns.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    if torch.cuda.is_available():
        return DDP(model, device_ids=[local_rank], output_device=local_rank)
    return DDP(model)  # gloo/CPU fallback for smoke tests
