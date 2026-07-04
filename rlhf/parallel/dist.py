"""torch.distributed bootstrap for multi-GPU runs (launched with torchrun).

Reads the standard ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` env vars that
torchrun sets, binds each rank to its own CUDA device, and initializes the NCCL
process group. Falls back to a single-process ``gloo`` group so the same code
runs (and tests) on a CPU-only laptop.
"""
import os

import torch
import torch.distributed as dist


def is_distributed_launch():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    """Init the process group and return (rank, local_rank, world_size, device)."""
    if is_distributed_launch():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        return rank, local_rank, world_size, device

    # Single-process fallback (laptop / CI): still a valid 1-rank group.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0
