"""Multi-rank correctness check for the from-scratch Megatron tensor-parallel linears.

Launched under torchrun (``--nproc_per_node=N``): shards a known dense weight across
the ``N`` ranks, runs the column->row parallel block, and asserts the result matches a
plain ``nn.Linear`` computed from the *full* weight. Unlike the single-process unit test
(where every collective is a no-op), this exercises the real NCCL collectives:

* all-gather        — ColumnParallelLinear(gather_output=True) reassembling the output.
* forward all-reduce — RowParallelLinear summing each rank's partial product.
* backward all-reduce — _CopyToModelParallel summing the input gradient across ranks.

Deterministic setup: every rank builds the *same* full weight from a fixed seed, then
keeps only its own shard, so the sharded math is checked against an unambiguous dense
reference. Exits non-zero if any rank disagrees, so a Slurm log shows pass/fail plainly.

    torchrun --standalone --nproc_per_node=4 -m experiments.tp_check
"""
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rlhf.parallel import (
    ColumnParallelLinear, RowParallelLinear,
    setup_distributed, cleanup_distributed, is_main_process,
)

D_IN, D_HIDDEN, D_OUT, BATCH = 64, 128, 64, 8
ATOL = 1e-4


def _full_weights(device):
    """Identical (seeded) dense reference weights on every rank."""
    g = torch.Generator(device=device).manual_seed(0)
    w_col = torch.randn(D_HIDDEN, D_IN, generator=g, device=device) * 0.02
    b_col = torch.randn(D_HIDDEN, generator=g, device=device) * 0.02
    w_row = torch.randn(D_OUT, D_HIDDEN, generator=g, device=device) * 0.02
    b_row = torch.randn(D_OUT, generator=g, device=device) * 0.02
    x = torch.randn(BATCH, D_IN, generator=g, device=device)  # replicated TP input
    return w_col, b_col, w_row, b_row, x


@torch.no_grad()
def _load_shard(param, value):
    param.copy_(value)


def _check_column_gather(rank, world, device, w_col, b_col, x):
    """ColumnParallelLinear(gather_output=True) == dense Linear (exercises all-gather)."""
    op = D_HIDDEN // world
    col = ColumnParallelLinear(D_IN, D_HIDDEN, bias=True, gather_output=True).to(device)
    _load_shard(col.weight, w_col[rank * op:(rank + 1) * op])
    _load_shard(col.bias, b_col[rank * op:(rank + 1) * op])
    got = col(x)
    ref = F.linear(x, w_col, b_col)
    return (got - ref).abs().max().item()


def _check_column_row(rank, world, device, w_col, b_col, w_row, b_row, x):
    """Column(no gather)->Row == dense two-layer, incl. input grad.

    Exercises the forward all-reduce (Row) and the backward all-reduce (_CopyToModelParallel).
    """
    op = D_HIDDEN // world  # column shards the hidden (output) dim
    ip = D_HIDDEN // world  # row shards the hidden (input) dim
    col = ColumnParallelLinear(D_IN, D_HIDDEN, bias=True, gather_output=False).to(device)
    row = RowParallelLinear(D_HIDDEN, D_OUT, bias=True, input_is_parallel=True).to(device)
    _load_shard(col.weight, w_col[rank * op:(rank + 1) * op])
    _load_shard(col.bias, b_col[rank * op:(rank + 1) * op])
    _load_shard(row.weight, w_row[:, rank * ip:(rank + 1) * ip])
    _load_shard(row.bias, b_row)  # row bias is replicated, added once after the reduce

    xt = x.clone().requires_grad_(True)
    out = row(col(xt))
    out.sum().backward()

    xr = x.clone().requires_grad_(True)
    ref = F.linear(F.linear(xr, w_col, b_col), w_row, b_row)
    ref.sum().backward()

    fwd_err = (out - ref).abs().max().item()
    grad_err = (xt.grad - xr.grad).abs().max().item()
    return fwd_err, grad_err


def main():
    rank, local_rank, world, device = setup_distributed()
    w_col, b_col, w_row, b_row, x = _full_weights(device)

    e_gather = _check_column_gather(rank, world, device, w_col, b_col, x)
    e_fwd, e_grad = _check_column_row(rank, world, device, w_col, b_col, w_row, b_row, x)

    errs = torch.tensor([e_gather, e_fwd, e_grad], device=device)
    if dist.is_initialized() and world > 1:
        dist.all_reduce(errs, op=dist.ReduceOp.MAX)  # worst error across all ranks
    e_gather, e_fwd, e_grad = errs.tolist()
    ok = max(e_gather, e_fwd, e_grad) < ATOL

    if is_main_process():
        print(f"[tp_check] world_size={world} device={device.type} atol={ATOL}", flush=True)
        print(f"[tp_check] column all-gather   max|err| = {e_gather:.2e}", flush=True)
        print(f"[tp_check] column->row forward  max|err| = {e_fwd:.2e}", flush=True)
        print(f"[tp_check] backward input-grad  max|err| = {e_grad:.2e}", flush=True)
        print(f"[tp_check] {'PASS' if ok else 'FAIL'} — "
              f"sharded block matches dense nn.Linear across {world} ranks", flush=True)

    cleanup_distributed()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
