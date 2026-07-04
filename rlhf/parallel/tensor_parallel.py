"""Megatron-style tensor-parallel linear layers, from scratch.

A single ``Linear(d_in, d_out)`` is sharded across ``world_size`` GPUs so no one
device holds the full weight — this is the core trick that lets a matmul too big
for one GPU run on many.

Two complementary shardings, composed as Column -> Row inside an MLP/attention
block so the two all-* collectives cancel to one all-reduce per block:

* ColumnParallelLinear  — shard the *output* dim. Each rank computes a slice of
  the output columns; ``gather_output=True`` all-gathers them back to full width.
* RowParallelLinear     — shard the *input* dim. Each rank multiplies its input
  slice by its weight slice; results are summed with an all-reduce.

Custom autograd functions insert the transpose collective on the backward pass
(all-reduce of grads for column-parallel; identity for row-parallel input),
which is the part most "use FSDP and move on" implementations never show.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _rank_world():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class _CopyToModelParallel(torch.autograd.Function):
    """Identity in forward; all-reduce gradients in backward (column-parallel input)."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(grad)
        return grad


class _ReduceFromModelParallel(torch.autograd.Function):
    """All-reduce in forward; identity in backward (row-parallel output)."""

    @staticmethod
    def forward(ctx, x):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


def _all_gather_last_dim(x):
    rank, world = _rank_world()
    if world == 1:
        return x
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=-1)


class ColumnParallelLinear(nn.Module):
    """Linear with the output dimension sharded across ranks.

    Y = X @ Wᵀ, W split row-wise (i.e. output cols) so each rank owns
    ``out_features // world_size`` output units.
    """

    def __init__(self, in_features, out_features, bias=True, gather_output=True):
        super().__init__()
        rank, world = _rank_world()
        assert out_features % world == 0, "out_features must divide world_size"
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_rank = out_features // world
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        x = _CopyToModelParallel.apply(x)  # replicate input, sum grads on backward
        y = F.linear(x, self.weight, self.bias)
        return _all_gather_last_dim(y) if self.gather_output else y


class RowParallelLinear(nn.Module):
    """Linear with the input dimension sharded across ranks.

    Expects an already-sharded input (e.g. the output of a preceding
    ColumnParallelLinear with ``gather_output=False``); sums partials with an
    all-reduce.
    """

    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True):
        super().__init__()
        rank, world = _rank_world()
        assert in_features % world == 0, "in_features must divide world_size"
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_rank = in_features // world
        self.input_is_parallel = input_is_parallel
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        if not self.input_is_parallel:
            rank, world = _rank_world()
            x = x.chunk(world, dim=-1)[rank]
        partial = F.linear(x, self.weight)  # bias added after reduce
        out = _ReduceFromModelParallel.apply(partial)
        if self.bias is not None:
            out = out + self.bias
        return out
