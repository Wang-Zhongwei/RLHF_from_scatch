"""Correctness test for tensor-parallel linears.

On a single process (world_size=1) the sharded layers must be numerically
identical to a plain nn.Linear — this guards the sharding/collective math. The
true multi-rank equivalence is exercised under torchrun in CI/slurm.
"""
import torch

from rlhf.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


def test_column_parallel_matches_dense_single_rank():
    torch.manual_seed(0)
    layer = ColumnParallelLinear(8, 16, bias=True, gather_output=True)
    x = torch.randn(3, 8)
    ref = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(layer(x), ref, atol=1e-5)


def test_row_parallel_matches_dense_single_rank():
    torch.manual_seed(0)
    layer = RowParallelLinear(8, 16, bias=True, input_is_parallel=True)
    x = torch.randn(3, 8)
    ref = torch.nn.functional.linear(x, layer.weight) + layer.bias
    assert torch.allclose(layer(x), ref, atol=1e-5)


def test_column_then_row_backward():
    col = ColumnParallelLinear(8, 16, gather_output=False)
    row = RowParallelLinear(16, 8, input_is_parallel=True)
    x = torch.randn(2, 8, requires_grad=True)
    row(col(x)).sum().backward()
    assert x.grad is not None
