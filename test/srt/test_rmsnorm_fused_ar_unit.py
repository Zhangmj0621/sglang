"""CPU unit tests for the rmsnorm-fused-ar shard math and marker state.

Runnable without GPU/distributed:
    pytest test/srt/test_rmsnorm_fused_ar_unit.py -v
(Not run on the dev machine per project convention; execute during GPU
validation.)
"""

import pytest
import torch

from sglang.srt.layers.rmsnorm_fused_ar import _token_shard


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 7, 8, 63, 64, 129, 4096])
def test_token_shard_partition(num_tokens, world_size):
    """Shards are disjoint, ordered, and cover [0, num_tokens)."""
    prev_end = 0
    total = 0
    for rank in range(world_size):
        start, end = _token_shard(num_tokens, rank, world_size)
        assert start == prev_end
        assert end >= start
        total += end - start
        prev_end = end
    assert prev_end == num_tokens
    assert total == num_tokens


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_token_shard_balance(world_size):
    """No shard exceeds another by more than one token."""
    for num_tokens in range(1, 40):
        sizes = [
            e - s
            for s, e in (
                _token_shard(num_tokens, r, world_size) for r in range(world_size)
            )
        ]
        assert max(sizes) - min(sizes) <= 1
        # Remainder goes to the lowest ranks.
        assert sizes == sorted(sizes, reverse=True)


def test_shard_marker_roundtrip():
    """The marker is a plain tensor attribute: settable, readable, deletable."""
    residual = torch.zeros(8, 16)
    assert getattr(residual, "_mega_residual_shard", None) is None
    setattr(residual, "_mega_residual_shard", (2, 4, None))
    assert getattr(residual, "_mega_residual_shard", None) == (2, 4, None)
    delattr(residual, "_mega_residual_shard")
    assert getattr(residual, "_mega_residual_shard", None) is None


def test_shard_marker_not_inherited_by_new_tensors():
    """Arithmetic produces fresh tensors without the marker — the reason the
    fused path forbids post_residual_addition after sharding."""
    residual = torch.zeros(8, 16)
    setattr(residual, "_mega_residual_shard", (0, 4, None))
    derived = residual + 1.0
    assert getattr(derived, "_mega_residual_shard", None) is None
