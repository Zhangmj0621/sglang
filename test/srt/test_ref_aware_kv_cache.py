"""Unit tests for RefAwareHiRadixCache tiered eviction."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.ref_aware_hiradix_cache import (
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    _classify_node_tier,
)


class TestClassifyNodeTier(unittest.TestCase):
    def test_unused(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 0
        assert _classify_node_tier(node) == TIER_UNUSED

    def test_low_ref(self):
        node = TreeNode()
        node.high_ref = 0
        node.low_ref = 1
        assert _classify_node_tier(node) == TIER_LOW_REF

    def test_high_ref(self):
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 0
        assert _classify_node_tier(node) == TIER_HIGH_REF

    def test_high_ref_overrides_low(self):
        node = TreeNode()
        node.high_ref = 1
        node.low_ref = 5
        assert _classify_node_tier(node) == TIER_HIGH_REF


class TestTreeNodeRefFields(unittest.TestCase):
    def test_default_zero(self):
        node = TreeNode()
        assert node.high_ref == 0
        assert node.low_ref == 0


if __name__ == "__main__":
    unittest.main()
