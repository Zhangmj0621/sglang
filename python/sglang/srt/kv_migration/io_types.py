"""Internal dataclasses for KVMigrationManager state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


@dataclass
class PendingMigration:
    """In-flight migration on the target side, awaiting /commit or watchdog."""

    input_ids: List[int]
    extra_key: Optional[str]
    full_key: "RadixKey"
    matched_aligned: int
    matched_node: "TreeNode"
    host_locked_nodes: List["TreeNode"]
    host_tail_indices: "torch.Tensor"
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class TransferTarget:
    """One peer rank's metadata for /transfer_request_kvcache."""

    tp: int
    pp: int
    session_id: str
    host_kv_data_ptrs: List[int]
    host_kv_item_lens: List[int]
    kv_indices: List[int]
