from __future__ import annotations

"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Page-aligned memory pool.
"""

import abc
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


def alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
    device,
):
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)
    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        if prefix_lens[i] + num1 == seq_lens[i]:
            continue

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages using a dynamic blocked loop.
    # The loop bound is derived from num_part2 (runtime value), so Triton
    # generates a real loop instead of unrolling — no constexpr dependency
    # on extend size and only one kernel compilation.
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)

class DynamicPagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Paged allocator with per-request priority, dynamic slot-width halving,
    and global-pressure-driven backpressure.

    All priorities start at slot_width = layer_num (full layers, zero extra
    load/store overhead).  Both share free_pages freely with no per-priority
    cap.  When free_pages run out, ``ensure_capacity`` triggers backpressure:

    1. If low_priority_used > low_max_pages: halve low-priority slot_width
       AND cross-page repack (release physical pages back to free_pages).
    2. If high_priority_used > high_reserved_pages: halve high-priority
       slot_width (partial slots only, no physical page release).
    3. Forced halving of low, then high, as last resort.

    Per-request priority: ``is_high_priority`` can be a ``torch.Tensor[bs]``
    of bools.  Mixed-priority batches are split into two internal allocation
    calls and results are merged back in original request order.

    Low-priority repack migrates KV data across pages and updates
    ``req_to_token_pool`` (injected at construction) via a GPU lookup-table
    remap.

    Call ``reset_slot_widths()`` to restore both priorities to full layer_num.

    alloc/alloc_extend/alloc_decode return ``(primary_indices, dynamic_indices)``
    where ``dynamic_indices`` is ``None`` when slot_width == layer_num.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
        req_to_token_pool=None,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self._req_to_token_pool = req_to_token_pool
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")

        if not hasattr(kvcache, "get_layer_num"):
            raise TypeError(
                "DynamicPagedTokenToKVPoolAllocator expects a KV cache with "
                "`get_layer_num()`"
            )

        self.layer_num = int(kvcache.get_layer_num())
        self.high_priority_layers = int(
            getattr(kvcache, "high_priority_layers", self.layer_num)
        )
        self.low_priority_layers = int(getattr(kvcache, "low_priority_layers", 2))

        # Page budget formulas
        self.low_max_pages = int(
            self.num_pages * self.low_priority_layers
            / (self.layer_num + self.low_priority_layers)
        )
        self.dynamic_max_pages = int(
            (self.num_pages * 2 + self.layer_num + 2 - 1) // (self.layer_num + 2)
        )
        self.high_reserved_pages = (
            self.num_pages - self.low_max_pages - self.dynamic_max_pages
        )

        self.min_slot_width = 2
        self.high_current_slot_width = self.layer_num
        self.low_current_slot_width = self.layer_num

        # Per-layer stride in the flat KV buffer: (num_pages + 1) * page_size
        # = size + page_size.  Used by the flat-buffer encoding so that
        # slot_start maps to a layer offset within the flattened buffer.
        self.pool_stride = (self.num_pages + 1) * self.page_size

        self.clear()

    def debug_print(self) -> str:
        return ""

    def backup_state(self):
        return {
            "free_pages": self.free_pages.clone(),
            "release_pages": self.release_pages.clone(),
            "page_priority": self.page_priority.clone(),
            "slot_width": self.slot_width.clone(),
            "is_not_in_free_group": self.is_not_in_free_group,
            "free_group": list(self.free_group),
            "high_priority_used": self.high_priority_used,
            "low_priority_used": self.low_priority_used,
            "dynamic_priority_used": self.dynamic_priority_used,
            "_high_priority_partial_pages": self._high_priority_partial_pages.clone(),
            "_high_priority_partial_slot_starts": self._high_priority_partial_slot_starts.clone(),
            "_low_priority_partial_pages": self._low_priority_partial_pages.clone(),
            "_low_priority_partial_slot_starts": self._low_priority_partial_slot_starts.clone(),
            "_dynamic_partial_pages": self._dynamic_partial_pages.clone(),
            "_dynamic_partial_slot_starts": self._dynamic_partial_slot_starts.clone(),
            "high_current_slot_width": self.high_current_slot_width,
            "low_current_slot_width": self.low_current_slot_width,
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            raise TypeError(
                "DynamicPagedTokenToKVPoolAllocator.restore_state expects the "
                "dict returned by backup_state()"
            )

        self.free_pages = state["free_pages"].clone()
        self.release_pages = state["release_pages"].clone()
        self.page_priority = state["page_priority"].clone()
        self.slot_width = state["slot_width"].clone()
        self.is_not_in_free_group = state["is_not_in_free_group"]
        self.free_group = list(state["free_group"])
        self.high_priority_used = state["high_priority_used"]
        self.low_priority_used = state["low_priority_used"]
        self.dynamic_priority_used = state["dynamic_priority_used"]
        self._high_priority_partial_pages = state["_high_priority_partial_pages"].clone()
        self._high_priority_partial_slot_starts = state["_high_priority_partial_slot_starts"].clone()
        self._low_priority_partial_pages = state["_low_priority_partial_pages"].clone()
        self._low_priority_partial_slot_starts = state["_low_priority_partial_slot_starts"].clone()
        self._dynamic_partial_pages = state["_dynamic_partial_pages"].clone()
        self._dynamic_partial_slot_starts = state["_dynamic_partial_slot_starts"].clone()
        self.high_current_slot_width = state["high_current_slot_width"]
        self.low_current_slot_width = state["low_current_slot_width"]

    def clear(self):
        # Need to track the priority of each page
        self.page_priority = torch.zeros(
            (self.num_pages,), dtype=torch.int64, device=self.device
        )
        # Need to track the slot_width of each page
        self.slot_width = torch.zeros(
            (self.num_pages,), dtype=torch.int64, device=self.device
        )

        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

        self.is_not_in_free_group = True
        self.free_group = []
        self.high_priority_used = 0
        self.low_priority_used = 0
        self.dynamic_priority_used = 0
        self.high_current_slot_width = self.layer_num
        self.low_current_slot_width = self.layer_num

        # Each entry represents a single available slot on a partial page;
        # ``len(...)`` gives the number of free partial slots
        self._high_priority_partial_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self._high_priority_partial_slot_starts = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self._low_priority_partial_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self._low_priority_partial_slot_starts = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self._dynamic_partial_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self._dynamic_partial_slot_starts = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )

    def available_size(
        self,
        is_high_priority: bool = True,
        priority_code: Optional[int] = None,
    ):
        if priority_code is None:
            priority_code = self._get_priority_code(is_high_priority)

        if priority_code == 1:
            partial_count = len(self._high_priority_partial_pages)
            slot_width = self.high_current_slot_width
        elif priority_code == 2:
            partial_count = len(self._low_priority_partial_pages)
            slot_width = self.low_current_slot_width
        elif priority_code == 3:
            partial_count = len(self._dynamic_partial_pages)
            slot_width = 1
        else:
            raise ValueError(f"Unknown priority_code {priority_code}")

        slots_per_page = self.layer_num // slot_width
        free_count = len(self.free_pages) + len(self.release_pages)
        return (free_count * slots_per_page + partial_count) * self.page_size

    @staticmethod
    def _resolve_priority(is_high_priority, bs: int) -> list:
        """Normalize is_high_priority into a CPU list of bools."""
        if is_high_priority is None:
            return [True] * bs
        if isinstance(is_high_priority, bool):
            return [is_high_priority] * bs
        if isinstance(is_high_priority, torch.Tensor):
            return is_high_priority.bool().tolist()
        return list(is_high_priority)

    def _allocate_and_stamp(self, num_pages, priority_code):
        """_allocate_pages + set page_priority. Returns (pages, slot_starts) or None."""
        alloc_result = self._allocate_pages(
            num_pages, is_high_priority=(priority_code == 1),
            return_slot_indices=True, priority_code=priority_code,
        )
        if alloc_result is None:
            return None
        pages, slot_starts = alloc_result
        if pages.numel() > 0:
            unique_rows = torch.unique(pages - 1)
            self.page_priority[unique_rows] = priority_code
        return pages, slot_starts

    def _allocate_dynamic(self, num_pages, priority_code, slot_width_override=None):
        """Allocate dynamic pages and stamp them.

        When slot_width_override is given, each dynamic page has that many
        physical positions (used as ping-pong buffer for non-resident layers).
        """
        alloc_result = self._allocate_pages(
            num_pages,
            is_high_priority=(priority_code == 1),
            return_slot_indices=True,
            priority_code=3,
            slot_width_override=slot_width_override,
        )
        if alloc_result is None:
            return None
        pages, slot_starts = alloc_result
        if pages.numel() > 0:
            unique_rows = torch.unique(pages - 1)
            self.page_priority[unique_rows] = 3
        return pages, slot_starts

    def alloc(
        self,
        need_size: int,
        is_high_priority=None,
    ):
        if self.debug_mode:
            assert need_size % self.page_size == 0

        num_pages = need_size // self.page_size
        if num_pages == 0:
            empty = torch.empty((0,), dtype=torch.int64, device=self.device)
            return (empty, None)

        hp_list = self._resolve_priority(is_high_priority, 1)
        priority_code = 1 if hp_list[0] else 2
        self.ensure_capacity(num_pages, priority_code)

        slot_width = self._get_current_slot_width(priority_code)
        alloc_result = self._allocate_and_stamp(num_pages, priority_code)
        if alloc_result is None:
            return None

        out_pages, out_slot_starts = alloc_result
        out_indices = out_pages + (out_slot_starts - 1) * (self.num_pages + 1)
        out_indices = (
            out_indices[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        dynamic_indices = None
        if slot_width < self.layer_num:
            dyn_result = self._allocate_dynamic(
                num_pages, priority_code, slot_width_override=slot_width
            )
            if dyn_result is None:
                return None
            dyn_pages, dyn_ss = dyn_result
            dyn_combined = dyn_pages + (dyn_ss - 1) * (self.num_pages + 1)
            dynamic_indices = (
                dyn_combined[:, None] * self.page_size
                + torch.arange(self.page_size, device=self.device)
            ).reshape(-1)

        return (out_indices, dynamic_indices)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        is_high_priority=None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        hp_list = self._resolve_priority(is_high_priority, bs)
        high_idx = [i for i, h in enumerate(hp_list) if h]
        low_idx = [i for i, h in enumerate(hp_list) if not h]
        has_high = len(high_idx) > 0
        has_low = len(low_idx) > 0

        if has_high and has_low:
            high_pages_needed = get_num_new_pages(
                seq_lens_cpu[high_idx], self.page_size,
                prefix_lens_cpu[high_idx],
            )
            low_pages_needed = get_num_new_pages(
                seq_lens_cpu[low_idx], self.page_size,
                prefix_lens_cpu[low_idx],
            )
        elif has_high:
            high_pages_needed = get_num_new_pages(
                seq_lens_cpu, self.page_size, prefix_lens_cpu,
            )
            low_pages_needed = 0
        else:
            high_pages_needed = 0
            low_pages_needed = get_num_new_pages(
                seq_lens_cpu, self.page_size, prefix_lens_cpu,
            )

        if high_pages_needed > 0:
            self.ensure_capacity(high_pages_needed, 1)
        if low_pages_needed > 0:
            self.ensure_capacity(low_pages_needed, 2)

        if has_high:
            high_result = self._allocate_and_stamp(high_pages_needed, 1)
            if high_result is None:
                return None
            high_slot_width = self.high_current_slot_width
        if has_low:
            low_result = self._allocate_and_stamp(low_pages_needed, 2)
            if low_result is None:
                return None
            low_slot_width = self.low_current_slot_width

        if has_high and has_low:
            # Pass just page_ids (within-layer addressing)
            high_pages = high_result[0]
            low_pages = low_result[0]

            reorder = torch.tensor(
                high_idx + low_idx, dtype=torch.int64, device=self.device,
            )
            prefix_lens_r = prefix_lens[reorder]
            seq_lens_r = seq_lens[reorder]
            last_loc_r = last_loc[reorder]
            combined = torch.cat([high_pages, low_pages])

            out_indices_r = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device,
            )
            alloc_extend_kernel[(bs,)](
                prefix_lens_r, seq_lens_r, last_loc_r, combined,
                out_indices_r, next_power_of_2(bs), self.page_size,
            )

            extend_lens_r = seq_lens_r - prefix_lens_r
            end_pos_r = torch.cumsum(extend_lens_r, 0)
            start_pos_r = end_pos_r - extend_lens_r

            extend_lens_orig = seq_lens - prefix_lens
            end_pos_orig = torch.cumsum(extend_lens_orig, 0)
            start_pos_orig = end_pos_orig - extend_lens_orig

            out_indices = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device,
            )
            for i in range(bs):
                orig_i = (high_idx + low_idx)[i]
                src_s = start_pos_r[i].item()
                src_e = end_pos_r[i].item()
                dst_s = start_pos_orig[orig_i].item()
                dst_e = end_pos_orig[orig_i].item()
                out_indices[dst_s:dst_e] = out_indices_r[src_s:src_e]

            slot_width = min(high_slot_width, low_slot_width)
        else:
            priority_code = 1 if has_high else 2
            result = high_result if has_high else low_result
            combined = result[0]  # just page_ids
            slot_width = self._get_current_slot_width(priority_code)

            out_indices = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device,
            )
            alloc_extend_kernel[(bs,)](
                prefix_lens, seq_lens, last_loc, combined,
                out_indices, next_power_of_2(bs), self.page_size,
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        dynamic_indices = None
        if slot_width < self.layer_num:
            total_pages = high_pages_needed + low_pages_needed
            dyn_result = self._allocate_dynamic(
                total_pages, 3, slot_width_override=slot_width
            )
            if dyn_result is None:
                return None
            dyn_pages = dyn_result[0]  # just page_ids

            dynamic_indices = torch.empty(
                (extend_num_tokens,), dtype=torch.int64, device=self.device,
            )
            alloc_extend_kernel[(bs,)](
                prefix_lens, seq_lens, last_loc, dyn_pages,
                dynamic_indices, next_power_of_2(bs), self.page_size,
            )
            if self.debug_mode:
                assert len(torch.unique(dynamic_indices)) == len(dynamic_indices)

        return (out_indices, dynamic_indices)

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        is_high_priority=None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        hp_list = self._resolve_priority(is_high_priority, bs)
        high_idx = [i for i, h in enumerate(hp_list) if h]
        low_idx = [i for i, h in enumerate(hp_list) if not h]
        has_high = len(high_idx) > 0
        has_low = len(low_idx) > 0

        if has_high and has_low:
            high_pages_needed = get_num_new_pages(
                seq_lens_cpu[high_idx], self.page_size, decode=True,
            )
            low_pages_needed = get_num_new_pages(
                seq_lens_cpu[low_idx], self.page_size, decode=True,
            )
        elif has_high:
            high_pages_needed = get_num_new_pages(
                seq_lens_cpu, self.page_size, decode=True,
            )
            low_pages_needed = 0
        else:
            high_pages_needed = 0
            low_pages_needed = get_num_new_pages(
                seq_lens_cpu, self.page_size, decode=True,
            )

        if high_pages_needed > 0:
            self.ensure_capacity(high_pages_needed, 1)
        if low_pages_needed > 0:
            self.ensure_capacity(low_pages_needed, 2)

        if has_high:
            high_result = self._allocate_and_stamp(high_pages_needed, 1)
            if high_result is None:
                return None
            high_slot_width = self.high_current_slot_width
        if has_low:
            low_result = self._allocate_and_stamp(low_pages_needed, 2)
            if low_result is None:
                return None
            low_slot_width = self.low_current_slot_width

        if has_high and has_low:
            high_pages = high_result[0]
            low_pages = low_result[0]

            reorder = torch.tensor(
                high_idx + low_idx, dtype=torch.int64, device=self.device,
            )
            seq_lens_r = seq_lens[reorder]
            last_loc_r = last_loc[reorder]
            combined = torch.cat([high_pages, low_pages])

            out_indices_r = torch.empty((bs,), dtype=torch.int64, device=self.device)
            alloc_decode_kernel[(bs,)](
                seq_lens_r, last_loc_r, combined,
                out_indices_r, next_power_of_2(bs), self.page_size,
            )

            out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
            out_indices[reorder] = out_indices_r

            slot_width = min(high_slot_width, low_slot_width)
        else:
            priority_code = 1 if has_high else 2
            result = high_result if has_high else low_result
            combined = result[0]  # just page_ids
            slot_width = self._get_current_slot_width(priority_code)

            out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
            alloc_decode_kernel[(bs,)](
                seq_lens, last_loc, combined,
                out_indices, next_power_of_2(bs), self.page_size,
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        dynamic_indices = None
        if slot_width < self.layer_num:
            total_pages = high_pages_needed + low_pages_needed
            dyn_result = self._allocate_dynamic(
                total_pages, 3, slot_width_override=slot_width
            )
            if dyn_result is None:
                return None
            dyn_pages = dyn_result[0]  # just page_ids

            dynamic_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
            alloc_decode_kernel[(bs,)](
                seq_lens, last_loc, dyn_pages,
                dynamic_indices, next_power_of_2(bs), self.page_size,
            )
            if self.debug_mode:
                assert len(torch.unique(dynamic_indices)) == len(dynamic_indices)

        return (out_indices, dynamic_indices)

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(
        self,
        free_index: torch.Tensor,
    ):
        '''
        Only put page to free_pages when the whole page is freed.
        If a page is partially freed, put it to the appropriate partial pool
        (high/low/dynamic) based on page_priority.
        '''

        if free_index.numel() == 0:
            return

        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return

        # Decode (page_id, slot_start) from encoded indices.
        # alloc() encodes: idx = (page_id + (slot_start-1)*(num_pages+1))*page_size + pos
        # So: combined = idx // page_size = page_id + (slot_start-1)*(num_pages+1)
        #     page_id   = combined % (num_pages+1)          # in [1, num_pages]
        #     slot_start = combined // (num_pages+1) + 1    # 1-indexed layer start
        combined = torch.unique(free_index.reshape(-1) // self.page_size)
        page_ids = combined % (self.num_pages + 1)
        slot_starts = combined // (self.num_pages + 1) + 1

        # Filter out the dummy slot/page 0 (used for padded tokens)
        valid = page_ids > 0
        page_ids = page_ids[valid]
        slot_starts = slot_starts[valid]

        if page_ids.numel() == 0:
            return

        # For each unique page, count how many distinct slots are being freed.
        unique_pages, inverse, freed_per_page = torch.unique(
            page_ids, return_inverse=True, return_counts=True
        )
        unique_page_rows = unique_pages - 1
        unique_priorities = self.page_priority[unique_page_rows]   # 1=high, 2=low, 3=dynamic
        unique_slot_widths = self.slot_width[unique_page_rows]
        total_slots_per_page = self.layer_num // unique_slot_widths

        # Count how many slots of each freed page are already sitting in the
        # partial pools (available but not yet re-allocated). bincount's output
        # size is fixed (minlength), so no host-device sync is required.
        def _partial_counts(partial_pages: torch.Tensor) -> torch.Tensor:
            if partial_pages.numel() == 0:
                return torch.zeros(
                    self.num_pages + 1, dtype=torch.int64, device=self.device
                )
            return torch.bincount(partial_pages, minlength=self.num_pages + 1)

        high_partial_counts = _partial_counts(self._high_priority_partial_pages)
        low_partial_counts = _partial_counts(self._low_priority_partial_pages)
        dynamic_partial_counts = _partial_counts(self._dynamic_partial_pages)

        # Select the right partial count based on priority
        partial_for_freed = torch.where(
            unique_priorities == 1,
            high_partial_counts[unique_pages],
            torch.where(
                unique_priorities == 2,
                low_partial_counts[unique_pages],
                dynamic_partial_counts[unique_pages],
            ),
        )

        # A page is fully available when:
        #   freed_slots_now + already_partial_slots == total_slots_per_page
        fully_freed_mask = (freed_per_page + partial_for_freed) == total_slots_per_page
        partially_freed_mask = ~fully_freed_mask

        # ------------------------------------------------------------------
        # Fully freed pages → remove leftover partial-pool entries and return
        # the page to free_pages.
        # ------------------------------------------------------------------
        fully_freed_pages = unique_pages[fully_freed_mask]
        if fully_freed_pages.numel() > 0:
            # Purge any leftover partial-pool entries for these pages.
            if self._high_priority_partial_pages.numel() > 0:
                keep = ~torch.isin(self._high_priority_partial_pages, fully_freed_pages)
                self._high_priority_partial_pages = self._high_priority_partial_pages[keep]
                self._high_priority_partial_slot_starts = self._high_priority_partial_slot_starts[keep]
            if self._low_priority_partial_pages.numel() > 0:
                keep = ~torch.isin(self._low_priority_partial_pages, fully_freed_pages)
                self._low_priority_partial_pages = self._low_priority_partial_pages[keep]
                self._low_priority_partial_slot_starts = self._low_priority_partial_slot_starts[keep]
            if self._dynamic_partial_pages.numel() > 0:
                keep = ~torch.isin(self._dynamic_partial_pages, fully_freed_pages)
                self._dynamic_partial_pages = self._dynamic_partial_pages[keep]
                self._dynamic_partial_slot_starts = self._dynamic_partial_slot_starts[keep]

            fully_freed_priorities = unique_priorities[fully_freed_mask]
            
            # Notably, thougth we add .cpu latency here, it's likely still faster than the alternative of
            # iterating on CPU or doing multiple device synchronizations to count per-priority freed pages.
            priority_counts = torch.stack([
                (fully_freed_priorities == 1).sum(),
                (fully_freed_priorities == 2).sum(),
                (fully_freed_priorities == 3).sum(),
            ]).cpu()
            self.high_priority_used -= int(priority_counts[0])
            self.low_priority_used -= int(priority_counts[1])
            self.dynamic_priority_used -= int(priority_counts[2])

            fully_freed_rows = fully_freed_pages - 1
            self.page_priority[fully_freed_rows] = 0
            self.slot_width[fully_freed_rows] = 0
            if self.need_sort:
                self.release_pages = torch.cat((fully_freed_pages, self.release_pages))
            else:
                self.free_pages = torch.cat((fully_freed_pages, self.free_pages))

        # ------------------------------------------------------------------
        # Partially freed slots → return them to the appropriate partial pool
        # so future _allocate_pages calls can reuse them.
        # ------------------------------------------------------------------
        if partially_freed_mask.any():
            partial_slot_mask = partially_freed_mask[inverse]
            partial_page_ids = page_ids[partial_slot_mask]
            partial_slot_starts_freed = slot_starts[partial_slot_mask]

            partial_priorities = self.page_priority[partial_page_ids - 1]
            high_mask = partial_priorities == 1
            low_mask = partial_priorities == 2
            dynamic_mask = partial_priorities == 3

            if high_mask.any():
                self._high_priority_partial_pages = torch.cat(
                    (self._high_priority_partial_pages, partial_page_ids[high_mask])
                )
                self._high_priority_partial_slot_starts = torch.cat(
                    (self._high_priority_partial_slot_starts,
                     partial_slot_starts_freed[high_mask])
                )
            if low_mask.any():
                self._low_priority_partial_pages = torch.cat(
                    (self._low_priority_partial_pages, partial_page_ids[low_mask])
                )
                self._low_priority_partial_slot_starts = torch.cat(
                    (self._low_priority_partial_slot_starts,
                     partial_slot_starts_freed[low_mask])
                )
            if dynamic_mask.any():
                self._dynamic_partial_pages = torch.cat(
                    (self._dynamic_partial_pages, partial_page_ids[dynamic_mask])
                )
                self._dynamic_partial_slot_starts = torch.cat(
                    (self._dynamic_partial_slot_starts,
                     partial_slot_starts_freed[dynamic_mask])
                )

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            for free_index in self.free_group:
                self.free(free_index)
            self.free_group = []

    def _get_priority_code(self, is_high_priority: bool) -> int:
        return 1 if is_high_priority else 2

    def _account_alloc(self, num_pages: int, priority_code: int):
        if priority_code == 1:
            self.high_priority_used += num_pages
        elif priority_code == 2:
            self.low_priority_used += num_pages
        elif priority_code == 3:
            self.dynamic_priority_used += num_pages

    def _get_partial_pool_attrs(self, priority_code: int):
        """Return (partial_pages_attr, partial_slot_starts_attr) for the given priority_code."""
        if priority_code == 1:
            return "_high_priority_partial_pages", "_high_priority_partial_slot_starts"
        elif priority_code == 2:
            return "_low_priority_partial_pages", "_low_priority_partial_slot_starts"
        elif priority_code == 3:
            return "_dynamic_partial_pages", "_dynamic_partial_slot_starts"
        raise ValueError(f"Unknown priority_code {priority_code}")

    def _get_max_fresh_pages(self, priority_code: int) -> int:
        """Return how many fresh raw pages this priority can still open."""
        if priority_code == 1:
            return max(self.high_reserved_pages - self.high_priority_used, 0)
        elif priority_code == 2:
            return max(self.low_max_pages - self.low_priority_used, 0)
        elif priority_code == 3:
            return max(self.dynamic_max_pages - self.dynamic_priority_used, 0)
        raise ValueError(f"Unknown priority_code {priority_code}")

    def _halve_slot_width(self, priority_code: int):
        if priority_code == 1:
            old_width = self.high_current_slot_width
        elif priority_code == 2:
            old_width = self.low_current_slot_width
        else:
            return

        if old_width <= self.min_slot_width:
            return

        new_width = old_width // 2

        page_mask = (self.page_priority == priority_code) & (self.slot_width == old_width)
        target_page_rows = torch.nonzero(page_mask, as_tuple=True)[0]
        target_page_ids = target_page_rows + 1

        if priority_code == 1:
            self.high_current_slot_width = new_width
        else:
            self.low_current_slot_width = new_width

        if target_page_ids.numel() == 0:
            return

        pp_attr, pss_attr = self._get_partial_pool_attrs(priority_code)
        partial_pages = getattr(self, pp_attr)
        partial_slot_starts = getattr(self, pss_attr)

        if partial_pages.numel() > 0:
            target_partial_mask = torch.isin(partial_pages, target_page_ids)
            target_partial_pages_t = partial_pages[target_partial_mask]
            target_partial_ss_t = partial_slot_starts[target_partial_mask]
            remaining_partial_pages = partial_pages[~target_partial_mask]
            remaining_partial_ss = partial_slot_starts[~target_partial_mask]
        else:
            target_partial_pages_t = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            target_partial_ss_t = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            remaining_partial_pages = partial_pages
            remaining_partial_ss = partial_slot_starts

        target_partial_pages_cpu = target_partial_pages_t.cpu().tolist()
        target_partial_ss_cpu = target_partial_ss_t.cpu().tolist()
        free_slots_per_page: dict = {}
        for pid, ss in zip(target_partial_pages_cpu, target_partial_ss_cpu):
            free_slots_per_page.setdefault(pid, set()).add(ss)

        old_slot_starts = list(range(1, self.layer_num + 1, old_width))

        new_partial_page_ids: list = []
        new_partial_slot_starts: list = []

        k_buf_flat = self._kvcache.k_buffer_flat
        v_buf_flat = self._kvcache.v_buffer_flat
        target_page_ids_cpu = target_page_ids.cpu().tolist()

        for old_ss in old_slot_starts:
            allocated_pids = [
                pid
                for pid in target_page_ids_cpu
                if old_ss not in free_slots_per_page.get(pid, set())
            ]

            if allocated_pids:
                allocated_pids_tensor = torch.tensor(
                    allocated_pids, dtype=torch.int64, device=self.device
                )
                # Within-layer locations: page_id * page_size + offset
                within_layer_locs = (
                    allocated_pids_tensor[:, None] * self.page_size
                    + torch.arange(self.page_size, device=self.device)
                ).reshape(-1)

                for j in range(1, old_width // 2):
                    src_layer = (old_ss - 1) + 2 * j
                    dst_layer = (old_ss - 1) + j
                    src_flat = src_layer * self.pool_stride + within_layer_locs
                    dst_flat = dst_layer * self.pool_stride + within_layer_locs
                    k_buf_flat[dst_flat] = k_buf_flat[src_flat]
                    v_buf_flat[dst_flat] = v_buf_flat[src_flat]

                for pid in allocated_pids:
                    new_partial_page_ids.append(pid)
                    new_partial_slot_starts.append(old_ss + new_width)

            for pid in target_page_ids_cpu:
                if old_ss in free_slots_per_page.get(pid, set()):
                    new_partial_page_ids.append(pid)
                    new_partial_slot_starts.append(old_ss)
                    new_partial_page_ids.append(pid)
                    new_partial_slot_starts.append(old_ss + new_width)

        self.slot_width[target_page_rows] = new_width

        if new_partial_page_ids:
            new_pp = torch.tensor(
                new_partial_page_ids, dtype=torch.int64, device=self.device
            )
            new_pss = torch.tensor(
                new_partial_slot_starts, dtype=torch.int64, device=self.device
            )
            final_pp = torch.cat([remaining_partial_pages, new_pp])
            final_pss = torch.cat([remaining_partial_ss, new_pss])
        else:
            final_pp = remaining_partial_pages
            final_pss = remaining_partial_ss

        setattr(self, pp_attr, final_pp)
        setattr(self, pss_attr, final_pss)

    def _halve_and_repack(self, priority_code: int, max_keep_pages: int):
        """Halve a priority's slot_width AND repack excess pages to release them.

        Phase A: in-page halving (compact layers, create partial slots).
        Phase B: cross-page repack — only for pages exceeding max_keep_pages.
                 Pair donor/receiver pages, migrate donor's occupied slot to
                 receiver's free slot, release donor back to free_pages.
        Phase C: update req_to_token_pool via lookup-table remap.

        Args:
            priority_code: 1 (high) or 2 (low).
            max_keep_pages: pages within this limit keep partial slots only
                            (no cross-page migration). Pages beyond this
                            limit are paired and repacked to release physical
                            pages.
        """
        if priority_code == 1:
            old_width = self.high_current_slot_width
            used_attr = "high_priority_used"
        else:
            old_width = self.low_current_slot_width
            used_attr = "low_priority_used"

        if old_width <= self.min_slot_width:
            return

        new_width = old_width // 2

        # Phase A: in-page halving
        self._halve_slot_width(priority_code)

        # Phase B: cross-page repack for excess pages
        new_slots_per_page = self.layer_num // new_width
        if new_slots_per_page < 2:
            return

        current_used = getattr(self, used_attr)
        excess = current_used - max_keep_pages
        if excess <= 0:
            return

        pp_attr, pss_attr = self._get_partial_pool_attrs(priority_code)
        partial_pages = getattr(self, pp_attr)
        partial_ss = getattr(self, pss_attr)

        if partial_pages.numel() == 0:
            return

        page_counts = torch.bincount(partial_pages, minlength=self.num_pages + 1)

        candidate_mask = (
            (self.page_priority == priority_code)
            & (self.slot_width == new_width)
            & (page_counts[1:] == 1)
        )
        candidate_rows = torch.nonzero(candidate_mask, as_tuple=True)[0]
        if candidate_rows.numel() < 2:
            return

        candidate_page_ids = candidate_rows + 1

        max_pairs = min(candidate_page_ids.numel() // 2, (excess + 1) // 2)
        receivers = candidate_page_ids[:max_pairs]
        donors = candidate_page_ids[max_pairs : 2 * max_pairs]

        donors_cpu = donors.cpu().tolist()
        receivers_cpu = receivers.cpu().tolist()
        partial_pages_cpu = partial_pages.cpu().tolist()
        partial_ss_cpu = partial_ss.cpu().tolist()

        free_ss_by_page: dict = {}
        for pid, ss in zip(partial_pages_cpu, partial_ss_cpu):
            free_ss_by_page.setdefault(pid, []).append(ss)

        all_slot_starts = list(range(1, self.layer_num + 1, new_width))

        donor_occupied_ss = []
        receiver_free_ss = []
        valid_pairs = []
        for i in range(len(donors_cpu)):
            d_free = free_ss_by_page.get(donors_cpu[i], [])
            r_free = free_ss_by_page.get(receivers_cpu[i], [])
            if len(d_free) != 1 or len(r_free) != 1:
                continue
            d_occ = [ss for ss in all_slot_starts if ss not in d_free]
            if len(d_occ) != new_slots_per_page - 1:
                continue
            donor_occupied_ss.append(d_occ[0])
            receiver_free_ss.append(r_free[0])
            valid_pairs.append(i)

        if not valid_pairs:
            return

        valid_idx = torch.tensor(valid_pairs, dtype=torch.long, device=self.device)
        donors = donors[valid_idx]
        receivers = receivers[valid_idx]

        k_buf_flat = self._kvcache.k_buffer_flat
        v_buf_flat = self._kvcache.v_buffer_flat

        all_old_locs = []
        all_new_locs = []

        page_offsets = torch.arange(self.page_size, device=self.device)

        for i in range(len(valid_pairs)):
            d_pid = donors[i]
            r_pid = receivers[i]
            d_ss = donor_occupied_ss[i]
            r_ss = receiver_free_ss[i]

            d_within_layer = d_pid * self.page_size + page_offsets
            r_within_layer = r_pid * self.page_size + page_offsets

            # Encoded locs for req_to_token remap
            d_combined = d_pid + (d_ss - 1) * (self.num_pages + 1)
            r_combined = r_pid + (r_ss - 1) * (self.num_pages + 1)
            d_encoded = d_combined * self.page_size + page_offsets
            r_encoded = r_combined * self.page_size + page_offsets

            for k in range(new_width):
                src_layer = (d_ss - 1) + k
                dst_layer = (r_ss - 1) + k
                src_flat = src_layer * self.pool_stride + d_within_layer
                dst_flat = dst_layer * self.pool_stride + r_within_layer
                k_buf_flat[dst_flat] = k_buf_flat[src_flat]
                v_buf_flat[dst_flat] = v_buf_flat[src_flat]

            all_old_locs.append(d_encoded)
            all_new_locs.append(r_encoded)

        # Phase C: remap req_to_token_pool (primary + dynamic)
        if self._req_to_token_pool is not None and all_old_locs:
            old_locs_cat = torch.cat(all_old_locs)
            new_locs_cat = torch.cat(all_new_locs)

            pool = self._req_to_token_pool.req_to_token
            max_val = int(pool.max().item()) + 1
            remap = torch.arange(max_val, device=self.device, dtype=pool.dtype)
            remap[old_locs_cat.to(pool.dtype)] = new_locs_cat.to(pool.dtype)
            pool[:] = remap[pool.long()]

        remove_set = torch.cat([donors, receivers])
        keep_mask = ~torch.isin(partial_pages, remove_set)
        setattr(self, pp_attr, partial_pages[keep_mask])
        setattr(self, pss_attr, partial_ss[keep_mask])

        donor_rows = donors - 1
        self.page_priority[donor_rows] = 0
        self.slot_width[donor_rows] = 0
        setattr(self, used_attr, getattr(self, used_attr) - len(donors))
        if self.need_sort:
            self.release_pages = torch.cat([donors, self.release_pages])
        else:
            self.free_pages = torch.cat([donors, self.free_pages])

    def _get_current_slot_width(self, priority_code: int) -> int:
        if priority_code == 1:
            return self.high_current_slot_width
        elif priority_code == 2:
            return self.low_current_slot_width
        return 1

    def _get_partial_slot_count(self, priority_code: int) -> int:
        pp_attr, _ = self._get_partial_pool_attrs(priority_code)
        return len(getattr(self, pp_attr))

    def ensure_capacity(self, num_new_slots: int, priority_code: int):
        """Ensure enough free pages + partial slots for an upcoming allocation.

        Triggers backpressure (halving) if free_pages are insufficient.
        Called BEFORE alloc/alloc_extend/alloc_decode Triton kernels so that
        existing req_to_token_pool entries are never invalidated.
        """
        if num_new_slots <= 0:
            return

        while True:
            slot_width = self._get_current_slot_width(priority_code)
            slots_per_page = self.layer_num // slot_width
            partial_slots = self._get_partial_slot_count(priority_code)
            need_fresh = max(0, num_new_slots - partial_slots)
            need_fresh_pages = (need_fresh + slots_per_page - 1) // slots_per_page

            total_free = len(self.free_pages) + len(self.release_pages)
            if need_fresh_pages <= total_free:
                return

            if not self._try_backpressure(priority_code):
                return

    def _try_backpressure(self, requesting_priority_code: int) -> bool:
        """Decide who backs off and trigger halving. Returns True if progress was made."""
        # 1. Low over budget → halve + full repack (release all excess pages)
        if (
            self.low_priority_used > self.low_max_pages
            and self.low_current_slot_width > self.min_slot_width
        ):
            self._halve_and_repack(2, self.low_max_pages)
            return True

        # 2. High over budget → halve + partial repack (only excess over reserved)
        if (
            self.high_priority_used > self.high_reserved_pages
            and self.high_current_slot_width > self.min_slot_width
        ):
            self._halve_and_repack(1, self.high_reserved_pages)
            return True

        # In fact, due to our proactive control, the following two cases
        # will not be happened if can't allocate pages.
        # 3. Force low halve + repack
        if self.low_current_slot_width > self.min_slot_width:
            self._halve_and_repack(2, self.low_max_pages)
            return True

        # 4. Force high halve + partial repack
        if self.high_current_slot_width > self.min_slot_width:
            self._halve_and_repack(1, self.high_reserved_pages)
            return True

        return False

    def get_resident_layers(self, slot_width: int) -> list:
        """Return the evenly-spaced resident layer IDs for a given slot_width.

        With stride = layer_num // slot_width, resident layers are
        {0, stride, 2*stride, ...} — `slot_width` layers total.
        """
        stride = self.layer_num // slot_width
        return [k * stride for k in range(slot_width)]

    def reset_slot_widths(self):
        """Reset per-priority slot widths to layer_num for new allocations.

        Existing pages retain their per-page slot_width until fully freed.
        Partial pool entries from halved pages remain valid because each
        entry carries its own (page_id, slot_start) and free() reads the
        per-page slot_width array.
        """
        self.high_current_slot_width = self.layer_num
        self.low_current_slot_width = self.layer_num

    def _allocate_pages(
        self,
        num_pages: int,
        is_high_priority: bool,
        return_slot_indices: bool = False,
        priority_code: Optional[int] = None,
        slot_width_override: Optional[int] = None,
    ):
        if num_pages == 0:
            empty_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
            if return_slot_indices:
                empty_slots = torch.empty((0,), dtype=torch.int64, device=self.device)
                return empty_pages, empty_slots
            return empty_pages

        if priority_code is None:
            priority_code = self._get_priority_code(is_high_priority)

        slot_width = slot_width_override if slot_width_override is not None else (
            self.high_current_slot_width if is_high_priority else self.low_current_slot_width
        )
        if slot_width <= 0:
            raise RuntimeError("priority layer count must be positive")
        if self.layer_num % slot_width != 0:
            raise RuntimeError(
                "layer_num must be divisible by the priority slot width"
            )
        slots_per_page = self.layer_num // slot_width

        partial_pages_attr, partial_slot_starts_attr = self._get_partial_pool_attrs(
            priority_code
        )
        partial_pages = getattr(self, partial_pages_attr)
        partial_slot_starts = getattr(self, partial_slot_starts_attr)

        total_partial_slots = len(partial_pages)

        need_pages = num_pages
        need_fresh_slots = max(need_pages - total_partial_slots, 0)
        need_fresh_page_count = (need_fresh_slots + slots_per_page - 1) // slots_per_page

        if self.need_sort and need_fresh_page_count > len(self.free_pages):
            self.merge_and_sort_free()

        available_free = min(need_fresh_page_count, len(self.free_pages))
        available_with_free = total_partial_slots + available_free * slots_per_page
        if need_pages > available_with_free:
            return None

        out_page_chunks = []
        out_slot_chunks = []

        if total_partial_slots > 0:
            reusable_count = min(need_pages, total_partial_slots)
            out_page_chunks.append(partial_pages[:reusable_count])
            out_slot_chunks.append(partial_slot_starts[:reusable_count])
            partial_pages = partial_pages[reusable_count:]
            partial_slot_starts = partial_slot_starts[reusable_count:]
            need_pages -= reusable_count

        if need_pages > 0:
            fresh_page_count = min(
                (need_pages + slots_per_page - 1) // slots_per_page,
                len(self.free_pages),
            )
            fresh_pages = self.free_pages[:fresh_page_count]
            self.free_pages = self.free_pages[fresh_page_count:]

            # Update self.slot_width for the fresh pages
            page_rows = fresh_pages - 1
            self.slot_width[page_rows] = slot_width

            # Expand each fresh page into ``slots_per_page`` (page_id, slot_start)
            # pairs. Row-major order matches the original allocation order.
            slot_start_template = torch.arange(
                1,
                self.layer_num + 1,
                slot_width,
                dtype=torch.int64,
                device=self.device,
            )
            fresh_page_expanded = fresh_pages.repeat_interleave(slots_per_page)
            fresh_slot_expanded = slot_start_template.repeat(fresh_page_count)

            out_page_chunks.append(fresh_page_expanded[:need_pages])
            out_slot_chunks.append(fresh_slot_expanded[:need_pages])

            leftover_pages = fresh_page_expanded[need_pages:]
            leftover_slot_starts = fresh_slot_expanded[need_pages:]
            if partial_pages.numel() == 0:
                partial_pages = leftover_pages
                partial_slot_starts = leftover_slot_starts
            elif leftover_pages.numel() > 0:
                partial_pages = torch.cat((partial_pages, leftover_pages), dim=0)
                partial_slot_starts = torch.cat(
                    (partial_slot_starts, leftover_slot_starts), dim=0
                )

            # Update priority account of new fresh pages
            self._account_alloc(fresh_page_count, priority_code)

        setattr(self, partial_pages_attr, partial_pages)
        setattr(self, partial_slot_starts_attr, partial_slot_starts)

        out_pages = torch.cat(out_page_chunks, dim=0)
        if return_slot_indices:
            out_slot_starts = torch.cat(out_slot_chunks, dim=0)
            return out_pages, out_slot_starts
        return out_pages

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
