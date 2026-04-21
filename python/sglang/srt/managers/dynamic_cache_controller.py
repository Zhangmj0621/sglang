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

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheAck,
    HiCacheController,
)
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import DynamicPagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

device_module = get_device_module()


class SlotLoadingEvent:
    """Per-layer CUDA events for lazy KV cache loading.

    Supports a ping-pong protocol between load stream and compute stream:
    - ready_events[layer_id]: recorded by load stream after H2D copy completes.
    - consumed_events[layer_id]: recorded by compute stream when the next layer
      is accessed (meaning this layer's attention kernel is done).

    Events are created for ALL layers (not just covered ones), so that future
    load logic for uncovered layers can plug into the same protocol.

    The load stream inserts wait_consumed barriers so that each layer (except the
    first) only begins loading after the previous layer has been consumed. This
    prevents data corruption when layers share the same physical device buffer.
    """

    def __init__(self, layer_num: int = 0):
        self.start_event = device_module.Event()
        self.layer_num = layer_num
        self.ready_events: List[Any] = [
            device_module.Event() for _ in range(layer_num)
        ]
        self.consumed_events: List[Any] = [
            device_module.Event() for _ in range(layer_num)
        ]
        self.layer_to_slot: List[int] = []
        self.covered_layers: set = set()
        self.ordered_layers: List[int] = []
        self._finish_event = None
        # Write-back completion events for dynamic page ping-pong (2 positions)
        self.wb_complete_events: List[Any] = [
            device_module.Event() for _ in range(2)
        ]

    def add_layer(self, layer_id: int):
        """Mark a layer as covered (will be loaded by start_loading)."""
        self.covered_layers.add(layer_id)

    def complete_layer(self, layer_id: int):
        """Record ready event after loading layer_id (called on load stream)."""
        self.ready_events[layer_id].record()

    def wait_ready(self, layer_id: int):
        """Make current stream wait until layer_id is loaded."""
        if layer_id < 0 or layer_id >= self.layer_num:
            return
        device_module.current_stream().wait_event(self.ready_events[layer_id])

    def signal_consumed(self, layer_id: int):
        """Record consumed event for layer_id (called on compute stream)."""
        if layer_id < 0 or layer_id >= self.layer_num:
            return
        self.consumed_events[layer_id].record()

    def wait_consumed(self, layer_id: int):
        """Make current stream wait until layer_id is consumed."""
        if layer_id < 0 or layer_id >= self.layer_num:
            return
        device_module.current_stream().wait_event(self.consumed_events[layer_id])

    def build_layer_mapping(self, slot_groups: Dict[int, int], layer_num: int):
        """Build layer_to_slot: for each 0-indexed layer, which slot_start covers it.

        Args:
            slot_groups: {slot_start: slot_width} where slot_start is 1-indexed.
            layer_num: total number of layers.
        """
        self.layer_to_slot = [-1] * layer_num
        for ss, sw in slot_groups.items():
            for layer in range(ss - 1, ss - 1 + sw):
                if 0 <= layer < layer_num:
                    self.layer_to_slot[layer] = ss

    # backward compat alias
    def wait(self, layer_id: int):
        self.wait_ready(layer_id)

    @property
    def finish_event(self):
        if self._finish_event is not None:
            return self._finish_event
        if not self.ordered_layers:
            self._finish_event = device_module.Event()
            return self._finish_event
        last_layer = self.ordered_layers[-1]
        self._finish_event = self.ready_events[last_layer]
        return self._finish_event


class ForwardLayerSchedule:
    """Pre-computed load/write-back schedule for one forward pass.

    When slot_width < layer_num, only `slot_width` evenly-spaced layers are
    resident in HBM.  The remaining layers must be streamed through 2 dynamic
    buffer positions (ping-pong) during the forward pass.

    Attributes:
        resident_layers: set of layer IDs that live on the primary page.
        non_resident: list of non-resident layer IDs in forward order.
        dyn_pos: {layer_id: 0 or 1} — which dynamic buffer position holds it.
        wb_before_load: {layer_to_load: layer_to_writeback} — before loading
            ``layer_to_load`` into a dyn pos, the previous occupant
            ``layer_to_writeback`` must be written back first.
        final_wb: layers whose data must be flushed after the last compute.
    """

    def __init__(self, layer_num: int, slot_width: int):
        self.layer_num = layer_num
        self.slot_width = slot_width
        self.stride = layer_num // slot_width

        self.resident_layers = set(k * self.stride for k in range(slot_width))
        self.non_resident = [
            i for i in range(layer_num) if i not in self.resident_layers
        ]

        self.dyn_pos: Dict[int, int] = {}
        for idx, lid in enumerate(self.non_resident):
            self.dyn_pos[lid] = idx % 2

        self.wb_before_load: Dict[int, int] = {}
        self.final_wb: List[int] = []
        self._compute_wb_schedule()

    def _compute_wb_schedule(self):
        last_on_pos: Dict[int, Optional[int]] = {0: None, 1: None}
        for lid in self.non_resident:
            pos = self.dyn_pos[lid]
            prev = last_on_pos[pos]
            if prev is not None:
                self.wb_before_load[lid] = prev
            last_on_pos[pos] = lid
        self.final_wb = [lid for lid in last_on_pos.values() if lid is not None]

    def is_resident(self, layer_id: int) -> bool:
        return layer_id in self.resident_layers

    def get_dyn_pos(self, layer_id: int) -> int:
        return self.dyn_pos[layer_id]


class SlotDoneCounter:
    """Drop-in replacement for LayerDoneCounter using per-layer lazy loading events.

    Maintains the same external interface: update_producer(), set_consumer(),
    wait_until(layer_id), reset(). Registered on the KV pool via
    register_layer_transfer_counter().

    Every layer participates in the consumed/ready protocol (not just covered
    layers), so that future load logic for currently-uncovered layers can plug
    in without changing the synchronization mechanism.
    """

    def __init__(self, layer_num: int):
        self.layer_num = layer_num
        self.num_counters = 3
        self.events: List[SlotLoadingEvent] = [
            SlotLoadingEvent(layer_num) for _ in range(self.num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1
        self._last_waited_layer: int = -1

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        prev = self.events[self.producer_index]
        if prev.ordered_layers and not prev.finish_event.query():
            raise RuntimeError(
                "Producer finish event should be ready before being reused."
            )
        self.events[self.producer_index] = SlotLoadingEvent(self.layer_num)
        return self.producer_index

    def get_current_producer_event(self) -> SlotLoadingEvent:
        return self.events[self.producer_index]

    def set_consumer(self, index: int):
        self.consumer_index = index
        self._last_waited_layer = -1

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        evt = self.events[self.consumer_index]
        if self._last_waited_layer >= 0:
            evt.signal_consumed(self._last_waited_layer)
        evt.wait_ready(threshold)
        self._last_waited_layer = threshold

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1
        self._last_waited_layer = -1


class DynamicCacheOperation(CacheOperation):
    """CacheOperation extended with per-token slot metadata."""

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        slot_starts: Optional[torch.Tensor] = None,
        slot_widths: Optional[torch.Tensor] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.slot_starts = slot_starts
        self.slot_widths = slot_widths

    @staticmethod
    def merge_ops(ops: List[DynamicCacheOperation]) -> DynamicCacheOperation:
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)

        slot_starts = None
        slot_widths = None
        if ops[0].slot_starts is not None:
            slot_starts = torch.cat([op.slot_starts for op in ops])
        if ops[0].slot_widths is not None:
            slot_widths = torch.cat([op.slot_widths for op in ops])

        merged = DynamicCacheOperation(
            host_indices, device_indices, -1, priority, slot_starts, slot_widths
        )
        merged.node_ids = node_ids
        return merged

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class DynamicHiCacheController(HiCacheController):
    """HiCacheController variant for DynamicPagedTokenToKVPoolAllocator.

    Key differences from HiCacheController:
    - Uses SlotDoneCounter instead of LayerDoneCounter for per-slot_start signaling.
    - Only transfers the layers each token's slot actually covers (bandwidth savings).
    - Supports mixed-priority batches (tokens with different slot_starts in one load).
    """

    def __init__(
        self,
        token_to_kv_pool_allocator: DynamicPagedTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        attn_cp_rank: int = 0,
        attn_cp_size: int = 1,
        enable_storage_metrics: bool = False,
    ):
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            pp_rank=pp_rank,
            pp_size=pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            enable_storage_metrics=enable_storage_metrics,
        )

        self.allocator = token_to_kv_pool_allocator

        self.layer_done_counter = SlotDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

    def _decode_slot_info(
        self, device_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode per-token (slot_start, slot_width) from encoded device indices.

        Returns tensors on the same device as device_indices.
        """
        combined = device_indices // self.page_size
        page_ids = combined % (self.allocator.num_pages + 1)
        slot_starts = combined // (self.allocator.num_pages + 1) + 1
        slot_widths = self.allocator.slot_width[page_ids - 1]
        return slot_starts, slot_widths

    def _group_by_slot(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        slot_starts_cpu: torch.Tensor,
        slot_widths_cpu: torch.Tensor,
    ) -> Dict[int, Tuple[int, torch.Tensor, torch.Tensor]]:
        """Group tokens by slot_start.

        Returns:
            {slot_start: (slot_width, group_host_indices, group_device_indices)}
        """
        groups: Dict[int, Tuple[int, List[int]]] = {}
        for i in range(len(slot_starts_cpu)):
            ss = int(slot_starts_cpu[i].item())
            sw = int(slot_widths_cpu[i].item())
            if ss not in groups:
                groups[ss] = (sw, [])
            groups[ss][1].append(i)

        result = {}
        for ss, (sw, indices) in groups.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            result[ss] = (
                sw,
                host_indices[idx_tensor],
                device_indices[idx_tensor],
            )
        return result

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        slot_starts, slot_widths = self._decode_slot_info(device_indices)
        self.write_queue.append(
            DynamicCacheOperation(
                host_indices, device_indices, node_id, priority,
                slot_starts, slot_widths,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = DynamicCacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()

        slot_starts_cpu = op.slot_starts.cpu()
        slot_widths_cpu = op.slot_widths.cpu()
        groups = self._group_by_slot(
            host_indices, device_indices, slot_starts_cpu, slot_widths_cpu
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()


        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)

            for ss in sorted(groups.keys()):
                sw, group_host, group_device = groups[ss]
                for layer_offset in range(sw):
                    layer_id = (ss - 1) + layer_offset
                    self.mem_pool_host.backup_from_device_per_layer(
                        self.mem_pool_device,
                        group_host,
                        group_device,
                        layer_id,
                        self.io_backend,
                    )

            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(
            HiCacheAck(start_event, finish_event, op.node_ids)
        )

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        slot_starts, slot_widths = self._decode_slot_info(device_indices)
        self.load_queue.append(
            DynamicCacheOperation(
                host_indices, device_indices, node_id, priority,
                slot_starts, slot_widths,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = DynamicCacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(op)
        self.load_queue.clear()

        producer_event: SlotLoadingEvent = (
            self.layer_done_counter.get_current_producer_event()
        )

        slot_starts_cpu = op.slot_starts.cpu()
        slot_widths_cpu = op.slot_widths.cpu()
        groups = self._group_by_slot(
            host_indices, device_indices, slot_starts_cpu, slot_widths_cpu
        )

        slot_groups: Dict[int, int] = {}
        ordered_layers: List[int] = []
        for ss in sorted(groups.keys()):
            sw, _, _ = groups[ss]
            slot_groups[ss] = sw
            for layer_offset in range(sw):
                layer_id = (ss - 1) + layer_offset
                producer_event.add_layer(layer_id)
                ordered_layers.append(layer_id)
        producer_event.ordered_layers = ordered_layers
        producer_event.build_layer_mapping(slot_groups, self.layer_num)

        for layer_id in range(self.layer_num):
            if layer_id not in producer_event.covered_layers:
                producer_event.complete_layer(layer_id)

        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)

            for i, layer_id in enumerate(ordered_layers):
                if layer_id > 0:
                    producer_event.wait_consumed(layer_id - 1)

                ss = producer_event.layer_to_slot[layer_id]
                _, group_host, group_device = groups[ss]
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    group_host,
                    group_device,
                    layer_id,
                    self.io_backend,
                )
                producer_event.complete_layer(layer_id)

            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    # ------------------------------------------------------------------
    # Forward-pass streaming: interleaved load / write-back for non-
    # resident layers via the 2-position dynamic page ping-pong buffer.
    # ------------------------------------------------------------------

    def start_forward_streaming(
        self,
        schedule: ForwardLayerSchedule,
        host_indices: torch.Tensor,
        dynamic_device_indices: torch.Tensor,
    ) -> int:
        """Kick off the layer-wise load/write-back pipeline for a forward pass.

        Three CUDA streams cooperate:
        * **load_stream** — loads non-resident layers from DRAM into the
          dynamic page, waits on write-back completion before reusing a
          buffer position.
        * **write_stream** — writes back computed non-resident layers from
          the dynamic page to DRAM.
        * **compute stream** (default) — runs the model forward; synchronised
          via ``SlotDoneCounter.wait_until`` / ``signal_consumed``.

        Args:
            schedule: pre-computed ``ForwardLayerSchedule``.
            host_indices: token indices in the host memory pool.
            dynamic_device_indices: flat device indices covering *all*
                ``slot_width`` positions of the dynamic page. Positions are
                contiguous: ``[pos0_tokens..., pos1_tokens..., ...]``.

        Returns:
            producer_id for the ``SlotDoneCounter``.
        """
        if not schedule.non_resident:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        producer_event: SlotLoadingEvent = (
            self.layer_done_counter.get_current_producer_event()
        )

        page_size = self.page_size
        num_dyn_pos = 2  # ping-pong uses exactly 2 buffer positions

        def _dyn_indices(pos: int) -> torch.Tensor:
            return dynamic_device_indices[pos * page_size : (pos + 1) * page_size]

        producer_event.start_event.record()

        # Build a quick lookup: for each non-resident layer, which is the
        # *next* non-resident layer that reuses the same dyn pos?
        next_on_same_pos: Dict[int, Optional[int]] = {}
        last_seen: Dict[int, int] = {}
        for lid in reversed(schedule.non_resident):
            pos = schedule.dyn_pos[lid]
            next_on_same_pos[lid] = last_seen.get(pos, None)
            last_seen[pos] = lid

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)

            # --- Pre-load: load the first non-resident layers into each pos
            # before the forward pass reaches them.  We can pre-load up to
            # min(2, len(non_resident)) layers (one per pos).
            preloaded = set()
            for lid in schedule.non_resident:
                pos = schedule.dyn_pos[lid]
                if pos in preloaded:
                    break
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    _dyn_indices(pos),
                    lid,
                    self.io_backend,
                )
                producer_event.complete_layer(lid)
                preloaded.add(pos)
                if len(preloaded) >= min(2, num_dyn_pos):
                    break

            # --- Main loop: iterate in layer order.  For each non-resident
            # layer *after* the pre-loaded ones, we need to:
            #   1. wait for compute to consume the previous occupant
            #   2. trigger write-back of the previous occupant (write stream)
            #   3. wait for write-back to finish (so buffer is safe)
            #   4. load the new layer from DRAM
            #   5. signal ready
            for layer_id in range(schedule.layer_num):
                if schedule.is_resident(layer_id):
                    producer_event.complete_layer(layer_id)
                    continue

                if layer_id in preloaded:
                    # Already pre-loaded — ready event was already recorded.
                    # After compute consumes it, we may need to write back.
                    producer_event.wait_consumed(layer_id)

                    pos = schedule.dyn_pos[layer_id]
                    # Write back on the write stream
                    wb_event = producer_event.wb_complete_events[pos]
                    with device_module.stream(self.write_stream):
                        self.mem_pool_host.backup_from_device_per_layer(
                            self.mem_pool_device,
                            host_indices,
                            _dyn_indices(pos),
                            layer_id,
                            self.io_backend,
                        )
                        wb_event.record()
                    preloaded.discard(layer_id)
                    continue

                pos = schedule.dyn_pos[layer_id]

                # The previous occupant of this dyn pos must finish writing
                # back before we can load new data into it.
                if layer_id in schedule.wb_before_load:
                    wb_layer = schedule.wb_before_load[layer_id]
                    # Wait for compute to finish with the previous layer
                    producer_event.wait_consumed(wb_layer)
                    # Write back on the write stream
                    wb_event = producer_event.wb_complete_events[pos]
                    with device_module.stream(self.write_stream):
                        self.mem_pool_host.backup_from_device_per_layer(
                            self.mem_pool_device,
                            host_indices,
                            _dyn_indices(pos),
                            wb_layer,
                            self.io_backend,
                        )
                        wb_event.record()
                    # Wait for write-back completion before loading
                    device_module.current_stream().wait_event(wb_event)

                # Load this layer from DRAM into the dynamic buffer
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    _dyn_indices(pos),
                    layer_id,
                    self.io_backend,
                )
                producer_event.complete_layer(layer_id)

                # After compute consumes, trigger write-back
                producer_event.wait_consumed(layer_id)
                wb_event = producer_event.wb_complete_events[pos]
                with device_module.stream(self.write_stream):
                    self.mem_pool_host.backup_from_device_per_layer(
                        self.mem_pool_device,
                        host_indices,
                        _dyn_indices(pos),
                        layer_id,
                        self.io_backend,
                    )
                    wb_event.record()

        return producer_id

    def finish_forward_streaming(
        self,
        schedule: ForwardLayerSchedule,
        host_indices: torch.Tensor,
        dynamic_device_indices: torch.Tensor,
    ) -> None:
        """Flush remaining write-backs after the forward pass completes.

        Called after the model forward is done and all compute-stream work
        for the batch has finished.  Writes back the last occupants of
        each dynamic buffer position that haven't been written back yet.
        """
        if not schedule.final_wb:
            return

        page_size = self.page_size

        def _dyn_indices(pos: int) -> torch.Tensor:
            return dynamic_device_indices[pos * page_size : (pos + 1) * page_size]

        with device_module.stream(self.write_stream):
            for lid in schedule.final_wb:
                pos = schedule.dyn_pos[lid]
                self.mem_pool_host.backup_from_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    _dyn_indices(pos),
                    lid,
                    self.io_backend,
                )
