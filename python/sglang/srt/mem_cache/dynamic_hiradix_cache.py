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
from typing import TYPE_CHECKING

from sglang.srt.managers.dynamic_cache_controller import DynamicHiCacheController
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DynamicHiRadixCache(HiRadixCache):
    """HiRadixCache variant for DynamicPagedTokenToKVPoolAllocator.

    Replaces the default HiCacheController with DynamicHiCacheController which
    supports per-slot_start layer transfers and per-slot_start completion signals.

    All other radix tree, eviction, prefetch, and storage logic is inherited
    unchanged from HiRadixCache.
    """

    def _create_cache_controller(self, params, server_args, extra_config,
                                  prefetch_threshold):
        """Override to create DynamicHiCacheController instead of HiCacheController."""
        return DynamicHiCacheController(
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            mem_pool_host=self.token_to_kv_pool_host,
            page_size=self.page_size,
            tp_group=self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=self.attn_cp_rank,
            attn_cp_size=self.attn_cp_size,
            enable_storage_metrics=self.enable_storage_metrics,
        )

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        # We need to intercept HiRadixCache.__init__ at the point where it
        # creates HiCacheController. Since that's inline in __init__, the
        # cleanest approach without modifying the parent is to:
        # 1. Call super().__init__() which creates a HiCacheController
        # 2. Stop the old controller properly
        # 3. Replace it with DynamicHiCacheController
        super().__init__(params=params, server_args=server_args)

        old_controller = self.cache_controller

        # Stop old controller's threads and release its resources
        old_controller.stop_event.set()
        old_controller.storage_stop_event.set()
        old_controller.write_buffer.clear()
        old_controller.load_buffer.clear()
        if old_controller.enable_storage:
            old_controller._stop_storage_threads()

        # Re-parse extra_config for the new controller (same logic as parent __init__)
        (
            extra_config,
            prefetch_threshold,
            _prefetch_timeout_base,
            _prefetch_timeout_per_ki_token,
            _hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )

        self.cache_controller = self._create_cache_controller(
            params, server_args, extra_config, prefetch_threshold
        )
