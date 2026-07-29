from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.ref_aware_mamba_cache_mixin import RefAwareMambaCacheMixin

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


class RefAwareMambaRadixCache(RefAwareMambaCacheMixin, MambaRadixCache):
    """MambaRadixCache with priority-aware tiered eviction for both the
    full-KV and mamba-state resources."""

    def __init__(self, params: CacheInitParams, server_args: ServerArgs = None):
        # Must precede super().__init__: MambaRadixCache.__init__ calls
        # self.reset(), which dispatches to our reset() below.
        self._init_ref_aware_state(server_args)
        self.is_eagle = False
        super().__init__(params)

    def reset(self):
        self._reset_ref_aware_state()
        super().reset()
