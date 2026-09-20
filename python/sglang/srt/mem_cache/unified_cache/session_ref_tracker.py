"""Session references and priority state for UnifiedRadixCache.

Components own frontier coverage; this tracker keeps only O(number of sessions)
state. Candidate lengths are computed on host pressure, never on cache hits.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        TreeComponent,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

logger = logging.getLogger(__name__)

_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192


@dataclass
class SessionPriorityState:
    # Declared priority is independent of temporary cache demotion.
    priority: Optional[int] = None
    effective_high: bool = True
    initialized: bool = False
    active_requests: int = 0


@dataclass(kw_only=True)
class UnifiedSessionRefTracker:
    components: tuple[TreeComponent, ...]
    tree_core: UnifiedTreeCore
    enable_session_radix_cache: bool
    enable_priority_scheduling: bool = False
    high_priority_threshold: int = 1

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._closed_session_ids: OrderedDict[str, None] = OrderedDict()
        # Do not reuse an incarnation across reset: late finish/abort events
        # still carry the previous generation on their Req.
        self._session_incarnation_counter = getattr(
            self, "_session_incarnation_counter", 0
        )
        self._session_generations: dict[str, int] = {}
        self._session_states: dict[str, SessionPriorityState] = {}
        self._adaptively_demoted_sessions: OrderedDict[str, int] = OrderedDict()
        for component in self.components:
            component.reset_session_state()

    def is_high_priority(self, priority: Optional[int]) -> bool:
        return not self.enable_priority_scheduling or (
            (0 if priority is None else priority) >= self.high_priority_threshold
        )

    def session_is_high(self, session_id: str) -> bool:
        state = self._session_states.get(session_id)
        return (
            state.effective_high if state is not None else self.is_high_priority(None)
        )

    def current_generation(self, session_id: str) -> Optional[int]:
        return self._session_generations.get(session_id)

    def session_id_for_req(self, req: Req) -> Optional[str]:
        session_id = req.session_id
        if session_id is None and req.session is not None:
            session_id = req.session.session_id
        return session_id

    def _eligible_session_id(self, req: Req) -> Optional[str]:
        if not self.enable_session_radix_cache:
            return None
        if req.session is not None and req.session.streaming:
            return None
        return self.session_id_for_req(req)

    def _set_initial_priority(
        self, session_id: str, priority: Optional[int]
    ) -> SessionPriorityState:
        state = self._session_states[session_id]
        if not state.initialized:
            state.priority = priority
            state.effective_high = self.is_high_priority(priority)
            state.initialized = True
        return state

    def begin_request(self, req: Req) -> tuple[bool, str]:
        """Count one accepted request; requeue/retraction calls are idempotent."""
        session_id = self._eligible_session_id(req)
        if session_id is None:
            return True, ""
        generation = self.current_generation(session_id)
        if generation is None or req.session_generation != generation:
            return False, "Session is closed or the request has a stale generation."
        activity = (session_id, generation)
        if getattr(req, "_session_ref_activity", None) == activity:
            return True, ""
        priority = getattr(req, "priority", None)
        state = self._set_initial_priority(session_id, priority)
        if self.is_high_priority(priority) != self.is_high_priority(state.priority):
            return False, (
                "Session priority class mismatch; use /update_session_priority "
                "to change the session's priority class."
            )
        state.priority = priority
        state.active_requests += 1
        req._session_ref_activity = activity
        return True, ""

    def end_request(self, req: Req) -> None:
        """Release activity once; an old incarnation cannot decrement a new one."""
        activity = getattr(req, "_session_ref_activity", None)
        if activity is None:
            return
        req._session_ref_activity = None
        session_id, generation = activity
        if self.current_generation(session_id) != generation:
            return
        state = self._session_states[session_id]
        assert state.active_requests > 0
        state.active_requests -= 1

    def register_session_ref(self, req: Req) -> None:
        """Tag completed requests, after insertion has updated their last node."""
        session_id = self._eligible_session_id(req)
        if session_id is None or session_id in self._closed_session_ids:
            return
        generation = self.current_generation(session_id)
        if generation is None or req.session_generation != generation:
            logger.warning("register_session_ref called for stale request; Skip it.")
            return
        state = self._set_initial_priority(session_id, getattr(req, "priority", None))
        # Requests were validated at admission. The declared state may have
        # changed through an explicit control update while a request was active.
        if self.is_high_priority(state.priority):
            self.restore_session(session_id)
        assert req.last_node is not None
        last_node = self.tree_core.node_by_id(req.last_node)
        if last_node is self.tree_core.root_node:
            return
        for component in self.components:
            leaf = component.resolve_session_leaf(req, last_node)
            component.register_session_leaf(session_id, leaf)

    def _change_effective_priority(self, session_id: str, is_high: bool) -> None:
        state = self._session_states[session_id]
        if state.effective_high == is_high:
            return
        for component in self.components:
            component.change_session_priority(session_id, state.effective_high, is_high)
        state.effective_high = is_high

    def update_priority(self, session_id: str, priority: int) -> tuple[bool, str]:
        if not self.enable_session_radix_cache:
            return False, "Session radix cache is disabled."
        if session_id not in self._session_generations:
            return False, f"Session {session_id!r} is unknown or closed."
        if isinstance(priority, bool) or not isinstance(priority, int):
            return False, "Session priority must be an integer."
        self._change_effective_priority(session_id, self.is_high_priority(priority))
        state = self._session_states[session_id]
        state.priority = priority
        state.initialized = True
        self._adaptively_demoted_sessions.pop(session_id, None)
        return True, f"Updated session {session_id!r} priority to {priority}."

    def idle_hp_candidates(self) -> list[str]:
        """Snapshot shortest idle HP sessions using current logical Full KV.

        Coverage iterators can repeat shared
        ancestors, so deduplicate within a session. Device/host copies count
        once. This temporary set replaces a persistent owner-to-all-nodes index.
        """
        if not self.enable_session_radix_cache:
            return []
        full = next(
            c for c in self.components if c.component_type == BASE_COMPONENT_TYPE
        )
        candidates = []
        for session_id, state in self._session_states.items():
            if (
                not state.initialized
                or not state.effective_high
                or state.active_requests
            ):
                continue
            nodes = set(full.session_nodes(session_id))
            tokens = sum(
                len(node.key)
                for node in nodes
                if (
                    node.component_data[full.component_type].value is not None
                    or node.component_data[full.component_type].host_value is not None
                )
            )
            if tokens:
                candidates.append((tokens, session_id))
        candidates.sort()
        return [session_id for _, session_id in candidates]

    def demote_session(self, session_id: str) -> bool:
        state = self._session_states.get(session_id)
        if (
            not self.enable_session_radix_cache
            or state is None
            or not state.initialized
            or not state.effective_high
            or state.active_requests
            or not any(c._session_leaves.get(session_id) for c in self.components)
        ):
            return False
        self._change_effective_priority(session_id, False)
        self._adaptively_demoted_sessions[session_id] = self._session_generations[
            session_id
        ]
        return True

    def restore_session(self, session_id: str) -> bool:
        generation = self._adaptively_demoted_sessions.pop(session_id, None)
        if generation is None or self.current_generation(session_id) != generation:
            return False
        self._change_effective_priority(session_id, True)
        return True

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> Optional[int]:
        # Explicit reuse starts a new incarnation, including clean coverage.
        if session_id in self._session_generations:
            self.release_radix_session(session_id)
        self._closed_session_ids.pop(session_id, None)
        self._session_incarnation_counter += 1
        self._session_generations[session_id] = self._session_incarnation_counter
        self._session_states[session_id] = SessionPriorityState(
            effective_high=self.is_high_priority(None)
        )
        return self._session_incarnation_counter

    def ensure_session_generation(self, session_id: str) -> int:
        generation = self._session_generations.get(session_id)
        if generation is None:
            generation = self.open_radix_session(session_id)
        return generation

    def release_radix_session(self, session_id: str) -> int:
        if not self.enable_session_radix_cache or session_id is None:
            return 0
        if session_id in self._closed_session_ids:
            return 0
        state = self._session_states.get(session_id)
        was_high = (
            state is not None
            and state.effective_high
            and any(c._session_leaves.get(session_id) for c in self.components)
        )
        # Keep the effective tier available until all component references
        # have been removed; release_session uses session_is_high.
        indexed = sum(c.release_session(session_id) for c in self.components)
        self._remember_closed_session(session_id)
        self._session_generations.pop(session_id, None)
        self._session_states.pop(session_id, None)
        self._adaptively_demoted_sessions.pop(session_id, None)
        if was_high and self._adaptively_demoted_sessions:
            self.restore_session(next(iter(self._adaptively_demoted_sessions)))
        logger.info(
            "release_session %s: indexed %d component leaves", session_id, indexed
        )
        return 0
