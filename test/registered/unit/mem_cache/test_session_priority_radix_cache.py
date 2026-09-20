"""CPU behavior tests for priority-aware Unified session cache components."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import random
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from session_priority_test_utils import (
    COMPONENT_CASES,
    FULL,
    MAMBA,
    SWA,
    attach_host_cache,
    backup,
    insert,
    make_cache,
    match_len,
    register,
    request,
)

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InitLoadBackParams
from sglang.srt.mem_cache.hicache_storage import PoolName


def assert_refs(node, component, total, high):
    data = node.component_data[component]
    assert data.session_ref == total
    assert data.session_high_ref == high
    assert 0 <= data.session_high_ref <= data.session_ref


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_shared_priority_updates_and_close(components):
    cache = make_cache(components)
    leaf = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "low", priority=0)
    register(cache, [1, 2, 3, 4], "high", priority=1)
    register(cache, [1, 2, 3, 4], "high", priority=9)
    for component in components:
        assert_refs(leaf, component, 2, 1)
    assert cache.session_refs.update_priority("low", 2)[0]
    for component in components:
        assert_refs(leaf, component, 2, 2)
    assert cache.session_refs.update_priority("high", 0)[0]
    for component in components:
        assert_refs(leaf, component, 2, 1)
    cache.release_radix_session("low")
    cache.release_radix_session("low")
    for component in components:
        assert_refs(leaf, component, 1, 0)
    cache.release_radix_session("high")
    for component in components:
        assert_refs(leaf, component, 0, 0)
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_split_preserves_coverage_and_mamba_checkpoint_scope(components):
    cache = make_cache(components)
    old_leaf = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "high", priority=1)
    new_leaf = insert(cache, [1, 2, 8, 9])
    register(cache, [1, 2, 8, 9], "low", priority=0)
    parent = old_leaf.parent
    assert parent is new_leaf.parent
    for component in components:
        assert_refs(old_leaf, component, 1, 1)
        assert_refs(new_leaf, component, 1, 0)
        if component == MAMBA:
            assert_refs(parent, component, 0, 0)
        else:
            assert_refs(parent, component, 2, 1)
    cache.release_radix_session("high")
    for component in components:
        assert_refs(old_leaf, component, 0, 0)
        assert_refs(new_leaf, component, 1, 0)
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_demotion_restore_keeps_shared_hp_protection(components):
    cache = make_cache(components)
    leaf = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "a", priority=7)
    register(cache, [1, 2, 3, 4], "b", priority=3)
    assert cache.session_refs.demote_session("a")
    for component in components:
        assert_refs(leaf, component, 2, 1)
    assert cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted == 0
    assert match_len(cache, [1, 2, 3, 4]) == 4
    assert cache.session_refs.restore_session("a")
    for component in components:
        assert_refs(leaf, component, 2, 2)
    assert cache.session_refs._session_states["a"].priority == 7
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_three_tiers_and_explicit_hp_permission(components):
    cache = make_cache(components)
    high = insert(cache, [1, 2])
    low = insert(cache, [3, 4])
    insert(cache, [5, 6])
    register(cache, [1, 2], "high", priority=1)
    register(cache, [3, 4], "low", priority=0)
    assert cache.evict(EvictParams(num_tokens=2)).num_tokens_evicted == 2
    assert match_len(cache, [5, 6]) == 0
    assert_refs(high, FULL, 1, 1)
    assert_refs(low, FULL, 1, 0)
    assert cache.evict(EvictParams(num_tokens=2)).num_tokens_evicted == 2
    assert match_len(cache, [3, 4]) == 0
    assert cache.evict(EvictParams(num_tokens=2)).num_tokens_evicted == 0
    result = cache.evict_for_session(EvictParams(num_tokens=2), allow_high=True)
    assert result.num_tokens_evicted == 2
    assert match_len(cache, [1, 2]) == 0
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_lock_protection_and_tier_size_accounting(components):
    cache = make_cache(components)
    leaf = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "s", priority=1)
    assert cache.session_evictable_size() == 0
    assert cache.session_evictable_size(allow_high=True) == 4
    lock = cache.inc_lock_ref(leaf.id)
    assert cache.session_evictable_size(allow_high=True) == 0
    assert (
        cache.evict_for_session(
            EvictParams(num_tokens=4), allow_high=True
        ).num_tokens_evicted
        == 0
    )
    cache.dec_lock_ref(leaf.id, lock.to_dec_params())
    assert cache.session_refs.demote_session("s")
    assert cache.session_evictable_size() == 4
    assert cache.session_refs.restore_session("s")
    assert cache.session_evictable_size() == 0
    assert cache.session_evictable_size(allow_high=True) == 4
    cache.sanity_check()


def test_full_low_refs_then_lru_and_high_refs_then_mru():
    cache = make_cache()
    a, b, c = ([1, 2], [3, 4], [5, 6])
    for tokens in (a, b, c):
        insert(cache, tokens)
    register(cache, a, "a1", priority=0)
    register(cache, a, "a2", priority=0)
    register(cache, b, "b", priority=0)
    register(cache, c, "c", priority=0)
    match_len(cache, c)
    cache.evict(EvictParams(num_tokens=2))
    assert match_len(cache, b) == 0
    assert match_len(cache, a) == 2
    assert match_len(cache, c) == 2
    for sid in ("a1", "a2", "c"):
        assert cache.session_refs.update_priority(sid, 1)[0]
    cache.evict_for_session(EvictParams(num_tokens=2), allow_high=True)
    assert match_len(cache, c) == 0
    assert match_len(cache, a) == 2

    cache = make_cache()
    insert(cache, a)
    insert(cache, b)
    register(cache, a, "a", priority=1)
    register(cache, b, "b", priority=1)
    match_len(cache, a)
    cache.evict_for_session(EvictParams(num_tokens=2), allow_high=True)
    assert match_len(cache, a) == 0  # HP ties preserve the old MRU policy.
    assert match_len(cache, b) == 2


@pytest.mark.parametrize("aux", (SWA, MAMBA))
def test_aux_tiers_keep_lru_and_reject_protected_cascade(aux):
    cache = make_cache((FULL, aux))
    a = insert(cache, [1, 2])
    b = insert(cache, [3, 4])
    register(cache, [1, 2], "a", priority=1)
    register(cache, [3, 4], "b", priority=1)
    # Remove auxiliary coverage only. The auxiliary may be tombstoned, but
    # FULL must still block a leaf cascade that would discard its HP value.
    cache.components[aux].release_session("a")
    request_size = {
        SWA: EvictParams(swa_num_tokens=2),
        MAMBA: EvictParams(mamba_num=1),
    }[aux]
    cache.evict(request_size)
    assert a.component_data[FULL].value is not None
    assert a.component_data[aux].value is None
    assert b.component_data[aux].value is not None
    cache.sanity_check()

    cache = make_cache((FULL, aux))
    a = insert(cache, [1, 2])
    b = insert(cache, [3, 4])
    register(cache, [1, 2], "a", priority=0)
    register(cache, [3, 4], "b", priority=0)
    cache.evict(request_size)
    assert a.component_data[aux].value is None
    assert b.component_data[aux].value is not None


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_generation_events_do_not_retag_or_decrement_new_incarnation(components):
    cache = make_cache(components)
    leaf = insert(cache, [1, 2, 3, 4])
    old = request(cache, [1, 2, 3, 4], "s", priority=1)
    assert cache.session_refs.begin_request(old)[0]
    cache.release_radix_session("s")
    cache.open_radix_session("s")
    current = request(cache, [1, 2, 3, 4], "s", priority=0)
    assert cache.session_refs.begin_request(current)[0]
    cache.session_refs.register_session_ref(old)
    cache.session_refs.end_request(old)
    assert cache.session_refs._session_states["s"].active_requests == 1
    cache.session_refs.register_session_ref(current)
    cache.session_refs.end_request(current)
    for component in components:
        assert_refs(leaf, component, 1, 0)
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_host_pressure_shortest_idle_demotion_and_real_pool_free(components):
    cache = make_cache(components)
    host = attach_host_cache(cache, size=16)
    short = insert(cache, [1, 2])
    long = insert(cache, [3, 4, 5, 6])
    register(cache, [1, 2], "short", priority=1)
    register(cache, [3, 4, 5, 6], "long", priority=1)
    assert backup(cache, short) == 2
    assert backup(cache, long) == 4
    cache.evict_for_session(EvictParams(num_tokens=6), allow_high=True)
    assert short.evicted and long.evicted
    assert cache.evict_host(2) == 0
    before = host.mem_pool_host.available_size()
    assert cache.evict_host(2, allow_high=True) == 2
    assert host.mem_pool_host.available_size() == before + 2
    assert not cache.session_refs.session_is_high("short")
    assert cache.session_refs.session_is_high("long")
    assert long.component_data[FULL].host_value is not None
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_backup_ack_and_host_lock_are_respected(components):
    cache = make_cache(components)
    host = attach_host_cache(cache)
    leaf = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "s", priority=0)
    assert backup(cache, leaf, finish=False) == 4
    assert cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted == 0
    cache._finish_write_through_ack(leaf.id)
    assert cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted == 4
    lock = cache.inc_host_lock_ref(leaf.id)
    assert cache.evict_host(4, allow_high=True) == 0
    cache.dec_host_lock_ref(leaf.id, lock.to_dec_params())
    assert cache.evict_host(4, allow_high=True) == 4
    assert host.mem_pool_host.available_size() == host.mem_pool_host.size
    cache.sanity_check()


def test_nested_scope_and_exception_restore_permissions():
    cache = make_cache()
    insert(cache, [1, 2])
    register(cache, [1, 2], "s", priority=1)
    with pytest.raises(RuntimeError):
        with cache.scoped_evict(allow_high=True):
            with cache.scoped_evict(allow_high=False):
                assert cache.evict(EvictParams(num_tokens=2)).num_tokens_evicted == 0
            assert cache.tree_core.session_allow_high
            raise RuntimeError("abort this allocation")
    assert cache.evict(EvictParams(num_tokens=2)).num_tokens_evicted == 0


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_feature_disabled_preserves_unref_eviction(components):
    cache = make_cache(components, enable_session=False)
    leaf = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "s", priority=10)
    for component in components:
        assert_refs(leaf, component, 0, 0)
    assert cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted == 4


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_match_and_quota_do_not_scan_session_coverage(components):
    cache = make_cache(components)
    insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "s", priority=1)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                cache.session_refs,
                "idle_hp_candidates",
                side_effect=AssertionError("hot path scanned sessions"),
            )
        )
        for component in cache.components.values():
            stack.enter_context(
                patch.object(
                    component,
                    "session_nodes",
                    side_effect=AssertionError("hot path scanned coverage"),
                )
            )
        for _ in range(20):
            assert match_len(cache, [1, 2, 3, 4]) == 4
            assert cache.session_evictable_size(allow_high=True) == 4


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_seeded_reference_oracle_and_bounded_churn(components):
    cache = make_cache(components)
    paths = [[1, 2], [3, 4], [5, 6]]
    leaves = [insert(cache, tokens) for tokens in paths]
    rng = random.Random(717)
    model = {}  # session -> (declared_priority, effective_high, set(path indexes))
    for _ in range(250):
        sid = str(rng.randrange(7))
        operation = rng.choice(("register", "update", "demote", "restore"))
        if sid not in model:
            priority = rng.randrange(2)
            model[sid] = (priority, bool(priority), set())
            cache.ensure_session_generation(sid)
            cache.session_refs.update_priority(sid, priority)
        priority, high, owned = model[sid]
        if operation == "register":
            which = rng.randrange(3)
            register(cache, paths[which], sid, priority=priority)
            owned.add(which)
            high = bool(priority)
        elif operation == "update":
            priority = rng.randrange(2)
            high = bool(priority)
            assert cache.session_refs.update_priority(sid, priority)[0]
        elif operation == "demote":
            if cache.session_refs.demote_session(sid):
                high = False
        elif operation == "restore":
            if cache.session_refs.restore_session(sid):
                high = True
        model[sid] = (priority, high, owned)
        for index, leaf in enumerate(leaves):
            expected = sum(index in owned for _, _, owned in model.values())
            expected_high = sum(
                high and index in owned for _, high, owned in model.values()
            )
            for component in components:
                assert_refs(leaf, component, expected, expected_high)
    for sid in tuple(model):
        cache.release_radix_session(sid)
    for i in range(500):
        sid = f"churn-{i}"
        register(cache, paths[0], sid, priority=1)
        assert cache.session_refs.demote_session(sid)
        assert cache.session_refs.restore_session(sid)
        cache.release_radix_session(sid)
    assert not cache.session_refs._session_states
    assert not cache.session_refs._adaptively_demoted_sessions
    for component in cache.components.values():
        assert not component._session_leaves
    for leaf in leaves:
        for component in components:
            assert_refs(leaf, component, 0, 0)
    cache.sanity_check()


def test_swa_window_advance_and_eviction_recede_update_priority_coverage():
    cache = make_cache((FULL, SWA))
    tokens = list(range(1, 9))
    chain = [insert(cache, tokens[:length]) for length in (2, 4, 6, 8)]
    register(cache, tokens[:4], "s", priority=1)
    register(cache, tokens, "s", priority=1)
    for i, node in enumerate(chain):
        assert_refs(node, FULL, 1, 1)
        # W=4 plus one page: three two-token nodes cover the new frontier.
        assert_refs(node, SWA, int(i > 0), int(i > 0))
    cache.evict_for_session(EvictParams(num_tokens=2), allow_high=True)
    with pytest.raises(KeyError):
        cache.tree_core.node_by_id(chain[-1].id)
    for node in chain[:-1]:
        assert_refs(node, FULL, 1, 1)
        assert_refs(node, SWA, 1, 1)
    assert cache.session_refs.update_priority("s", 0)[0]
    for node in chain[:-1]:
        assert_refs(node, FULL, 1, 0)
        assert_refs(node, SWA, 1, 0)
    cache.release_radix_session("s")
    for node in chain[:-1]:
        assert_refs(node, FULL, 0, 0)
        assert_refs(node, SWA, 0, 0)
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_host_pressure_excludes_active_demotion_and_allows_hp_fallback(components):
    cache = make_cache(components)
    attach_host_cache(cache)
    short = insert(cache, [1, 2])
    long = insert(cache, [3, 4, 5, 6])
    register(cache, [1, 2], "active", priority=1)
    register(cache, [3, 4, 5, 6], "idle", priority=1)
    backup(cache, short)
    backup(cache, long)
    cache.evict_for_session(EvictParams(num_tokens=6), allow_high=True)
    active = request(cache, [1, 2], "active", priority=1)
    assert cache.session_refs.begin_request(active)[0]
    assert cache.session_refs.idle_hp_candidates() == ["idle"]
    assert cache.evict_host(2, allow_high=True) == 4
    assert short.component_data[FULL].host_value is not None
    assert cache.session_refs.session_is_high("active")
    assert not cache.session_refs.session_is_high("idle")
    # No idle candidate remains. Explicit HP permission permits final HP
    # eviction, while the active request's declared/effective tier stays HP.
    assert cache.evict_host(2, allow_high=True) == 2
    assert cache.session_refs.session_is_high("active")
    assert cache.session_refs._session_states["active"].active_requests == 1
    cache.session_refs.end_request(active)
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_backup_capacity_failure_does_not_leak_locks_or_pending_transfer(components):
    cache = make_cache(components)
    host = attach_host_cache(cache, size=2)
    node = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "s", priority=1)
    assert backup(cache, node, finish=False) == 0
    assert not cache.ongoing_write_through
    assert host.mem_pool_host.available_size() == 2
    for component in components:
        data = node.component_data[component]
        assert data.host_value is None
        assert data.lock_ref == data.host_lock_ref == 0
        assert data.value is not None
    cache.sanity_check()


@pytest.mark.parametrize("aux", (SWA, MAMBA))
def test_backup_aux_allocation_failure_rolls_back_transfer_boundary(aux):
    cache = make_cache((FULL, aux))
    host = attach_host_cache(cache, size=4)
    node = insert(cache, [1, 2, 3, 4])
    pool_name = PoolName.SWA if aux == SWA else PoolName.MAMBA
    held = host.pools[pool_name].alloc(4)
    assert held is not None
    assert backup(cache, node, finish=False) == 0
    assert host.mem_pool_host.available_size() == 4
    assert not cache.ongoing_write_through
    for component in (FULL, aux):
        data = node.component_data[component]
        assert data.host_value is None
        assert data.lock_ref == data.host_lock_ref == 0
    host.pools[pool_name].free(held)
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
@pytest.mark.parametrize("priority", (0, 1))
def test_init_load_back_uses_request_permission_and_releases_locks(
    components, priority
):
    cache = make_cache(components, size=8)
    host = attach_host_cache(cache)
    target = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "target", priority=priority)
    backup(cache, target)
    cache.evict_for_session(EvictParams(num_tokens=4), allow_high=True)
    blocker = insert(cache, list(range(11, 19)))
    register(cache, list(range(11, 19)), "blocker", priority=1)
    req = request(cache, [1, 2, 3, 4], "target", priority=priority)
    assert cache.session_refs.begin_request(req)[0]
    mamba_before = (
        cache.req_to_token_pool.mamba_allocator.available_size()
        if MAMBA in components
        else None
    )
    indices, node_id = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=target.id,
            host_hit_length=4,
            req=req,
        )
    )
    if priority == 0:
        assert len(indices) == 0
        assert target.evicted
        assert blocker.component_data[FULL].value is not None
        assert not cache.ongoing_load_back
        if MAMBA in components:
            assert req.mamba_pool_idx is None
            assert (
                cache.req_to_token_pool.mamba_allocator.available_size() == mamba_before
            )
    else:
        assert len(indices) == 4
        assert node_id == target.id
        assert target.component_data[FULL].value is not None
        # SWA may evict only the capped suffix; Full-only/Mamba may evict
        # the whole leaf. In either case the protected prefix was reclaimed.
        assert match_len(cache, list(range(11, 19))) < 8
        assert target.id in cache.ongoing_load_back
        cache.loading_check(finish_count=1)
        assert not cache.ongoing_load_back
        assert not host.ack_load_queue
    for component in components:
        data = target.component_data[component]
        assert data.lock_ref == data.host_lock_ref == 0
    assert not cache.tree_core.session_allow_high
    cache.session_refs.end_request(req)
    cache.sanity_check()


@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_load_transfer_failure_cleans_host_device_and_mamba_request_locks(components):
    cache = make_cache(components)
    host = attach_host_cache(cache)
    target = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "target", priority=1)
    backup(cache, target)
    cache.evict_for_session(EvictParams(num_tokens=4), allow_high=True)
    host.fail_load = True
    req = request(cache, [1, 2, 3, 4], "target", priority=1)
    mamba_before = (
        cache.req_to_token_pool.mamba_allocator.available_size()
        if MAMBA in components
        else None
    )
    indices, _ = cache.init_load_back(
        InitLoadBackParams(
            best_match_node=target.id,
            host_hit_length=4,
            req=req,
        )
    )
    assert len(indices) == 0
    assert not cache.ongoing_load_back
    assert not host.ack_load_queue
    for component in components:
        data = target.component_data[component]
        assert data.value is None
        assert data.host_value is not None
        assert data.lock_ref == data.host_lock_ref == 0
    if MAMBA in components:
        assert req.mamba_pool_idx is None
        assert cache.req_to_token_pool.mamba_allocator.available_size() == mamba_before
    cache.sanity_check()


@pytest.mark.parametrize("priority", (0, 1))
@pytest.mark.parametrize("components", COMPONENT_CASES)
def test_write_back_host_pressure_inherits_request_permission(components, priority):
    cache = make_cache(components)
    host = attach_host_cache(cache, size=4)
    protected = insert(cache, [1, 2, 3, 4])
    register(cache, [1, 2, 3, 4], "protected", priority=1)
    assert backup(cache, protected) == 4
    cache.evict_for_session(EvictParams(num_tokens=4), allow_high=True)
    target = insert(cache, [5, 6, 7, 8])
    register(cache, [5, 6, 7, 8], "target", priority=priority)
    with cache.scoped_evict(allow_high=cache.is_high_priority(priority)):
        written = backup(cache, target, write_back=True)
    if priority == 0:
        assert written == 0
        assert protected.component_data[FULL].host_value is not None
        assert target.component_data[FULL].host_value is None
        assert cache.session_refs.session_is_high("protected")
    else:
        assert written == 4
        with pytest.raises(KeyError):
            cache.tree_core.node_by_id(protected.id)
        assert target.component_data[FULL].host_value is not None
        assert not cache.session_refs.session_is_high("protected")
    assert host.mem_pool_host.available_size() == 0
    assert not cache.ongoing_write_through
    assert not cache.tree_core.session_allow_high
    cache.sanity_check()


def _run_repartitioned_cursor_case(aux):
    cache = make_cache((FULL, aux), size=64)
    first = insert(cache, [1, 2])
    second = insert(cache, [3, 4])
    victim_paths = ([1, 2], [3, 4], [5, 6], [7, 8])
    for tokens in victim_paths[2:]:
        insert(cache, list(tokens))
    other = insert(cache, [9, 10])
    for tokens in victim_paths:
        register(cache, list(tokens), "victim", priority=1)
    register(cache, [9, 10], "other", priority=1)

    tracker = {FULL: 0, aux: 0}
    with cache.scoped_evict(allow_low=True, allow_high=True):
        cache.tree_core.evict_device_start(aux, 100)
        try:
            selected = cache._evict_device_next_node(aux, tracker)
            assert selected in {first.id, second.id}
            assert cache.session_refs.demote_session("victim")
            cache._evict_device_leaf(selected, tracker)
            while (node_id := cache._evict_device_next_node(aux, tracker)) is not None:
                cache._evict_device_leaf(node_id, tracker)
        finally:
            cache.tree_core.evict_device_end(aux)

    # All five leaves are eligible in this allow-low+allow-high audit scope.
    return tracker[aux], cache


def test_mamba_cursor_sees_nodes_repartitioned_during_walk():
    freed, cache = _run_repartitioned_cursor_case(MAMBA)
    assert freed == 5
    cache.sanity_check()


def test_swa_cursor_sees_nodes_repartitioned_during_walk():
    freed, cache = _run_repartitioned_cursor_case(SWA)
    assert freed == 10
    cache.sanity_check()


def test_public_writeback_host_pressure_repartitions_mamba_walk():
    cache = make_cache((FULL, MAMBA), size=64)
    host = attach_host_cache(cache, size=1)

    backed = insert(cache, [100])
    register(cache, [100], "victim", priority=1)
    assert backup(cache, backed) == 1
    assert (
        cache.evict_for_session(
            EvictParams(num_tokens=1), allow_high=True
        ).num_tokens_evicted
        == 1
    )
    assert backed.evicted
    assert host.mem_pool_host.available_size() == 0

    for token in (1, 2, 3, 4):
        insert(cache, [token])
        register(cache, [token], "victim", priority=1)
    insert(cache, list(range(10, 20)))
    register(cache, list(range(10, 20)), "other", priority=1)

    cache.cache_controller.write_policy = "write_back"
    cache.is_write_back = True
    cache.tree_core.is_write_back = True

    def finish_writes(*, write_back=False, **_kwargs):
        for node_id in tuple(cache.ongoing_write_through):
            cache._finish_write_through_ack(node_id)

    cache.writing_check = finish_writes
    result = cache.evict_for_session(
        EvictParams(mamba_num=100), allow_low=True, allow_high=True
    )
    assert result.mamba_num_evicted == 5
    cache.sanity_check()


def test_empty_hp_close_does_not_restore_demotion():
    cache = make_cache(enable_priority_scheduling=False)
    insert(cache, [1])
    register(cache, [1], "demoted", priority=0)
    assert cache.session_refs.demote_session("demoted")
    cache.open_radix_session("empty")
    cache.release_radix_session("empty")
    assert not cache.session_refs.session_is_high("demoted")


def test_aux_cursor_keeps_unused_before_newly_demoted_lp():
    cache = make_cache((FULL, MAMBA), size=64)
    unused = [insert(cache, [token]) for token in (1, 2, 3)]
    for token in (10, 11, 12, 13):
        insert(cache, [token])
        register(cache, [token], "victim", priority=1)

    tracker = {FULL: 0, MAMBA: 0}
    selected = []
    with cache.scoped_evict(allow_low=True, allow_high=True):
        cache.tree_core.evict_device_start(MAMBA, 100)
        try:
            node_id = cache._evict_device_next_node(MAMBA, tracker)
            selected.append(node_id)
            assert node_id in {node.id for node in unused}
            assert cache.session_refs.demote_session("victim")
            cache._evict_device_leaf(node_id, tracker)
            while len(selected) < len(unused):
                node_id = cache._evict_device_next_node(MAMBA, tracker)
                selected.append(node_id)
                cache._evict_device_leaf(node_id, tracker)
        finally:
            cache.tree_core.evict_device_end(MAMBA)

    assert set(selected) == {node.id for node in unused}
    cache.sanity_check()


@pytest.mark.parametrize("aux", (SWA, MAMBA))
@pytest.mark.parametrize("host_layer", (False, True))
def test_feature_disabled_aux_drivers_keep_original_eviction(aux, host_layer):
    cache = make_cache((FULL, aux), enable_session=False)
    node = insert(cache, [1, 2])
    if host_layer:
        attach_host_cache(cache, size=8)
        assert backup(cache, node) == 2
        assert cache.evict(EvictParams(num_tokens=2)).num_tokens_evicted == 2
        assert cache.evict_host(1, component_type=aux) > 0
    else:
        params = (
            EvictParams(swa_num_tokens=1) if aux == SWA else EvictParams(mamba_num=1)
        )
        cache.evict(params)
    assert node.id not in cache.tree_core._node_arena
    cache.sanity_check()


def test_host_pressure_rekey_heap_is_bounded_with_shared_sessions():
    cache = make_cache()
    attach_host_cache(cache, size=8)
    node = insert(cache, [1, 2])
    for index in range(100):
        register(cache, [1, 2], f"shared-{index}", priority=1)
    assert backup(cache, node) == 2
    assert (
        cache.evict_for_session(
            EvictParams(num_tokens=2), allow_high=True
        ).num_tokens_evicted
        == 2
    )
    comp = cache.components[FULL]
    original = comp._rekey_session_leaf
    observed = []

    def check_heap(heap, leaves, changed):
        original(heap, leaves, changed)
        observed.append(len(heap))
        assert len(heap) <= 3 * len(leaves) + 32

    with patch.object(comp, "_rekey_session_leaf", side_effect=check_heap):
        assert cache.evict_host(2, allow_high=True) == 2
    assert len(observed) >= 100
    assert comp._session_host_heap is None
    cache.sanity_check()
