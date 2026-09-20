"""CPU lifecycle tests using the real Unified radix tree and components."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from session_priority_test_utils import FULL, insert, make_cache, register, request

from sglang.srt.mem_cache.unified_cache import session_ref_tracker


class TestSessionPriorityTracker(unittest.TestCase):
    def test_threshold_and_disabled_priority_scheduling(self):
        cache = make_cache(high_priority_threshold=3)
        self.assertFalse(cache.is_high_priority(2))
        self.assertTrue(cache.is_high_priority(3))
        self.assertFalse(cache.is_high_priority(None))
        disabled = make_cache(enable_priority_scheduling=False)
        self.assertTrue(disabled.is_high_priority(-100))
        self.assertTrue(disabled.is_high_priority(None))

    def test_admission_preserves_resolved_priority_and_rejects_class_change(self):
        cache = make_cache()
        insert(cache, [1, 2])
        first = request(cache, [1, 2], "s", priority=-123)
        refs = cache.session_refs
        self.assertTrue(refs.begin_request(first)[0])
        second = request(cache, [1, 2], "s", priority=1)
        ok, message = refs.begin_request(second)
        self.assertFalse(ok)
        self.assertIn("update_session_priority", message)
        self.assertIsNone(getattr(second, "_session_ref_activity", None))
        self.assertEqual(refs._session_states["s"].priority, -123)
        self.assertEqual(refs._session_states["s"].active_requests, 1)
        refs.end_request(first)

    def test_requeue_and_double_end_are_idempotent(self):
        cache = make_cache()
        insert(cache, [1])
        req = request(cache, [1], "s", priority=1)
        refs = cache.session_refs
        self.assertTrue(refs.begin_request(req)[0])
        self.assertTrue(refs.begin_request(req)[0])
        self.assertEqual(refs._session_states["s"].active_requests, 1)
        refs.end_request(req)
        refs.end_request(req)
        self.assertEqual(refs._session_states["s"].active_requests, 0)

    def test_concurrent_requests_remain_active_until_last_finish(self):
        cache = make_cache()
        insert(cache, [1, 2])
        register(cache, [1, 2], "s", priority=1)
        refs = cache.session_refs
        a = request(cache, [1, 2], "s", priority=1)
        b = request(cache, [1, 2], "s", priority=1)
        self.assertTrue(refs.begin_request(a)[0])
        self.assertTrue(refs.begin_request(b)[0])
        refs.end_request(a)
        self.assertEqual(refs.idle_hp_candidates(), [])
        self.assertFalse(refs.demote_session("s"))
        refs.end_request(b)
        self.assertEqual(refs.idle_hp_candidates(), ["s"])

    def test_close_reopen_ignores_old_end_and_registration(self):
        cache = make_cache()
        node = insert(cache, [1, 2])
        refs = cache.session_refs
        old = register(cache, [1, 2], "s", priority=1)
        self.assertTrue(refs.begin_request(old)[0])
        cache.release_radix_session("s")
        new = request(cache, [1, 2], "s", priority=0)
        self.assertNotEqual(old.session_generation, new.session_generation)
        self.assertTrue(refs.begin_request(new)[0])
        refs.register_session_ref(old)
        refs.end_request(old)
        self.assertEqual(refs._session_states["s"].active_requests, 1)
        self.assertEqual(node.component_data[FULL].session_ref, 0)
        refs.register_session_ref(new)
        refs.end_request(new)
        self.assertEqual(node.component_data[FULL].session_ref, 1)
        self.assertEqual(node.component_data[FULL].session_high_ref, 0)

    def test_reset_does_not_reuse_generation(self):
        cache = make_cache()
        insert(cache, [1])
        old = request(cache, [1], "s", priority=1)
        refs = cache.session_refs
        self.assertTrue(refs.begin_request(old)[0])
        cache.reset()
        insert(cache, [1])
        new = request(cache, [1], "s", priority=1)
        self.assertGreater(new.session_generation, old.session_generation)
        self.assertTrue(refs.begin_request(new)[0])
        refs.end_request(old)
        refs.register_session_ref(old)
        self.assertEqual(refs._session_states["s"].active_requests, 1)
        refs.end_request(new)

    def test_fifo_restore_only_one_session_on_hp_close(self):
        cache = make_cache()
        for token, sid in ((1, "a"), (2, "b"), (3, "c")):
            insert(cache, [token])
            register(cache, [token], sid, priority=5)
        refs = cache.session_refs
        self.assertTrue(refs.demote_session("a"))
        self.assertTrue(refs.demote_session("b"))
        self.assertFalse(refs.demote_session("a"))
        cache.release_radix_session("c")
        self.assertTrue(refs.session_is_high("a"))
        self.assertFalse(refs.session_is_high("b"))
        self.assertEqual(list(refs._adaptively_demoted_sessions), ["b"])
        cache.release_radix_session("c")
        self.assertFalse(refs.session_is_high("b"))

    def test_closing_demoted_session_does_not_restore_another(self):
        cache = make_cache()
        for token, sid in ((1, "a"), (2, "b")):
            insert(cache, [token])
            register(cache, [token], sid, priority=1)
            self.assertTrue(cache.session_refs.demote_session(sid))
        cache.release_radix_session("a")
        self.assertFalse(cache.session_refs.session_is_high("b"))

    def test_new_hp_completion_restores_without_changing_declared_priority(self):
        cache = make_cache()
        node = insert(cache, [1, 2])
        register(cache, [1, 2], "s", priority=7)
        refs = cache.session_refs
        self.assertTrue(refs.demote_session("s"))
        self.assertEqual(refs._session_states["s"].priority, 7)
        req = request(cache, [1, 2], "s", priority=7)
        self.assertTrue(refs.begin_request(req)[0])
        self.assertFalse(refs.session_is_high("s"))
        refs.register_session_ref(req)
        refs.end_request(req)
        self.assertTrue(refs.session_is_high("s"))
        self.assertEqual(node.component_data[FULL].session_high_ref, 1)

    def test_explicit_priority_update_cancels_demotion(self):
        cache = make_cache()
        node = insert(cache, [1, 2])
        register(cache, [1, 2], "s", priority=5)
        refs = cache.session_refs
        self.assertTrue(refs.demote_session("s"))
        self.assertTrue(refs.update_priority("s", 0)[0])
        self.assertFalse(refs.restore_session("s"))
        self.assertFalse(refs.session_is_high("s"))
        self.assertTrue(refs.update_priority("s", 8)[0])
        self.assertTrue(refs.session_is_high("s"))
        self.assertEqual(node.component_data[FULL].session_high_ref, 1)

    def test_unknown_update_and_invalid_priority_do_not_create_state(self):
        cache = make_cache()
        refs = cache.session_refs
        self.assertFalse(refs.update_priority("missing", 1)[0])
        self.assertEqual(refs._session_states, {})
        cache.open_radix_session("s")
        self.assertFalse(refs.update_priority("s", True)[0])
        self.assertFalse(refs.update_priority("s", 1.5)[0])
        self.assertFalse(refs.demote_session("s"))
        cache.release_radix_session("s")
        self.assertFalse(refs.update_priority("s", 1)[0])

    def test_candidates_deduplicate_shared_ancestors_within_session(self):
        cache = make_cache()
        insert(cache, [1, 2, 3, 4])
        register(cache, [1, 2, 3, 4], "fork", priority=1)
        insert(cache, [1, 2, 5, 6])
        register(cache, [1, 2, 5, 6], "fork", priority=1)
        insert(cache, [7, 8, 9, 10, 11, 12, 13])
        register(cache, [7, 8, 9, 10, 11, 12, 13], "other", priority=1)
        # fork has 6 unique tokens, although coverage visits 8 tokens.
        self.assertEqual(cache.session_refs.idle_hp_candidates(), ["fork", "other"])

    def test_session_churn_retains_no_history_except_bounded_tombstones(self):
        cache = make_cache()
        node = insert(cache, [1, 2])
        refs = cache.session_refs
        with patch.object(session_ref_tracker, "_CLOSED_SESSION_TOMBSTONE_LIMIT", 16):
            for index in range(100):
                sid = f"s{index}"
                register(cache, [1, 2], sid, priority=1)
                self.assertTrue(refs.demote_session(sid))
                self.assertTrue(refs.restore_session(sid))
                cache.release_radix_session(sid)
        self.assertEqual(refs._session_states, {})
        self.assertEqual(refs._session_generations, {})
        self.assertEqual(refs._adaptively_demoted_sessions, {})
        self.assertEqual(len(refs._closed_session_ids), 16)
        self.assertEqual(node.component_data[FULL].session_ref, 0)
        self.assertEqual(node.component_data[FULL].session_high_ref, 0)
        self.assertFalse(cache.components[FULL]._session_leaves)

    def test_feature_disabled_and_streaming_requests_have_no_activity(self):
        cache = make_cache(enable_session=False)
        insert(cache, [1])
        req = request(cache, [1], "s", priority=1)
        self.assertTrue(cache.session_refs.begin_request(req)[0])
        self.assertIsNone(getattr(req, "_session_ref_activity", None))
        self.assertEqual(cache.session_refs.idle_hp_candidates(), [])
        self.assertFalse(cache.session_refs.update_priority("s", 1)[0])
        enabled = make_cache()
        insert(enabled, [1])
        req = request(enabled, [1], "s", priority=1)
        from types import SimpleNamespace

        req.session = SimpleNamespace(streaming=True, session_id="s")
        self.assertTrue(enabled.session_refs.begin_request(req)[0])
        enabled.session_refs.register_session_ref(req)
        self.assertIsNone(getattr(req, "_session_ref_activity", None))
        self.assertEqual(enabled.session_refs.idle_hp_candidates(), [])

    def test_closing_empty_hp_session_does_not_restore_demoted_refs(self):
        cache = make_cache(enable_priority_scheduling=False)
        insert(cache, [1, 2])
        register(cache, [1, 2], "demoted", priority=1)
        self.assertTrue(cache.session_refs.demote_session("demoted"))
        cache.ensure_session_generation("never_started")
        cache.release_radix_session("never_started")
        self.assertFalse(cache.session_refs.session_is_high("demoted"))
        req = request(cache, [1, 2], "aborted", priority=1)
        self.assertTrue(cache.session_refs.begin_request(req)[0])
        cache.session_refs.end_request(req)
        cache.release_radix_session("aborted")
        self.assertFalse(cache.session_refs.session_is_high("demoted"))


if __name__ == "__main__":
    unittest.main()
