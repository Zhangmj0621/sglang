"""CPU tests for the session priority control plane and validation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.managers.io_struct import (
    UpdateSessionPriorityReqInput,
    UpdateSessionPriorityReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.server_args import ServerArgs, prepare_server_args


class _ControlHarness(TokenizerControlMixin):
    def __init__(self, results=(), enabled=True):
        self.server_args = SimpleNamespace(enable_session_radix_cache=enabled)
        self.results = results
        self.calls = []

    def auto_create_handle_loop(self):
        pass

    async def update_session_priority_communicator(self, obj):
        self.calls.append(obj)
        return self.results


class TestSessionPriorityControl(unittest.IsolatedAsyncioTestCase):
    async def test_session_on_one_dp_rank_updates_successfully(self):
        owner = UpdateSessionPriorityReqOutput(
            success=True, message="updated", found=True
        )
        nonowner = UpdateSessionPriorityReqOutput(success=True, message="", found=False)
        manager = _ControlHarness([owner, nonowner])
        request = UpdateSessionPriorityReqInput(session_id="s", priority=3)
        response = await manager.update_session_priority(request)
        self.assertTrue(response.success)
        self.assertTrue(response.found)
        self.assertEqual(response.message, "updated")
        self.assertEqual(manager.calls, [request])

    async def test_unknown_on_every_rank_returns_failure(self):
        manager = _ControlHarness(
            [UpdateSessionPriorityReqOutput(success=True, message="", found=False)]
        )
        response = await manager.update_session_priority(
            UpdateSessionPriorityReqInput(session_id="s", priority=3)
        )
        self.assertFalse(response.success)
        self.assertFalse(response.found)
        self.assertIn("unknown or closed", response.message)

    async def test_rank_failure_is_not_hidden_by_successful_rank(self):
        manager = _ControlHarness(
            [
                UpdateSessionPriorityReqOutput(
                    success=True, message="updated", found=True
                ),
                UpdateSessionPriorityReqOutput(
                    success=False, message="backend disabled", found=False
                ),
            ]
        )
        response = await manager.update_session_priority(
            UpdateSessionPriorityReqInput(session_id="s", priority=3)
        )
        self.assertFalse(response.success)
        self.assertTrue(response.found)
        self.assertIn("backend disabled", response.message)

    async def test_feature_disabled_never_dispatches(self):
        manager = _ControlHarness(enabled=False)
        response = await manager.update_session_priority(
            UpdateSessionPriorityReqInput(session_id="s", priority=3)
        )
        self.assertFalse(response.success)
        self.assertEqual(manager.calls, [])

    async def test_http_route_reports_success_and_unknown_session(self):
        # Load the real route into a minimal app: importing the serving entry
        # point would initialize unrelated optional ASR/model dependencies.
        import ast
        import json
        from pathlib import Path
        from typing import Annotated

        from fastapi import Body, FastAPI, Request
        from fastapi.responses import ORJSONResponse

        from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/entrypoints/http_server.py"
        )
        source = ast.parse(path.read_text())
        route = next(
            node
            for node in source.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "update_session_priority"
        )
        success = UpdateSessionPriorityReqOutput(
            success=True, message="updated", found=True
        )
        manager = _ControlHarness([success])
        namespace = {
            "Annotated": Annotated,
            "Body": Body,
            "Request": Request,
            "ORJSONResponse": ORJSONResponse,
            "app": FastAPI(),
            "UpdateSessionPriorityReqInput": UpdateSessionPriorityReqInput,
            "msgspec_to_builtins": msgspec_to_builtins,
            "_global_state": SimpleNamespace(tokenizer_manager=manager),
        }
        exec(
            compile(ast.Module(body=[route], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        obj = UpdateSessionPriorityReqInput(session_id="s", priority=3)
        reply = await namespace["update_session_priority"](obj, None)
        self.assertEqual(reply.status_code, 200)
        self.assertTrue(json.loads(reply.body)["success"])
        manager.results = [
            UpdateSessionPriorityReqOutput(success=True, message="", found=False)
        ]
        reply = await namespace["update_session_priority"](obj, None)
        self.assertEqual(reply.status_code, 400)
        self.assertFalse(json.loads(reply.body)["success"])

    def test_ipc_round_trip_preserves_priority_and_found(self):
        request = UpdateSessionPriorityReqInput(session_id="s", priority=-4)
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(request), type=UpdateSessionPriorityReqInput
        )
        self.assertEqual(decoded, request)
        reply = UpdateSessionPriorityReqOutput(
            success=True, message="updated", found=True
        )
        self.assertEqual(
            msgspec.msgpack.decode(
                msgspec.msgpack.encode(reply), type=UpdateSessionPriorityReqOutput
            ),
            reply,
        )


class TestSessionPriorityConfiguration(unittest.TestCase):
    def test_threshold_cli_option_is_parsed(self):
        args = prepare_server_args(
            ["--model-path", "dummy", "--high-priority-threshold", "7"]
        )
        self.assertEqual(args.high_priority_threshold, 7)

    def test_reversed_priority_order_is_rejected_for_session_cache(self):
        args = ServerArgs(
            model_path="dummy",
            served_model_name="dummy",
            chunked_prefill_size=32,
            page_size=1,
            enable_session_radix_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        with self.assertRaisesRegex(ValueError, "higher priority values first"):
            args.check_server_args()

    def test_reversed_order_remains_available_without_session_cache(self):
        args = ServerArgs(
            model_path="dummy",
            served_model_name="dummy",
            chunked_prefill_size=32,
            page_size=1,
            enable_session_radix_cache=False,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        args.check_server_args()

    def test_default_threshold_and_priority_scheduling_stay_independent(self):
        args = ServerArgs(
            model_path="dummy",
            served_model_name="dummy",
            chunked_prefill_size=32,
            page_size=1,
            enable_session_radix_cache=True,
        )
        self.assertEqual(args.high_priority_threshold, 1)
        self.assertFalse(args.enable_priority_scheduling)
        args.check_server_args()


if __name__ == "__main__":
    unittest.main()
