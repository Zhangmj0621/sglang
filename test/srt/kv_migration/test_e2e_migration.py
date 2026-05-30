"""End-to-end test for HTTP-driven KV cache migration between two sglang servers.

Requires two GPUs and Mooncake-capable RDMA. Skipped automatically unless
SGLANG_KV_MIGRATION_E2E=1 is set in the environment.

Run:

    SGLANG_KV_MIGRATION_E2E=1 \\
    pytest test/srt/kv_migration/test_e2e_migration.py -v -s
"""

import os
import unittest

import requests

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    popen_launch_server,
)


@unittest.skipIf(
    os.environ.get("SGLANG_KV_MIGRATION_E2E", "0") != "1",
    "Set SGLANG_KV_MIGRATION_E2E=1 to run two-server migration E2E "
    "(requires 2 GPUs and Mooncake)",
)
class TestKVMigrationE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        common_args = [
            "--enable-ref-aware-kv-buffer",
            "--enable-kv-migration",
            "--page-size",
            "64",
        ]
        cls.url_a = "http://127.0.0.1:7501"
        cls.url_b = "http://127.0.0.1:7502"
        cls.proc_a = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            base_url=cls.url_a,
            timeout=120,
            other_args=[*common_args, "--gpu-id", "0"],
        )
        cls.proc_b = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            base_url=cls.url_b,
            timeout=120,
            other_args=[*common_args, "--gpu-id", "1"],
        )

    @classmethod
    def tearDownClass(cls):
        for proc in (getattr(cls, "proc_a", None), getattr(cls, "proc_b", None)):
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()

    def test_migration_round_trip(self):
        # 1) Discover topology of both servers.
        topo_a = requests.get(f"{self.url_a}/get_transfer_session_ids").json()
        topo_b = requests.get(f"{self.url_b}/get_transfer_session_ids").json()
        self.assertEqual(
            topo_a["topology"]["tp_size"], topo_b["topology"]["tp_size"]
        )

        # 2) Run /generate on A so it builds up KV cache for some prefix.
        rid = "rid-test-1"
        prompt = "The quick brown fox jumps over the lazy dog. " * 20
        gen_resp = requests.post(
            f"{self.url_a}/generate",
            json={
                "text": prompt,
                "rid": rid,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            },
        )
        self.assertTrue(gen_resp.ok, gen_resp.text)

        tok_resp = requests.post(f"{self.url_a}/tokenize", json={"text": prompt})
        self.assertTrue(tok_resp.ok, tok_resp.text)
        input_ids = tok_resp.json()["input_ids"]

        # 3) Ask target B how much it is missing for input_ids.
        extra_resp = requests.post(
            f"{self.url_b}/get_request_extra_token_size",
            json={"input_ids": input_ids, "extra_key": None},
        ).json()
        self.assertTrue(extra_resp["success"], extra_resp)
        extra_token_size = extra_resp["extra_token_size"]
        matched_token_size = extra_resp["matched_token_size"]
        self.assertGreater(extra_token_size, 0)

        # 4) Pre-allocate on B.
        alloc_resp = requests.post(
            f"{self.url_b}/allocate_token_for_transfer_request",
            json={
                "input_ids": input_ids,
                "extra_key": None,
                "extra_token_size": extra_token_size,
            },
        ).json()
        self.assertTrue(alloc_resp["success"], alloc_resp)
        migration_id = alloc_resp["migration_id"]

        # 5) Build target_per_rank from B's topology + alloc response.
        target_per_rank = []
        for rank in topo_b["ranks"]:
            kv_indices = next(
                p["kv_indices"]
                for p in alloc_resp["per_rank_kv_indices"]
                if p["tp"] == rank["tp"] and p["pp"] == rank["pp"]
            )
            target_per_rank.append(
                {
                    "tp": rank["tp"],
                    "pp": rank["pp"],
                    "session_id": rank["session_id"],
                    "host_kv_data_ptrs": rank["host_kv_data_ptrs"],
                    "host_kv_item_lens": rank["host_kv_item_lens"],
                    "kv_indices": kv_indices,
                }
            )

        # 6) On A, transfer.
        xfer_resp = requests.post(
            f"{self.url_a}/transfer_request_kvcache",
            json={
                "input_ids": input_ids,
                "extra_key": None,
                "matched_token_size": matched_token_size,
                "extra_token_size": extra_token_size,
                "target_per_rank": target_per_rank,
            },
        ).json()
        self.assertTrue(xfer_resp["success"], xfer_resp)

        # 7) On B, commit the transfer.
        commit_resp = requests.post(
            f"{self.url_b}/commit_transfer_request_kvcache",
            json={"migration_id": migration_id},
        ).json()
        self.assertTrue(commit_resp["success"], commit_resp)

        # 8) Verify B can hit the migrated cache.
        extra_after = requests.post(
            f"{self.url_b}/get_request_extra_token_size",
            json={"input_ids": input_ids, "extra_key": None},
        ).json()
        self.assertGreater(
            extra_after["matched_token_size"],
            matched_token_size,
            msg=(
                f"matched did not grow: before={matched_token_size}, "
                f"after={extra_after['matched_token_size']}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
