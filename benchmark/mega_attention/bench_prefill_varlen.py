#!/usr/bin/env python3
"""megaAttention vs baseline prefill-forward latency on SGLang, exact varlen shapes.

Reuses sglang.bench_one_batch internals (load_model / prepare_synthetic_inputs_for_latency_test
with custom varlen inputs / _TorchBenchRunner.extend / .clear) and times ONLY the prefill forward
(torch.cuda.synchronize + wall clock, warm median). The exact shape is warmed first so flashinfer
per-shape JIT and the megaAttention cute.compile are amortized.

Two regimes (mirrors megaAttention benchmark bench_fused_fa_oproj_ar.py):
  * FULL prefill  (q == k, no KV prefix):
        _CHUNK_SMALL = [6912,3200,1792,1088,512,256]          6 seqs ~13.8K
        _CHUNK_BIG   = [7680,5376,3968,2560,1664,1024,640,256] 8 seqs ~23.2K
  * CHUNKED prefill (0 < q < k, KV prefix already cached — bottom-right causal): q = _CHUNK_SMALL,
    prefix pre-filled so k = 2x q and k = 4x q (the megaAttention q<k ratio sweep). This exercises
    reading a paged KV prefix from cache.

Reproduce (from /root/sglang), TP4:
  # baseline (mega off)
  python benchmark/mega_attention/bench_prefill_varlen.py \
      --model-path /models/Qwen3-30B-A3B --tp 4 --page-size 128 --attention-backend fa3 \
      --chunked-prefill-size 32768 --disable-cuda-graph --dtype bfloat16 --max-running-requests 256
  # mega on
  SGLANG_USE_MEGA_ATTENTION=1 PYTHONPATH=/root/sglang/3rdparty/megaAttention/src \
  python benchmark/mega_attention/bench_prefill_varlen.py --model-path /models/Qwen3-30B-A3B --tp 4 --page-size 128 --attention-backend fa3 \
      --chunked-prefill-size 32768 --disable-cuda-graph --dtype bfloat16 --max-running-requests 256

chunked-prefill-size must exceed the largest per-forward NEW-token total (23.2K) so the batch is
ONE prefill forward; context length (40960) must exceed the largest k (4x13.8K/6seq -> max seq
4*6912=27648) so the prefix fits.
"""

import argparse
import multiprocessing
import os
import statistics
import time
from array import array

import numpy as np

from sglang.bench_one_batch import (
    BenchArgs,
    _set_envs_and_config,
    load_model,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, kill_process_tree

BenchArgs.output_len = 1  # prefill + single sample; avoids tuple max_new_tokens

# FULL prefill sweep 14K -> 64K total NEW tokens per forward (ragged; every seq <= 40960
# context). Purpose: watch whether mega's fused-attention gain grows as the prefill gets
# bigger and the attention segment's share of the whole MoE layer rises.
FULL_SHAPES = {
    "~14K": [6912, 3200, 1792, 1088, 512, 256],  # 13760, 6 seq
    "~32K": [12288, 8192, 5120, 3072, 2048, 1024, 768],  # 32512, 7 seq
    "~48K": [16384, 12288, 8192, 5120, 3072, 2048, 1024],  # 48128, 7 seq
    "~64K": [20480, 16384, 12288, 8192, 4096, 2048, 1024],  # 64512, 7 seq
}
_CS = [6912, 3200, 1792, 1088, 512, 256]  # ~13.8K
_CH = [3456, 1600, 896, 544, 256, 128]  # ~6.9K
_C2B = [8192, 6144, 5120, 4096, 2560, 1536, 512]  # ~28.2K (k=2q -> per-seq k<=16384)
_C4B = [4096, 2560, 1536, 1024, 512, 256]  # ~10K (k=4q -> per-seq k<=16384)
# name -> (q_lens, prefix_lens);  k = q + prefix.  Every per-seq k must stay <= model
# context (40960): k=2q caps single q at ~20K, so the big buckets use more/shorter seqs.
CHUNKED_CASES = {
    "k=2q (q~13.8K)": (_CS, [s * 1 for s in _CS]),  # prefix ~13.8K, k ~27.5K
    "k=2q (q~28K)": (_C2B, [s * 1 for s in _C2B]),  # prefix ~28K, k ~56K
    "k=4q (q~6.9K)": (_CH, [s * 3 for s in _CH]),  # prefix ~20.6K, k ~27.5K
    "k=4q (q~10K)": (_C4B, [s * 3 for s in _C4B]),  # prefix ~30K, k ~40K
}
N_WARM = 2
N_TIMED = 3


def _rand_ids(n):
    return np.random.randint(1, 10000, n, dtype=np.int32).tolist()


def _build_full_reqs(seqlens):
    return prepare_synthetic_inputs_for_latency_test(
        len(seqlens), max(seqlens), [_rand_ids(L) for L in seqlens]
    )


def _time_full_prefill(model_runner, seqlens):
    # model_runner is _TorchBenchRunner: .extend(reqs) runs the prefill forward,
    # .clear() frees the KV / req pools so iterations don't accumulate.
    for _ in range(N_WARM):
        model_runner.extend(_build_full_reqs(seqlens))
        model_runner.synchronize()
        model_runner.clear()
    ts = []
    for _ in range(N_TIMED):
        reqs = _build_full_reqs(seqlens)
        model_runner.synchronize()
        t0 = time.perf_counter()
        model_runner.extend(reqs)
        model_runner.synchronize()
        ts.append(time.perf_counter() - t0)
        model_runner.clear()
    return ts


def _time_chunked_prefill(model_runner, q_lens, prefix_lens):
    # q_lens[i] NEW tokens this chunk; prefix_lens[i] tokens already in KV (bottom-right causal).
    req_to_token = model_runner.torch_runner.req_to_token_pool.req_to_token

    def run_once(timed):
        # phase 1 (untimed): prefill the prefix -> populates KV, assigns req_pool_idx.
        reqs = prepare_synthetic_inputs_for_latency_test(
            len(q_lens), max(prefix_lens), [_rand_ids(p) for p in prefix_lens]
        )
        model_runner.extend(reqs)
        model_runner.synchronize()
        # phase 2 (timed): append q new tokens, point prefix_indices at cached prefix, extend.
        for i, req in enumerate(reqs):
            req.full_untruncated_fill_ids = array(
                "q", list(req.origin_input_ids) + _rand_ids(q_lens[i])
            )
            req.fill_len = len(req.full_untruncated_fill_ids)
            req.prefix_indices = req_to_token[req.req_pool_idx, : prefix_lens[i]].to(
                req.prefix_indices.dtype
            )
            req.logprob_start_len = -1
            req.set_extend_input_len(req.fill_len - len(req.prefix_indices))
        dt = None
        if timed:
            model_runner.synchronize()
            t0 = time.perf_counter()
        model_runner.extend(reqs)
        model_runner.synchronize()
        if timed:
            dt = time.perf_counter() - t0
        model_runner.clear()
        return dt

    for _ in range(N_WARM):
        run_once(False)
    return [run_once(True) for _ in range(N_TIMED)]


def work(server_args, port_args, gpu_id, tp_rank):
    _set_envs_and_config(server_args)
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rp = print if tp_rank == 0 else (lambda *a, **k: None)
    model_runner, _ = load_model(server_args, port_args, gpu_id, tp_rank)
    mega = os.environ.get("SGLANG_USE_MEGA_ATTENTION", "0") == "1"

    rp(f"\n===== prefill-forward latency  (mega={'ON' if mega else 'off'}) =====")
    rp("-- FULL prefill (q==k, no KV prefix) --")
    for name, seqlens in FULL_SHAPES.items():
        ts = _time_full_prefill(model_runner, seqlens)
        med = statistics.median(ts)
        tot = sum(seqlens)
        rp(
            f"[{name:11s} n={len(seqlens)} q=k={tot:6d}] median = {med*1000:8.2f} ms   "
            f"{tot/med:9.0f} tok/s   iters(ms)={[round(t*1e3, 2) for t in ts]}"
        )

    # ---- correctness gate (all ranks print): after real prefills, confirm mega actually
    #      engaged on THIS rank (not silently fell back). launch_count>0 & not disabled means
    #      the q_per_kv=16 workspace built + launched. In baseline (mega off) runtime is None. ----
    from sglang.srt.layers.attention.mega_attention_runtime import get_mega_runtime

    _rt = get_mega_runtime()
    _eng = _rt is not None and not _rt._disabled and _rt.mega_launch_count > 0
    print(
        f"[GATE] TP{tp_rank} mega_engaged={_eng} "
        f"launch_count={_rt.mega_launch_count if _rt else 0} disabled={_rt._disabled if _rt else 'n/a'}",
        flush=True,
    )

    rp("-- CHUNKED prefill (q<k, KV prefix cached) --")
    for name, (q_lens, prefix_lens) in CHUNKED_CASES.items():
        ts = _time_chunked_prefill(model_runner, q_lens, prefix_lens)
        med = statistics.median(ts)
        totq, totk = sum(q_lens), sum(q_lens) + sum(prefix_lens)
        rp(
            f"[{name:17s} q={totq:6d} k={totk:6d}] median = {med*1000:8.2f} ms   "
            f"{totq/med:9.0f} q-tok/s   iters(ms)={[round(t*1e3, 2) for t in ts]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ServerArgs.add_cli_args(ap)
    args = ap.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    if server_args.tp_size == 1:
        work(server_args, port_args, 0, 0)
    else:
        procs = []
        for tp_rank in range(server_args.tp_size):
            p = multiprocessing.Process(
                target=work, args=(server_args, port_args, tp_rank, tp_rank)
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    try:
        main()
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
