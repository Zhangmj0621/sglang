"""
Benchmark: ref-aware KV cache vs baseline HiCache under multi-turn RL workload.

Simulates RL multi-turn rollouts where each sample goes through up to 20 turns,
with random sleep between turns to model environment interaction. Measures
throughput, cache hit rate, TTFT, and load-back frequency across different
batch sizes.

Usage:
  # Start server (baseline):
  python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct \
      --enable-hierarchical-cache --hicache-write-policy write_through

  # Start server (ref-aware):
  python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct \
      --enable-hierarchical-cache --hicache-write-policy write_through \
      --enable-ref-aware-kv-buffer --high-priority-threshold 1

  # Run benchmark:
  python benchmark/hicache/bench_ref_aware_multiturn.py \
      --num-clients 128 --max-turns 20 --request-length 256 --output-length 32 \
      --interaction-delay-min 0.1 --interaction-delay-max 2.0

  # Sweep batch sizes:
  python benchmark/hicache/bench_ref_aware_multiturn.py \
      --sweep-clients 32,64,128,256,512
"""

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import numpy as np

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


@dataclass
class TurnResult:
    success: bool = False
    ttft: float = 0.0
    latency: float = 0.0
    prompt_len: int = 0
    cached_tokens: int = 0
    generated_len: int = 0
    output_ids: List[int] = field(default_factory=list)
    error: str = ""
    itl: List[float] = field(default_factory=list)


async def send_generate(
    url: str,
    input_ids: List[int],
    output_len: int,
    rid: str,
    priority: int,
    is_first_turn: bool,
) -> TurnResult:
    payload = {
        "input_ids": input_ids,
        "rid": rid,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "priority": priority,
        "is_first_turn": is_first_turn,
    }

    result = TurnResult()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        st = time.perf_counter()
        most_recent_ts = st
        try:
            async with session.post(url=url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        text = chunk_bytes.decode("utf-8")
                        if text.startswith("data: "):
                            text = text[6:]
                        if text == "[DONE]":
                            continue
                        data = json.loads(text)
                        if data.get("output_ids"):
                            result.output_ids = data["output_ids"]
                            ts = time.perf_counter()
                            if result.ttft == 0.0:
                                result.ttft = ts - st
                                meta = data.get("meta_info") or {}
                                result.prompt_len = meta.get("prompt_tokens", 0)
                                result.cached_tokens = meta.get("cached_tokens", 0)
                            else:
                                result.itl.append(ts - most_recent_ts)
                            most_recent_ts = ts
                    result.latency = time.perf_counter() - st
                    result.generated_len = len(result.itl) + 1
                    result.success = True
                else:
                    result.error = f"HTTP {response.status}: {response.reason}"
        except Exception as e:
            result.error = str(e)
    return result


async def send_release_ref(base_url: str, rid: str):
    url = f"{base_url}/release_ref"
    payload = {"rid": rid}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(url=url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                return {"success": False, "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "message": str(e)}


async def send_update_ref(base_url: str, rid: str, new_priority: int):
    url = f"{base_url}/update_ref"
    payload = {"rid": rid, "new_priority": new_priority}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(url=url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                return {"success": False, "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "message": str(e)}


async def run_one_rollout(
    client_id: int,
    base_url: str,
    initial_input_ids: List[int],
    sub_question_pool: List[List[int]],
    num_turns: int,
    output_len: int,
    priority: int,
    interaction_delay_min: float,
    interaction_delay_max: float,
    enable_ref_api: bool,
    semaphore: asyncio.Semaphore,
    priority_flip_prob: float = 0.0,
    high_priority_threshold: int = 1,
) -> Dict:
    generate_url = f"{base_url}/generate"
    rid = f"rollout-{client_id}"
    history = list(initial_input_ids)
    turn_results = []

    for turn in range(num_turns):
        is_first_turn = (turn == 0)
        async with semaphore:
            result = await send_generate(
                generate_url, history, output_len, rid, priority, is_first_turn
            )

        turn_results.append({
            "turn": turn,
            "success": result.success,
            "ttft": result.ttft,
            "latency": result.latency,
            "prompt_len": result.prompt_len,
            "cached_tokens": result.cached_tokens,
            "generated_len": result.generated_len,
            "cache_hit_rate": (
                result.cached_tokens / result.prompt_len
                if result.prompt_len > 0 else 0.0
            ),
        })

        if not result.success:
            print(f"  [client {client_id}] turn {turn} FAILED: {result.error}")
            break

        history.extend(result.output_ids)

        if turn < num_turns - 1:
            # Simulate environment interaction delay
            delay = random.uniform(interaction_delay_min, interaction_delay_max)
            await asyncio.sleep(delay)

            # Optionally flip priority mid-rollout
            if enable_ref_api and priority_flip_prob > 0 and random.random() < priority_flip_prob:
                new_priority = high_priority_threshold if priority < high_priority_threshold else 0
                await send_update_ref(base_url, rid, new_priority)
                priority = new_priority

            # Append sub-question tokens for next turn
            sub_q = random.choice(sub_question_pool)
            history.extend(sub_q)

    # Release ref after rollout completes
    if enable_ref_api:
        await send_release_ref(base_url, rid)

    return {
        "client_id": client_id,
        "rid": rid,
        "num_turns_completed": len(turn_results),
        "priority": priority,
        "turns": turn_results,
    }


async def run_benchmark(args):
    base_url = f"http://{args.host}:{args.port}"

    # Detect if server has ref-aware enabled
    enable_ref_api = False
    try:
        resp_data = await send_release_ref(base_url, "__probe__")
        msg = resp_data.get("message", "")
        enable_ref_api = "not found" in msg or "released" in msg
        if not enable_ref_api and "not enabled" in msg:
            enable_ref_api = False
    except Exception:
        pass
    print(f"Ref-aware API detected: {enable_ref_api}")

    # Generate random token sequences for prompts and sub-questions
    random.seed(args.seed)
    np.random.seed(args.seed)
    vocab_size = args.vocab_size

    initial_inputs = [
        [random.randint(1, vocab_size - 1) for _ in range(args.request_length)]
        for _ in range(args.num_clients)
    ]
    sub_question_pool = [
        [random.randint(1, vocab_size - 1) for _ in range(args.sub_question_length)]
        for _ in range(args.sub_question_pool_size)
    ]

    # Assign priorities: mix of high and low
    priorities = []
    for i in range(args.num_clients):
        if random.random() < args.high_priority_ratio:
            priorities.append(args.high_priority_threshold)
        else:
            priorities.append(0)

    # Assign per-client turn count
    turn_counts = [
        random.randint(args.min_turns, args.max_turns)
        for _ in range(args.num_clients)
    ]

    total_turns = sum(turn_counts)
    print(f"Clients: {args.num_clients}, Total turns: {total_turns}, "
          f"High-priority ratio: {sum(1 for p in priorities if p >= args.high_priority_threshold) / len(priorities):.2f}")

    # Flush cache
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(f"{base_url}/flush_cache")
            await asyncio.sleep(1)
        except Exception:
            pass

    semaphore = asyncio.Semaphore(args.max_parallel)
    start_time = time.perf_counter()

    tasks = [
        run_one_rollout(
            client_id=i,
            base_url=base_url,
            initial_input_ids=initial_inputs[i],
            sub_question_pool=sub_question_pool,
            num_turns=turn_counts[i],
            output_len=args.output_length,
            priority=priorities[i],
            interaction_delay_min=args.interaction_delay_min,
            interaction_delay_max=args.interaction_delay_max,
            enable_ref_api=enable_ref_api,
            semaphore=semaphore,
            priority_flip_prob=args.priority_flip_prob,
            high_priority_threshold=args.high_priority_threshold,
        )
        for i in range(args.num_clients)
    ]

    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time

    # Aggregate metrics
    all_turns = []
    per_turn_metrics = defaultdict(lambda: {
        "ttft": [], "latency": [], "cache_hit_rate": [],
        "prompt_len": [], "cached_tokens": [],
    })
    high_turns = []
    low_turns = []

    for r in results:
        for t in r["turns"]:
            if t["success"]:
                all_turns.append(t)
                per_turn_metrics[t["turn"]]["ttft"].append(t["ttft"])
                per_turn_metrics[t["turn"]]["latency"].append(t["latency"])
                per_turn_metrics[t["turn"]]["cache_hit_rate"].append(t["cache_hit_rate"])
                per_turn_metrics[t["turn"]]["prompt_len"].append(t["prompt_len"])
                per_turn_metrics[t["turn"]]["cached_tokens"].append(t["cached_tokens"])

                if r["priority"] >= args.high_priority_threshold:
                    high_turns.append(t)
                else:
                    low_turns.append(t)

    def stats(values):
        if not values:
            return {"mean": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
        s = sorted(values)
        return {
            "mean": np.mean(s),
            "p50": np.percentile(s, 50),
            "p90": np.percentile(s, 90),
            "p99": np.percentile(s, 99),
            "max": s[-1],
        }

    total_prompt_tokens = sum(t["prompt_len"] for t in all_turns)
    total_cached_tokens = sum(t["cached_tokens"] for t in all_turns)
    total_generated_tokens = sum(t["generated_len"] for t in all_turns)
    successful_turns = len(all_turns)
    failed_turns = total_turns - successful_turns

    # Turn >= 1 cache hit rate (excludes cold-start turn 0)
    subsequent_turns = [t for t in all_turns if t["turn"] > 0]
    subsequent_cache_hit = (
        sum(t["cached_tokens"] for t in subsequent_turns)
        / sum(t["prompt_len"] for t in subsequent_turns)
        if subsequent_turns else 0.0
    )

    summary = {
        "config": {
            "num_clients": args.num_clients,
            "max_turns": args.max_turns,
            "max_parallel": args.max_parallel,
            "request_length": args.request_length,
            "output_length": args.output_length,
            "interaction_delay": f"{args.interaction_delay_min}-{args.interaction_delay_max}s",
            "high_priority_ratio": args.high_priority_ratio,
            "ref_aware_enabled": enable_ref_api,
        },
        "duration_s": duration,
        "total_turns": total_turns,
        "successful_turns": successful_turns,
        "failed_turns": failed_turns,
        "throughput_turns_per_s": successful_turns / duration,
        "input_token_throughput": total_prompt_tokens / duration,
        "output_token_throughput": total_generated_tokens / duration,
        "overall_cache_hit_rate": total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0,
        "subsequent_turn_cache_hit_rate": subsequent_cache_hit,
        "ttft": stats([t["ttft"] for t in all_turns]),
        "latency": stats([t["latency"] for t in all_turns]),
    }

    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS: num_clients={args.num_clients}, ref_aware={enable_ref_api}")
    print(f"{'='*70}")
    print(f"  Duration:              {duration:.2f}s")
    print(f"  Turns completed:       {successful_turns}/{total_turns} ({failed_turns} failed)")
    print(f"  Turn throughput:       {summary['throughput_turns_per_s']:.2f} turns/s")
    print(f"  Output tok throughput: {summary['output_token_throughput']:.0f} tok/s")
    print(f"  Overall cache hit:     {summary['overall_cache_hit_rate']:.4f}")
    print(f"  Turn>=1 cache hit:     {subsequent_cache_hit:.4f}  <-- KEY METRIC")
    print(f"  TTFT mean/p90/p99:     {summary['ttft']['mean']:.3f} / {summary['ttft']['p90']:.3f} / {summary['ttft']['p99']:.3f}s")
    print(f"  Latency mean/p90/p99:  {summary['latency']['mean']:.3f} / {summary['latency']['p90']:.3f} / {summary['latency']['p99']:.3f}s")

    if high_turns and low_turns:
        high_cache_hit = sum(t["cached_tokens"] for t in high_turns if t["turn"] > 0) / max(sum(t["prompt_len"] for t in high_turns if t["turn"] > 0), 1)
        low_cache_hit = sum(t["cached_tokens"] for t in low_turns if t["turn"] > 0) / max(sum(t["prompt_len"] for t in low_turns if t["turn"] > 0), 1)
        print(f"  High-priority cache hit (turn>=1): {high_cache_hit:.4f}")
        print(f"  Low-priority cache hit (turn>=1):  {low_cache_hit:.4f}")
        summary["high_priority_cache_hit"] = high_cache_hit
        summary["low_priority_cache_hit"] = low_cache_hit

    # Per-turn breakdown
    print(f"\n  Per-turn breakdown:")
    for turn_num in sorted(per_turn_metrics.keys()):
        m = per_turn_metrics[turn_num]
        avg_hit = np.mean(m["cache_hit_rate"]) if m["cache_hit_rate"] else 0
        avg_ttft = np.mean(m["ttft"]) if m["ttft"] else 0
        count = len(m["ttft"])
        print(f"    Turn {turn_num:2d}: n={count:4d}, cache_hit={avg_hit:.4f}, ttft={avg_ttft:.3f}s")

    # Log to file
    if args.log_file:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tag": args.tag,
            **summary,
        }
        with open(args.log_file, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")
        print(f"\n  Logged to {args.log_file}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ref-aware KV cache under multi-turn RL workload"
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--num-clients", type=int, default=128,
                        help="Number of concurrent rollout clients (effective batch size)")
    parser.add_argument("--max-parallel", type=int, default=64,
                        help="Max concurrent in-flight requests to server")
    parser.add_argument("--min-turns", type=int, default=5,
                        help="Min turns per rollout")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max turns per rollout")
    parser.add_argument("--request-length", type=int, default=256,
                        help="Initial prompt length in tokens")
    parser.add_argument("--sub-question-length", type=int, default=128,
                        help="Sub-question length appended each turn")
    parser.add_argument("--sub-question-pool-size", type=int, default=500,
                        help="Pool size of pre-generated sub-questions")
    parser.add_argument("--output-length", type=int, default=32,
                        help="Max new tokens per turn")
    parser.add_argument("--interaction-delay-min", type=float, default=0.1,
                        help="Min sleep (seconds) between turns simulating env interaction")
    parser.add_argument("--interaction-delay-max", type=float, default=2.0,
                        help="Max sleep (seconds) between turns simulating env interaction")
    parser.add_argument("--high-priority-ratio", type=float, default=0.5,
                        help="Fraction of clients that are high-priority")
    parser.add_argument("--high-priority-threshold", type=int, default=1,
                        help="Priority >= threshold is high-priority")
    parser.add_argument("--priority-flip-prob", type=float, default=0.0,
                        help="Probability of flipping priority via /update_ref between turns")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocab size for random token generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", type=str, default="bench_ref_aware_multiturn.jsonl")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for this run (e.g., 'baseline' or 'ref-aware')")
    parser.add_argument("--sweep-clients", type=str, default="",
                        help="Comma-separated list of num_clients to sweep, e.g., '32,64,128,256'")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sweep_clients:
        client_counts = [int(x.strip()) for x in args.sweep_clients.split(",")]
        print(f"Sweeping batch sizes: {client_counts}")
        all_results = []
        for nc in client_counts:
            args.num_clients = nc
            print(f"\n{'#'*70}")
            print(f"  Running with num_clients={nc}")
            print(f"{'#'*70}")
            result = asyncio.run(run_benchmark(args))
            all_results.append({"num_clients": nc, **result})

        # Print comparison table
        print(f"\n{'='*90}")
        print(f"  SWEEP SUMMARY")
        print(f"{'='*90}")
        print(f"  {'clients':>8} {'turns/s':>10} {'out_tok/s':>12} {'cache_hit':>10} "
              f"{'turn1+_hit':>12} {'ttft_p90':>10} {'lat_p90':>10}")
        print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        for r in all_results:
            print(f"  {r['num_clients']:>8} "
                  f"{r['throughput_turns_per_s']:>10.2f} "
                  f"{r['output_token_throughput']:>12.0f} "
                  f"{r['overall_cache_hit_rate']:>10.4f} "
                  f"{r['subsequent_turn_cache_hit_rate']:>12.4f} "
                  f"{r['ttft']['p90']:>10.3f} "
                  f"{r['latency']['p90']:>10.3f}")
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
