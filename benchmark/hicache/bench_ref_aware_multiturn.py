import argparse
import asyncio
import json
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
    completion_tokens: int = 0
    generated_len: int = 0
    output_ids: List[int] = field(default_factory=list)
    error: str = ""
    itl: List[float] = field(default_factory=list)
    meta_info: Dict = field(default_factory=dict)
    finish_reason: Optional[object] = None
    server_decode_throughput: Optional[float] = None
    response_sent_to_client_ts: Optional[float] = None
    queue_time: Optional[float] = None
    start_ts: float = 0.0
    first_token_ts: float = 0.0
    end_ts: float = 0.0


async def send_generate(
    url: str,
    input_ids: List[int],
    output_len: int,
    rid: str,
    priority: int,
    is_first_turn: bool,
    request_log_metrics: bool,
) -> TurnResult:
    payload = {
        "input_ids": input_ids,
        "rid": rid,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "priority": priority,
        "is_first_turn": is_first_turn,
    }
    if request_log_metrics:
        payload["log_metrics"] = True

    result = TurnResult()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        st = time.perf_counter()
        result.start_ts = st
        most_recent_ts = st
        saw_output_chunk = False
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
                        if data.get("error"):
                            error = data["error"]
                            if isinstance(error, dict):
                                result.error = error.get("message", json.dumps(error))
                            else:
                                result.error = str(error)
                            break
                        meta = data.get("meta_info") or {}
                        if meta:
                            result.meta_info = meta
                            result.prompt_len = meta.get(
                                "prompt_tokens", result.prompt_len
                            )
                            result.cached_tokens = meta.get(
                                "cached_tokens", result.cached_tokens
                            )
                            result.completion_tokens = meta.get(
                                "completion_tokens", result.completion_tokens
                            )
                            result.server_decode_throughput = meta.get(
                                "decode_throughput", result.server_decode_throughput
                            )
                            result.response_sent_to_client_ts = meta.get(
                                "response_sent_to_client_ts",
                                result.response_sent_to_client_ts,
                            )
                            result.queue_time = meta.get(
                                "queue_time", result.queue_time
                            )
                            finish_reason = meta.get("finish_reason")
                            if finish_reason is not None:
                                result.finish_reason = finish_reason
                                if (
                                    isinstance(finish_reason, dict)
                                    and finish_reason.get("type") == "abort"
                                ):
                                    result.error = finish_reason.get(
                                        "message", json.dumps(finish_reason)
                                    )
                                    if "output_ids" in data:
                                        result.output_ids = data["output_ids"] or []
                                    break

                        if data.get("output_ids") is not None:
                            saw_output_chunk = True
                            result.output_ids = data["output_ids"] or []
                            ts = time.perf_counter()
                            if result.ttft == 0.0:
                                result.ttft = ts - st
                                result.first_token_ts = ts
                            else:
                                result.itl.append(ts - most_recent_ts)
                            most_recent_ts = ts
                    result.end_ts = time.perf_counter()
                    result.latency = result.end_ts - st

                    if saw_output_chunk:
                        # Prefer engine-reported completion token count.
                        if result.completion_tokens > 0:
                            result.generated_len = result.completion_tokens
                        else:
                            # Fallback: final output_ids length is still much better than
                            # counting streaming chunks, though some old versions may differ.
                            result.generated_len = len(result.output_ids)
                            result.completion_tokens = result.generated_len
                        result.success = not result.error
                    elif not result.error:
                        if result.finish_reason is not None:
                            result.error = f"No output_ids before finish_reason={result.finish_reason}"
                        else:
                            result.error = (
                                "No output_ids received before stream completion"
                            )
                else:
                    error_body = (await response.text()).strip()
                    result.error = f"HTTP {response.status}: {response.reason}"
                    if error_body:
                        result.error += f" | {error_body}"
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
    rid_prefix: str,
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
    request_log_metrics: bool = True,
) -> Dict:
    generate_url = f"{base_url}/generate"
    rid = f"{rid_prefix}-{client_id}"
    history = list(initial_input_ids)
    turn_results = []

    for turn in range(num_turns):
        is_first_turn = turn == 0
        async with semaphore:
            result = await send_generate(
                generate_url,
                history,
                output_len,
                rid,
                priority,
                is_first_turn,
                request_log_metrics,
            )

        turn_results.append(
            {
                "turn": turn,
                "priority": priority,
                "is_high_priority": priority >= high_priority_threshold,
                "success": result.success,
                "ttft": result.ttft,
                "latency": result.latency,
                "prompt_len": result.prompt_len,
                "cached_tokens": result.cached_tokens,
                "completion_tokens": result.completion_tokens,
                "generated_len": result.generated_len,
                "cache_hit_rate": (
                    result.cached_tokens / result.prompt_len
                    if result.prompt_len > 0
                    else 0.0
                ),
                "server_decode_throughput": result.server_decode_throughput,
                "queue_time": result.queue_time,
                "start_ts": result.start_ts,
                "first_token_ts": result.first_token_ts,
                "end_ts": result.end_ts,
            }
        )

        if not result.success:
            print(f"  [client {client_id}] turn {turn} FAILED: {result.error}")
            break

        history.extend(result.output_ids)

        if turn < num_turns - 1:
            delay = random.uniform(interaction_delay_min, interaction_delay_max)
            await asyncio.sleep(delay)

            if (
                enable_ref_api
                and priority_flip_prob > 0
                and random.random() < priority_flip_prob
            ):
                new_priority = (
                    high_priority_threshold if priority < high_priority_threshold else 0
                )
                await send_update_ref(base_url, rid, new_priority)
                priority = new_priority

            sub_q = random.choice(sub_question_pool)
            history.extend(sub_q)

    if enable_ref_api:
        await send_release_ref(base_url, rid)

    return {
        "client_id": client_id,
        "rid": rid,
        "num_turns_completed": len(turn_results),
        "priority": priority,
        "turns": turn_results,
    }


def interval_union_length(intervals: List[Tuple[float, float]]) -> float:
    valid = [(s, e) for s, e in intervals if e > s]
    if not valid:
        return 0.0
    valid.sort()
    total = 0.0
    cur_s, cur_e = valid[0]
    for s, e in valid[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def weighted_mean(values: List[Tuple[float, float]]) -> float:
    # values = [(metric, weight), ...]
    total_weight = sum(w for _, w in values if w > 0)
    if total_weight <= 0:
        return 0.0
    return sum(v * w for v, w in values if w > 0) / total_weight


def fixed_priority_split(
    num_clients: int, high_priority_threshold: int
) -> Tuple[List[int], int, int]:
    if num_clients % 3 != 0:
        raise ValueError(
            f"num_clients must be divisible by 3 to enforce an exact 2/3 high-priority "
            f"and 1/3 low-priority split, got {num_clients}"
        )

    high_priority_count = num_clients * 2 // 3
    low_priority_count = num_clients - high_priority_count
    priorities = [high_priority_threshold] * high_priority_count + [
        0
    ] * low_priority_count
    return priorities, high_priority_count, low_priority_count


async def run_benchmark(args):
    base_url = f"http://{args.host}:{args.port}"
    rid_prefix = args.rid_prefix or (
        f"rollout-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    )

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    vocab_size = args.vocab_size

    if args.priority_flip_prob != 0.0:
        raise ValueError(
            "priority_flip_prob must be 0.0 when using the fixed 2/3 high-priority "
            "and 1/3 low-priority benchmark configuration"
        )

    # Respect the requested lengths directly.
    initial_inputs = [
        [random.randint(1, vocab_size - 1) for _ in range(args.request_length)]
        for _ in range(args.num_clients)
    ]
    sub_question_pool = [
        [random.randint(1, vocab_size - 1) for _ in range(args.sub_question_length)]
        for _ in range(args.sub_question_pool_size)
    ]

    priorities, high_priority_count, low_priority_count = fixed_priority_split(
        args.num_clients, args.high_priority_threshold
    )
    effective_max_parallel = high_priority_count + 100

    turn_counts = [
        random.randint(args.min_turns, args.max_turns) for _ in range(args.num_clients)
    ]

    total_turns = sum(turn_counts)
    print(
        f"Clients: {args.num_clients}, Total turns: {total_turns}, "
        f"High-priority clients: {high_priority_count}, "
        f"Low-priority clients: {low_priority_count}, "
        f"High-priority ratio: {high_priority_count / len(priorities):.2f}"
    )
    print(f"Forced max_parallel: {effective_max_parallel}")
    print(f"RID prefix: {rid_prefix}")

    async with aiohttp.ClientSession() as session:
        try:
            await session.post(f"{base_url}/flush_cache")
            await asyncio.sleep(1)
        except Exception:
            pass

    semaphore = asyncio.Semaphore(effective_max_parallel)
    start_time = time.perf_counter()

    tasks = [
        run_one_rollout(
            client_id=i,
            rid_prefix=rid_prefix,
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
            request_log_metrics=args.request_log_metrics,
        )
        for i in range(args.num_clients)
    ]

    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time

    all_turns = []
    focus_all_turns = []
    focus_turns = []
    per_turn_metrics = defaultdict(
        lambda: {
            "ttft": [],
            "latency": [],
            "cache_hit_rate": [],
            "prompt_len": [],
            "cached_tokens": [],
            "completion_tokens": [],
            "server_decode_throughput": [],
        }
    )
    num_high_priority_clients = high_priority_count

    for r in results:
        for t in r["turns"]:
            if t["is_high_priority"]:
                focus_all_turns.append(t)
            if t["success"]:
                all_turns.append(t)
                if t["is_high_priority"]:
                    focus_turns.append(t)
                    per_turn_metrics[t["turn"]]["ttft"].append(t["ttft"])
                    per_turn_metrics[t["turn"]]["latency"].append(t["latency"])
                    per_turn_metrics[t["turn"]]["cache_hit_rate"].append(
                        t["cache_hit_rate"]
                    )
                    per_turn_metrics[t["turn"]]["prompt_len"].append(t["prompt_len"])
                    per_turn_metrics[t["turn"]]["cached_tokens"].append(
                        t["cached_tokens"]
                    )
                    per_turn_metrics[t["turn"]]["completion_tokens"].append(
                        t["completion_tokens"]
                    )
                    if t["server_decode_throughput"] is not None:
                        per_turn_metrics[t["turn"]]["server_decode_throughput"].append(
                            t["server_decode_throughput"]
                        )

    def stats(values):
        if not values:
            return {"mean": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
        s = sorted(values)
        return {
            "mean": float(np.mean(s)),
            "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)),
            "p99": float(np.percentile(s, 99)),
            "max": float(s[-1]),
        }

    total_prompt_tokens = sum(t["prompt_len"] for t in focus_turns)
    total_cached_tokens = sum(t["cached_tokens"] for t in focus_turns)
    total_completion_tokens = sum(t["completion_tokens"] for t in focus_turns)
    successful_turns = len(focus_turns)
    failed_turns = len([t for t in focus_all_turns if not t["success"]])

    subsequent_turns = [t for t in focus_turns if t["turn"] > 0]
    first_turns = [t for t in focus_turns if t["turn"] == 0]
    subsequent_cache_hit = (
        sum(t["cached_tokens"] for t in subsequent_turns)
        / sum(t["prompt_len"] for t in subsequent_turns)
        if subsequent_turns
        else 0.0
    )
    high_priority_makespan = (
        max((t["end_ts"] for t in focus_turns), default=start_time) - start_time
    )

    decode_intervals = [
        (t["start_ts"] + t["ttft"], t["end_ts"])
        for t in focus_turns
        if t["success"]
        and t["completion_tokens"] > 0
        and t["end_ts"] > (t["start_ts"] + t["ttft"])
    ]
    active_decode_wall_time = interval_union_length(decode_intervals)
    output_token_throughput_wall = (
        total_completion_tokens / duration if duration > 0 else 0.0
    )
    output_token_throughput_active_decode = (
        total_completion_tokens / active_decode_wall_time
        if active_decode_wall_time > 0
        else 0.0
    )

    server_decode_samples = [
        (t["server_decode_throughput"], t["completion_tokens"])
        for t in focus_turns
        if t["server_decode_throughput"] is not None and t["completion_tokens"] > 0
    ]
    server_decode_weighted_mean = weighted_mean(server_decode_samples)
    server_decode_stats = (
        stats([v for v, _ in server_decode_samples]) if server_decode_samples else None
    )

    summary = {
        "config": {
            "num_clients": args.num_clients,
            "num_high_priority_clients": num_high_priority_clients,
            "num_low_priority_clients": low_priority_count,
            "max_turns": args.max_turns,
            "max_parallel": effective_max_parallel,
            "request_length": args.request_length,
            "output_length": args.output_length,
            "interaction_delay": f"{args.interaction_delay_min}-{args.interaction_delay_max}s",
            "high_priority_ratio": high_priority_count / args.num_clients,
            "ref_aware_enabled": enable_ref_api,
            "request_log_metrics": args.request_log_metrics,
            "metrics_scope": "high_priority_only",
            "priority_split_mode": "fixed_2_to_1",
        },
        "duration_s": duration,
        "high_priority_makespan_s": high_priority_makespan,
        "active_decode_wall_time_s": active_decode_wall_time,
        "total_turns": len(focus_all_turns),
        "successful_turns": successful_turns,
        "failed_turns": failed_turns,
        "throughput_turns_per_s": successful_turns / duration if duration > 0 else 0.0,
        "input_token_throughput_wall": (
            total_prompt_tokens / duration if duration > 0 else 0.0
        ),
        "output_token_throughput_wall": output_token_throughput_wall,
        "output_token_throughput_active_decode": output_token_throughput_active_decode,
        "total_completion_tokens": total_completion_tokens,
        "overall_cache_hit_rate": (
            total_cached_tokens / total_prompt_tokens
            if total_prompt_tokens > 0
            else 0.0
        ),
        "subsequent_turn_cache_hit_rate": subsequent_cache_hit,
        "first_turn_ttft": stats([t["ttft"] for t in first_turns]),
        "first_turn_latency": stats([t["latency"] for t in first_turns]),
        "subsequent_ttft": stats([t["ttft"] for t in subsequent_turns]),
        "subsequent_latency": stats([t["latency"] for t in subsequent_turns]),
        "ttft": stats([t["ttft"] for t in focus_turns]),
        "latency": stats([t["latency"] for t in focus_turns]),
    }
    if server_decode_samples:
        summary["engine_decode_throughput_weighted_mean"] = server_decode_weighted_mean
        summary["engine_decode_throughput_stats"] = server_decode_stats

    print(f"\n{'='*78}")
    print(
        f"  HIGH-PRIORITY RESULTS: num_clients={args.num_clients}, ref_aware={enable_ref_api}"
    )
    print(f"{'='*78}")
    print(
        f"  High-priority clients:           {num_high_priority_clients}/{args.num_clients}"
    )
    print(f"  Duration:                        {duration:.2f}s")
    print(f"  High-priority makespan:          {high_priority_makespan:.2f}s")
    print(f"  Active decode wall time:         {active_decode_wall_time:.2f}s")
    print(f"  High-priority turns completed:   {successful_turns}")
    print(f"  High-priority completion tokens: {total_completion_tokens}")
    print(
        f"  High-priority out tok (wall):    {output_token_throughput_wall:.0f} tok/s"
    )
    print(
        f"  High-priority out tok (decode):  {output_token_throughput_active_decode:.0f} tok/s"
    )
    if server_decode_samples:
        print(
            f"  High-priority engine decode:     {server_decode_weighted_mean:.0f} tok/s "
            f"(weighted mean from meta_info)"
        )
        print(
            f"  High-priority engine p50/p90/p99:{server_decode_stats['p50']:.0f} / "
            f"{server_decode_stats['p90']:.0f} / {server_decode_stats['p99']:.0f} tok/s"
        )
    else:
        print(
            "  High-priority engine decode:     N/A (server did not return decode_throughput)"
        )
    print(f"  High-priority cache hit:         {summary['overall_cache_hit_rate']:.4f}")
    print(f"  High-priority turn>=1 cache hit: {subsequent_cache_hit:.4f}")
    print(
        f"  High-priority TTFT mean/p90/p99: {summary['ttft']['mean']:.3f} / "
        f"{summary['ttft']['p90']:.3f} / {summary['ttft']['p99']:.3f}s"
    )
    print(
        f"  High-priority Lat mean/p90/p99:  {summary['latency']['mean']:.3f} / "
        f"{summary['latency']['p90']:.3f} / {summary['latency']['p99']:.3f}s"
    )
    print(
        f"  High-priority turn0 TTFT p90/p99:{summary['first_turn_ttft']['p90']:.3f} / "
        f"{summary['first_turn_ttft']['p99']:.3f}s"
    )
    print(
        f"  High-priority turn>=1 TTFT p90/p99:"
        f"{summary['subsequent_ttft']['p90']:.3f} / "
        f"{summary['subsequent_ttft']['p99']:.3f}s"
    )

    print(f"\n  High-priority per-turn breakdown:")
    for turn_num in sorted(per_turn_metrics.keys()):
        m = per_turn_metrics[turn_num]
        avg_hit = np.mean(m["cache_hit_rate"]) if m["cache_hit_rate"] else 0
        avg_ttft = np.mean(m["ttft"]) if m["ttft"] else 0
        avg_out = np.mean(m["completion_tokens"]) if m["completion_tokens"] else 0
        count = len(m["ttft"])
        line = (
            f"    Turn {turn_num:2d}: n={count:4d}, avg_out={avg_out:7.1f}, "
            f"cache_hit={avg_hit:.4f}, ttft={avg_ttft:.3f}s"
        )
        if m["server_decode_throughput"]:
            line += (
                f", engine_decode={np.mean(m['server_decode_throughput']):.0f} tok/s"
            )
        print(line)

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
    parser.add_argument(
        "--num-clients",
        type=int,
        default=126,
        help="Number of rollout clients (not equal to actual in-flight batch).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=256,
        help="Deprecated: ignored. Effective max_parallel is forced to the number of high-priority clients (2/3 of num_clients).",
    )
    parser.add_argument(
        "--min-turns", type=int, default=5, help="Min turns per rollout"
    )
    parser.add_argument(
        "--max-turns", type=int, default=20, help="Max turns per rollout"
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=15360,
        help="Initial prompt length in tokens",
    )
    parser.add_argument(
        "--sub-question-length",
        type=int,
        default=1024,
        help="Sub-question length appended each turn",
    )
    parser.add_argument(
        "--sub-question-pool-size",
        type=int,
        default=512,
        help="Pool size of pre-generated sub-questions",
    )
    parser.add_argument(
        "--output-length", type=int, default=256, help="Max new tokens per turn"
    )
    parser.add_argument(
        "--interaction-delay-min",
        type=float,
        default=5,
        help="Min sleep (seconds) between turns simulating env interaction",
    )
    parser.add_argument(
        "--interaction-delay-max",
        type=float,
        default=10,
        help="Max sleep (seconds) between turns simulating env interaction",
    )
    parser.add_argument(
        "--high-priority-ratio",
        type=float,
        default=2 / 3,
        help="Deprecated: ignored. The benchmark always uses a fixed 2/3 high-priority and 1/3 low-priority split.",
    )
    parser.add_argument(
        "--high-priority-threshold",
        type=int,
        default=1,
        help="Priority >= threshold is high-priority",
    )
    parser.add_argument(
        "--priority-flip-prob",
        type=float,
        default=0.0,
        help="Probability of flipping priority via /update_ref between turns",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocab size for random token generation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--request-log-metrics",
        action="store_true",
        help="Ask SGLang to include per-request metrics such as decode_throughput in meta_info.",
    )
    parser.add_argument(
        "--log-file", type=str, default="bench_ref_aware_multiturn.jsonl"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag for this run (e.g., 'baseline' or 'ref-aware')",
    )
    parser.add_argument(
        "--rid-prefix",
        type=str,
        default="",
        help="Optional request ID prefix. Defaults to a unique per-run namespace.",
    )
    parser.add_argument(
        "--sweep-clients",
        type=str,
        default="",
        help="Comma-separated list of num_clients to sweep. Each value must be divisible by 3, e.g. '30,60,126,255'",
    )
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

        print(f"\n{'='*120}")
        print(f"  HIGH-PRIORITY SWEEP SUMMARY")
        print(f"{'='*120}")
        print(
            f"  {'clients':>8} {'hp_wall/s':>12} {'hp_decode/s':>14} {'hp_eng/s':>14} "
            f"{'hp_hit':>10} {'hp_ttft':>10} {'hp_lat':>10}"
        )
        print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
        for r in all_results:
            engine_dec = r.get("engine_decode_throughput_weighted_mean", 0.0)
            print(
                f"  {r['num_clients']:>8} "
                f"{r['output_token_throughput_wall']:>12.0f} "
                f"{r['output_token_throughput_active_decode']:>14.0f} "
                f"{engine_dec:>14.0f} "
                f"{r['overall_cache_hit_rate']:>10.4f} "
                f"{r['ttft']['p90']:>10.3f} "
                f"{r['latency']['p90']:>10.3f}"
            )
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
