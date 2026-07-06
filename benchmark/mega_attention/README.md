# megaAttention × SGLang — prefill 性能对比(可复现)

对比 `SGLANG_USE_MEGA_ATTENTION` 开/关时,Qwen3-30B-A3B（TP4，8×H200，torch 2.12 源码栈）
的 **prefill 前向延迟**。megaAttention 只替换 prefill 的 `attention + o_proj + TP AllReduce`，
decode 走 fa3 fallback 不变，因此只比 prefill。

环境搭建见仓库根 memory / `开发文档.md`（源码编译 sgl_kernel、flashinfer、tvm-ffi 0.1.12、
卸载 tilelang）。mega 路径需 `SGLANG_USE_MEGA_ATTENTION=1` +
`PYTHONPATH=/root/sglang/3rdparty/megaAttention/src`（用 submodule 那份新 API）。

## 脚本

### 1. `bench_prefill_varlen.py` —— 精确 varlen 形状（推荐，最可靠）
用 megaAttention benchmark 的 ragged chunk 形状（`_CHUNK_SMALL`/`_CHUNK_BIG`）在 SGLang 上跑
单次 prefill 前向，`torch.cuda.synchronize` + wall clock、先 warmup 同 shape（摊掉 flashinfer
per-shape JIT 与 mega cute.compile）、取 warm 中位数。复用 `sglang.bench_one_batch` 的
`load_model` / `prepare_synthetic_inputs_for_latency_test(custom_inputs=...)` / `_TorchBenchRunner`。

```bash
cd /root/sglang
# baseline (mega off)
python benchmark/mega_attention/bench_prefill_varlen.py \
  --model-path /models/Qwen3-30B-A3B --tp 4 --page-size 128 --attention-backend fa3 \
  --chunked-prefill-size 32768 --disable-cuda-graph --dtype bfloat16 --max-running-requests 256
# mega on
SGLANG_USE_MEGA_ATTENTION=1 PYTHONPATH=/root/sglang/3rdparty/megaAttention/src \
python benchmark/mega_attention/bench_prefill_varlen.py \
  --model-path /models/Qwen3-30B-A3B --tp 4 --page-size 128 --attention-backend fa3 \
  --chunked-prefill-size 32768 --disable-cuda-graph --dtype bfloat16 --max-running-requests 256
```
`--chunked-prefill-size 32768` 须 > 最大总 token（23.2K）以保证单次前向不被切；
`--max-running-requests 256` 定 mega workspace 容量 + warmup/计时迭代的 req 池。

### 2. 均匀 shape（用官方 `sglang.bench_one_batch`，单 combo 单独跑保证 warmup 干净）
```bash
for BS_IL in "1 2048" "1 4096" "8 1024"; do set -- $BS_IL
  [SGLANG_USE_MEGA_ATTENTION=1 PYTHONPATH=...] python -m sglang.bench_one_batch \
    --model-path /models/Qwen3-30B-A3B --tp 4 --page-size 128 --attention-backend fa3 \
    --chunked-prefill-size 8192 --disable-cuda-graph --dtype bfloat16 --max-running-requests 48 \
    --batch-size $1 --input-len $2 --output-len 1   # 取 "Benchmark" 段的 Prefill. latency
done
```
注意：多 combo 一次跑会因 bench_one_batch 只 warmup 首个 combo + flashinfer per-shape JIT
导致非首 combo 数字被污染，务必**一个 combo 一次调用**。

## 结果（warm 中位数,单次 prefill 前向,含整层 MoE,TP4,launch-config 已按首个 shape 调优 w_fa=8）

`bench_prefill_varlen.py`(最可靠):

| 场景 | q | k | baseline | mega | 比值 |
|---|---|---|---|---|---|
| FULL small (q==k) | 13760 | 13760 | 100.87 ms | 103.44 ms | 0.98× |
| FULL big (q==k) | 23168 | 23168 | 164.85 ms | 169.23 ms | 0.97× |
| CHUNKED k=2q (复用 KV) | 13760 | 27520 | 119.38 ms | 128.82 ms | 0.93× |
| CHUNKED k=4q (复用 KV) | 6880 | 27520 | 66.06 ms | 72.11 ms | 0.92× |

（更早的均匀 `bench_one_batch` 单 combo 数字方差较大,以上 varlen warm-median 为准。）

## 结论与注意

- 功能正确（mega engaged=4 真触发,greedy 与 fa3 基线逐字一致,见 memory）。
- **端到端 prefill 未跑赢 baseline,反而一致慢 2–9%**(full 2–3%、chunked 7–9%),即使接了
  `choose_launch_config`(选 w_fa=8,相比默认 w_fa=4 数字基本不变)。
- 原因:(1) MoE experts 占整层大头,mega 只加速 attention 段 → 被稀释;(2) baseline 的
  fa3 + cuBLAS + flashinfer fused-allreduce 已高度优化;(3) chunked prefill(小 q / 大 KV 前缀)下
  o_proj/AR 工作量(随 q)相对 attention(随 q×k)很小 → mega 融合内核的重叠收益缩水,退化更大;
  (4) compile-once 只按首个 shape 选一次 config,两桶配置差别有限。
- megaAttention 单卡 benchmark 的 1.1–1.5× 是 **attention 段单独** 的比值;集成到 MoE 模型后未
  转化成端到端净收益。
- **可能的深挖方向**:torch profiler 拆出 attention 段单独比值(定位 mega 在整层的占比与真实
  加速/退化);paged-KV(chunked)路径的 mega kernel 调参;或该方案更适合 attention 占比更高的
  dense 模型。
