# nano-adaptive-vllm

Latency-aware mixed prefill/decode scheduling and KV-pressure admission on a fork of
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) (single GPU, dense Qwen3 and Llama).

**Question:** when long and short requests arrive together or in bursts, a fixed prefill
budget trades off time-to-first-token, inter-token latency and KV pressure. Can admission
driven by measured latency and queue state raise *goodput* (requests that meet a latency SLO
per second)?

## What this fork adds

Upstream nano-vLLM already provides the paged KV cache, prefix cache, CUDA Graph decode,
chunked prefill and tensor-parallel layers. This fork adds:

| Component | Files |
|---|---|
| Decode-first mixed scheduling with one varlen forward per step; fixed / adaptive prefill budget | `engine/adaptive_scheduler.py`, `engine/policy.py` |
| Online step-cost model (fixed overhead + per-token cost) replacing an AIMD controller | `engine/cost_model.py` |
| Prefill-slot admission driven by queue wait and predicted slot release, with decode rotation under the same hard limits | `engine/adaptive_scheduler.py` |
| KV-pressure admission that reserves near-term block growth for running requests | `engine/adaptive_scheduler.py`, `engine/block_manager.py` |
| Batch-invariant Triton linear and paged attention (fixed tiles, no autotune / split-K) with FP32 logits | `layers/invariant.py` |
| Packed pinned H2D decode metadata bound directly to CUDA Graph inputs; CPU / CUDA-event stage profiling | `engine/input_buffers.py`, `engine/profiling.py` |
| Dense Llama support (default / linear / llama3 RoPE), model registry | `models/llama.py`, `models/registry.py`, `models/rope_config.py` |
| Trace replay, fixed-budget sweeps, ablations, randomized-order GPU A/B with output-equality gate | `benchmarks/` |

Dynamic scheduling itself is not claimed as a new idea; see [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Results

RTX 3090 24GB, FP16, PyTorch 2.8 + Triton 3.4. Synthetic token traces (no HTTP/tokenizer),
`max_num_seqs=16`, 2048-token step cap. Goodput = completed requests with TTFT ≤ 1000 ms and
mean TPOT ≤ 50 ms, per second. Parameters were tuned on seed 7 only, frozen, then evaluated on
held-out seeds 101 / 307 (3 repeats each, medians). All policies share the same math backend.

**1. Output consistency first.** Different schedules produced different greedy outputs on long
traces. The first divergence was traced to batch-shape-dependent FP16 reductions and logits
rounding (an FP32 output projection alone did not fix it). With fixed-reduction Triton linear /
attention and FP32 logits, 5 policies × 3 seeds — 360 request runs, 34,370 free-running tokens —
were token-identical. All later comparisons required identical outputs.

**2. Admission, not dynamic budgeting, is what helps** — long-output *pressure* workload, goodput (req/s):

| KV pool | seed | best plain fixed budget | fixed 2048 + predicted admission | change |
|---|---|---:|---:|---:|
| 64 blocks | 101 | 3.348 | 4.229 | +26.3% |
| 64 blocks | 307 | 3.163 | 4.800 | +51.7% |
| 128 blocks | 101 | 3.347 | 5.448 | +62.8% |
| 128 blocks | 307 | 3.158 | 5.582 | +76.7% |

- Mixed and burst workloads changed by < 1% (treated as flat).
- With 64 blocks, the gain is SLO attainment, not raw throughput: on seed 101 token/s fell ~8%,
  ITL P99 rose 42.9 → 59.4 ms and preemptions rose 0 → 6. With 128 blocks token/s also rose 8.5–17.8%.
- The online cost model did not beat a fixed budget with the same admission; a simple wait
  threshold was about as good as completion-time prediction. Both are kept as opt-in switches.

**3. Executor overhead.** Packing decode metadata into one pinned H2D copy and binding it to CUDA
Graph inputs: end-to-end +0.98–3.52% (Qwen3-0.6B), +0.90–2.01% (Llama-3.2-1B), +0.45–0.70%
(Qwen3-4B), outputs identical. Reusing separate tensors alone gave no stable gain.

Full tables, failed iterations and caveats: [docs/GPU_ROUND2.md](docs/GPU_ROUND2.md),
[docs/NUMERICAL_CORRECTNESS.md](docs/NUMERICAL_CORRECTNESS.md),
[docs/BUFFER_GPU_VALIDATION.md](docs/BUFFER_GPU_VALIDATION.md),
[docs/LLAMA_GPU_VALIDATION.md](docs/LLAMA_GPU_VALIDATION.md),
[docs/QWEN4B_GPU_VALIDATION.md](docs/QWEN4B_GPU_VALIDATION.md).
Raw per-request JSON (~580 MB) is kept out of the repo; summary statistics are in
[docs/ROUND2_RESULTS.json](docs/ROUND2_RESULTS.json).

## Usage

```python
from nanovllm import LLM, SamplingParams
from nanovllm.engine.policy import SchedulingOptions

llm = LLM("models/Qwen3-0.6B", dtype="float16", batch_invariant=True,
          max_num_seqs=16, max_num_batched_tokens=2048, kv_cache_block_cap=64,
          scheduling=SchedulingOptions(mode="fixed", prefill_budget=2048,
                                       admission_wait_ms=100, admission_target_ms=800,
                                       admission_max_slots=4))
outputs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=64))
```

CPU-only tests and the scheduling simulator run without a GPU:

```bash
pip install -e '.[test]' && pytest -q
python -m benchmarks.workload --scenario burst --count 60 --seed 7 --output results/burst.jsonl
python -m benchmarks.run --backend sim --trace results/burst.jsonl \
  --policy adaptive --kv-mode adaptive --blocks 32 --output results/adaptive.json
```

GPU setup and experiment protocol: [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md),
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Limitations

Single GPU, models up to 4B, synthetic traces with known `max_tokens`, `max_num_seqs=16`, and a
deliberately small KV pool. The fixed-reduction backend is slower than FlashAttention / cuBLAS;
its cost has not been measured separately, so absolute numbers should not be compared with
upstream nano-vLLM or vLLM. With the scheduling parameters tuned on 0.6B, Qwen3-4B met the SLO for
only 0–45.8% of requests.

## License

MIT, same as upstream. Original nano-vLLM by Xingkai Yu; see [LICENSE](LICENSE).
