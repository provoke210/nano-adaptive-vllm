"""Replay open-loop arrivals between engine steps; save raw data + manifest."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from nanovllm.engine.policy import SchedulingOptions
from nanovllm.sampling_params import SamplingParams
from benchmarks.metrics import summarize
from benchmarks.workload import load_trace


def replay(engine, trace, clock, sleep, max_steps=1_000_000):
    start = clock()
    index = 0
    sequences = []
    steps = []
    while index < len(trace) or not engine.is_finished():
        if len(steps) >= max_steps:
            raise RuntimeError("step limit exceeded; possible scheduler livelock")
        now_ms = (clock() - start) * 1000
        while index < len(trace) and trace[index]["arrival_ms"] <= now_ms + 1e-7:
            row = trace[index]
            seq = engine.add_request(row["token_ids"], SamplingParams(
                temperature=0, max_tokens=row["max_tokens"], ignore_eos=True))
            # Include submission delayed by a running step in queueing latency.
            seq.arrival_time = start + row["arrival_ms"] / 1000
            sequences.append(seq)
            index += 1
        if engine.is_finished():
            if index < len(trace):
                sleep(max(0, (trace[index]["arrival_ms"] - now_ms) / 1000))
            continue
        engine.step()
        steps.append(dict(time_ms=(clock() - start) * 1000, **engine.last_step))
    duration = clock() - start
    records = []
    for row, seq in zip(trace, sequences):
        records.append(dict(id=row["id"], arrival_ms=row["arrival_ms"],
                            prompt_tokens=len(row["token_ids"]),
                            token_times_ms=[(t - start) * 1000 for t in seq.token_times],
                            output_ids=seq.completion_token_ids,
                            preemptions=seq.preemptions, recomputed_tokens=seq.recomputed_tokens))
    return records, steps, duration


def manifest():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    packages = {}
    for name in ["numpy", "xxhash", "torch", "triton", "transformers", "flash-attn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        revision, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    sources = [root / "pyproject.toml"]
    sources += sorted((root / "nanovllm").rglob("*.py"))
    sources += sorted((root / "benchmarks").rglob("*.py"))
    for path in sources:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=packages, revision=revision, dirty=dirty, source_sha256=digest.hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["sim", "gpu"], default="sim")
    p.add_argument("--model")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    p.add_argument('--batch-invariant', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--controller', choices=['aimd', 'cost'], default='aimd')
    p.add_argument('--coalesce-ms', type=float, default=0)
    p.add_argument("--policy", choices=["native", "fixed", "adaptive"], default="adaptive")
    p.add_argument("--prefill-budget", type=int, default=512)
    p.add_argument("--min-prefill-budget", type=int, default=32)
    p.add_argument("--max-prefill-budget", type=int, default=2048)
    p.add_argument("--target-step-ms", type=float, default=20)
    p.add_argument("--aging-ms", type=float, default=500)
    p.add_argument("--kv-mode", choices=["none", "fixed", "adaptive"], default="none")
    p.add_argument("--reserve-blocks", type=int, default=0)
    p.add_argument("--decode-lookahead", type=int, default=16)
    p.add_argument("--blocks", type=int, default=64, help="same physical KV block cap for every policy")
    p.add_argument("--max-seqs", type=int, default=16)
    p.add_argument("--token-budget", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--ttft-slo-ms", type=float, default=1000)
    p.add_argument("--tpot-slo-ms", type=float, default=50)
    p.add_argument("--eager", action="store_true")
    a = p.parse_args()
    if min(a.blocks, a.max_seqs, a.token_budget, a.max_model_len) < 1 or a.token_budget < a.max_seqs:
        p.error("positive capacities and token-budget >= max-seqs required")
    options = SchedulingOptions(mode=a.policy, prefill_budget=a.prefill_budget,
        min_prefill_budget=a.min_prefill_budget, max_prefill_budget=a.max_prefill_budget,
        target_step_ms=a.target_step_ms, aging_ms=a.aging_ms, kv_mode=a.kv_mode,
        reserve_blocks=a.reserve_blocks, decode_lookahead=a.decode_lookahead)
    options = replace(options, controller=a.controller, coalesce_ms=a.coalesce_ms)
    trace = load_trace(a.trace)
    metadata = manifest()
    if a.backend == "sim":
        from benchmarks.simulator import SimEngine
        engine = SimEngine(options, blocks=a.blocks, token_budget=a.token_budget,
                           max_seqs=a.max_seqs, max_model_len=a.max_model_len)
        clock, sleep = engine.clock, engine.clock.sleep
    else:
        if not a.model:
            p.error("--model is required for GPU")
        import torch
        if not torch.cuda.is_available():
            p.error("GPU backend requires a CUDA GPU; use --backend sim on Mac")
        from nanovllm import LLM
        from nanovllm.engine.adaptive_scheduler import make_scheduler
        engine = LLM(a.model, scheduling=options, kv_cache_block_cap=a.blocks,
                     dtype=a.dtype,
                     batch_invariant=a.batch_invariant,
                     max_num_seqs=a.max_seqs, max_num_batched_tokens=a.token_budget,
                     max_model_len=a.max_model_len, enforce_eager=a.eager)
        try:
            if engine.config.num_kvcache_blocks != a.blocks:
                raise ValueError("requested KV block cap does not fit; lower --blocks for ALL policies")
            if any(max(row["token_ids"]) >= engine.config.hf_config.vocab_size for row in trace):
                raise ValueError("trace contains token IDs outside model vocabulary")
            # Exercise prefill/partial/mixed/decode paths before timed replay.
            engine.add_request([11] * min(512, a.max_model_len - 8), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
            engine.step()
            engine.add_request([12] * min(256, a.max_model_len - 8), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
            while not engine.is_finished():
                engine.step()
            engine.scheduler = make_scheduler(engine.config)
            metadata.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                            actual_dtype=str(engine.model_runner.dtype),
                            batch_invariant=engine.config.batch_invariant,
                            actual_kv_blocks=engine.config.num_kvcache_blocks,
                            gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
                            model_config_sha256=hashlib.sha256((Path(a.model) / "config.json").read_bytes()).hexdigest())
        except BaseException:
            engine.exit()
            raise
        clock, sleep = time.perf_counter, time.sleep
    try:
        records, steps, duration = replay(engine, trace, clock, sleep)
        report = dict(schema_version=1, evidence="synthetic_simulation" if a.backend == "sim" else "cuda_measurement",
                      manifest=metadata, arguments=vars(a), scheduling=asdict(options),
                      trace_sha256=hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(),
                      summary=summarize(records, duration, a.ttft_slo_ms, a.tpot_slo_ms),
                      requests=records, steps=steps)
        dest = Path(a.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(evidence=report["evidence"], **report["summary"]), indent=2))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
