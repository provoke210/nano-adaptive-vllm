"""Run paired traces with fixed-budget sweep and independent ablations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from benchmarks.workload import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results/matrix")
    p.add_argument("--backend", choices=["sim", "gpu"], default="sim")
    p.add_argument("--model")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    p.add_argument("--scenarios", nargs="+", choices=["short", "mixed", "burst", "pressure"],
                   default=["mixed", "burst"])
    p.add_argument("--blocks", type=int, default=32)
    p.add_argument("--rate", type=float, default=30)
    p.add_argument("--eager", action="store_true")
    a = p.parse_args()
    if a.backend == "gpu" and not a.model:
        p.error("--model required for GPU")
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    variants = [("native", "native", 512, "none", 0)]
    variants += [(f"fixed-{n}", "fixed", n, "none", 0) for n in [128, 256, 512, 1024, 2048]]
    variants += [("fixed-kv", "fixed", 512, "fixed", 2),
                 ("dynamic-kv-only", "fixed", 512, "adaptive", 0),
                 ("dynamic-budget-only", "adaptive", 512, "none", 0),
                 ("combined", "adaptive", 512, "adaptive", 0)]
    summaries = []
    for scenario in a.scenarios:
        for seed in a.seeds:
            trace = root / f"trace-{scenario}-{seed}.jsonl"
            trace.write_text("".join(json.dumps(r) + "\n" for r in generate(a.count, seed, scenario, a.rate)))
            for name, mode, budget, kv, reserve in variants:
                output = root / f"{scenario}-{seed}-{name}.json"
                command = [sys.executable, "-m", "benchmarks.run", "--backend", a.backend,
                           "--trace", str(trace), "--output", str(output), "--policy", mode,
                           "--prefill-budget", str(budget), "--kv-mode", kv,
                           "--reserve-blocks", str(reserve), "--blocks", str(a.blocks)]
                if a.model:
                    command += ["--model", a.model, "--dtype", a.dtype]
                if a.eager:
                    command += ["--eager"]
                proc = subprocess.run(command, capture_output=True, text=True)
                if proc.returncode:
                    raise RuntimeError(f"{name} failed:\n{proc.stdout}\n{proc.stderr}")
                report = json.loads(output.read_text())
                summaries.append(dict(scenario=scenario, seed=seed, variant=name,
                                      evidence=report["evidence"], **report["summary"]))
                print(f"{scenario}/{seed}/{name}: completed ({report['evidence']})", flush=True)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    columns = ["scenario", "seed", "variant", "goodput_requests_per_s", "ttft_p99_ms", "itl_p99_ms", "preemptions"]
    lines = ["# " + ("SIMULATION ONLY — not GPU evidence" if a.backend == "sim" else "CUDA measurements"),
             "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in summaries:
        lines.append("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
