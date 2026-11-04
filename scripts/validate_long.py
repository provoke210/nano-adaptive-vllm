"""Free-generation regression across batch schedules; no teacher forcing.

The FP32 head is an experimental ablation, applied identically to every policy.
This deterministic step-arrival test is correctness evidence, not a benchmark.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.adaptive_scheduler import make_scheduler
from nanovllm.engine.policy import SchedulingOptions
from nanovllm.utils.context import get_context
from benchmarks.workload import generate
from benchmarks.run import manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--head', choices=['model', 'fp32', 'fp32-mm'], default='model')
    p.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 43])
    p.add_argument('--eager', action='store_true')
    p.add_argument('--batch-invariant', action='store_true')
    a = p.parse_args()
    engine = LLM(a.model, dtype='float16', max_num_seqs=16,
                 max_num_batched_tokens=2048, kv_cache_block_cap=64,
                 enforce_eager=a.eager, batch_invariant=a.batch_invariant)
    if a.head in ('fp32', 'fp32-mm'):
        head = engine.model_runner.model.lm_head
        weight = head.weight.detach().float() if a.head == 'fp32' else None

        def high_head(x):
            ctx = get_context()
            if ctx.is_prefill:
                x = x[ctx.sample_indices].contiguous()
            if a.head == 'fp32-mm':
                return torch.mm(x, head.weight.t(), out_dtype=torch.float32)
            return torch.nn.functional.linear(x.float(), weight)

        head.forward = high_head
    report = dict(manifest=manifest(), gpu=torch.cuda.get_device_name(), arguments=vars(a),
                  evidence='cuda_free_generation', runs=[], all_equal=True)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        for seed in a.seeds:
            reference = None
            for mode, budget in [('native', 512), ('fixed', 128), ('fixed', 512),
                                 ('fixed', 2048), ('adaptive', 512)]:
                options = SchedulingOptions(mode=mode, prefill_budget=budget)
                engine.config.scheduling = options
                engine.scheduler = make_scheduler(engine.config)
                pending = list(generate(count=24, seed=seed))
                seqs = []
                step = 0
                while pending or not engine.is_finished():
                    if pending and (step % 8 == 0 or engine.is_finished()):
                        row = pending.pop(0)
                        seqs.append(engine.add_request(row['token_ids'], SamplingParams(
                            temperature=0, max_tokens=row['max_tokens'], ignore_eos=True)))
                    engine.step()
                    step += 1
                outputs = [s.completion_token_ids for s in seqs]
                if reference is None:
                    reference = outputs
                differences = []
                for i, (expected, actual) in enumerate(zip(reference, outputs)):
                    if expected != actual:
                        first = next(j for j, (x, y) in enumerate(zip(expected, actual)) if x != y)
                        differences.append(dict(request=i, token_index=first,
                                                expected=expected[first], actual=actual[first]))
                assert not engine.scheduler.block_manager.used_block_ids
                report['all_equal'] &= not differences
                report['runs'].append(dict(seed=seed, scheduling=asdict(options),
                    differences=differences, outputs=outputs,
                    preemptions=[s.preemptions for s in seqs]))
                dest.write_text(json.dumps(report, indent=2))
                print(seed, mode, budget, 'differences', differences, flush=True)
    finally:
        engine.exit()
    if not report['all_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
