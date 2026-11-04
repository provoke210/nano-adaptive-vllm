"""Randomized repeated GPU comparison on a shared numerical backend.

One model load, a fresh scheduler/cache manager per replay, identical per-policy
warmup, and raw token/step records. Initialization and warmup are excluded.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from benchmarks.run import manifest, replay
from benchmarks.metrics import summarize
from benchmarks.workload import generate
from nanovllm.engine.policy import SchedulingOptions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[7])
    p.add_argument('--scenarios', nargs='+', default=['mixed', 'burst'])
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--rate', type=float, default=15)
    p.add_argument('--target-step-ms', type=float, default=40)
    p.add_argument('--initial-budget', type=int, default=512)
    p.add_argument('--decode-lookahead', type=int, default=128)
    p.add_argument('--coalesce-ms', type=float, default=24)
    p.add_argument('--admission-wait-ms', type=float, default=100)
    p.add_argument('--admission-target-ms', type=float, default=800)
    p.add_argument('--admission-max-slots', type=int, default=4)
    p.add_argument('--blocks', type=int, default=64)
    p.add_argument('--max-seqs', type=int, default=16)
    p.add_argument('--token-budget', type=int, default=2048)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--ttft-slo-ms', type=float, default=1000)
    p.add_argument('--tpot-slo-ms', type=float, default=50)
    p.add_argument('--variants', nargs='+')
    a = p.parse_args()
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.adaptive_scheduler import make_scheduler
    dest = Path(a.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    variants = {'native': SchedulingOptions(mode='native')}
    for budget in [128, 256, 512, 1024, 2048]:
        variants[f'fixed-{budget}'] = SchedulingOptions(mode='fixed', prefill_budget=budget)
    variants['aimd'] = SchedulingOptions(mode='adaptive')
    variants['cost'] = SchedulingOptions(mode='adaptive', controller='cost', target_step_ms=a.target_step_ms,
                                         prefill_budget=a.initial_budget)
    variants['cost-coalesce'] = SchedulingOptions(mode='adaptive', controller='cost',
        target_step_ms=a.target_step_ms, coalesce_ms=a.coalesce_ms, prefill_budget=a.initial_budget)
    variants['cost-kv'] = SchedulingOptions(mode='adaptive', controller='cost', target_step_ms=a.target_step_ms,
        prefill_budget=a.initial_budget, kv_mode='adaptive', decode_lookahead=a.decode_lookahead)
    variants['fixed-2048-kv'] = SchedulingOptions(mode='fixed', prefill_budget=2048,
        kv_mode='adaptive', decode_lookahead=a.decode_lookahead)
    variants['fixed-2048-coalesce'] = SchedulingOptions(mode='fixed', controller='cost', prefill_budget=2048,
        coalesce_ms=a.coalesce_ms)
    for budget in [128, 256, 512, 1024, 2048]:
        variants[f'fixed-{budget}-admit'] = SchedulingOptions(mode='fixed', prefill_budget=budget,
            admission_wait_ms=a.admission_wait_ms)
    variants['cost-admit'] = SchedulingOptions(mode='adaptive', controller='cost',
        target_step_ms=a.target_step_ms, prefill_budget=a.initial_budget,
        admission_wait_ms=a.admission_wait_ms)
    variants['cost-admit-kv'] = SchedulingOptions(mode='adaptive', controller='cost',
        target_step_ms=a.target_step_ms, prefill_budget=a.initial_budget,
        admission_wait_ms=a.admission_wait_ms, kv_mode='adaptive', decode_lookahead=a.decode_lookahead)
    for budget in [128, 256, 512, 1024, 2048]:
        variants[f'fixed-{budget}-aware'] = SchedulingOptions(mode='fixed', prefill_budget=budget,
            admission_wait_ms=a.admission_wait_ms, admission_target_ms=a.admission_target_ms,
            admission_max_slots=a.admission_max_slots)
    variants['cost-aware'] = SchedulingOptions(mode='adaptive', controller='cost',
        target_step_ms=a.target_step_ms, prefill_budget=a.initial_budget,
        admission_wait_ms=a.admission_wait_ms, admission_target_ms=a.admission_target_ms,
        admission_max_slots=a.admission_max_slots)
    if a.variants:
        variants = {key: variants[key] for key in a.variants}
    engine = LLM(a.model, dtype='float16', batch_invariant=True,
                 max_num_seqs=a.max_seqs, max_num_batched_tokens=a.token_budget,
                 max_model_len=a.max_model_len, kv_cache_block_cap=a.blocks)
    metadata = manifest()
    metadata.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        actual_dtype=str(engine.model_runner.dtype), actual_kv_blocks=engine.config.num_kvcache_blocks,
        gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
        model_config_sha256=hashlib.sha256((Path(a.model)/'config.json').read_bytes()).hexdigest(),
        batch_invariant=True)
    assert engine.config.num_kvcache_blocks == a.blocks
    index = []
    try:
        for seed in a.seeds:
            for scenario in a.scenarios:
                trace = generate(a.count, seed, scenario, a.rate)
                trace_data = ''.join(json.dumps(row)+'\n' for row in trace)
                (dest/f'trace-{scenario}-{seed}.jsonl').write_text(trace_data)
                reference = None
                for repeat in range(a.repeats):
                    order = list(variants)
                    random.Random(seed * 100 + repeat).shuffle(order)
                    for name in order:
                        options = variants[name]
                        engine.config.scheduling = options
                        engine.scheduler = make_scheduler(engine.config)
                        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
                        engine.add_request([11]*512, params)
                        engine.step()
                        engine.add_request([12]*256, params)
                        while not engine.is_finished():
                            engine.step()
                        engine.scheduler = make_scheduler(engine.config)
                        records, steps, duration = replay(engine, trace, time.perf_counter, time.sleep)
                        outputs = [r['output_ids'] for r in records]
                        if reference is None:
                            reference = outputs
                        equal = outputs == reference
                        report = dict(schema_version=1, evidence='cuda_measurement',
                            manifest=metadata, arguments=dict(vars(a), backend='gpu', dtype='float16', eager=False),
                            scheduling=asdict(options), variant=name, repeat=repeat,
                            trace_sha256=hashlib.sha256(trace_data.encode()).hexdigest(),
                            summary=summarize(records, duration, a.ttft_slo_ms, a.tpot_slo_ms),
                            outputs_equal_to_first_run=equal, requests=records, steps=steps)
                        filename = f'{scenario}-{seed}-{name}-r{repeat}.json'
                        (dest/filename).write_text(json.dumps(report, indent=2, allow_nan=False))
                        index.append(dict(file=filename, seed=seed, scenario=scenario,
                                          variant=name, repeat=repeat, equal=equal, **report['summary']))
                        (dest/'index.json').write_text(json.dumps(index, indent=2))
                        print(filename, json.dumps(report['summary']), 'equal', equal, flush=True)
                        if not equal:
                            raise RuntimeError('Cross-schedule output divergence; stop performance comparison')
    finally:
        engine.exit()


if __name__ == '__main__':
    main()
