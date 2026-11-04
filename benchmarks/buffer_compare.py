"""Same-process randomized A/B of Decode input reuse. No profiling in timed runs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from benchmarks.metrics import summarize
from benchmarks.run import manifest, replay
from benchmarks.workload import generate
from nanovllm import LLM, SamplingParams
from nanovllm.engine.adaptive_scheduler import make_scheduler
from nanovllm.engine.policy import SchedulingOptions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--suite', choices=['micro', 'e2e', 'profile'], required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 307])
    p.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 32, 64])
    p.add_argument('--contexts', type=int, nargs='+', default=[32, 1024])
    p.add_argument('--decode-steps', type=int, default=96)
    p.add_argument('--blocks', type=int, default=384)
    p.add_argument('--eager', action='store_true')
    a = p.parse_args()
    import torch
    dest = Path(a.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    micro = a.suite != 'e2e'
    engine = LLM(a.model, dtype='float16', reuse_decode_buffers=False, profile_stages=False,
                 max_num_seqs=max(a.batches) if micro else 16, max_num_batched_tokens=2048,
                 max_model_len=4096, kv_cache_block_cap=a.blocks, enforce_eager=a.eager,
                 scheduling=SchedulingOptions(mode='native'))
    metadata = manifest()
    metadata.update(gpu=torch.cuda.get_device_name(), model_type=engine.config.hf_config.model_type,
                    model_config_sha256=hashlib.sha256((Path(a.model)/'config.json').read_bytes()).hexdigest(),
                    kv_blocks=engine.config.num_kvcache_blocks)
    (dest/'manifest.json').write_text(json.dumps(dict(manifest=metadata, arguments=vars(a)), indent=2))
    assert engine.config.num_kvcache_blocks == a.blocks
    index, references = [], {}

    def reset(reuse, options):
        assert engine.is_finished()
        engine.config.reuse_decode_buffers = reuse
        engine.config.profile_stages = False
        engine.config.scheduling = options
        engine.scheduler = make_scheduler(engine.config)
        engine.model_runner.profiler.collect(wait=True)

    def save(name, payload):
        (dest/(name+'.json')).write_text(json.dumps(payload, indent=2, allow_nan=False))
        index.append(dict(file=name+'.json', reuse=payload['reuse'], key=payload['key'],
                          repeat=payload['repeat'], **payload['summary']))
        (dest/'index.json').write_text(json.dumps(index, indent=2))
        print(name, json.dumps(payload['summary']), 'equal=True', flush=True)

    try:
        cases = [(b, c) for c in a.contexts for b in a.batches] if micro else [(s, seed) for seed in a.seeds for s in ['mixed', 'burst', 'pressure']]
        for case_id, case in enumerate(cases):
            key = '-'.join(map(str, case))
            repeats = 1 if a.suite == 'profile' else a.repeats
            for repeat in range(repeats):
                variants = [False, True]
                random.Random(1009 + case_id*17 + repeat).shuffle(variants)
                for reuse in variants:
                    options = SchedulingOptions(mode='native') if micro else SchedulingOptions(
                        mode='fixed', prefill_budget=2048, admission_wait_ms=100,
                        admission_target_ms=800, admission_max_slots=4)
                    reset(reuse, options)
                    if micro:
                        batch, context = case
                        # Native prompt-first schedule fills all prompts before any Decode.
                        rng = random.Random(1009 + case_id)
                        prompts = [[rng.randrange(10, min(1000, engine.config.hf_config.vocab_size)) for _ in range(context)] for _ in range(batch)]
                        params = SamplingParams(temperature=0, max_tokens=a.decode_steps+33, ignore_eos=True)
                        seqs = [engine.add_request(prompt, params) for prompt in prompts]
                        while engine.scheduler.waiting:
                            engine.step()
                        for _ in range(16):
                            engine.step()
                        assert len(engine.scheduler.running) == batch
                        before = [len(s.completion_token_ids) for s in seqs]
                        engine.config.profile_stages = a.suite == 'profile'
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(a.decode_steps):
                            engine.step()
                        torch.cuda.synchronize()
                        duration = time.perf_counter()-start
                        outputs = [s.completion_token_ids[begin:] for s, begin in zip(seqs, before)]
                        assert all(len(row) == a.decode_steps for row in outputs)
                        summary = dict(duration_s=duration, output_tokens_per_s=batch*a.decode_steps/duration,
                                       mean_decode_step_ms=duration*1000/a.decode_steps)
                        profiles = engine.model_runner.profiler.collect(wait=True)
                        engine.config.profile_stages = False
                        while not engine.is_finished():
                            engine.step()
                        trace_hash = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
                        records = None
                    else:
                        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
                        engine.add_request([11]*512, params)
                        engine.step()
                        engine.add_request([12]*256, params)
                        while not engine.is_finished():
                            engine.step()
                        engine.scheduler = make_scheduler(engine.config)
                        scenario, seed = case
                        trace = generate(24, seed, scenario, 30)
                        trace_hash = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
                        records, steps, duration = replay(engine, trace, time.perf_counter, time.sleep)
                        outputs = [r['output_ids'] for r in records]
                        summary = summarize(records, duration, 1000, 50)
                        profiles = []
                    if key not in references:
                        references[key] = outputs
                    if outputs != references[key]:
                        (dest/'output-mismatch.json').write_text(json.dumps(dict(key=key, reuse=reuse, outputs=outputs, reference=references[key])))
                        raise AssertionError(f'output mismatch: {key}')
                    payload = dict(manifest=metadata, key=key, repeat=repeat, reuse=reuse, suite=a.suite,
                        profile_enabled=a.suite == 'profile', scheduling=asdict(options), trace_sha256=trace_hash,
                        summary=summary, outputs=outputs, requests=records, profiles=profiles)
                    save(f'{key}-r{repeat}-reuse{int(reuse)}', payload)
    finally:
        engine.exit()


if __name__ == '__main__':
    main()
