"""Teacher-forced diagnostic: compare policies at identical token histories."""
import json
import os
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.policy import SchedulingOptions

model = os.environ.get('NANO_MODEL', 'models/Qwen3-0.6B')
prompts = ['Explain KV caching in one sentence.', 'The capital of France is', 'Explain attention. ' * 90]
reference = {}
report = []
for mode in ['native', 'fixed', 'adaptive']:
    engine = LLM(model, scheduling=SchedulingOptions(mode=mode, prefill_budget=32),
                 batch_invariant=False,
                 max_num_seqs=8, max_num_batched_tokens=256, max_model_len=4096,
                 enforce_eager=True, kv_cache_block_cap=64)
    runner = engine.model_runner
    original_model, original_run = runner.run_model, runner.run
    capture = []
    identities = {}

    def capture_model(*args):
        logits = original_model(*args)
        capture[:] = [logits.detach().float().cpu()]
        return logits

    def compare_run(seqs, is_prefill):
        sampled = original_run(seqs, is_prefill)
        row = 0
        for i, seq in enumerate(seqs):
            if sampled[i] is None:
                continue
            key = (identities[seq.seq_id], seq.num_completion_tokens)
            logits = capture[0][row]
            row += 1
            if mode == 'native':
                reference[key] = (logits, sampled[i])
            else:
                ref, expected = reference[key]
                delta = logits - ref
                top = ref.topk(2)
                report.append(dict(mode=mode, request=key[0], token_index=key[1],
                    reference_token=expected, actual_token=sampled[i],
                    max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                    relative_l2=float(delta.norm()/ref.norm()),
                    reference_margin=float(top.values[0]-top.values[1]),
                    reference_top5=list(zip(ref.topk(5).indices.tolist(),ref.topk(5).values.tolist())),
                    actual_top5=list(zip(logits.topk(5).indices.tolist(),logits.topk(5).values.tolist()))))
                sampled[i] = expected  # Hold the next history identical to native.
        return sampled

    runner.run_model, runner.run = capture_model, compare_run
    params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    try:
        first = engine.add_request(prompts[0], params)
        identities[first.seq_id] = 0
        for _ in range(3): engine.step()
        for i, prompt in enumerate(prompts[1:], 1):
            seq = engine.add_request(prompt, params)
            identities[seq.seq_id] = i
        while not engine.is_finished(): engine.step()
    finally:
        engine.exit()
dest = Path('results/logit-diagnostic.json')
dest.write_text(json.dumps(report, indent=2))
print(json.dumps({'comparisons': len(report),
    'argmax_mismatches': sum(r['actual_token'] != r['reference_token'] for r in report),
    'max_abs': max(r['max_abs'] for r in report),
    'max_relative_l2': max(r['relative_l2'] for r in report)}, indent=2))
