"""Teacher-force saved long-trace histories, recording only suspect positions.

Diagnostic timings include CPU transfers and must never be used as benchmarks.
Run from repository root with PYTHONPATH=.
"""
import argparse
import json
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.adaptive_scheduler import make_scheduler
from nanovllm.engine.policy import SchedulingOptions
from nanovllm.utils.context import get_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fp32-head', action='store_true')
    p.add_argument('--batch-invariant', action='store_true')
    p.add_argument('--hf-reference', action='store_true', help='full-context FP32 Transformers reference at known divergences')
    a = p.parse_args()
    fixture = json.loads(Path(a.fixture).read_text())
    engine = LLM(a.model, dtype='float16', max_num_seqs=16,
                 max_num_batched_tokens=2048, kv_cache_block_cap=64,
                 enforce_eager=False, batch_invariant=a.batch_invariant)
    runner = engine.model_runner
    head = runner.model.lm_head
    original_head, original_run = head.forward, runner.run
    high_weight = head.weight.detach().float()
    captured = {}
    identities = {}
    rows = []
    selected = []
    reference_logits = {}

    def capture_head(x):
        low = original_head(x)
        ctx = get_context()
        if ctx.is_prefill:
            x = x[ctx.sample_indices].contiguous()
        # Deliberately diagnostic FP32 matmul; no hidden performance claim.
        high = torch.nn.functional.linear(x.float(), high_weight)
        captured['low'] = low.detach().float().cpu()
        captured['high'] = high.detach().cpu()
        return high if a.fp32_head else low

    def compare(seqs, is_prefill):
        values = original_run(seqs, is_prefill)
        row_index = 0
        for i, seq in enumerate(seqs):
            if values[i] is None:
                continue
            request = identities[seq.seq_id]
            token_index = seq.num_completion_tokens
            expected = fixture[request]['output_ids'][token_index]
            if request in (10, 11) and token_index in range(0, 100):
                low, high = captured['low'][row_index], captured['high'][row_index]
                top = low.topk(5)
                high_top = high.topk(5)
                key = (request, token_index)
                if selected[0].startswith('native'):
                    reference_logits[key] = low.clone()
                delta = low - reference_logits[key]
                rows.append(dict(policy=selected[0], request=request, token_index=token_index,
                    expected=expected, actual=values[i], is_mixed=is_prefill,
                    max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                    low_top=list(zip(top.indices.tolist(), top.values.tolist())),
                    high_top=list(zip(high_top.indices.tolist(), high_top.values.tolist())),
                    low_margin=float(top.values[0]-top.values[1]),
                    high_margin=float(high_top.values[0]-high_top.values[1])))
            values[i] = expected
            row_index += 1
        return values

    head.forward, runner.run = capture_head, compare
    try:
        for mode, budget in [('native', 512), ('fixed', 128), ('fixed', 512), ('fixed', 2048)]:
            selected[:] = [f'{mode}-{budget}']
            engine.config.scheduling = SchedulingOptions(mode=mode, prefill_budget=budget)
            engine.scheduler = make_scheduler(engine.config)
            identities.clear()
            # Deterministic staggered admission independent of diagnostic overhead.
            pending = list(enumerate(fixture))
            step = 0
            while pending or not engine.is_finished():
                if pending and (step % 8 == 0 or engine.is_finished()):
                    request, item = pending.pop(0)
                    seq = engine.add_request(item['token_ids'], SamplingParams(
                        temperature=0, max_tokens=len(item['output_ids']), ignore_eos=True))
                    identities[seq.seq_id] = request
                engine.step()
                step += 1
            print(selected[0], 'done', flush=True)
    finally:
        engine.exit()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(rows, indent=2))
    print(json.dumps([r for r in rows if (r['request'], r['token_index']) in [(10, 89), (11, 19)]], indent=2))
    if a.hf_reference:
        from transformers import AutoModelForCausalLM
        # Match the experimental weight rounding, then perform the reference
        # computation in FP32 without a paged cache or mixed/chunked batching.
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
                                                   attn_implementation='eager').float().cuda().eval()
        references = []
        with torch.inference_mode():
            for request, index in [(10, 89), (11, 19)]:
                item = fixture[request]
                tokens = item['token_ids'] + item['output_ids'][:index]
                logits = model(torch.tensor([tokens], device='cuda'),
                               use_cache=False, logits_to_keep=1).logits[0, -1].float()
                top = logits.topk(5)
                references.append(dict(request=request, token_index=index,
                    token=int(logits.argmax()), top=list(zip(top.indices.tolist(), top.values.tolist()))))
        Path(a.output).with_suffix('.hf.json').write_text(json.dumps(references, indent=2))
        print('HF_REFERENCE', json.dumps(references), flush=True)


if __name__ == '__main__':
    main()
