"""Independent FP32 HF full-history reference for a pretrained CUDA checkpoint.

Diagnostic CPU transfers and teacher forcing are excluded from performance runs.
FP32 reference uses the same FP16-rounded weights as the tested engine.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.engine.policy import SchedulingOptions
from benchmarks.run import manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    texts = ['What is 2 + 2? Answer briefly.', 'Explain KV caching in one sentence.',
             'What is the capital of France?']
    cases = [dict(name=f'chat-{i}', tokens=tok.apply_chat_template([dict(role='user', content=text)],
                           tokenize=True, add_generation_prompt=True)) for i, text in enumerate(texts)]
    rng = random.Random(20261022)
    for length in [257, 1537]:
        cases.append(dict(name=f'random-{length}', tokens=[rng.randrange(10, 1000) for _ in range(length)]))
    references = {}
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
          attn_implementation='eager', local_files_only=True).float().cuda().eval()
    with torch.inference_mode():
        for case in cases:
            history = list(case['tokens'])
            records = []
            for _ in range(8):
                logits = hf(torch.tensor([history], dtype=torch.long, device='cuda'),
                            use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
                token = int(logits.argmax())
                records.append((logits, token))
                history.append(token)
            references[case['name']] = records
            case['hf_output_ids'] = history[len(case['tokens']):]
            case['hf_text'] = tok.decode(case['hf_output_ids'])
            print('HF reference complete', case['name'], flush=True)
    del hf
    gc.collect(); torch.cuda.empty_cache()
    engine = LLM(a.model, dtype='float16', max_num_seqs=8, max_num_batched_tokens=256,
                 max_model_len=4096, kv_cache_block_cap=64, enforce_eager=False,
                 reuse_decode_buffers=True, batch_invariant=True,
                 scheduling=SchedulingOptions(mode='fixed', prefill_budget=128))
    runner = engine.model_runner
    original_model, original_run = runner.run_model, runner.run
    captured, identities, comparisons = [], {}, []

    def capture(*args):
        logits = original_model(*args)
        captured[:] = [logits.detach().float().cpu()]
        return logits

    def compare(seqs, is_prefill):
        sampled = original_run(seqs, is_prefill)
        row = 0
        for i, seq in enumerate(seqs):
            if sampled[i] is None:
                continue
            name, index = identities[seq.seq_id], seq.num_completion_tokens
            expected, token = references[name][index]
            actual = captured[0][row]
            row += 1
            delta = actual-expected
            top = expected.topk(2)
            comparisons.append(dict(case=name, index=index, is_prefill=is_prefill,
                hf_token=token, actual_token=sampled[i], max_abs=float(delta.abs().max()),
                rms=float(delta.square().mean().sqrt()),
                relative_l2=float(delta.norm()/expected.norm()),
                hf_margin=float(top.values[0]-top.values[1])))
            sampled[i] = token  # Hold history identical, report real pre-forcing argmax above.
        return sampled

    runner.run_model, runner.run = capture, compare
    report = dict(manifest=manifest(), model=a.model, gpu=torch.cuda.get_device_name(),
                  evidence='independent_fp32_hf_same_fp16_weights_full_history',
                  limits=dict(max_abs=.15, relative_l2=.01, argmax_mismatches=0), cases=cases)
    try:
        for case in cases:
            seq = engine.add_request(case['tokens'], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
            identities[seq.seq_id] = case['name']
        while not engine.is_finished():
            engine.step()
        assert not engine.scheduler.block_manager.used_block_ids
    finally:
        engine.exit()
    report['comparisons'] = comparisons
    report['summary'] = dict(comparisons=len(comparisons), max_abs=max(x['max_abs'] for x in comparisons),
        max_relative_l2=max(x['relative_l2'] for x in comparisons),
        argmax_mismatches=sum(x['actual_token'] != x['hf_token'] for x in comparisons))
    report['passed'] = (len(comparisons) == 40 and report['summary']['max_abs'] <= .15 and
                        report['summary']['max_relative_l2'] <= .01 and report['summary']['argmax_mismatches'] == 0)
    dest = Path(a.output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary']), 'passed', report['passed'], flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
