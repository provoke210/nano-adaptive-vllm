"""Fail-closed audit of repeated CUDA comparisons and per-trace fixed-budget oracle."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def audit(directory):
    index = json.loads((directory / 'index.json').read_text())
    assert index and len({x['file'] for x in index}) == len(index)
    traces = {hashlib.sha256(p.read_bytes()).hexdigest():
              [json.loads(line) for line in p.read_text().splitlines()]
              for p in directory.glob('trace-*.jsonl')}
    groups = defaultdict(list)
    signatures = set()
    sources = set()
    configs = {}
    for entry in index:
        report = json.loads((directory / entry['file']).read_text())
        assert report['evidence'] == 'cuda_measurement'
        args, manifest = report['arguments'], report['manifest']
        sources.add(manifest['source_sha256'])
        signatures.add(json.dumps({
            'environment': {k: manifest[k] for k in ('gpu', 'cuda', 'actual_dtype',
                'actual_kv_blocks', 'model_config_sha256', 'packages', 'batch_invariant')},
            'limits': {k: args[k] for k in ('max_seqs', 'token_budget', 'max_model_len',
                'eager', 'ttft_slo_ms', 'tpot_slo_ms')},
        }, sort_keys=True))
        variant = report['variant']
        config = json.dumps(report['scheduling'], sort_keys=True)
        assert configs.setdefault(variant, config) == config
        assert entry['variant'] == variant and entry['repeat'] == report['repeat']
        expected = {r['id']: r for r in traces[report['trace_sha256']]}
        records = report['requests']
        assert len(records) == len(expected) == report['summary']['completed']
        assert {r['id'] for r in records} == set(expected)
        for r in records:
            q = expected[r['id']]
            times = r['token_times_ms']
            assert len(times) == len(r['output_ids']) == q['max_tokens']
            assert r['arrival_ms'] == q['arrival_ms']
            assert r['prompt_tokens'] == len(q['token_ids'])
            assert all(math.isfinite(t) for t in times) and times[0] >= q['arrival_ms']
            assert all(b >= a for a, b in zip(times, times[1:]))
        assert all(math.isfinite(v) for v in report['summary'].values() if isinstance(v, (float, int)))
        groups[(entry['scenario'], entry['seed'], report['trace_sha256'])].append(report)
    assert len(sources) == len(signatures) == 1, 'Mixed code/environment/limits'
    assert {(scenario, seed) for scenario, seed, _ in groups} == {
        (scenario, seed) for scenario in args['scenarios'] for seed in args['seeds']
    }, 'Missing complete trace group'
    if args['variants']:
        assert set(configs) == set(args['variants']), 'Missing complete variant'
    rows, comparisons = [], []
    request_executions = output_tokens = 0
    for (scenario, seed, trace_hash), reports in groups.items():
        reference = {r['id']: r['output_ids'] for r in reports[0]['requests']}
        by_variant = defaultdict(list)
        for r in reports:
            assert {q['id']: q['output_ids'] for q in r['requests']} == reference, 'Output divergence'
            by_variant[r['variant']].append(r)
            request_executions += len(r['requests'])
            output_tokens += sum(len(q['output_ids']) for q in r['requests'])
        assert set(by_variant) == set(configs), 'Missing variant'
        local = []
        for variant, runs in by_variant.items():
            expected_repeats = runs[0]['arguments']['repeats']
            assert sorted(r['repeat'] for r in runs) == list(range(expected_repeats)), 'Missing/duplicate repeat'
            row = dict(scenario=scenario, seed=seed, trace_sha256=trace_hash, variant=variant,
                       repeats=expected_repeats)
            for metric in ('goodput_requests_per_s', 'output_tokens_per_s', 'ttft_p99_ms',
                           'itl_p99_ms', 'slo_attainment', 'preemptions', 'recomputed_tokens'):
                values = [r['summary'][metric] for r in runs]
                row[metric] = dict(median=median(values), min=min(values), max=max(values))
            local.append(row)
        fixed = [r for r in local if r['variant'].startswith('fixed-') and r['variant'][6:].isdigit()]
        best = max(fixed, key=lambda r: r['goodput_requests_per_s']['median'])
        admission_fixed = [r for r in local if r['variant'].startswith('fixed-')
                           and r['variant'].endswith('-admit')]
        best_admission = max(admission_fixed, key=lambda r: r['goodput_requests_per_s']['median']) if admission_fixed else None
        aware_fixed = [r for r in local if r['variant'].startswith('fixed-')
                       and r['variant'].endswith('-aware')]
        best_aware = max(aware_fixed, key=lambda r: r['goodput_requests_per_s']['median']) if aware_fixed else None
        for r in local:
            if r['variant'].startswith('cost') or r['variant'] == 'fixed-2048-aware':
                base = best['goodput_requests_per_s']['median']
                comparison = dict(scenario=scenario, seed=seed, variant=r['variant'],
                    best_fixed=best['variant'], goodput_change_pct=
                    100 * (r['goodput_requests_per_s']['median'] / base - 1))
                if best_admission:
                    comparison.update(best_fixed_with_admission=best_admission['variant'],
                        versus_fixed_with_admission_pct=100 * (
                            r['goodput_requests_per_s']['median'] /
                            best_admission['goodput_requests_per_s']['median'] - 1))
                if best_aware:
                    comparison.update(best_fixed_with_aware_admission=best_aware['variant'],
                        versus_fixed_with_aware_admission_pct=100 * (
                            r['goodput_requests_per_s']['median'] /
                            best_aware['goodput_requests_per_s']['median'] - 1))
                comparisons.append(comparison)
        rows.extend(local)
    return dict(reports=len(index), traces=len(groups), request_executions=request_executions,
                output_tokens=output_tokens, all_outputs_identical=True,
                comparability_checks_passed=True, source_sha256=next(iter(sources)),
                rows=rows, comparisons=comparisons,
                limitation='Finite synthetic traces; medians/ranges are descriptive, not statistical significance.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    result = audit(a.directory)
    (a.directory / 'audit-repeated.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
