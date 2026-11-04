"""Audit exact-output pairs and report per-case medians; stdlib only."""
import argparse
import json
from pathlib import Path
from statistics import median


def audit(root):
    report = {}
    for suite in ['micro', 'e2e', 'profile']:
        directory = root / suite
        if not (directory / 'index.json').exists():
            continue
        arguments = json.loads((directory / 'manifest.json').read_text())['arguments']
        expected_repeats = 1 if suite == 'profile' else arguments['repeats']
        expected_keys = ({f'{b}-{c}' for b in arguments['batches'] for c in arguments['contexts']}
                         if suite != 'e2e' else {f'{scenario}-{seed}' for scenario in ['mixed', 'burst', 'pressure'] for seed in arguments['seeds']})
        rows = json.loads((directory / 'index.json').read_text())
        assert len(rows) == len(expected_keys) * expected_repeats * 2
        assert len({row['file'] for row in rows}) == len(rows)
        groups = {}
        for row in rows:
            data = json.loads((directory / row['file']).read_text())
            assert data['profile_enabled'] == (suite == 'profile')
            groups.setdefault(data['key'], {}).setdefault(data['repeat'], {})[data['reuse']] = data
        assert set(groups) == expected_keys
        cases = []
        for key, repeats in groups.items():
            assert set(repeats) == set(range(expected_repeats))
            paired = []
            baseline, reused = [], []
            reference = None
            for repeat, variants in sorted(repeats.items()):
                assert set(variants) == {False, True}, (suite, key, repeat)
                a, b = variants[False], variants[True]
                assert a['trace_sha256'] == b['trace_sha256']
                assert a['manifest'] == b['manifest']
                assert a['scheduling'] == b['scheduling']
                assert a['outputs'] == b['outputs'], (suite, key, repeat)
                if reference is None:
                    reference = a['outputs']
                assert reference == a['outputs']
                paired.append(100 * (b['summary']['output_tokens_per_s'] / a['summary']['output_tokens_per_s'] - 1))
                baseline.append(a)
                reused.append(b)
            case = dict(key=key, pairs=len(paired), exact_output_equal=True)
            if suite == 'profile':
                for name, records in [('baseline', baseline), ('reuse', reused)]:
                    traces = [t for record in records for t in record['profiles']]
                    assert traces and all(not t['is_prefill'] for t in traces)
                    case[name] = {clock: {phase: median(t[clock][phase] for t in traces)
                        for phase in traces[0][clock]} for clock in ['cpu_ms', 'cuda_stream_ms', 'engine_cpu_ms']}
            else:
                case.update(median_paired_speedup_pct=median(paired), min_paired_speedup_pct=min(paired), max_paired_speedup_pct=max(paired), paired_speedups_pct=paired)
                for name, records in [('baseline', baseline), ('reuse', reused)]:
                    case[name] = {metric: median(record['summary'][metric] for record in records)
                                  for metric in records[0]['summary'] if records[0]['summary'][metric] is not None}
            cases.append(case)
        report[suite] = cases
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / 'audit.json').write_text(json.dumps(result, indent=2))
    for suite, cases in result.items():
        print(suite)
        for case in cases:
            if suite == 'profile':
                print(case['key'], 'prepare CPU ms', case['baseline']['cpu_ms']['prepare'], '->', case['reuse']['cpu_ms']['prepare'])
            else:
                print(case['key'], 'pairs', case['pairs'], 'speedup median/range %', *[round(case[k], 2) for k in ['median_paired_speedup_pct', 'min_paired_speedup_pct', 'max_paired_speedup_pct']])
