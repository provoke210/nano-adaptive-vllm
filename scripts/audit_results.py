"""Audit CUDA report comparability and expose output differences (not a quality gate)."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--expected-reports', type=int)
a = p.parse_args()
groups = defaultdict(list)
reports = []
for path in sorted(a.directory.glob('*.json')):
    r = json.loads(path.read_text())
    if not isinstance(r, dict) or 'evidence' not in r:
        continue
    assert r['evidence'] == 'cuda_measurement', path
    assert r['summary']['completed'] == len(r['requests']) == r['summary']['requests'], path
    for q in r['requests']:
        times = q['token_times_ms']
        assert times and len(times) == len(q['output_ids']), path
        assert all(y >= x for x, y in zip(times, times[1:])), path
    groups[r['trace_sha256']].append((path, r))
    reports.append(r)
assert reports, 'No CUDA reports'
if a.expected_reports is not None:
    assert len(reports) == a.expected_reports, (len(reports), a.expected_reports)
assert len({r['manifest']['source_sha256'] for r in reports}) == 1, 'Mixed source versions'
rows = []
for trace_hash, entries in groups.items():
    natives = [r for _, r in entries if r['arguments']['policy'] == 'native']
    assert len(natives) == 1, 'Need exactly one native per trace'
    reference = {q['id']: q['output_ids'] for q in natives[0]['requests']}
    signatures = set()
    for path, r in entries:
        args, m = r['arguments'], r['manifest']
        signatures.add(tuple(m[k] for k in ('gpu', 'actual_dtype', 'actual_kv_blocks', 'model_config_sha256'))
                       + tuple(args[k] for k in ('max_seqs', 'token_budget', 'max_model_len', 'eager',
                                                'ttft_slo_ms', 'tpot_slo_ms'))
                       + (json.dumps(m['packages'], sort_keys=True),))
        trace = [json.loads(line) for line in (a.directory / Path(args['trace']).name).read_text().splitlines()]
        lengths = {q['id']: q['max_tokens'] for q in trace}
        assert set(reference) == {q['id'] for q in r['requests']}, path
        assert all(len(q['output_ids']) == lengths[q['id']] for q in r['requests']), path
        differences = [q['id'] for q in r['requests'] if q['output_ids'] != reference[q['id']]]
        rows.append(dict(file=path.name, trace_sha256=trace_hash,
                         output_different_request_ids=differences,
                         max_step_ms=max(s['elapsed_ms'] for s in r['steps']),
                         **r['summary']))
    assert len(signatures) == 1, 'Configuration mismatch within trace'
audit = dict(reports=len(reports), traces=len(groups), comparability_checks_passed=True,
             all_outputs_identical=all(not r['output_different_request_ids'] for r in rows),
             limitation='Output differences are reported, not accepted as a numerical correctness pass.',
             source_sha256=reports[0]['manifest']['source_sha256'], rows=rows)
(a.directory / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'rows'}, indent=2))
