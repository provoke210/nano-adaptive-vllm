"""Audit the frozen 4B suite and summarize sampled device memory by phase."""
import argparse
import csv
from datetime import datetime
import json
import hashlib
import tarfile
from pathlib import Path
import xml.etree.ElementTree as ET

from audit_buffers import audit

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
root = a.directory
assert (root/'exit-code.txt').read_text().strip() == '0'
report = audit(root)
assert set(report) == {'micro', 'e2e', 'profile'}
session = json.loads((root/'session.json').read_text())
source = session['remote_manifest_source_sha256']
with tarfile.open(root/'executed-core-source.tar.gz') as archive:
    files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
for include_metadata, expected in [(True, source), (False, session['core_source_sha256'])]:
    paths = ['pyproject.toml']
    for directory in ['nanovllm/', 'benchmarks/']:
        paths += sorted(n for n in files if n.startswith(directory) and n.endswith('.py')
                        and (include_metadata or not Path(n).name.startswith('._')))
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode() + b'\0' + files[name] + b'\0')
    assert digest.hexdigest() == expected
for suite, count in [('micro', 80), ('e2e', 36), ('profile', 12)]:
    directory = root/suite
    metadata = json.loads((directory/'manifest.json').read_text())
    assert metadata['manifest']['source_sha256'] == source
    assert metadata['manifest']['model_type'] == 'qwen3'
    assert metadata['manifest']['kv_blocks'] == (128 if suite == 'e2e' else 192)
    rows = json.loads((directory/'index.json').read_text())
    assert len(rows) == count
    for row in rows:
        data = json.loads((directory/row['file']).read_text())
        assert data['manifest'] == metadata['manifest']
hf = json.loads((root/'hf-reference.json').read_text())
assert hf['passed'] and hf['summary']['comparisons'] == 40
assert hf['manifest']['source_sha256'] == source
long = json.loads((root/'long-correctness.json').read_text())
assert long['manifest']['source_sha256'] == source and long['all_equal']
assert len(long['runs']) == 15 and sum(len(r['outputs']) for r in long['runs']) == 360
assert all(not r['differences'] for r in long['runs'])
suites = list(ET.parse(root/'tests.xml').getroot().iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites) == 134
assert all(int(s.attrib[k]) == 0 for s in suites for k in ['errors', 'failures', 'skipped'])
assert (root/'regression-exit-code.txt').read_text().strip() == '0'
for name in ['qwen06', 'llama']:
    regression = list(ET.parse(root/f'regression-{name}.xml').getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in regression) == 7
    assert all(int(s.attrib[k]) == 0 for s in regression for k in ['errors', 'failures', 'skipped'])
phases = []
for line in (root/'phases.log').read_text().splitlines():
    stamp, name = line.split()
    phases.append((datetime.fromisoformat(stamp).replace(tzinfo=None), name))
rows = list(csv.reader((root/'gpu-memory.csv').open()))[1:]
samples = [(datetime.strptime(r[0], '%Y/%m/%d %H:%M:%S.%f'), int(r[1].split()[0])) for r in rows]
memory = {}
for (start, phase), (end, _) in zip(phases, phases[1:]):
    if phase == 'audit':
        continue
    values = [used for time, used in samples if start <= time < end]
    assert values, phase
    memory[phase] = dict(samples=len(values), max_observed_mib=max(values), max_observed_gib=max(values)/1024)
result = dict(source_sha256=source, tests_passed=134, hf=hf['summary'], long_request_executions=360,
              buffer_runs=128, e2e_request_executions=864, regression_gpu_tests_per_model=7, memory=memory,
              memory_note='nvidia-smi whole-device samples every 1 second; not an instantaneous allocator peak.')
(root/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
(root/'validation-audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
