"""Check matching experiment plans, then compare within-model A/B speedups."""
import argparse
import json
from pathlib import Path
from audit_buffers import audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qwen', required=True, type=Path)
    p.add_argument('--llama', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    qwen, llama = audit(a.qwen), audit(a.llama)
    result = {}
    for suite in ['micro', 'e2e', 'profile']:
        qm = json.loads((a.qwen/suite/'manifest.json').read_text())
        lm = json.loads((a.llama/suite/'manifest.json').read_text())
        assert qm['manifest']['model_type'] == 'qwen3'
        assert lm['manifest']['model_type'] == 'llama'
        assert qm['manifest']['source_sha256'] == lm['manifest']['source_sha256']
        assert qm['manifest']['gpu'] == lm['manifest']['gpu']
        assert qm['manifest']['packages'] == lm['manifest']['packages']
        qa = {k:v for k,v in qm['arguments'].items() if k not in ('model', 'output_dir')}
        la = {k:v for k,v in lm['arguments'].items() if k not in ('model', 'output_dir')}
        assert qa == la, (suite, qa, la)
        assert {r['key'] for r in qwen[suite]} == {r['key'] for r in llama[suite]}
        if suite != 'profile':
            qr = {r['key']:r for r in qwen[suite]}
            result[suite] = [dict(key=r['key'], qwen_paired_median_pct=qr[r['key']]['median_paired_speedup_pct'],
                llama_paired_median_pct=r['median_paired_speedup_pct'],
                llama_paired_min_pct=r['min_paired_speedup_pct'], llama_paired_max_pct=r['max_paired_speedup_pct']) for r in llama[suite]]
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
