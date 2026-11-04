"""Download public Llama 3.2 weights from ModelScope and verify upstream SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--destination', required=True)
a = p.parse_args()
root = Path(a.destination)
root.mkdir(parents=True, exist_ok=True)
repo = 'LLM-Research/Llama-3.2-1B-Instruct'
upstream_sha = '1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f'
with urllib.request.urlopen(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true', timeout=60) as response:
    metadata = json.load(response)
(root/'download-manifest.json').write_text(json.dumps(metadata, indent=2))
names = {'config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'LICENSE.txt', 'USE_POLICY.md', 'README.md'}
for item in metadata['Data']['Files']:
    name = item['Path']
    if name not in names:
        continue
    if name == 'model.safetensors':
        assert item['Sha256'] == upstream_sha
    dest = root/name
    if dest.exists() and hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest() == item['Sha256']:
        print('verified cached', name, flush=True)
        continue
    partial = dest.with_suffix(dest.suffix+'.partial')
    digest = hashlib.sha256()
    count, checkpoint = 0, 0
    url = f"https://modelscope.cn/models/{repo}/resolve/{item['Revision']}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
        while chunk := response.read(8*1024*1024):
            output.write(chunk); digest.update(chunk); count += len(chunk)
            if count-checkpoint >= 128*1024*1024:
                print(name, count, '/', item['Size'], flush=True); checkpoint = count
    assert count == item['Size'] and digest.hexdigest() == item['Sha256'], name
    partial.replace(dest)
    print('verified', name, count, flush=True)
assert all((root/name).exists() for name in names)
print('DOWNLOAD_VERIFIED', flush=True)
