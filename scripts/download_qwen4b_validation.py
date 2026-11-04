"""Download official Qwen3-4B files from ModelScope; verify pinned weight SHA256."""
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
repo = 'Qwen/Qwen3-4B'
upstream_shas = {'model-00001-of-00003.safetensors': '328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223', 'model-00002-of-00003.safetensors': '6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5', 'model-00003-of-00003.safetensors': 'e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1'}
with urllib.request.urlopen(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true', timeout=60) as response:
    metadata = json.load(response)
(root/'download-manifest.json').write_text(json.dumps(metadata, indent=2))
names = set(upstream_shas) | {'config.json', 'generation_config.json', 'model.safetensors.index.json',
    'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'LICENSE', 'README.md'}
for item in metadata['Data']['Files']:
    name = item['Path']
    if name not in names:
        continue
    if name in upstream_shas:
        assert item['Sha256'] == upstream_shas[name]
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
