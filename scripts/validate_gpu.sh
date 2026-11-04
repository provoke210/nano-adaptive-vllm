#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: bash scripts/validate_gpu.sh MODEL_DIR OUTPUT_DIR}"
: "${2:?Output directory required}"
export NANO_MODEL="$1"
export NANO_DTYPE="${NANO_DTYPE:-auto}"
output_dir="$2"
mkdir -p "$output_dir"
python - <<'PY'
import torch, flash_attn, transformers
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name())
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('flash_attn:', flash_attn.__version__, 'transformers:', transformers.__version__)
PY
python -m pip freeze > "$output_dir/packages.txt"
nvidia-smi > "$output_dir/nvidia-smi.txt"
python -m pytest -q -m 'not gpu' --junitxml="$output_dir/cpu-tests.xml" 2>&1 | tee "$output_dir/cpu-tests.log"
python -m pytest -q -m gpu --junitxml="$output_dir/gpu-tests.xml" 2>&1 | tee "$output_dir/gpu-tests.log"
python - "$output_dir/gpu-tests.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert cases and not any(c.find('skipped') is not None or c.find('failure') is not None or c.find('error') is not None for c in cases), 'GPU gate did not pass; benchmarks blocked'
PY
python -m benchmarks.matrix --backend gpu --model "$NANO_MODEL" \
  --dtype "$NANO_DTYPE" \
  --count 24 --seeds 7 --scenarios mixed burst --blocks 64 \
  --output-dir "$output_dir/matrix" 2>&1 | tee "$output_dir/matrix.log"
