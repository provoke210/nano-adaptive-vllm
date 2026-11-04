# GPU 验收流程

先用 Linux、单张 NVIDIA 24GB GPU、Qwen3-0.6B 验收，再扩大模型。
2026-10-19 已在 RTX 3090、PyTorch 2.8.0+cu128、FlashAttention 2.8.3、
Transformers 4.55.2 上完成 FP16 的有限样本正确性验收。
这是历史旧数学路径的结果。2026-10-21 默认已切换为固定归约后端，FP16 长负载一致性回归通过；
BF16 长负载尚未在新路径重做验收。默认 `auto` 保留模型原 dtype，不隐式改为 FP16。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 triton==3.4.0
python -m pip install packaging ninja psutil setuptools wheel
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install transformers==4.55.2 tqdm
python -m pip install -e '.[test]'
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
export NANO_MODEL="$PWD/models/Qwen3-0.6B"
export NANO_DTYPE=float16
python -m pytest -q
```

FlashAttention 构建需要匹配的 CUDA toolkit / 编译工具，优先选择配置齐全的镜像。
若默认 pip 镜像找不到 ninja，可改用 `PIP_INDEX_URL=https://pypi.org/simple`。
若 flash-attn 落入耗时的源码编译，可使用 [官方 release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3)
的 `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`。
必须先核对 `torch._C._GLIBCXX_USE_CXX11_ABI` 为 True、Python 3.12 和 torch 2.8。
该文件 SHA-256：`f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb`。

GPU gate 同时覆盖 eager 与 CUDA Graph；greedy 不一致时，先保留失败并定位 logits/attention 路径。
该 gate 只是有限样本，尚不能替代共享前缀与抢占边界的 GPU 专项测试。

```bash
python -m benchmarks.matrix --backend gpu --model "$NANO_MODEL" \
  --dtype float16 \
  --count 24 --seeds 7 --scenarios mixed burst --blocks 64 \
  --output-dir results/gpu-smoke
```

每组独立进程加载模型、预热，保持同样的 GPU 空闲条件与包版本。
blocks 是自动容量上限；装不下会报错，不能让不同配置暗中使用不同池大小。
64 × 256 token 是压力测试容量，之后增加 blocks 检查适用范围。
正式实验增加请求数、到达率、独立种子和重复运行，按 [协议](EXPERIMENTS.md) 冻结参数。
JSON 保存逐请求输出和时间、逐步预算和缓存，以及 GPU/包版本。
还需归档模型 revision / 权重 manifest；工具只自动记录 config.json hash，不扫描整个权重文件。

只有通过真实模型正确性后，才开始形成性能结论。失败时不要只给一个配置换 dtype / 图设置后继续比较。

本次 BF16 失败后，使用 `scripts/diagnose_logits.py` 做固定历史诊断，96 次对照中 4 次
argmax 分歧都对应原版 top-2 分数并列。随后所有策略统一指定 FP16 重跑原严格断言，
没有放宽 equality、换提示词或忽略失败。这个结果支持舍入影响的解释，但不是任意形状下
缓存/attention 正确性的完整证明，也不意味着 FP16 天然保证 batch invariance。
相关背景见 [PyTorch 数值精度](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
和 [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)。

## 第二轮：先数值回归，再重复性能实验

```bash
export NANO_MODEL=/path/to/Qwen3-0.6B
NANO_DTYPE=float16 python -m pytest -q
python -m scripts.validate_long --model "$NANO_MODEL" --batch-invariant \
  --seeds 7 19 43 --output results/long-invariant.json
python -m benchmarks.gpu_compare --model "$NANO_MODEL" \
  --output-dir results/train --seeds 7 --scenarios mixed burst pressure \
  --count 24 --repeats 3 --rate 30 --target-step-ms 50 \
  --initial-budget 2048 --coalesce-ms 96
python scripts/audit_repeated.py results/train
```

`gpu_compare` 在一个模型进程中依次运行，每次重置 scheduler/cache manager，执行同样的专门预热，
随后再次重置调度器开始计时；每次重复打乱配置顺序。全部策略统一启用 batch_invariant。
报告保存每次运行的输出、时间、在线拟合参数和来源；跨策略输出不一致时立即停止。
`audit_repeated.py` 重新检查完整性、资源/源码一致性、trace hash、token 数与时间单调性，
按同一 trace 的多次重复中位数选择最佳 fixed，并给出 adaptive 的相对变化。

上面命令为调参实验，不是已经冻结的推荐参数；最终评估应采用报告中记录的冻结参数及独立种子。
始终保留失败结果。不能用较旧数学后端的 fixed 去和新后端的 adaptive 比速度。
