# 本地验收记录

2026-10-21：本地 73 项 CPU 测试通过、5 项 GPU 测试跳过；GPU 完整门禁 78 项全部通过。
长负载固定归约回归与新调度实验见 [第二轮报告](GPU_ROUND2.md)。下文均为历史记录。

2026-10-19 补充：已在 RTX 3090 / Qwen3-0.6B 上通过 FP16 的
60 项 CPU 测试与 3 项 GPU 测试；BF16 严格 greedy 对照未通过。
以下保留 2026-10-18 的历史本地记录，不能把历史“尚未验收”视为最新状态。
最新完整结果与失败诊断见 [首轮 GPU 实测](GPU_ROUND1.md)。

日期：2026-10-18（Asia/Singapore）。环境：Apple Silicon macOS，Python 3.12.14。
没有 NVIDIA GPU，未安装本项目的 CUDA 依赖，没有执行真实模型推理。

## 已执行

| 检查 | 结果 |
|---|---|
| `python -m pip install -e '.[test]'` | 安装成功 |
| `python -m pytest -q` | **60 passed, 2 skipped** |
| `python -m compileall -q nanovllm benchmarks tests` | 通过 |
| `git diff --check` | 通过 |
| `python -m pip check` | 无损坏依赖 |
| 仿真矩阵 | **60 次完成，每次 24 请求** |
| 同 trace 跨策略输出对照 | **6 条 trace 各自的 10 组输出完全一致** |

2 个跳过项是 eager / CUDA Graph 的真实模型 greedy 测试，须设置 NANO_MODEL 并提供 CUDA。
安装后的本机主要依赖：numpy 2.5.4、xxhash 3.8.2、pytest 9.1.1；每份实验报告另记录完整相关包版本。

矩阵命令：

```bash
.venv/bin/python -m benchmarks.matrix --count 24 --seeds 7 19 \
  --scenarios mixed burst pressure --output-dir results/local-smoke
```

原始报告及汇总在本地 results/local-smoke/，标记 synthetic_simulation，不作为 GPU 性能证据。
在 seed=7 的压力负载里，native / fixed-512 / combined 的模拟抢占次数为 10 / 8 / 5；
这仅证明压力场景确实触发抢占路径，不说明真实 GPU 上的吞吐或收益。

## 覆盖

随机到达的 15 种种子 × 3 种策略；每步验证物理块所有权与引用计数；
混合 Decode/Prefill、非整块分块、prefix cache 重用、抢占后生成历史重算、
中间 chunk 不采样、取消、EOS、完成回收、单请求不可能装入池的拒绝、
控制器回退/恢复/老化、有界 token 预算、可复现重放、联合 SLO goodput 口径。

## 尚未验收

- CUDA 模型前向、FlashAttention paged varlen 混合批次的数值一致性。
- eager 与 CUDA Graph 的真实模型输出和性能。
- 有代表性的自然语言负载、较大模型、不同 GPU 与正常 KV 容量。
- 拟合延迟预测器、正式 held-out 评估、重复测量置信度。

下一步按 GPU_RUNBOOK.md 先做正确性 gate，再 profiling 与性能实验。
