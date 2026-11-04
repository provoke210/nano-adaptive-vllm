# Llama 单卡验证：正确性与输入缓冲区复用

日期：2026-10-22。Meta Llama-3.2-1B-Instruct，RTX 3090 24GB，FP16，CUDA Graph，batch_invariant=True。

## 结论

Llama 适配通过真实预训练权重的 CUDA 验证。与同一引擎关闭输入复用相比，纯 Decode 吞吐提升 **0.76%–3.08%**，端到端提升 **0.90%–2.01%**，均为逐场景配对中位数。50 对 micro 和 18 对 e2e 测量均为正收益。输入复用的收益在 Qwen3-0.6B 和 Llama-3.2-1B 上均复现，幅度有限。

这轮测量固定了调度参数，只切换输入复用。它没有证明动态 token 预算优于最佳固定预算，也没有验证历史槽位准入 goodput 增幅在 Llama 上成立。

## 模型与环境

- 官方模型：[Meta Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)。从公开 [ModelScope 镜像](https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct) 下载，按每文件 revision、大小和 SHA256 校验，模型许可证及使用政策随文件保存。
- 权重 SHA256：`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`，与官方 HF 文件元数据一致；文件 2,471,645,608 bytes。来源证据见 `model-provenance.json` 与 `model-download-manifest.json`。
- 16 层、hidden=2048、32 query heads / 8 KV heads、共享 embedding/LM head、llama3 RoPE、多 EOS。原始 checkpoint 为 BF16，本次执行转换为 FP16。
- torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.55.2 / flash-attn 2.8.3；驱动 570.124.04。
- 核心源码 SHA256：`b5f8ec7346dbb8484609f3f8fc9f3ce85da9a99a2294687c6adec0f2f0a9c345`，与最终 Qwen3 A/B 完全相同。本轮新增验证/下载/审计脚本，没有修改引擎实现。

## 正确性证据

1. 全测试套件 **133 passed，0 skipped，0 failed**，GPU 测试使用 Llama 预训练权重；详见 `tests.xml`。
2. 独立 HF 对照：3 个聊天模板输入及 257/1537 token 随机输入，每个后续 8 步，共 **40 个 logits 位置**。HF 使用相同的 FP16 量化后权重转 FP32、eager attention、完整历史重算；引擎使用 FP16、分页 KV、分块/混合调度与 CUDA Graph。预设门槛为最大绝对误差≤0.15、相对 L2≤0.01、argmax 不同数=0。实测分别为 **0.060606、0.005319、0**。参考脚本在同一历史上比较实际 argmax，再用 HF token 延续历史；本次所有实际 argmax 都相同。不是 logits 逐位完全相等，也不是原始 BF16 执行精度测试。
3. 自由生成回归：seeds 7/19/43 × native512/fixed128/fixed512/fixed2048/adaptive512 × 24 请求，共 **360 次请求执行**；每次输出 64–128 token，跨调度输出全部逐 token 一致，无 teacher forcing。这是正确性检查，不作性能对比。
4. 输入复用 A/B：100 次 micro、36 次 e2e、12 次 profile，共 **148 次运行**，每对输出完全一致；e2e 共 **864 次请求执行**。本地严格审计完整性、配置、trace、源码和输出；跨模型审计确认两模型采用相同实验计划、核心源码、GPU 型号与包版本。不同模型间不比较输出 token 是否相同。

## 测量方法

与 Qwen3 的最终 v3 实验计划一致：同进程同模型、随机 A/B 次序、每次预热，两组 KV 容量相同且均预分配输入池。正式吞吐计时关闭 profiling，阶段统计独立采样。

- micro：batch=1/4/16/32/64 × 初始 context=32/1024，每组 5 对重复；首个输出及 16 步预热后计时 96 个 Decode 步，KV blocks=384。context 随生成增长。
- e2e：mixed/burst/pressure × seeds 101/307，每组 3 对重复，每次 24 请求；KV blocks=128，max_num_seqs=16，token budget=2048，固定 prefill_budget=2048，加相同的现有准入控制（wait=100ms、target=800ms、max_slots=4）。
- profile：batch=1/16/64 × 初始 context=32/1024，1 对、48 步，只解释阶段开销，不用于吞吐结论。
- 百分比为每对 `reuse/baseline - 1` 后取中位数；独立 token/s 中位数的比值可能略有不同。全部 e2e 的 SLO 达标率均为 100%，抢占均为 0。SLO 定义为 TTFT≤1000ms 且请求平均 TPOT≤50ms，不是每个 ITL≤50ms。

## 纯 Decode 吞吐

| batch / 初始 context | Llama 配对中位数 | Llama 配对范围 | Qwen3 配对中位数 |
|---|---:|---:|---:|
| 1 / 32 | +3.04% | +1.37% ~ +3.12% | +4.36% |
| 4 / 32 | +3.08% | +3.04% ~ +3.17% | +4.48% |
| 16 / 32 | +2.92% | +2.75% ~ +3.00% | +4.04% |
| 32 / 32 | +2.40% | +2.36% ~ +2.61% | +3.44% |
| 64 / 32 | +1.00% | +0.84% ~ +1.16% | +1.89% |
| 1 / 1024 | +2.89% | +2.79% ~ +3.07% | +3.52% |
| 4 / 1024 | +2.85% | +2.79% ~ +2.98% | +3.50% |
| 16 / 1024 | +2.19% | +2.13% ~ +2.50% | +2.32% |
| 32 / 1024 | +1.70% | +1.23% ~ +1.80% | +1.36% |
| 64 / 1024 | +0.76% | +0.64% ~ +0.86% | +0.60% |

## 端到端吞吐与尾延迟

下表 token/s、P99 均先按每次运行计算，再取三次中位数。

| 场景 | 配对吞吐提升中位数 | 配对范围 | token/s：关→开 | TTFT P99 ms：关→开 | ITL P99 ms：关→开 |
|---|---:|---:|---:|---:|---:|
| mixed-101 | +1.29% | +0.68% ~ +1.33% | 930.02→941.07 | 802.92→792.16 | 136.09→136.64 |
| burst-101 | +0.90% | +0.87% ~ +1.02% | 970.15→979.52 | 552.96→550.47 | 147.45→147.14 |
| pressure-101 | +1.76% | +1.50% ~ +2.28% | 1759.64→1792.93 | 551.01→549.24 | 44.79→44.62 |
| mixed-307 | +1.81% | +1.78% ~ +1.89% | 1401.54→1428.09 | 426.41→448.00 | 75.24→75.18 |
| burst-307 | +1.24% | +1.13% ~ +1.28% | 1321.69→1338.10 | 346.55→346.10 | 16.84→16.81 |
| pressure-307 | +2.01% | +1.95% ~ +2.05% | 2073.04→2115.00 | 321.97→322.17 | 11.57→11.28 |

吞吐提升不等于尾延迟全面改善。mixed-307 的 TTFT P99 从 426.41ms 升至 448.00ms（约 +5.06%），mixed-101 的 ITL P99 从 136.09ms 升至 136.64ms，pressure-307 的 TTFT P99 也小幅上升。不能宣称提高了 SLO 达标率，或本次通过缓解抢占获得收益。

## 阶段统计

| batch / 初始 context | prepare CPU μs：关→开 | model 提交 CPU μs：关→开 |
|---|---:|---:|
| 1 / 32 | 166.3→117.6 | 258.9→159.9 |
| 16 / 32 | 186.5→137.9 | 267.4→157.3 |
| 64 / 32 | 245.2→235.2 | 265.8→181.6 |
| 1 / 1024 | 164.3→114.0 | 256.0→154.9 |
| 16 / 1024 | 194.7→154.9 | 267.4→167.6 |
| 64 / 1024 | 315.5→301.5 | 345.6→216.4 |

输入复用减少 CPU 输入构造与 Graph 入参搬运开销，计算占比增大时收益趋小。阶段计时含 profiling 开销；model CPU 时间包含 Graph 入参准备、replay 和 logits 发射，不能解释成 GPU 矩阵计算本身加速。CUDA event span 包含等待及发射间隙，与 CPU 时间重叠，不能相加。

## 复核入口与边界

原始结果在 `results/llama-20261022/`，包括 `audit.json`、`cross-model-audit.json`、`hf-reference.json`、`long-correctness.json`、日志、环境及压缩归档。

```bash
.venv/bin/python scripts/audit_cross_model_buffers.py \
  --qwen results/autodl-20261022/buffer-v3-20261022 \
  --llama results/llama-20261022 \
  --output results/llama-20261022/cross-model-audit.json

NANO_MODEL=/path/to/Llama-3.2-1B-Instruct NANO_DTYPE=float16 python -m pytest -q
python scripts/validate_hf_model.py --model /path/to/Llama-3.2-1B-Instruct \
  --output results/llama-hf.json
python scripts/validate_long.py --model /path/to/Llama-3.2-1B-Instruct \
  --output results/llama-long.json --batch-invariant
```

- 本轮覆盖单卡、小模型、FP16、合成 token 性能负载；未验证 BF16、大模型、多卡、真实生产流量或 128k 上下文。引擎上限设为 4096，不代表测到该上限。
- 同样的整数 token 在不同 tokenizer 下含义不同，因此跨模型只比较各自 A/B 的加速比例，不比较模型质量或把绝对吞吐差异归因于某个架构细节。
- 纯 Decode 使用输入池；Prefill/mixed 保持原路径，仍存在 sampler.tolist() 同步。较快执行还会改变请求到达落点与准入反馈，端到端效果包含这些间接变化。
- 只有一次 GPU 会话、有限重复，表中是观察到的范围，不是置信区间；小百分比收益仍需更多硬件和流量复测。默认复用开关仍关闭。
- 历史 Qwen3 调度收益与本轮输入复用收益不能直接相加；Llama 上最佳固定预算与动态预算的完整性能扫描尚未进行。
