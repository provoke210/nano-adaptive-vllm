# 长负载输出差异：定位与修复

## 结论与范围

2026-10-21 在同一 RTX 3090 / Qwen3-0.6B 上，新增固定归约路径通过了 3 个种子、5 种调度配置、360 次请求执行的自由生成回归，合计 34,370 个输出 token。所有配置的输出逐 token 完全一致，没有放宽原来的输出相等检查。

这是已测模型、设备、软件版本与负载上的实证；并非对所有模型、GPU、编译器版本的普遍保证。推理仍使用 FP16/BF16 权重和 KV；新路径把 logits 保留为 FP32，并固定主要归约的形状和顺序。

本轮模型回归使用 FP16、temperature=0。未验证新路径的 BF16 长生成；温度采样也未实现按请求固定的随机数流，
因此 `batch_invariant` 开关不代表随机采样结果跨调度完全相同。

## 为什么旧路径会分歧

首先在旧 mixed seed 7 的保存结果中定位首个差异，再强制所有策略使用相同的历史 token，避免第一次分歧之后的内容变化污染诊断。native 输出只作为保持历史的参考，不能当作绝对正确答案。

- 请求 11 的输出索引 19：两个候选的 FP16 logits 会舍入成相同的 `22.53125`。Greedy argmax 选择了较小 token ID 5975；同一隐藏状态的 FP32 输出投影能区分 19482 与 5975。
- 请求 10 的输出索引 89：旧 native 输出 280；不同批次形状可输出 401。
- 独立 Transformers FP32 完整上下文重算（不用本项目的分页缓存、chunk 或调度器），分别选择 19482 和 401。
- 新路径在这两个位置之前的历史与保存样本相同，所选 token 均与上述独立参考一致。

这些现象指向输出投影舍入与上游随批次形状变化的归约误差，不能仅凭“FP16 更精确”就判定修复。仅把输出投影改成 FP32 后，seed 43 的完整生成仍有差异。

| 消融 | 种子数 × 策略数 | 与同组 native 不同的请求执行数 | 结果 |
|---|---:|---:|---|
| 原 FlashAttention/cuBLAS 路径 | 1 × 5 | 2 | 失败 |
| 仅 FP32 输出投影 | 3 × 5 | 5 | 失败 |
| 固定归约 linear + attention + FP32 logits | 3 × 5 | 0 | 通过 |

自由生成没有 teacher forcing。每个种子包含 24 个合成请求；种子为 7、19、43，策略为 native、fixed128、fixed512、fixed2048、原 adaptive。请求每 8 个执行步骤进入一个，用于稳定地制造不同批次形状；该回归的时间不能当作开放环性能指标。

## 实现改变

`nanovllm/layers/invariant.py` 的矩阵乘法固定为 M=16、N=64、K=64 的分块，不根据批次大小选择算法，也不执行 split-K。累加使用 FP32；普通层输出恢复模型 dtype，LM head 输出保持 FP32。

Attention 在 prefill、混合批次和 decode 中均使用同一个固定形状算法，按相同顺序遍历 64-token K 块。逻辑 token 顺序与物理分页地址分开处理，使用相同 causal mask 和 online softmax。纯 decode 仍可由 CUDA Graph 重放。

`Config.batch_invariant=True` 为当前默认值。设为 False 可恢复原数学执行路径，便于消融，但不得将不同数学后端的耗时直接当作调度加速比。固定归约有性能代价，必须同时在所有性能基线上启用。

## 验证方法

- 矩阵乘法对比独立 PyTorch FP32 数学参考；整批与不对齐拆分批次要求逐位相等。
- Attention 对比独立 FP32 causal softmax 参考；测试跨 256-token 页边界、非连续物理页、未对齐 chunk 和单 token decode。分页/chunk/decode 与完整计算要求逐位相等。数学参考的误差容限用于低精度算子误差，不用于放宽最终 token 回归。
- 空闲 KV 槽填充 NaN，帮助检测越界读取。
- 修复后的完整测试门禁 73 项通过（68 CPU + 5 GPU），包含 eager/CUDA Graph、初始化失败清理及上述内核测试。后续新增槽位准入测试后，第三版完整门禁为 77 项通过（72 CPU + 5 GPU）；冻结第四版为 78 项通过（73 CPU + 5 GPU）。

远程证据位于 `results/round2/`；本地下载位于 `results/autodl-20261021/results/round2/`。`long-original.json`、`long-fp32.json` 保留失败结果，`long-invariant.json` 保留全部通过的 token 序列。该测试覆盖调度一致性，不等同于自然语言质量评测。

参考背景：[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)、[vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)。本项目内核为本地实现，不是直接移植 vLLM 的完整 batch-invariant 后端。
