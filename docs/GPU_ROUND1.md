# 首轮 GPU 验证（2026-10-19）

此页保存首轮历史结果。2026-10-21 的长负载修复与独立评估见 [第二轮报告](GPU_ROUND2.md)。

这是 Qwen3-0.6B 单卡 smoke 验证，不是正式性能结论或质量评测。

全部原始结果已回传本地。

## 环境与可追溯性

- 云 GPU：RTX 3090 24GB × 1，15 vCPU、90GB RAM。
- PyTorch 2.8.0+cu128、Triton 3.4.0、FlashAttention 2.8.3、Transformers 4.55.2、Python 3.12.3。
- 模型来自 ModelScope 官方 `Qwen/Qwen3-0.6B`；模型文件大小与 SHA-256 见原始 `model-manifest.json`。
- 实际测试源码包 SHA-256：`21de602f9d8a95054f4246552b05d033c9444454832cab8ef34b7e6d9ec80f5b`。
- GPU 报告源码 hash：`ec7680258215e2c6559a0c4018711b4fc26c9128d19bd243dc2cad373543443e`。
- `native` 是共享 runner 修复的同一 fork 中的上游调度策略，不是未经修改的上游程序。

## 正确性结果与修复

1. 首轮发现 Transformers 4.55.2 的配置只有 `torch_dtype`，原 runner 读取 `dtype` 后初始化失败。
   已兼容两种字段，并在模型构建失败时清理 NCCL、恢复默认 dtype/device 和执行上下文。
2. BF16 strict greedy 测试：eager 与 CUDA Graph 两项均失败。没有删除断言、换提示词或忽略失败。
3. 固定历史诊断：96 次 token 对照、4 次 argmax 分歧，全部对应 native top-2 logits 并列。
   例如 token 9625/15344 在 native 中均为 16.625，混合批次中一个变为 16.75。
   最大绝对 logit 差 0.4375；最大相对 L2 差约 3.68%。这支持舍入影响的解释，不能单独排除所有实现问题。
4. 所有策略统一切换为显式 FP16 后：60 项 CPU 测试、3 项 GPU 测试通过。
   GPU 项含原样保留的 eager/graph greedy 对照与新增初始化失败清理回归。
5. 更长的合成负载中，FP16 仍观察到部分跨策略输出不同。因此“小样本 gate 通过”不等于
   全量输出一致、batch invariance 或模型回答质量通过；性能数据只能作为探索性调度测量。

BF16 默认仍为模型的原始 dtype；代码不会静默将 `auto` 改成 FP16。
数值差异背景见 [PyTorch 文档](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
和 [vLLM batch invariance 文档](https://docs.vllm.ai/en/latest/features/batch_invariance/)。

## 实验口径

mixed / burst，seed=7，各 24 请求；10 个配置：native、fixed 128/256/512/1024/2048、
固定 KV 预留、仅动态 KV、仅动态预算、联合自适应。每组独立进程加载、图捕获及预热后计时。
所有组使用 FP16、全局预算 2048、最多 16 个活跃序列、64 × 256 tokens 的 KV 池。
SLO 为 TTFT ≤ 1000 ms 且平均 TPOT ≤ 50 ms；goodput 按同时满足两项的请求数计算。
无网络服务开销，不评估语义质量。单种子、单次测量、24 个请求的 P99 均不具备统计代表性。

## 初步观察

混合负载中，联合自适应降低 ITL P99，但恶化 TTFT 和 goodput，且不优于最佳 fixed。
当前 20 ms step 目标和反馈规则未针对这张卡、模型和混合 attention 固定开销做建模。
该结果应驱动下一轮 profiling，而不是调整测试 SLO 或事后挑选参数制造优势。

建议下一轮先完善长 trace 的 teacher-forced 数值检验，再拆解 CPU 调度、输入准备、
Prefill/Decode attention 和 sampling 的时间。用独立 profiling 数据估计不可压缩的
Decode/混合批次开销，在 held-out trace 上比较预测型预算与最佳固定预算。

## 完整实测结果

20/20 组完成，运行退出码 0；审计确认所有组配置可比、每组 24 请求全部完成，
逐 token 时间戳和输出长度合法。18 个非 native 配置都存在 1–2/24 个请求的输出差异；
这些差异尚未逐一定位，不能把完整负载宣称为严格正确性通过。最长实测 step 为 48.53 ms。

| 场景 | 配置 | goodput (请求/s) | TTFT P99 (ms) | ITL P99 (ms) | 抢占 | 输出不同请求数 / 24 |
|---|---|---:|---:|---:|---:|---:|
| mixed | native | 15.933 | 252.824 | 94.042 | 1 | 0 |
| mixed | fixed-128 | 2.148 | 2568.744 | 30.491 | 0 | 1 |
| mixed | fixed-256 | 6.900 | 1338.476 | 31.817 | 0 | 2 |
| mixed | fixed-512 | 13.603 | 665.368 | 29.545 | 0 | 2 |
| mixed | fixed-1024 | 15.649 | 439.178 | 38.781 | 0 | 1 |
| mixed | fixed-2048 | 16.608 | 355.305 | 37.090 | 0 | 1 |
| mixed | fixed-kv | 13.774 | 646.120 | 29.462 | 0 | 2 |
| mixed | dynamic-kv-only | 13.657 | 659.098 | 30.705 | 0 | 2 |
| mixed | dynamic-budget-only | 9.886 | 1121.266 | 30.599 | 0 | 2 |
| mixed | combined | 10.047 | 1089.789 | 28.326 | 0 | 1 |
| burst | native | 14.574 | 180.297 | 4.779 | 0 | 0 |
| burst | fixed-128 | 2.949 | 2196.604 | 30.188 | 0 | 1 |
| burst | fixed-256 | 9.959 | 948.574 | 30.606 | 0 | 1 |
| burst | fixed-512 | 13.301 | 483.390 | 28.851 | 0 | 2 |
| burst | fixed-1024 | 14.251 | 245.015 | 29.964 | 0 | 1 |
| burst | fixed-2048 | 14.716 | 185.044 | 43.880 | 0 | 2 |
| burst | fixed-kv | 13.305 | 460.186 | 28.934 | 0 | 2 |
| burst | dynamic-kv-only | 13.395 | 443.593 | 27.551 | 0 | 2 |
| burst | dynamic-budget-only | 9.271 | 999.229 | 29.073 | 0 | 1 |
| burst | combined | 9.219 | 1012.784 | 30.742 | 0 | 1 |

结论：联合自适应在 mixed 中降低 ITL 尾延迟，但 goodput 下降约 37%；在 burst 中，
TTFT、ITL 与 goodput 均劣于 native。本轮两种负载按 goodput 排名的最佳 fixed 都是 2048，
联合自适应均没有胜出。不能据此写“自适应提高吞吐”或“整体加速”。

mixed 的联合控制器有 133 个 `latency_backoff`、103 个 `aged_request_floor` 决策，
表明当前控制器在延迟目标与排队推进之间反复取舍。除 native/mixed 的一次抢占外，其余组没有抢占，
因此也不足以证明动态 KV 预留在高压场景中的收益。

原始数据位于本地 `results/autodl-20261019/`：JUnit XML、完整测试日志、20 份请求/step JSON、
模型 manifest、环境版本、BF16 诊断、`source-tested.tar.gz`。历史失败归档单独保留。

复核命令：
```bash
python scripts/audit_results.py results/autodl-20261019/matrix --expected-reports 20
```
审计输出 `all_outputs_identical=false` 是已发现的限制，不会被自动转换成正确性通过。

## 面试表述

> 我把模拟调度迁移到了真实 RTX 3090 上，补上 dtype 版本兼容和初始化失败清理，
> 跑通 eager 与 CUDA Graph 的有限样本测试，再做同 trace 的 20 组固定预算扫描和消融。
> 实测发现，自适应控制改善了部分混合负载的 token 间尾延迟，但损失 goodput，突发负载更差；
> 完整负载还有数值一致性覆盖缺口。因此我把它定位为有真实实验和失败分析的研究原型，
> 下一步优先做数值回归、分阶段 profiling 与预测型预算控制，而不是宣称算法已经领先。
