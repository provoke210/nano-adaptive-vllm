# 来源与边界

- 上游：https://github.com/GeeeekExplorer/nano-vllm
- 固定基线：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`
- 原作者 Xingkai Yu；保留完整 Git 历史、原 MIT LICENSE 与作者署名。
- 原 README：[UPSTREAM_README.md](UPSTREAM_README.md)。
- 本地名称 nano-adaptive-vllm，Python import 仍为 nanovllm；不要在同一环境安装原版与此 fork。

研究参考：

1. [nano-vLLM PR #276](https://github.com/GeeeekExplorer/nano-vllm/pull/276)：混合调度、Decode 优先、固定 KV 水位；2026-10-18 查询为 Open。
2. [vLLM 调优文档](https://docs.vllm.ai/en/stable/configuration/optimization/)：Prefill 预算与 TTFT / ITL 权衡。
3. [NKNaN 实现](https://github.com/NKNaN/nano-vllm-chunked-prefill)：分块 Prefill、重算与延迟观测。

此 fork 的改动直接在固定上游源码上实现，没有把上述 PR 当成已合并的上游功能。
反馈控制与动态 KV 预留是待验证的工程策略，不声明新算法或已证明优于 vLLM。
2026-10-21 新增在线成本模型、等待/完成时间驱动的槽位准入与 Decode 轮转；这些机制仍按组件消融验证。
`layers/invariant.py` 是本地实现的固定分块 linear / chronological attention，不是直接移植 vLLM 的完整后端。
数值背景参考 [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
与 [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)。它用于本 fork 各调度策略的共同数学执行路径，
不能把更换后端导致的性能变化当作自适应调度收益。


Llama 本地适配参考：

- [Transformers v4.55.2 Llama](https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/llama/modeling_llama.py)：作为独立随机模型的架构与数值参考。
- [Transformers v4.55.2 RoPE](https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/modeling_rope_utils.py)：校验 default / linear / llama3 频率定义。
- `models/llama.py` 在本项目已有 packed Linear、RMSNorm、Attention 和 Qwen3 解码器组织方式上实现；没有把 HF 模型替换为自己的实现来充当参考真值。
- 本地测试仅替换 CUDA Attention 为显式 CPU 数学实现，并绕过编译包装与单卡分布式元数据；不是实际 Triton/FlashAttention 测试。
