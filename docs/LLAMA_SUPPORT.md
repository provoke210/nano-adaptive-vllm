# Llama 适配与验证边界

状态：本地适配完成；2026-10-22 已完成 Llama-3.2-1B-Instruct / FP16 / RTX 3090 验证：133 项全通过，独立 HF 对照 40 个位置 greedy 一致，360 次跨调度请求执行输出一致。输入复用端到端吞吐提升 0.90%–2.01%，详见 [GPU 报告](LLAMA_GPU_VALIDATION.md)。
历史 Qwen3 第二轮调度收益仍见 `GPU_ROUND2.md`；不能把该历史收益直接算作 Llama 的结果。

## 支持范围

- HF 格式 dense text `model_type=llama`、`architectures=["LlamaForCausalLM"]`（architectures 可省略）。保留 dense Qwen3。
- Llama 的 MHA / GQA、显式 head_dim、可选 attention_bias / mlp_bias、共享或独立 LM head。
- 全维 RoPE：default、linear，以及 Llama 3.1/3.2 text 使用的 llama3 静态频率缩放。兼容旧 `type` 键。
- 单文件及带 `model.safetensors.index.json` 的分片浮点权重。严格核验 Q/K/V、gate/up 分片是否齐全、形状是否匹配。
- 按 generation_config.json → model config → tokenizer 的优先级读取 EOS，支持多个停止 token，保留 ignore_eos。
- 模型类型在配置阶段校验；未知架构、MoE、多模态、量化、部分维度 RoPE、滑窗、dynamic / YaRN / LongRoPE 显式拒绝。
- 不支持 GGUF、原始 Meta 格式或仅 pytorch_model.bin 的权重；需本地 HF safetensors 格式。
- 运行引擎仍限定单卡 CUDA，dtype 为 FP16/BF16。Llama-3.2-1B 的 FP16 CUDA 已验收；BF16 未完成 GPU 验收，Mac 只做 CPU 参考测试。

这是结构与配置适配范围，不是对所有挂名 Llama 的衍生模型或所有预训练权重的兼容承诺。

## 核心改动

1. `nanovllm/models/registry.py` 按 model_type 分发；`Config` 在初始化 GPU 前校验，`ModelRunner` 不再硬编码 Qwen3。
2. `models/llama.py` 使用已有 packed Linear、分页 Attention 接口与残差路径。Llama 明确没有 Q/K RMSNorm，不沿用 Qwen3 的 bias 判断启用它。
3. `layers/rotary_embedding.py` 增加 linear、llama3 静态缩放；缓存键包含缩放配置和设备。长上下文原始窗口参数不会被当前测试长度代替。
4. `utils/loader.py` 加入分片索引、缺失权重、形状、不认识的张量和冲突共享权重校验，避免 torch.empty 权重漏载后继续运行。共享 embedding/LM head 使用同一个 Parameter，兼容 HF 只保存一个别名。
5. `utils/generation.py` 与 Scheduler 处理多 EOS。tokenizer 在启动 runner 前加载，避免 tokenizer 初始化失败后遗留 GPU runner。
6. CUDA KV scatter 独立到 `layers/kv_cache.py`，Attention 延迟导入 CUDA 依赖；Mac 可以导入真实模型层。CPU 数学替代仅存在于测试，不添加隐式 CPU 生产后端。
7. RMSNorm 改为不原地覆盖输入。FP32 下 `.float()` 可能返回原张量，原地归一化会覆盖首层残差；新增输入/残差不变性测试，并用整模型 HF 对照检查。

## 首次本地适配证据（历史快照）

环境：macOS arm64，Python 3.12.14，PyTorch 2.8.0，Transformers 4.55.2，safetensors 0.8.0。
使用随机初始化的小模型，测试期间完全离线，无预训练权重或 tokenizer 下载。

```bash
.venv/bin/python -m pip install -e '.[model-test]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q \
  --junitxml=results/llama-local/pytest.xml
```

结果：**115 passed，5 skipped**。相对上一轮 73 个 CPU 测试，增加 42 项；5 个跳过项均为既有 CUDA 门禁。

覆盖：

- 从 HF 随机 Llama/Qwen3 保存实际 safetensors，再通过本项目 loader 加载；FP32 每个位置 logits 与独立 HF eager 前向比较，rtol=3e-5、atol=3e-6，并做多步 greedy 生成比较。
- MHA、GQA、非默认 head_dim、带 bias、共享 embedding、linear/llama3 RoPE；两组 FP16 Llama 参考检查（rtol=.01、atol=.003，不能等同于 GPU 精度界限）。
- 非连续物理页、非页边界 chunk、ragged 混合 Prefill/Decode 与后续 Decode，比较 HF 完整历史重算。
- RoPE 在位置 8191、8192、12001、16383 对照 HF，覆盖原始上下文边界与扩展位置。
- 丢失 K projection、权重形状错误、额外参数、缺失 shard、共享权重冲突均应报错。
- 多 EOS 的优先级、ignore_eos、三种调度的停止与 KV 回收；原调度测试全部保留。

测试仅替换 `Attention.forward` 为显式 CPU 数学参考，真实 packed Linear、模型层、RoPE、RMSNorm、权重加载和 logits 均参与运算；单卡 rank/world_size 固定为0/1，并绕过 torch.compile 包装。
**因此不覆盖实际 Triton、FlashAttention、CUDA Graph、GPU 输入准备与性能。** 同调度输出相等也不能替代独立 HF 数值检查。
CI 新增独立 model-reference job；最小 `.[test]` 安装仍能运行无 torch 的调度测试。

## GPU 验证进展与待办


1. 已完成：核对真实 Llama-3.2-1B 权重、config、tokenizer、EOS、FP16 dtype，保存文件 hash 和环境。
2. 已完成：Qwen3 与 Llama 使用同一核心源码，各 133 项测试通过。
3. 已完成：独立 HF 数值/greedy 参考、CUDA 测试与跨调度自由生成；本次自由生成没有发生抢占，不据此声称覆盖 Llama 高抢占压力。
4. 已完成同调度参数的输入复用 A/B。待做：Llama 普通固定预算扫描、固定预算加同样准入、动态预算加准入的完整性能消融，分别报告 TTFT、ITL、token/s 与 SLO goodput。

已有 GPU 跨调度门禁可直接复用，`NANO_MODEL` 指向 Llama 目录：

```bash
NANO_MODEL=/absolute/path/to/local-llama NANO_DTYPE=float16 \
  python -m pytest -q tests/test_gpu.py tests/test_invariant_gpu.py
```

该命令不包含独立 HF 预训练模型对照，也不是性能测量。当前可声明“两种模型上的输入复用收益已验证”，不能声明“Llama 调度 goodput 提升26%–77%”或不限定对象的“跨模型优化全面有效”。
