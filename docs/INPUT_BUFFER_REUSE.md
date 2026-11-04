# 分阶段统计与 Decode 输入缓冲区复用

这轮优化针对 Python 输入构造和 CUDA Graph 入参拷贝开销，调度策略保持不变。
灵感来自 DeepInfer 的输入张量复用；本项目自行实现，没有移植其分布式执行器。

## 实现

`Config.profile_stages` 和 `Config.reuse_decode_buffers` 均默认关闭，可以独立开启。

```python
llm = LLM(model_path, profile_stages=True, reuse_decode_buffers=True)
# 正常提交请求并调用 step / generate。
records = llm.model_runner.profiler.collect(wait=True)
```

`profiling.py` 在 runner 的输入准备、模型执行（含 logits）、采样、结果回传等待四个阶段记录 CPU 墙钟和 CUDA events；
`LLMEngine.step` 另记录 schedule、bookkeeping、runner、postprocess 的 CPU 墙钟。
事件在提交后延后查询，最终 collect(wait=True) 才等待最后一个事件；不为每个阶段加入同步。
CUDA event 间隔包含流内等待与 CPU 发射不及时导致的空闲，不等于纯 kernel 执行耗时；与 CPU 阶段重叠，不能相加。
性能 A/B 的正式计时关闭 profiling，阶段样本仅用于定位。

`input_buffers.py` 预分配 pinned CPU / GPU 的连续 byte slab，再按 8 字节对齐划分六个连续、互不重叠的 typed tensor 视图。CPU 侧用 NumPy 视图填入 token、position、slot、context length、block table 和 temperature。
纯 Decode 每步按当前 batch 和 block-table 宽度填值，再一次 H2D 上传整个 slab；CUDA Graph 直接绑定 GPU slab 视图，跳过原先额外的 D2D 搬运。
完整上传以少量额外字节换更少的调用：测试的 max_seqs=64、max_model_len=4096、block_size=256 时共 5,888 字节；极大容量时需要重新评估这一取舍。
Graph 的无效行设置 slot=-1、context length=0；旧 block-table 列只允许位于当前上下文范围之外。
上传函数运行在 inference mode，兼容 graph capture 时创建的 inference tensors。

CPU 在 sampler.tolist() 完成后才重新填充主机缓冲区。异常路径保留 pending_stream 标记，下次填充前等待旧 H2D，防止重写仍在传输的 pinned 内存。
因此当前实现仍是同步 Decode，不是异步流水或 GPU token 回灌。Prefill / mixed 路径保持现有逻辑。
输入池对 A/B 两组均在 KV 容量计算前分配，避免可用显存差异影响比较。

## 验证方式

本地测试覆盖缓冲区地址稳定、batch 缩小、ragged block table、graph padding、inference tensor、异常回收与事件延后收集。
真实模型门禁覆盖 eager / CUDA Graph × 复用开关，并比较 native / fixed / adaptive 的 greedy 输出。

```bash
NANO_MODEL=/path/to/Qwen3-0.6B NANO_DTYPE=float16 python -m pytest -q
python -m benchmarks.buffer_compare --model /path/to/Qwen3-0.6B \
  --suite profile --batches 1 16 64 --contexts 32 1024 --decode-steps 48 \
  --output-dir results/buffer/profile
python -m benchmarks.buffer_compare --model /path/to/Qwen3-0.6B \
  --suite micro --repeats 5 --output-dir results/buffer/micro
python -m benchmarks.buffer_compare --model /path/to/Qwen3-0.6B \
  --suite e2e --repeats 3 --blocks 128 --output-dir results/buffer/e2e
python scripts/audit_buffers.py results/buffer
```

- 单进程同模型、同 KV 容量，按预先固定的随机种子打乱 A/B 顺序；每组都预热。
- micro：batch=1/4/16/32/64，初始 context=32/1024，每组 5 对重复，每次计时 96 Decode 步。
- e2e：mixed / burst / pressure × seeds 101/307，每组 3 对重复、24 请求；固定 token budget=2048 加已有自适应槽位准入。
- 每组保存原始输出、trace/source hash、环境及指标；输出差异直接终止该 suite。
- micro 检查计时区间内完整输出；e2e 检查每个请求全部输出。记录所有 case，包含回退。
- e2e 调度参数一致，但真实运行速度变化会影响到达时刻落在哪一步与槽位反馈，因此端到端变化还包含这一效应。

已覆盖单卡 Qwen3-0.6B / 4B 与 Llama-3.2-1B 的 FP16 CUDA 正确性和同配置输入复用 A/B，见 [Llama 验证](LLAMA_GPU_VALIDATION.md)。4B 的显存、收益与低 SLO 达标率见 [4B 报告](QWEN4B_GPU_VALIDATION.md)；更大模型、BF16 和真实服务流量仍待验证。

## 与调度优化的关系

该开关不改变 token 预算、准入阈值或 KV 调度逻辑。它优化的是执行器把调度结果送入 GPU 的成本。
因此只能将同一调度配置下的复用开关 A/B 归因于这一改动，不能把结果解释为动态预算重新优于强固定基线，也不能与历史 goodput 增幅直接相加。
实际 GPU 数字、失败版本与适用范围见 [本轮验证报告](BUFFER_GPU_VALIDATION.md)。
