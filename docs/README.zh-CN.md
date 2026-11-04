# nano-adaptive-vllm

基于 nano-vLLM 的**延迟反馈混合调度与 KV 压力准入实验**，面向推理引擎 / AI Infra 项目实践。

**当前代码：已接入 dense Llama 与 Qwen3；Mac 本地 127 项通过、7 项 CUDA 测试跳过；RTX 3090 上 Qwen3-4B 的 134 项测试全通过。**
支持默认、linear、llama3 RoPE，详情与测试边界见 [Llama 本地适配](LLAMA_SUPPORT.md)。
Qwen3-0.6B、Llama-3.2-1B 已通过 CUDA 正确性与输入复用 A/B，4B 退出修复后两者各 7 项 GPU 回归通过；跨模型动态预算/准入收益尚未验证。以下调度收益数字属于第二轮冻结的 Qwen3 版本，当前输入复用优化另做 A/B 验证。

**历史 GPU 验证：RTX 3090 + Qwen3-0.6B / FP16，78 项通过（73 CPU + 5 GPU）。**
长负载输出差异已定位到批次形状相关的数值归约与低精度 logits 舍入。
固定归约 linear / attention + FP32 logits 的自由生成回归，在 3 个种子、5 种策略、360 次请求执行中全部逐 token 一致。
仅 FP32 输出投影仍失败，失败消融与独立数学参考保留在 [数值正确性报告](NUMERICAL_CORRECTNESS.md)。
该结论限于已测模型、设备、版本及负载；新路径尚未完成 BF16 长负载验收。
第二轮已完成：336 次独立评估/容量复测中输出全部相同。长输出负载上，固定 token 上限 + 自适应槽位准入的 goodput 相对普通最佳固定预算提高26%–77%；范围依赖KV容量。
收益主要来自准入和Decode轮转；动态token预算未胜过带同样准入的强固定基线，不能声称普遍加速。详见 [第二轮报告](AUTODL_ROUND2.md)。
第一轮历史数据见 [2026-10-19 实测报告](AUTODL_VALIDATION.md)。
Mac 上运行同一套 Scheduler、Sequence、BlockManager，配合 token-cache oracle 与合成时间模型。
这能验证缓存生命周期和调度状态，不能证明 attention 数值正确或预测 GPU 性能。

## 研究问题

长短请求混合或突发到达时，固定 Prefill 预算难以兼顾首 token 延迟、token 间延迟和缓存压力。
本项目验证：根据实测 step 延迟调整 Prefill 预算，并预留活跃请求近期增长所需的 KV blocks，
能否提高满足延迟目标的请求吞吐（goodput）？

第一版是有界反馈控制器；第二版增加在线固定开销 / token 成本拟合，以及满 Decode 批次时按队列等待时间预留 Prefill 槽位。
预算、槽位准入、KV 预留均可独立开启，保留固定预算加相同准入机制的强消融对照。
降低抢占但恶化排队 / goodput 的结果也必须保留。动态调度本身不是原创性声明。

## 已实现

| 能力 | 实现及验证边界 |
|---|---|
| 模型 | 显式分发 dense Qwen3 / Llama；分片 safetensors、共享词嵌入、多 EOS；Qwen3-0.6B / 4B、Llama-3.2-1B 已完成 FP16 CUDA 正确性与输入复用 A/B |
| native | 保留上游 Prefill 优先、Prefill/Decode 分开的调度策略 |
| fixed | Decode 优先 + 固定 Prefill 预算 + 单次 varlen 混合前向 |
| adaptive | 旧 AIMD 反馈与新在线成本模型并存；识别不可达目标，保留老化补偿 |
| 槽位准入 | 队首等待达到阈值时预留一个 Prefill 槽位；Decode 轮转，硬执行上限不变 |
| KV 准入 | none / fixed / adaptive；估计已持有 KV 的请求近期 block 增长 |
| 缓存 | 分页 KV、prefix hash、抢占重算、取消、EOS 与完成回收 |
| 采样 | 中间 Prefill chunk 不投影 logits、不采样；温度 0 支持 greedy |
| CUDA | 默认固定归约 tiled paged attention；纯 Decode 保留 CUDA Graph。暂限定单卡 |
| 输入复用 | 可选纯 Decode pinned CPU / GPU 输入池；CUDA Graph 直接写固定输入；Prefill / mixed 保持原路径 |
| 分阶段统计 | 可选 CPU 阶段计时与延后收集的 CUDA events；性能 A/B 关闭统计 |
| 实验 | 同 trace 重放、固定预算扫描、消融、逐请求和逐 step 原始 JSON |
| 追溯 | 上游 commit、代码 hash、trace hash、参数、包版本及 GPU 环境 |

native 是**同一 fork 中的上游调度策略**，不是未经修改的上游二进制：各组共享输入校验、greedy、
固定归约后端、按需 logits、计时与 CUDA Graph batch-size 修复。严格原版比较需另跑固定上游 commit。

## Mac 快速开始

Python 3.10–3.12。当前目录已建立 Python 3.12 的 `.venv`。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q

.venv/bin/python -m benchmarks.workload --scenario burst --count 60 --seed 7 \
  --output results/burst.jsonl
.venv/bin/python -m benchmarks.run --backend sim --trace results/burst.jsonl \
  --policy adaptive --kv-mode adaptive --blocks 32 --output results/adaptive.json

# 10 个配置：原策略、5 个固定预算、固定水位、两种单独改进、组合策略
.venv/bin/python -m benchmarks.matrix --count 24 --seeds 7 19 \
  --scenarios mixed burst pressure --output-dir results/local-smoke
```

仿真报告 evidence 为 synthetic_simulation；汇总表明确标记。results/ 默认不进 Git。

Mac 上额外运行模型适配测试（随机小模型，无需下载模型权重）：

```bash
.venv/bin/python -m pip install -e '.[model-test]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

这是 PyTorch CPU 参考验证，不是 Mac 上运行 CUDA 引擎。实际 GPU 入口仍是 `LLM(local_model_path, ...)`，
根据本地 `config.json` 的 `model_type` 选择 `llama` 或 `qwen3`。

**输入复用实测：** RTX 3090 / Qwen3-0.6B，合并 H2D 后纯 Decode 吞吐 +0.60%–4.48%，端到端 +0.98%–3.52%（逐场景配对中位数）。
Llama-3.2-1B 同配置复测：纯 Decode +0.76%–3.08%，端到端 +0.90%–2.01%；独立 HF 对照 40 个位置的 greedy token 全相同。
首次仅复用独立张量的版本无稳定收益；完整结果和尾延迟回退见 [Qwen3 报告](BUFFER_GPU_VALIDATION.md) 与 [Llama 报告](LLAMA_GPU_VALIDATION.md)。

**Qwen3-4B 扩展验证：** 24GiB 3090 可运行；纯 Decode 最大观察显存 14.86GiB、端到端 12.62GiB。吞吐分别 +0.44%–1.21%、+0.45%–0.70%，但沿用原参数的 SLO 达标率仅 0%–45.83%，未改善。验证中修复了重复创建引擎时权重循环引用导致的显存累积。详见 [4B 报告](QWEN4B_GPU_VALIDATION.md)。

分阶段统计和纯 Decode 输入复用的用法、同步边界与 A/B 方法见 [输入缓冲区复用](INPUT_BUFFER_REUSE.md)。

## GPU 实验配置

冻结第四版的“固定 token 上限 + 自适应槽位准入”可显式配置为：

```python
from nanovllm import LLM
from nanovllm.engine.policy import SchedulingOptions

llm = LLM(
    "models/Qwen3-0.6B", dtype="float16", batch_invariant=True,
    max_num_seqs=16, max_num_batched_tokens=2048, kv_cache_block_cap=64,
    scheduling=SchedulingOptions(
        mode="fixed", prefill_budget=2048,
        admission_wait_ms=100, admission_target_ms=800, admission_max_slots=4,
    ),
)
```

这里 `mode` 只控制 token 预算，准入仍按排队时间和预计槽位释放时间动态决定。
动态 token 成本控制可另外设 `mode="adaptive", controller="cost", target_step_ms=50`。
不要假设两者叠加必然更快，按同准入的强消融结果选择。
性能指标、适用范围和失败结果见 [第二轮报告](AUTODL_ROUND2.md)；
面试讲述见 [INTERVIEW.md](INTERVIEW.md)。

## 结构

```text
nanovllm/engine/
  scheduler.py             上游策略 + 通用校验、计时和取消
  adaptive_scheduler.py    混合调度、KV 预留
  policy.py                无 GPU 依赖的反馈控制器
  cost_model.py            在线固定开销 / token 增量成本回归
  model_runner.py          真实 CUDA 执行、dtype 兼容和失败清理
  input_buffers.py         纯 Decode 输入池与异常后的 H2D 生命周期保护
  profiling.py             CPU 阶段计时 / 延后收集 CUDA events
benchmarks/
  simulator.py             cache oracle + 合成执行成本
  workload.py              工作负载与 JSONL 校验
  run.py                   仿真 / CUDA 的统一 trace 重放
  matrix.py                固定预算扫描与消融
  gpu_compare.py           单模型加载、随机配置顺序、多次重复与输出相等门禁
  buffer_compare.py        同进程随机顺序输入复用 A/B、阶段采样与输出门禁
  metrics.py               TTFT、TPOT、ITL、goodput
 tests/                    CPU 测试与 opt-in GPU gate
 docs/                     设计、来源、GPU 操作与验收计划
nanovllm/layers/invariant.py 固定归约 linear 与分页 attention
scripts/audit_repeated.py   复核完整性、同后端对照与最佳 fixed
```

见 [设计](DESIGN.md)、[GPU 操作](GPU_RUNBOOK.md)、[实验协议](EXPERIMENTS.md)、
[本地验收](VALIDATION.md)、[AutoDL 实测](AUTODL_VALIDATION.md) 与 [来源](PROVENANCE.md)。
