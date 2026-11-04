# 调度设计

## 每一步

1. 优先给 running 请求安排一个 Decode token。容量不足时先回收未完成的 Prefill，再抢占未选中的 running 尾部请求。
2. 剩余硬预算为 max_num_batched_tokens - decode_count；配置要求全局预算不少于 max_num_seqs。
3. fixed 使用固定 Prefill 预算；adaptive 使用控制器预算，队首等待超阈值时施加初始预算下限。
4. 按 FCFS 为 waiting 请求分配块并安排 Prefill。沿用上游完整当前输入分配，**不是按 chunk 增量分配 KV**。
5. 默认采用固定归约顺序的 tiled paged attention；Decode query 长度为 1，位置与历史长度保持原义。关闭 batch_invariant 时走原 FlashAttention 路径。
6. 仅最后一个 Prefill chunk 和 Decode 行生成 token。中间 chunk 只推进已计算位置。
7. 记录 step 耗时、生成时间戳，更新控制器；完成请求释放引用，保留可回收 prefix cache。

## 第一版反馈策略（controller=aimd，保留作消融）

- target_step_ms 是 ITL 的代理，不能等同于服务 SLO。
- Prefill/mixed 样本更新 EWMA：0.3 × current + 0.7 × previous。
- 当前耗时或 EWMA 超过目标，预算减半至最小值。
- 连续 recovery_steps 个 Prefill/mixed 样本低于目标的 80%，预算增加一个最小预算单位。
- Decode-only 样本可触发缩小，不能证明增大 Prefill 的余量。
- 老化机制允许突破当前反馈预算，但不突破硬 token cap；其代价可能是 ITL 变差。
- 这不是基于 batch/context 特征拟合的预测控制，历史 EWMA 在负载切换时可能滞后。

## 在线成本控制器（controller=cost）

`cost_model.py` 用最近 32 个 Prefill/mixed 样本拟合 `step_ms = intercept + token_ms × prefill_tokens`。
至少 4 个样本且 token 数存在足够变化时才拟合，避免从同一形状误推边际成本。
Decode-only 使用单独 EWMA，不混入 Prefill 的回归。预算为
`(target_ms - intercept - 2 × median_absolute_residual) / token_ms`，再裁剪至配置范围。
这里的残差余量是启发式，不能当作置信区间或 SLO 保证。

旧策略的失败原因之一是：目标 20ms 低于实测 Prefill 固定开销约 28ms，减小 chunk 后仍超时，
预算不断缩小，反而增加执行次数。新策略识别不可达目标后采用最大允许预算摊薄固定开销；
尚未识别模型时探测更大 chunk。它显式选择吞吐进展，不能满足这个不可达的延迟目标。

可选 `coalesce_ms` 仅在仍有 Decode 可执行、无部分 Prefill、队列 token 少于 `coalesce_tokens`
时短暂延后 Prefill，让 Decode 继续推进；达到等待上限或老化阈值就恢复准入。
同样机制也可用于固定预算，实验必须保留该对照。所有策略都保留硬 token / KV 容量约束。
线性模型暂未显式建模上下文长度、Decode 数或 GPU 频率，形状变化可能导致误估。

## 满批次时的等待时间驱动准入

`admission_wait_ms > 0` 时，队首等待达到阈值且 KV 检查允许进入，就为 Prefill 保留一个执行槽位，
当步 Decode 配额由 `max_num_seqs` 降为 `max_num_seqs - 1`。仍然最多执行同样多的序列和 token。
单槽位配置不减少 Decode 配额，避免零进展。

这把“驻留请求数量”和“当步执行数量”分开：持有 KV 的 running 请求可以超过当步序列上限，
受物理 KV 容量约束。已执行 Decode 的请求移至 running 队尾，使本步跳过的请求下次优先，
避免旧的固定队首选择使新增请求一直无 Decode 机会。满批次时不再必须等一整段长生成结束才能产生新请求的首 token。
代价是每个请求未必每步都 Decode，ITL 和抢占风险可能增加，因此同时报告 TTFT、ITL 与重算。

槽位准入独立于 token 预算控制器，可与 fixed 组合。实验应扫描 `fixed-{128,256,512,1024,2048}-admit`
作为更强消融；相对没有准入改进的 fixed 的收益不能全部归给成本回归模型。

第四版的可选 `admission_target_ms` 增加完成时间预测：单独对 Decode-only 耗时做 EWMA，
用 running 中最早达到 max_tokens 的剩余步数乘估计步长，并按驻留数 / 执行槽位数修正轮转影响。
仅当“已等待 + 预计自然释放槽位时间”超过目标，或已经等待超目标，才介入。
`admission_max_slots` 将预留槽位数限定在等待请求数、配置上限和 `max_num_seqs - 1` 的最小值。
这样既保留 Decode 进展，也能把几个 Prefill 合入一次前向。

预测使用调用者的 max_tokens 上限，**并不知道实际 EOS**；输出提前结束可能使预测偏保守。
它也没有为每个请求拟合上下文成本，不能保证 TTFT。固定 token 预算加相同预测准入的 `fixed-*-aware`
对照用来区分“动态槽位准入”和“动态 token 预算”各自贡献。冻结参数见 [ROUND2_FROZEN.json](ROUND2_FROZEN.json)。

配置中的 `mode='fixed'` 只表示 Prefill **token 预算**固定，不会关闭独立的动态准入机制。
因此 `fixed-2048-aware` 实际是“固定 token 上限 + 自适应槽位准入”，并非完全静态的调度器。
必须在报告里写清基线：普通 fixed、固定预算加准入、动态预算加准入是不同层次的对照。

## 数值执行与调度分离

`layers/invariant.py` 固定 GEMM / attention 的归约块与累加顺序，LM head 保持 FP32 输出。
这消除了本次长负载回归中的批次形状敏感性，代价是增加自定义 kernel 的维护和性能验证工作。
它是所有策略共享的执行后端，不能把相对 FlashAttention 的变化计入自适应调度收益。
详细诊断、失败消融和有限验证范围见 [数值正确性](NUMERICAL_CORRECTNESS.md)。

## KV 预留

对每个已持有 KV 的请求，估计未来 decode_lookahead 个生成位置所需的新 block，
受 max_tokens 剩余额度约束。预留为增量之和加固定余量。
准入判断考虑已使用的共享前缀；软预留不替代硬容量检查，不强制驱逐已运行请求。
完全空闲时，允许一个可独立完成的队首请求越过软预留，避免永久空转。

入队检查 prompt + max_tokens 不超过模型长度，且独占时 KV 能装入池。
最后一个输出 token 不再进入模型，KV 上界按 prompt + max_tokens - 1 计算。

## 正确性

- 批次总 token / 序列数遵守硬预算。
- 物理块引用计数等于活跃页表引用数；free/used 互斥并覆盖池。
- 每个序列只读取已写入的历史 KV；位置、slot mapping、输出行一一对应。
- 抢占不丢失生成历史；重算历史不产生额外输出。
- Prefix cache 只复用已计算并登记 hash 的完整块，保留上游末块重算策略。
- 中间 Prefill chunk 不采样，不处理 EOS。
- cancel 仅在 step 之间调用；不是线程安全服务。

CPU oracle 存储物理 slot 的 token 并验证缓存历史，不执行 transformer。
attention 数值正确性必须通过真实模型测试验证。

## 当前范围

单卡、Qwen3 稠密文本、同步 engine、单 engine/进程。继承类级 block_size 与 NCCL 初始化方式。
TP 主动禁用；没有 HTTP、输出长度预测、KV swap、MoE 或 PD 分离。
过载时不承诺有限等待或 SLO 保证。
