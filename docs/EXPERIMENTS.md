# 实验协议与迭代

## 证据等级

1. CPU tests：调度与缓存正确性。
2. synthetic_simulation：实验工具和压力场景可执行，不能当硬件性能证据。
3. CUDA correctness：真实模型、混合批次与图执行验收。
4. cuda_measurement：配对硬件实验，才可用于性能简历数据。

## 对照

保持模型、dtype、图设置、max_num_seqs、全局 token 预算、KV block cap 相同：

- native；
- fixed，预算 128 / 256 / 512 / 1024 / 2048；
- fixed + 固定 KV；fixed + 动态 KV；
- adaptive + 无预留；adaptive + 动态 KV。

正式实验还应扫描固定水位与 lookahead，并对 aging 做开关消融。
在调优种子选参数，冻结到独立评估负载；不在测试集事后挑参。
最佳 fixed 扫描结果是必要强基线。每个 seed 是独立负载，不等同于同负载的硬件重复运行。

## 工作负载和计时

short / mixed / burst / pressure 是合成 token 负载，只支持形状/调度实验，不支持回答质量结论。
后续加入真实文本与固定公开请求集。
JSONL：唯一字符串 id、非递减 arrival_ms、非负 token_ids、正整数 max_tokens。

开环重放按原定到达时间，在 engine 两步间入队；延迟包含等待当前 step 完成的时间。
不包含网络、tokenization 或 HTTP 前端。
GPU 图捕获与专门预热不计入测量；新 shape 仍可能触发编译，需要 profiler 检查。

## 指标

- TTFT = 首次生成 - 原定到达。
- TPOT = 首至末生成耗时 / (输出长度 - 1)；单 token 定义为 0。
- ITL = 相邻生成间隔；P99 在全部 token 间隔上计算。
- goodput = 同时满足预先设置的 TTFT 和平均 TPOT 阈值的完成请求数 / replay 耗时。
- 分母从 replay 时间 0 到最后完成，包含空闲到达间隔与尾部 drain。
- tokens/s 只计生成 token；另外记录抢占、实际重算 token、KV 用量与决策原因。

正式报告使用多个种子、多次重复和离散度；少量 smoke 请求的 P99 不具备统计说服力。
真实池容量和刻意缩小容量的压力实验分开报告。

## 否定条件

不优于调优后的 fixed；减少抢占却恶化 goodput/长请求等待；收益只出现在极小 KV 池；
CPU 开销抵消 GPU 收益；控制振荡或 aging 持续压过反馈。
这些结果全部保留，不能改 SLO、丢弃慢请求或放宽正确性阈值制造优势。

## 后续顺序

1. 本地首版：三种调度、反馈控制、KV 准入、缓存 oracle、矩阵。
2. GPU 验收：greedy、chunk 边界 logits/hidden-state、图开关、前缀共享、抢占与取消。
3. Profiling：分离 scheduler CPU、输入准备、Prefill、Decode、sampling，建立稳定基线。
4. 第二版：用独立 profiling 样本拟合 decode 数、上下文长度、prefill 数对应的延迟模型；
   对比当前反馈控制和最佳 fixed，计入预测误差和控制开销。
5. 可复现报告：原始数据、曲线、消融、失败案例。GPU 验收前不宣称性能收益。
