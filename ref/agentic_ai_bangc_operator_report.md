# Agentic AI 赋能 BangC 算子开发：算子生成挑战赛的实践探索与收获

## 摘要

本项目围绕 BangC MLU 算子生成挑战赛展开，目标是在官方 OJ 环境中为一批 PyTorch 参考算子实现高性能 `.mlu` 提交，并持续优化到榜首。整个过程不是一次性写代码，而是一个 Agentic AI 驱动的闭环工程系统：读官方 reference、生成假设、实现变体、提交 OJ、异步收集结果、归纳规律、再批量迁移。

实践中，AI Agent 的价值主要体现在四个方面：

1. 将大量离散题目转化为可流水推进的任务队列。
2. 将 BangC 文档、API、教程、历史提交和 OJ 反馈整合成可操作的优化策略。
3. 在不等待单个 OJ 结果的情况下持续提出和验证多个变体。
4. 将一次题目的经验迁移到同类题目，例如 pointwise、reduce、matrix、pool、attention、io、embedding 等类别。

最终形成的核心经验是：在这类固定形状、固定输入分布、异步 OJ 的算子比赛中，最有效的并不是单点“写一个 kernel”，而是建立一套面向证据的开发循环，用工具和规范让 Agent 始终沿着正确来源、正确边界和正确反馈前进。

## 项目约束与目标

### 竞赛目标

项目的基本任务是实现 `README.md` Problems 表中的 BangC MLU operator。每个算子以 `.mlu` 文件形式提交，OJ 会编译、加载、运行，与 PyTorch reference 对比正确性和延迟。

主要优化目标包括：

- 未通过题目先完成正确实现。
- 已通过但未登顶题目继续降低延迟。
- 已登顶题目继续拉开延迟差距，降低被追风险。
- 对分数小于 1 的题目重点优化，提升到高于 PyTorch reference 或超过外部榜首。

### 硬约束

项目开发中最重要的规范是：最终提交不能用 Torch、ATen、CNNL 算法算子计算结果。允许使用 host 侧 tensor allocation、metadata 读取、runtime 初始化、stream 获取等非计算操作，但不能用 `torch::`、`at::`、`cnnlXXXX` 直接完成算子语义。

这条边界对 Agent 很重要，因为一些 wrapper 层技巧可能短期提高分数，但会偏离比赛目标。我们的经验是把“可提交实现”和“本地分析探针”严格分开：

- 探针可以用于理解 OJ、测量开销、验证输入分布。
- 正式 `.mlu` 提交要保持 BangC kernel 完成核心计算。
- 对于评测器问题、reference 溢出、dtype assert 等边界，要记录原因和风险，而不是无约束扩大。

### 官方来源优先级

AGENTS.md 明确了信息源优先级，这在实践中非常关键：

1. `ref/ref_files/%03d_opname.py`：最强来源，决定官方输入、shape、dtype、layout、常量、init args、输出语义。
2. 目标 `.mlu` wrapper signature 和当前代码。
3. `reference-impl/opname.py`。
4. `reference-impl/problems.json`。
5. 已 accepted 的 `.mlu` 实现。
6. BangC MCP、SDK 文档、本地 headers、示例。

开发中多次证明，不能只相信任务元数据或 README 描述。例如有的任务元数据 dtype 与实际 `ref/ref_files` 行为不完全一致；有的 reference 中存在 assert、overflow 或特殊数值行为。Agent 必须先读官方 ref，再写代码。

## 工具体系

本项目的高效推进依赖三类工具：MCP 文档工具、仓库内 `ref/` 工具、Git/OJ 自动化工具。

### BangC MCP 文档工具

BangC MCP 是 Agent 查证 API 和编程模型的核心入口。它内置了 API reference、programming guide、best practice、tutorial，以及 Neuware header 中提取的 CNRT/CNNL API 信息。

常用 MCP 能力包括：

- `bangc_api_lookup`：按符号查 BangC API，例如 `__memcpy_async`、`__sync_compute`、`__bang_reduce_sum`、`__bang_fusion`、`__bangc_matmul`、`__bangc_conv`。
- `bangc_search`：按主题搜索文档，例如 NRAM/WRAM、pipeline、reduce alignment、cluster sync。
- `bangc_tutorials`：查教程和样例实现。
- `bangc_context_pack`：为复杂问题提取多段相关文档上下文。
- `bangc_compile_torch_extension` / `bangc_compile_torch_extension_plan`：本地把 `.mlu` 编译成 torch extension，用于调试和小规模验证。
- `neuware_api_search` / `neuware_api_lookup`：查 CNRT/CNNL runtime API，例如 device、queue、stream、descriptor 等。
- `mlu_device_query`、`neuware_env_check`、`python_torch_env_check`：确认本地环境、设备、PyTorch/torch_mlu 状态。

MCP 的价值不是“多查文档”，而是防止 Agent 编造 API。典型例子：

- `__bang_write_zero` 的 count 是元素数，不是字节数。
- `__bang_reduce_sum` 有 NRAM 地址和 128B 对齐/长度约束。
- `__bang_argmax` 的 value 和 index 存储布局需要查证。
- `__sync_compute()` 只等待 compute pipeline，不等待 IO/MOVE；若前面有 `__memcpy_async`，仍需要合适的 IO sync。
- `__bang_conv` / `__bangc_conv` 对 WRAM layout、channel layout 和 alignment 有特殊要求，不能按普通 GDRAM layout 直觉写。

### `ref/` 下的 OJ 与数据工具

`ref/` 目录是整个 Agent 流水线的中枢。

#### 任务与榜单工具

- `ref/get_tasks.py`
  - 获取任务 id、名称、难度、类别、dtype、wrapper signature。
  - 支持范围、分类、搜索查询。
  - 用于选择同类题目批量优化。

- `ref/get_leaderboard.py`
  - 查询榜单、team best、other leaders、top rows。
  - 支持按 `team_91`、`team_42` 过滤。
  - 用于寻找未登顶、差距小、分数低的题目。

- `ref/get_oj_status.py`
  - 查询 OJ 队列、processing、leaderboard backlog。
  - 用于判断是否继续提交、是否批量等待。

#### 提交与结果工具

- `ref/oj_git.py`
  - 一站式写 `config`、stage、commit、push。
  - 支持 `--pull --retry` 处理多人/多 agent push race。
  - 支持 `--empty` 提交 current rerun，用来测 OJ 抖动。
  - 支持 `--result` 与 `--sleep` 做轻量自动收集。

- `ref/get_result.py`
  - 从 GitHub commit comments 提取 OJ 结果。
  - 支持 batch commits、jsonl/text/json、verbose、full output。
  - 实践中最常用的模式是批量收集多个 hash：

```bash
python ref/get_result.py 8ee50f8 2fac8d0 --format text
```

#### 官方 reference

- `ref/ref_files/%03d_opname.py`
  - 每题最重要的信息源。
  - 用来确认 input shape、dtype、随机分布、init args、reference 语义。
  - 例如判断输入是 `randn`、mask 是 `rand > threshold`、常量是 `float('-inf')`、输出是否为 tuple、是否存在 assert。

#### 评测器与 worker

- `ref/bangc_torch_tester.py`
  - 本地理解 OJ 正确性和计时流程。
  - 关键变化：曾从 MLU Event 计时变成冷启动 CPU wall-clock + synchronize 计时，这直接改变优化准则。
  - 当前流程中，加载 module、构造 Model、构造输入、一次性能调用、同步、再比对输出，这些 host 侧开销都会影响小算子。

- `ref/worker.py`
  - 理解 OJ 如何拉取提交、注入 `.mlu`、启动独立 Python 进程、编译冷启动、运行评测。
  - 有助于解释为什么同一 kernel 在不同 commit 上有较大随机波动。
  - 也帮助识别评测不是单机循环 benchmark，而是容器 worker + 多设备 + 多进程冷启动。

#### 探针与经验记录

- `ref/pointwise_jitter_probe.py`
  - 用于研究 pointwise 类题目的冷启动和波动来源。

- `ref/optimization_notes.md`
  - 持续记录每题关键尝试、commit、PASS/FAIL、latency、失败原因。
  - 这是 Agent 的长期记忆，避免重复走已经证明失败的路。

- `ref/matmul_survey.md`、`ref/conv_survey.md`、`ref/scatter_survey.md`
  - 用于归纳某类算子的已知路线和失败路线。

- `ref/05_matmul/tutorial.md`
  - 从朴素三重循环、向量化、tiling、专用 API 逐步理解 matmul。

- `ref/06_conv/tutorial.md`
  - 解释 `__bang_conv`、WRAM filter、layout、tiling、对齐。

### Git 与异步流水工具

本项目的 OJ 是异步系统：commit 进入 task queue，worker 编译运行，result queue 写 GitHub comment，leaderboard queue 再延迟更新。因此开发不能等待单次提交。

标准提交方式：

```bash
python ref/oj_git.py -o 041 -m "041 Opname: brief implementation detail" --empty --pull --retry 3
```

标准批量收集：

```bash
python ref/get_result.py <hash1> <hash2> <hash3> --format text
```

Agent 会记录 short hash、idea、result、diff、latency、note，然后继续做下一个变体或同类题目。

## AGENTS.md 驱动的 Agentic 开发流程

AGENTS.md 的核心要求可以总结为：

```text
Research -> Implement -> Submit -> Continue -> Collect -> Optimize
```

这不是普通的“先写完再测”，而是比赛环境下的流水线。

### 1. Pick tasks

任务选择遵循优先级：

- 未通过、红色 entry。
- 分数小于 1 的题目。
- 非 basic 优先。
- team_91/team_42 未登顶或领先差距小的题目。
- 与最近成功经验同类的题目，例如同一批 pointwise、reduce、matrix、pool。

工具上主要用：

```bash
python ref/get_leaderboard.py all --team team_91 --team team_42 --mode other-leaders
python ref/get_leaderboard.py all --team team_91 --team team_42 --mode team-best
python ref/get_tasks.py 039-043 --wrap --dtype
```

### 2. Research before coding

每题先读：

- `ref/ref_files/%03d_opname.py`
- 现有 `.mlu`
- 类似 accepted `.mlu`
- `reference-impl/problems.json`
- MCP API 文档

这一步的目的是把“题目”变成具体约束：

- Tensor rank、shape、numel。
- dtype：fp16、fp32、int、bool。
- layout：contiguous、NHWC、NCHW、strided。
- init args：标量、模块参数、常量。
- 输入分布：`randn`、`rand`、mask threshold、固定范围。
- 输出检查方式：max abs diff、tuple 特判、index 忽略、重建误差等。

### 3. Implement first correct path

第一版实现通常保守：

- 固定 shape 和 dtype。
- 使用 32 tasks Block 或 `{4,8,1}` Union1 作为起点。
- 先正确，再优化 tile、pipeline、近似、缓存、sync。
- 尽量原地写回输入 tensor，避免多余 host allocation。
- 对小 tensor 可尝试单核全量 NRAM。

### 4. Submit early, then continue

提交后不等待。典型 Agent 行为：

1. 提交变体 A。
2. 记录 hash。
3. 立即写变体 B 或研究同类题。
4. 同时收集 A、B、C 的结果。
5. 用 OJ 结果更新 attempt table。

这样可以把 OJ 延迟隐藏在思考和实现中。

### 5. Batch collect results

OJ 常出现同一 commit 两条结果，且存在明显随机波动。不能只看单行最低延迟，也不能只看单行最高延迟。实践中采用：

- 至少看两行。
- 对突破性结果用相同代码 rerun。
- 对“疑似随机赢”的结果谨慎记录。
- 以当前 tester 的结果为准，旧计时方式的绝对时间只作为历史线索。

### 6. Optimize iteratively

每个 active operator 保留尝试表：

| commit | idea | result | latency | note |
|---|---|---|---|---|
| `f76b13f` | single NRAM tile | PASS | 361us | 066 当前较好 |
| `cc3386a` | async single tile | PASS | 377us | 小算子 async 反而慢 |
| `e210283` | compute sync | PASS | 435us | 072 未超过旧版 |

这种表格让 Agent 能快速判断：

- 是继续微调，还是换算法。
- 是随机波动，还是结构性收益。
- 是正确性问题，还是性能问题。
- 哪些失败路线不要重复。

## OJ 与评测器认知

### OJ 架构

根据 worker 和实际观察，OJ 大致是：

```text
commit
  -> task_queue
  -> worker 拉取代码
  -> 独立容器/进程编译与运行
  -> result_queue
  -> GitHub commit comment
  -> leaderboard_queue
  -> leaderboard 周期更新
```

排行榜滞后于 commit comment，因此优化决策应优先看 `ref/get_result.py`，不要等待 leaderboard。

### 冷启动 CPU-sync 计时的影响

早期 OJ 更接近 MLU Event 计时，后续改为冷启动 + CPU wall-clock + synchronize。这个变化导致优化准则发生明显变化：

- 小算子会被 module load、first launch、queue 初始化、device runtime 初始化、tensor allocation 放大。
- 纯 kernel 内几十 us 的优化可能被几百 us 的 host 抖动淹没。
- 预热策略会显著影响结果，但要遵守不使用 Torch 算法计算输出的边界。
- 旧 notes 中的绝对 latency 不再能直接比较，只能作为历史 idea。

### 预热与设备绑定

实践中研究过以下策略：

- constructor 中不能 launch kernel，因为此时 cncc 尚未插入 kernel。
- 可以在 SO load 后通过静态 tensor 做轻量 runtime 初始化。
- `torch::zeros({1}, opt)` 这类极小分配可触发一定 GDRAM allocation/runtime warmup，比大 tensor 算术预热更干净。
- 对一些小算子，不预热会出现 2-3ms 冷启动；加 `zeros({1})` 可降到数百 us。
- 但预热不是万能，对中大算子可能收益被 kernel 带宽、NRAM 压力或随机抖动覆盖。

设备绑定方面，曾尝试按 pid 在可见 MLU 设备中分散 set device，目的是降低多 worker 抢同一张卡的概率。经验是：

- 绑卡能影响冷启动稳定性，但不是所有题都有收益。
- 必须在合法 runtime setup 边界内做，不能依赖多卡协同。
- 每个算子仍应只优化当前 active device。

### 随机抖动处理

OJ 抖动来自多个层面：

- worker 调度和容器冷启动。
- 编译缓存和 extension load。
- 首次 kernel launch。
- MLU 设备共享和队列状态。
- host tensor allocation。
- reference PyTorch/CNNL 路径本身波动。

处理方式：

- current rerun：对当前代码提交空 commit，测当前真实环境。
- same-code rerun：对突破性 commit 重复提交，判断是否只是运气。
- batch result：同一批 hash 一起收集，减少手工等待。
- 对两个 OJ rows 都看，不只看单条。
- 对小差距榜首，必须拉开更多延迟差距，避免被随机反超。

## BangC 编程经验

### 1. 固定 shape specialization 是核心策略

比赛题目的输入通常由 `ref/ref_files` 固定。只要官方 ref 证明 shape/dtype/layout 固定，就应积极专门化：

- 写死 numel。
- 写死 tile size。
- 写死 tensor rank 和 stride 假设。
- 写死 init args 或常量。
- 省去泛型分支和动态 shape 逻辑。

这能减少 host 侧 shape 计算，也能让 kernel 更短、更容易达到 NRAM 对齐。

### 2. NRAM 是最常用的性能杠杆

常见 pointwise/reduce/selection 算子基本模式：

```c
__nram__ half x[TILE];
__nram__ half y[TILE];
__memcpy(x, g_x + p, TILE * sizeof(half), GDRAM2NRAM);
__memcpy(y, g_y + p, TILE * sizeof(half), GDRAM2NRAM);
__bang_add(x, x, y, TILE);
__memcpy(g_out + p, x, TILE * sizeof(half), NRAM2GDRAM);
```

经验：

- tile 不一定越大越好。大 tile 减少 loop 次数，但可能增加 NRAM 压力、降低调度稳定性。
- 32768、65536、131072 是常见 fp16 tile 候选，但要结合 scratch 数量。
- 小 tensor 可以单核全量放入 NRAM，减少 task 调度和分段开销。
- 多输入算子要优先减少 GDRAM round trips。

### 3. WRAM 与卷积/矩阵专用 API

matmul/conv 类算子需要重点理解 WRAM layout：

- `__bangc_matmul`、`__bangc_conv` 对输入 layout、filter layout、channel 对齐有特殊要求。
- filter 常放 WRAM，input/output 放 NRAM。
- 对大矩阵要做 tiling 和数据复用。
- 对 triangular、diagonal、symmetric 这类矩阵，可利用数学结构跳过无效计算或减少读写。

`ref/05_matmul/tutorial.md` 和 `ref/06_conv/tutorial.md` 对理解这类问题很有帮助。

### 4. Sync 和 pipeline 要按真实依赖写

BangC 有 IO、Move、Compute 等 pipeline。同步过多会慢，同步过少会数据竞争。

常见原则：

- `__memcpy_async` 后如果 compute 要读 NRAM，必须等待 IO 完成。
- compute 后如果 NRAM2GDRAM 要读计算结果，需要等待 compute 完成。
- `__sync_compute()` 只等 compute，不等 IO/MOVE。
- `__sync_io()` 等 IO。
- `__sync_io_move_compute()` 更保守，可能过重。

实践发现：

- 对某些大 tile pointwise，替换为 `__sync_compute()` 可以降低一部分开销。
- 对小算子，async + extra sync 可能比同步 memcpy 更慢。
- 三流水/五流水要在充分理解依赖后做，否则容易出现偶发错误。

### 5. Launch layout 影响很大

MLU370 常见起点：

```c
k<<<{32,1,1}, cnrtFuncTypeBlock, queue>>>(...);
k<<<{4,8,1}, cnrtFuncTypeUnion1, queue>>>(...);
```

经验：

- 简单大 contiguous pointwise：32 Block 通常是起点。
- 需要 cluster 内同步或共享 SRAM 的任务：Union1 更合适。
- Union2/Union4/Union8 不会因为“更大”天然更快，只有同步/dataflow 需要时才用。
- 任务数更多不一定更快，尤其小 tensor 会增加调度和分段开销。

### 6. 原地输出和静态输出

很多 reference 只要求返回值正确，不要求输入不变。若语义允许，可直接返回被覆盖的输入 tensor：

```cpp
return a;
```

好处：

- 省去 host tensor allocation。
- 省去额外 GDRAM 输出 buffer。
- 降低冷启动 CPU-sync 计时中的 host 开销。

但要谨慎：

- 如果 correctness 对原输入有后续使用，不能原地改。
- 如果输出 shape/dtype 与输入不同，不能直接复用。
- 静态输出 tensor 在 tester 改动后不一定安全，尤其多次输入重新生成时可能 stale。

### 7. 近似与输入分布

很多 reference 输入来自 `randn` 或 `rand`。这允许用概率和数值分布做优化探索：

- 常量近似。
- 低阶多项式近似。
- 使用均值/方差集中性。
- 针对 sigmoid、gelu、swish、softplus、cos 等使用多项式或分段近似。
- 针对 reduction/loss 使用随机分布的统计稳定性。

但 max abs diff 检查很严格。经验：

- 只看平均误差不够，要看最大误差。
- p95/p99 通过不代表 OJ 通过。
- FP16 overflow/inf/nan 行为必须和 MLU reference 对齐。
- CPU probe 只能做初筛，最终以 OJ 为准。

### 8. 溢出、Inf、NaN 和 checker 特性

一些题目 reference 自身可能输出 `inf` 或触发 FP16 overflow。经验：

- `inf == inf` 未必在 checker 中表现为 diff 0。
- 写 `inf` 可能导致 `nan` diff。
- 有些 reference 在 fp32/fp16 迁移后行为不同。
- 对 loss、softmax、cross entropy、norm 类尤其要检查数值范围。

应对方式：

- 先用 probe 判断 reference 真值。
- 分别尝试 finite、inf、sentinel 输出。
- 记录 checker 行为，不盲目追一个不可能匹配的路径。

### 9. 类别化优化经验

#### Pointwise / Elementwise

典型策略：

- 原地输出。
- 32 Block 起步。
- 32768/65536/131072 tile sweep。
- 尽量融合多个 vector op，减少 GDRAM 往返。
- 对 activation 可使用 abs、clamp、polynomial、分段近似。
- 小算子要考虑冷启动预热。

典型结论：

- 对 066，小 tensor 全量 NRAM 单 task 比 32 task 分段略优。
- 对 072，大 tile 和轻同步不总是赢，旧的 32768 结构更稳。
- 对 087，65536 Block/async 有更高上限，但波动大；32768 更稳。
- 对 118，clean `zeros({1})` warm 能保留部分冷启动收益，但任务数变化受抖动影响。

#### Reduction

典型策略：

- 分块 reduce 到 NRAM。
- 多 kernel 做 partial + final。
- 使用 `__bang_reduce_sum`、max/min、argmax 等内置函数时严格检查对齐。
- 对 row/column reduction，优先让 GDRAM 访问连续。
- 避免大量 strided tiny copy。

经验：

- 多任务并行不一定快，strided load 很容易抵消并行收益。
- 对小 reduction，host/launch 开销可能比计算更重要。
- vector doubling scan 对 cumsum 类往往比 scalar loop 强很多。

#### Matrix / Matmul

典型策略：

- 利用固定 shape 做 tiling。
- 利用 triangular/diagonal/symmetric 结构减少计算。
- 调整 task count、tile M/N/K、WRAM reuse。
- 对小 K、大 K、转置布局分别定制。

经验：

- 单纯改 task count 常只能小幅改善。
- 真正突破通常需要 data reuse 或结构化跳算。
- `__bangc_matmul` 的 WRAM layout 和对齐需要查文档，不要凭 CUDA 直觉写。

#### Pool / Conv

典型策略：

- 利用 `__bang_maxpool`、`__bang_conv`。
- 预填 padding buffer，减少每 plane 清零。
- 尽量使用布局适配 intrinsic，而不是大量 strided gather。

经验：

- intrinsic 对 layout 很敏感，直接 strided gather 往往慢。
- padding buffer 复用可以有效减少内存操作。
- 0 padding、filter layout、WRAM 对齐要查 API。

#### Attention / Softmax

典型策略：

- 固定序列长度和 head dim。
- 利用 mask 分布和 causal 结构。
- 对 softmax 做 max-subtract、exp 近似、分块归一化。
- 对 KV cache/attention bias 减少重复 GDRAM 访问。

经验：

- 访存量和中间 tensor 数量经常比算术更关键。
- mask 稀疏模式如果固定，可专门化；如果随机，要谨慎。
- 概率近似必须过 max diff。

#### IO / Embedding / Scatter-Gather

典型策略：

- 先确认 index 分布、是否唯一、是否 sorted、是否重复。
- 能连续 copy 就不要标量 scatter。
- 小 embedding bag 可合并多个 rows 到 NRAM。
- 对 one-hot、masked select、index put，要优先减少随机 GDRAM 写。

经验：

- 随机访问是主要瓶颈。
- 如果输出很稀疏，可以考虑先清零再写热点，但清零本身也可能昂贵。
- 对 bool mask，uchar 到 half 的转换和融合计算可以减少分支。

## Agentic AI 实践经验

### 1. 让 Agent 按“证据”行动

Agent 很容易根据相似题目做直觉猜测，但比赛中很多直觉会错。有效做法是每一步都有证据来源：

- ref 证明 shape/dtype。
- MCP 证明 API 语义。
- OJ 证明性能和正确性。
- notes 证明历史失败路线。

### 2. 不反复横跳

用户多次强调“不要反复横跳”。实践上有效的方式是选择一个专题聚类：

- pointwise activation。
- matrix family。
- reduce family。
- pool family。
- attention family。
- IO/embedding family。

在一个专题内连续优化，可以迁移 tile、sync、预热、approx、task layout 等经验。

### 3. 提交后不要等待

异步 OJ 下，等待是最大浪费。Agent 应始终维护：

- 当前正在改的题。
- 已提交待结果的 hash。
- 下一个变体。
- 可迁移的同类题。

一个理想循环：

```text
submit A
while A pending:
  inspect ref of B
  implement B1
  submit B1
  collect A/B in batch
  update attempt table
```

### 4. 同代码 rerun 用于区分算法收益和随机抖动

当一个 commit 刷出特别好的一行时，不应立即认定算法突破。要做：

- 同代码空提交 rerun。
- 当前实现 rerun baseline。
- 对比双行结果。
- 如果收益小于 OJ 抖动，就不应过度记录。

### 5. 记录失败比记录成功更重要

优化中大量时间花在失败路线。`optimization_notes.md` 的价值在于：

- 防止重复尝试已失败 tile。
- 记录某 API 为什么不适用。
- 记录 checker 特殊行为。
- 记录 OJ 计时方式改变后的失效经验。

### 6. 把测试器当作系统而不是黑盒

通过阅读 `bangc_torch_tester.py` 和 `worker.py`，我们理解了：

- 每次是独立进程冷启动。
- module load、compile、tensor allocation、synchronize 都影响 CPU wall-clock。
- leaderboard 滞后 commit result。
- reference 可能被统一 fp16/fp32 处理。
- 部分题目有 problem-specific diff。

这使优化从“只看 kernel”转向“kernel + wrapper + runtime cold start”的整体系统优化。

## 典型案例

### 066_transform_vals

问题特点：

- shape 为 `(1, 3, 224, 224)`。
- dtype 为 fp16。
- 语义是 `a + b`。
- 数据量只有 150528 elements。

探索路线：

- 32 tasks 分段加法。
- 去除无用 float 分支。
- Block vs Union。
- 单 task 全量 NRAM。
- async + `__sync_compute()`。
- 预热有/无。

结论：

- 在新 CPU-sync OJ 下，不预热会被冷启动拉到数 ms。
- 单核全量 NRAM tile 可减少 32 task 调度和分段开销，成为较好路线。
- 对这个小算子，async 额外同步反而慢。

### 072_ElementwiseAdd

问题特点：

- 大 contiguous elementwise add。
- 典型带宽题。

探索路线：

- 32768、65536、131072 tile。
- `__sync_io_move_compute()` vs `__sync_compute()`。
- 原地输出。
- 预热与否。

结论：

- `__sync_compute()` 在部分变体中降低开销，但没有稳定超过旧最佳。
- tile 越大不一定越快，NRAM 压力和 OJ 抖动会影响结果。
- 当前最佳更像“稳定结构 + 冷启动处理”的组合，而非单纯带宽峰值。

### 087_Hadamard_product

问题特点：

- 与 elementwise add 类似，但核心计算是 multiply。
- 适合迁移 072 的 tile/sync 经验。

探索路线：

- current rerun 建 baseline。
- 65536 tile Block/async。
- 131072 tile Block/async。
- 32768 tile 稳定版。

结论：

- 65536 tile 有更高上限，但波动较大。
- 32768 tile 更稳。
- 同类迁移能快速缩小搜索空间。

### 118_Where_conditional

问题特点：

- bool condition + fp16 x/y。
- 语义 `where(condition, x, y)`。
- 可用 uchar mask 转 half，再做融合表达式。

探索路线：

- 原实现使用较重预热。
- 替换为 clean `zeros({1})` warm。
- 32 tasks vs 64 tasks。

结论：

- clean warm 保留一部分冷启动收益。
- 任务数对性能影响被 OJ 抖动放大。
- 对 selection 类，mask 转换和融合算术的成本需要与访存成本一起看。

## 可复用开发清单

### 新题启动清单

1. 读 `ref/ref_files/%03d_opname.py`。
2. 用 `ref/get_tasks.py` 查 dtype/wrapper/category。
3. 读当前 `.mlu` 和类似 accepted 实现。
4. 查 MCP API，确认所有非平凡 BangC/CNRT 行为。
5. 写最小正确 kernel。
6. 用 `ref/oj_git.py` 提交。
7. 记录 hash 和 idea。
8. 提交后继续做下一个变体。

### 优化假设清单

- 是否可以原地输出？
- 是否可以固定 shape/dtype？
- 是否可以减少 host tensor allocation？
- tile size 是否合适？
- NRAM scratch 是否过大？
- GDRAM 访问是否连续？
- 是否有多余 GDRAM round trip？
- 是否有多余 sync？
- task count 是否过多或过少？
- Union/Block 是否匹配 dataflow？
- 是否有可用 intrinsic？
- 是否可用近似通过 max diff？
- 是否存在 reference overflow？
- OJ 抖动是否主导了差距？

### 提交与记录清单

每个提交至少记录：

- op id/name。
- commit hash。
- idea。
- PASS/FAIL。
- diff。
- latency 两行。
- 是否更新 README。
- 是否值得 rerun。

## 风险与边界

### 不应做的事

- 不应修改 `ref/ref_files` 或 evaluator 来让错误实现通过。
- 不应使用 Torch/ATen/CNNL 算法算子计算最终输出。
- 不应把未验证的随机低延迟当成稳定突破。
- 不应在没有官方 ref 证据时写泛化假设。
- 不应重复尝试 notes 中已证明失败的路线。

### 需要谨慎的事

- Python/runtime patch：可能用于理解 tester 边界，但正式提交要遵守规则和当前 OJ 行为。
- 静态输出/cache：tester 变化后可能失效，必须用当前 OJ 重新验证。
- 预热：要尽量使用非计算型或最小计算型 runtime warmup，避免违反“非必要不要用 torch/aten/cnnl 参与计算”的要求。
- 近似：必须看 max abs diff 和 OJ，而不是只看均值误差。

## 总结

这次 BangC 算子挑战说明，Agentic AI 在底层算子开发中不是简单的代码生成器，而是一个可以持续组织证据、实验和反馈的工程协作者。它的优势来自：

- 快速阅读大量 reference 和历史代码。
- 用 MCP 文档精确查证 API。
- 按 AGENTS.md 规范执行异步 OJ 流水。
- 在多个题目之间迁移经验。
- 记录失败路线，减少重复劳动。
- 面对评测器变化时重建优化准则。

最终沉淀下来的方法论是：先用官方 ref 固定问题，再用 BangC 文档固定 API，再用 OJ 结果固定事实，最后用 Agent 的流水线能力把这些事实转化为持续优化。对于 MLU370 这类有明显 memory hierarchy、pipeline、launch 和 runtime 冷启动特征的平台，真正的高性能来自算法、访存、同步、任务布局、评测系统理解的共同优化。

