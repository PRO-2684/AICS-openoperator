下面是基于你贴出的三台宿主测评机 `10.200.32.40 / .41 / .42` 的硬件拓扑报告。重点结论是：**三台机器 CPU 基本同代同规格，均为双路 Xeon Gold 6348；MLU 卡分布和驱动版本有差异；40 和 42 更常被调度，41 较少出现，因此优化策略应优先兼容 40/42。**

---

# 三台 MLU 宿主测评机硬件拓扑分析报告

## 1. 总体概况

从 `lscpu` 和 `cnmon` 输出看，三台测评宿主均为双路 Intel Xeon Gold 6348 平台，NUMA 节点为 2 个，每路 28 核。MLU 主要为 **MLU370-X4 / MLU370-X4K**，单卡功耗上限 150 W，显存约 24 GiB。

| 宿主 IP          | CPU               | NUMA |      可见 CPU | MLU 数量 | 驱动/CNMON      | 出现概率 |
| -------------- | ----------------- | ---: | ----------: | -----: | ------------- | ---- |
| `10.200.32.40` | Xeon Gold 6348 ×2 |    2 | 可能 56 或 112 |      8 | 有 v5.10.22 截图 | 较高   |
| `10.200.32.41` | Xeon Gold 6348 ×2 |    2 |          56 |      8 | v6.2.10       | 较低   |
| `10.200.32.42` | Xeon Gold 6348 ×2 |    2 |          56 |     10 | v6.2.10       | 较高   |

需要注意：`.40` 的截图里有一次 `lscpu` 显示 **112 CPUs / 2 threads per core**，但之前的 `lscpu -e` 与另一些环境里看到的是 **56 CPUs / 1 thread per core**。这说明 `.40` 可能存在不同容器/namespace/cgroup 暴露 CPU 不一致，或者不同测评容器配置不一致。实际对算子测评而言，应以**当前容器内可见 CPU 集合**为准。

2026-06-01 的 OJ stdout 探针补充确认：BangC 扩展进程内可以直接读取 `WORKER_ID`，并且它与 Redis `bangc:processing` 中的 worker id 一致。探针提交 `b5900e7d/fd0ca525/c4ed5c37/a56aaf13`、`a8a504a8/2f2afba0/3d98b4f6/39b3524f/1c3f1d47/8dd78cd8`、batch2 `9e20b42c..4b609ce1` 以及 batch3 `cc01c269..c746bd6a` 显示，实际评测容器至少分成三类；分层索引见 `ref/docs/jit/worker_hardware_mapping.md`。

| 容器可见形态 | 可见 CPU affinity | 可见 MLU BDF | 样本 worker | 样本 028 均值范围 |
| --- | ---: | --- | --- | ---: |
| 10 卡 | 56 CPU | `4f,50,53,57,9c,9d,a0,a1,a4,b1` | `16,17,18,19,20,21,22,23,24,25` | `343.994-416.463 us` |
| 8 卡 A | 56 CPU | `4f,50,53,57,9c,9d,a0,a4` | `8,9,10,11,12,13,14,15` | `360.102-411.674 us` |
| 8 卡 B | 112 CPU | `4f,50,53,57,9c,9d,a0,a4` | `0,1,2,3,4,5,6,7` | `368.751-395.860 us` |

所有这些 stdout 样本中的设备属性一致为 MLU370、8 cluster、4 mcore/cluster、NRAM/core `786432` bytes、WRAM/core `1048576` bytes、CNRT lib `6.14.1`；driver 在 8 卡 B 类为 `5.10.22`，其他两类为 `6.2.10`。单次探针样本仍处在冷窗口和 stdout 开销下，不能证明某个 worker/机器稳定更快；目前只能作为被动分层信号，不能作为硬性 gating 结论。

---

## 2. CPU / NUMA 拓扑

三台机器的物理平台基本一致：

```text
CPU: Intel Xeon Gold 6348 @ 2.60GHz
Socket: 2
Core per socket: 28
NUMA nodes: 2
```

常见暴露形式是：

```text
NUMA node0 CPU(s): 0-27
NUMA node1 CPU(s): 28-55
```

`.40` 的一个截图中还出现过 SMT 暴露形式：

```text
CPU(s): 112
Thread(s) per core: 2
NUMA node0 CPU(s): 0-27,56-83
NUMA node1 CPU(s): 28-55,84-111
```

因此可以认为底层机器很可能本来支持超线程，但测评容器有时只暴露每核一个线程。

之前 `numactl -H` 也显示两路 NUMA 节点各约 512 GiB 内存，本地距离 10，跨 NUMA 距离 20，说明远端 NUMA 访问代价明显更高。

---

## 3. MLU 拓扑与卡号分布

### 3.1 `10.200.32.40`

`.40` 机器显示 8 张 MLU：

```text
Card 0: 0000:4F:00.0
Card 1: 0000:50:00.0
Card 2: 0000:53:00.0
Card 3: 0000:57:00.0
Card 4: 0000:9C:00.0
Card 5: 0000:9D:00.0
Card 6: 0000:A0:00.0
Card 7: 0000:A4:00.0
```

这非常像标准的 **4 + 4 分布**：

```text
低 bus 段:
  4F, 50, 53, 57  -> 可能靠近 NUMA node0

高 bus 段:
  9C, 9D, A0, A4  -> 可能靠近 NUMA node1
```

结合之前 `lspci -t` 中两棵明显的 PCIe switch 树：

```text
0000:4a -> 4b-58
0000:97 -> 98-a5
```

可以推断 4F/50/53/57 很可能挂在 `0000:4a` 这一侧，9C/9D/A0/A4 很可能挂在 `0000:97` 这一侧。

`.40` 的驱动截图是：

```text
CNMON v5.10.22
Driver v5.10.22
```

这和 `.41/.42` 的 v6.2.10 不同，意味着 `.40` 可能是老驱动环境，或者截图来自不同时间的宿主状态。

---

### 3.2 `10.200.32.41`

`.41` 机器也是 8 张 MLU，BDF 分布和 `.40` 基本一致：

```text
Card 0: 0000:4F:00.0
Card 1: 0000:50:00.0
Card 2: 0000:53:00.0
Card 3: 0000:57:00.0
Card 4: 0000:9C:00.0
Card 5: 0000:9D:00.0
Card 6: 0000:A0:00.0
Card 7: 0000:A4:00.0
```

这说明 `.41` 和 `.40` 的 PCIe/MLU 插卡拓扑大概率是同构的，都是：

```text
MLU 0-3 -> 一侧 PCIe switch / 一侧 NUMA
MLU 4-7 -> 另一侧 PCIe switch / 另一侧 NUMA
```

`.41` 的驱动为：

```text
CNMON v6.2.10
Driver v6.2.10
```

显存显示为：

```text
24576 MiB / card
```

而 `.40` 的 v5.10.22 输出里部分卡显示为 23292/23308 MiB，这更像是不同驱动版本下的可用显存统计口径差异。

`.41` 比较特殊的一点是：你说它**较难被分配到**。因此它对最终分数的统计影响可能较小，但仍需保证代码在 v6.2.10 驱动下行为正确。

---

### 3.3 `10.200.32.42`

`.42` 机器和前两台不同，它显示了 **10 张 MLU**：

```text
Card 0: 0000:4F:00.0
Card 1: 0000:50:00.0
Card 2: 0000:53:00.0
Card 3: 0000:57:00.0
Card 4: 0000:9C:00.0
Card 5: 0000:9D:00.0
Card 6: 0000:A0:00.0
Card 7: 0000:A1:00.0
Card 8: 0000:A4:00.0
Card 9: 0000:B1:00.0
```

这说明 `.42` 不是简单的 8 卡机器，而是：

```text
低 bus 段:
  4F, 50, 53, 57      -> 4 张

高 bus 段:
  9C, 9D, A0, A1, A4, B1 -> 6 张
```

即更像是 **4 + 6** 的非均匀分布。

对测评来说，这很重要：如果测评调度随机选择设备，`.42` 上高 bus / 可能 node1 侧设备数量更多，进程落在 node0 CPU 时更容易跨 NUMA 访问设备。

`.42` 驱动为：

```text
CNMON v6.2.10
Driver v6.2.10
```

显存同样是：

```text
24576 MiB / card
```

---

## 4. 三台机器的关键差异

### 差异一：MLU 数量不同

```text
.40: 8 卡
.41: 8 卡
.42: 10 卡
```

这会影响你之前用 `pid % device_count` 的策略。

如果在 `.40/.41` 上：

```text
device_count = 8
pid % 8 -> 0-7
```

但在 `.42` 上：

```text
device_count = 10
pid % 10 -> 0-9
```

那么同一个 pid 分布策略在不同机器上会选到不同物理位置的卡，尤其 `.42` 上可能选到 `A1` 或 `B1` 这种只在 `.42` 存在的卡。

---

### 差异二：驱动版本不同

```text
.40: v5.10.22
.41: v6.2.10
.42: v6.2.10
```

这可能影响：

```text
torch_mlu runtime 初始化时间
cnrt 队列行为
显存统计口径
kernel launch 开销
同步开销
部分 BangC intrinsic 或 runtime 行为
```

对于小算子，驱动差异可能比 kernel 本身更影响总耗时。

---

### 差异三：CPU 暴露不同

`.41/.42` 明确看到：

```text
CPU(s): 56
Thread(s) per core: 1
NUMA node0: 0-27
NUMA node1: 28-55
```

`.40` 至少有一次显示：

```text
CPU(s): 112
Thread(s) per core: 2
NUMA node0: 0-27,56-83
NUMA node1: 28-55,84-111
```

如果测评环境里 `.40` 暴露 112 CPU，而 `.41/.42` 暴露 56 CPU，那么 host 侧线程调度、OpenMP/Torch 线程池、系统噪声都会不同。

---

## 5. 对算子测评的影响

你的 BangC 算子多数是小 kernel，很多题目的计时在几十到几百微秒，甚至十微秒级。此时影响总时间的不只是 MLU kernel 本体，还包括：

```text
Python / C++ wrapper 开销
torch_mlu dispatch 开销
cnrt launch 开销
torch.mlu.synchronize() 等待开销
host CPU 调度
NUMA 远端访问
设备初始化/上下文切换
```

因此三台机器的影响大致如下：

### `.40`

优点：

```text
出现概率较大
8 卡拓扑较规整
```

风险：

```text
驱动可能是 v5.10.22，和 v6.2.10 行为不同
CPU 暴露可能不稳定，可能 56/112 两种形态
```

### `.41`

优点：

```text
8 卡拓扑规整
驱动 v6.2.10
CPU 暴露清晰：56 CPU，无 SMT
```

风险：

```text
较难被分配到，对线上结果影响较小
不应过度针对它调参
```

### `.42`

优点：

```text
出现概率较大
驱动 v6.2.10
CPU 拓扑清晰
```

风险：

```text
10 卡，且可能是 4+6 非均匀分布
pid % device_count 策略在这里会改变设备选择分布
可能更容易选到跨 NUMA 的卡
```

---


## 6. 结论

三台宿主测评机总体 CPU 平台一致，都是双路 Xeon Gold 6348、双 NUMA 结构，但 MLU 与驱动环境存在明显差异：

```text
10.200.32.40:
  8 卡，可能老驱动 v5.10.22，出现概率较大

10.200.32.41:
  8 卡，新驱动 v6.2.10，较难分配到

10.200.32.42:
  10 卡，新驱动 v6.2.10，出现概率较大
```

---

## 7. OJ Stdout / JIT Probe 补充画像

本节来自 OJ stdout、`worker.py`、`bangc_torch_tester.py` 和
`ref/jitter_experiments.md` 的重复探针结果。它描述的是当前 OJ 测评路径，
不是单纯硬件峰值。

### 7.1 测评 worker 的计时结构

当前 worker 对每个题目启动 3 个 fresh 子评测进程：

```text
worker.py
  evaluate_one(...)
    for run_idx in 1..EVAL_RUNS:
      subprocess.run(["python3", "bangc_torch_tester.py", ...])
    aggregate latency = average(run[1..3].bangc_us)
```

`bangc_torch_tester.py` 中被计时的区域是：

```text
torch.mlu.synchronize()
t0 = time.perf_counter()
output = model_new(*bangc_inputs)
torch.mlu.synchronize()
t1 = time.perf_counter()
```

因此每个 raw run 都会重新 import/load extension，并在计时区间内承受一次
fresh-process 下的 `bang_func` 调用、custom kernel launch 和最终全局同步。
这解释了小算子 wall time 远大于 device kernel hardware time。

### 7.2 OJ 中观测到的硬件类别

通过 CNRT stdout 探针，OJ 至少采样到以下类别：

```text
driver 5.10.22, card_count=8   -> 大概率对应 10.200.32.40
driver 6.2.10, card_count=8    -> 大概率对应 10.200.32.41
driver 6.2.10, card_count=10   -> 大概率对应 10.200.32.42
```

8 卡机器的 BDF 分布与前文 `.40/.41` 拓扑一致；10 卡机器与 `.42`
的 4+6 非均匀分布一致。不同类别可以带来数十微秒的 median/low-tail
差异，但所有类别在小 custom kernel 上仍然表现为 launch-bound。

### 7.3 基础开销分布

重复 OJ 探针显示，当前测评路径下各层开销量级大致为：

| 路径 | 代表实验 | 中位量级 |
| --- | --- | ---: |
| host-only return / stream lookup | `J027-HINFO0`, `J027-SINFO0` | `~40 us` |
| idle current queue query | `J027-QQUERY0` | `~44 us` |
| CNRT memory info query | `J027-MEMINFO0` | `~116 us` |
| empty custom kernel + final sync | `J027-KINFO0/KINFO3` | `~334-346 us` |
| memcpy-only custom kernel | `J027-MINFO0` | `~348 us` |
| no registered-op static warmup | `J027-NOWARM0` | `~2264 us` |

实际 026/028/029 的 CNRT notifier hardware time 很小：

```text
026 ELU:          8-9 us, 24 raw prints
028 HardSigmoid:  7-9 us, 21 raw prints
029 HardTanh:    12-13 us, 21 raw prints
```

这说明对这些小 pointwise 算子，主要开销不在 NRAM 计算或 GDRAM 读写本体，
而在 fresh-process custom kernel launch、runtime scheduling 和最终同步。

### 7.4 抖动来源排序

按当前证据，影响小算子 wall time 的主要因素大致是：

```text
1. 是否有有效的 static registered-op MLU warmup
2. fresh subprocess 下第一次 timed custom kernel launch + final sync
3. worker/设备类别：driver 5.10.22/8、driver 6.2.10/8、driver 6.2.10/10
4. worker/device contention 和 OJ 调度窗口
5. runtime 查询、内存池状态、偶发同步 outlier
6. CPU/NUMA 位置和 device selection，小幅且不稳定
7. 7-13us 量级的 kernel arithmetic/body 差异
```

CPU affinity、device0 固定、pid-spread、CPU-local device grouping、4GiB/16GiB
显存预留等方法都做过重复探针；它们可以改变个别批次或暴露拥塞/OOM，但没有形成
稳定的跨题降抖方案。

### 7.5 优化策略含义

对 026/028/029 这类实际 device kernel time 只有 `~10 us` 的题：

```text
即使算法 kernel 从 10us 优化到 5us，
收益也小于 OJ 常见 p25-p75 抖动和 worker 类别差异。
```

因此除非同窗口至少 8 行以上结果显示整体分布移动，否则不应把这类题的主要精力放在
Bang intrinsic 微调上。更有效的方向是：

```text
1. 保留有效 static warmup，避免 2ms+ cold start。
2. 避免多 kernel launch，能一核完成就一核完成。
3. 探索合法的 metadata/view 路径或无需 custom kernel 的语义特例。
4. 用 active-age queue gate 和低/中阈值 boost 做概率尾部采样，而不是按
   worker id 硬路由；`f3d2e5be` 的 `026,027,028,029` joint config
   确认多题可进入同一个 OJ task，但没有产生稳定的后续题预热收益。
4. 对已知 launch-bound 的正确实现做批量低尾采样。
5. 对真正大 kernel/大 IO 题，再回到 tiling、访存流水、NRAM/SRAM 优化。
```

更详细的计时模型和实验索引见：

```text
ref/docs/jit/measurement_model.md
ref/docs/jit/overhead_distribution.md
ref/docs/jit/system_overhead_analysis.md
ref/docs/jit/mitigation_playbook.md
ref/jitter_experiments.md
```

---

## 8. CNRT Probe Fields To Maintain

为了把宿主画像从“手工截图”升级为可重复的 OJ stdout 证据，当前使用
`085_Linear` 的 `Linear.mlu` 临时探针打印以下字段。该探针只用于测评系统画像，
不是有效 Linear 实现。

### 8.1 Runtime-level fields

```text
CNRTINFO lib=<major.minor.patch>
CNRTINFO count=<visible device count>
CNRTINFO current=<current device ordinal>
CNRTINFO flag=<cnrtGetDeviceFlag result>
CNRTINFO priority=[min,max]
```

含义：

```text
lib:
  编译/运行时可见 CNRT 库版本。注意它不等同于内核驱动版本；
  驱动版本仍需从 torch_mlu warning、cnmon 或已有 HWINFO 探针推断。

count:
  OJ worker 容器当前可见 MLU 卡数。8 基本对应 .40/.41 类，
  10 基本对应 .42 类。

flag:
  CNRT 同步策略。历史探针中尝试设置 block/yield 后仍打印为 0，
  说明该扩展上下文里 sync policy 不是稳定优化旋钮。
```

### 8.2 Per-device fields

每张卡打印：

```text
CNRTDEV id=<ordinal>
        bdf=<domain:bus:device.function>
        name=<device name>
        totalMB=<prop.totalMem>
        availMB=<prop.availableGlobalMemorySize>
        freeMB=<cnrtMemGetInfo free>
        cluster=<clusterCount>
        mcore=<McorePerCluster>
        nram=<NramSizePerMcore>
        wram=<WramSizePerMcore>
        sram=<SramSizePerMcore>
        l2=<maxL2CacheSize>
        gbus=<GmemBusWidth>
```

这些字段用于确认：

```text
1. MLU370-X4SN/X4K 的实际 cluster/mcore/NRAM/WRAM/SRAM 配置是否与
   AGENTS 中的 8 cluster、4 mcore/cluster、768KiB NRAM、1MiB WRAM一致。
2. 8 卡机器是否稳定呈现 4F/50/53/57 + 9C/9D/A0/A4。
3. 10 卡机器是否稳定呈现 4F/50/53/57 + 9C/9D/A0/A1/A4/B1。
4. freeMB/availMB 是否能解释个别 OOM、large-reserve 失败或极端慢行。
```

### 8.3 Host fields

探针还打印：

```text
hostname
/proc/self/status: Cpus_allowed_list, Mems_allowed_list
lscpu: model, sockets, cores, threads, NUMA node CPU lists
numactl -H: node memory and distance
cnmon info: if available
```

解释方式：

```text
Cpus_allowed_list:
  判断 worker 子进程是否被限制在 0-55、0-111 或其他 CPU 集合。

Mems_allowed_list / numactl:
  判断容器是否允许两个 host NUMA 节点，以及跨 NUMA distance 是否仍为 20。

hostname + count + BDF:
  交叉确认 .40/.41/.42 类别，而不是只靠 driver/count 推断。
```

### 8.4 当前待收集 hashes

```text
7cf7e7d0 74ff5f51 6709cc81 1d940a84 14ff9db8 3b7c57e1
```

收集命令：

```bash
python ref/get_result.py 7cf7e7d0 74ff5f51 6709cc81 1d940a84 14ff9db8 3b7c57e1 --format jsonl -j 12 --full --max-output-chars 12000
```

将结果合并入本文件时，必须区分：

```text
observed:
  stdout 直接打印的事实。

inferred:
  根据 hostname/count/BDF/driver 推断出的 .40/.41/.42 对应关系。

not proven:
  任何只在单次或少量 stdout 中出现的差异。
```

### 8.5 2026-05-31 Linear probe observed facts

`085_Linear` stdout 探针已经返回结果，覆盖了三类 worker。以下为 stdout 直接事实：

```text
CNRT library:
  6.14.1 on all sampled workers.

CNRT sync flag:
  cnrtGetDeviceFlag prints flag=3 on all sampled workers.
  This differs from earlier specialized HWINFO probes that printed flag=0 after
  different setup paths, so do not over-interpret flag alone without preserving
  probe placement.

Queue priority range:
  priority=[7,0] on sampled workers.
```

#### Class A: 8-card, v5.10.22, 112 visible CPUs

Observed in hashes including `7cf7e7d0`, `1d940a84`, `14ff9db8`.

```text
CNMON: v5.10.22
Product: MLU370-X4K
Driver: v5.10.22
Firmware: v1.1.6
Visible CPU(s): 112
Thread(s) per core: 2
NUMA node0 CPU(s): 0-27,56-83
NUMA node1 CPU(s): 28-55,84-111
Cpus_allowed_list: 0-111
Mems_allowed_list: 0-1
```

BDF list:

```text
0: 0000:4f:00.0
1: 0000:50:00.0
2: 0000:53:00.0
3: 0000:57:00.0
4: 0000:9c:00.0
5: 0000:9d:00.0
6: 0000:a0:00.0
7: 0000:a4:00.0
```

CNRT properties:

```text
name=MLU370
totalMB=24576
freeMB/available around 23292-23308 MB when idle
cluster=8
mcore=4
nram=786432
wram=1048576
sram=4194304
l2=2097152
gbus=384
```

This confirms the old-driver 8-card worker exposes SMT (`112 CPU`) in the OJ
container and matches the manually recorded `.40` topology.

#### Class B: 8-card, v6.2.10, 56 visible CPUs

Observed in hashes including `74ff5f51`, `6709cc81`, `14ff9db8`.

```text
CNMON: v6.2.10
Product: MLU370-X4
Driver: v6.2.10
Firmware: v1.1.6
Visible CPU(s): 56
Thread(s) per core: 1
NUMA node0 CPU(s): 0-27
NUMA node1 CPU(s): 28-55
Cpus_allowed_list: 0-55
Mems_allowed_list: 0-1
```

BDF list:

```text
0: 0000:4f:00.0
1: 0000:50:00.0
2: 0000:53:00.0
3: 0000:57:00.0
4: 0000:9c:00.0
5: 0000:9d:00.0
6: 0000:a0:00.0
7: 0000:a4:00.0
```

This matches the 8-card v6.2.10 worker class, likely `.41`.

#### Class C: 10-card, v6.2.10, 56 visible CPUs

Observed in hashes including `6709cc81`, `3b7c57e1`.

```text
CNMON: v6.2.10
Product: MLU370-X4
Driver: v6.2.10
Firmware: v1.1.6
Visible CPU(s): 56
Thread(s) per core: 1
NUMA node0 CPU(s): 0-27
NUMA node1 CPU(s): 28-55
Cpus_allowed_list: 0-55
Mems_allowed_list: 0-1
```

BDF list:

```text
0: 0000:4f:00.0
1: 0000:50:00.0
2: 0000:53:00.0
3: 0000:57:00.0
4: 0000:9c:00.0
5: 0000:9d:00.0
6: 0000:a0:00.0
7: 0000:a1:00.0
8: 0000:a4:00.0
9: 0000:b1:00.0
```

This confirms the 10-card `.42` class and its non-uniform 4+6 BDF layout.

#### Host NUMA

Both 56-CPU and 112-CPU forms report two host NUMA nodes of about 515-516 GB
each, with distance matrix:

```text
node   0   1
  0:  10  20
  1:  20  10
```

This keeps the earlier assumption valid: cross-host-NUMA placement can add
latency, but repeated affinity experiments show it is not a robust primary fix
for the small-kernel launch floor.

#### Linear probe timing note

The probe's timed `bang_func` only performs a `torch::empty` allocation with the
correct Linear output shape. It is expected to fail correctness with `diff=nan`.
Observed comment latencies ranged from about `182 us` to `1023 us`; these numbers
measure allocation/runtime behavior of the probe, not Linear algorithm speed.

`cnmon info` was initially too verbose and sometimes interleaved/truncated around
`@@RESULT@@`; future `Linear.mlu` probe revisions should keep only summarized
`cnmon` lines.
