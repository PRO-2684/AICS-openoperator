# Linear Stdout Hardware Probe

`Linear.mlu` is currently used as a diagnostic stdout probe for op `085_Linear`.
It is not a valid final Linear implementation.

## Purpose

Collect OJ worker hardware facts directly from commit stdout:

- CNRT library version;
- visible MLU device count;
- current device and sync flag;
- queue priority range;
- per-device BDF, memory size, free memory, clusters, mcores, NRAM/WRAM/SRAM,
  L2, global-memory bus width;
- host `hostname`, `lscpu`, process CPU/memory affinity, `numactl -H`, and
  `cnmon info` if available.

The probe returns `torch::empty({batch, seq, out_features}, x.options())` so the
shape is valid and the tester can emit `@@RESULT@@`, but correctness is expected
to fail.

## Submitted Probe Hashes

```text
7cf7e7d0 085 linear hardware stdout probe 1
74ff5f51 085 linear hardware stdout probe 2
6709cc81 085 linear hardware stdout probe 3
1d940a84 085 linear hardware stdout probe 4
14ff9db8 085 linear hardware stdout probe 5
3b7c57e1 085 linear hardware stdout probe 6
```

Collection command:

```bash
python ref/get_result.py 7cf7e7d0 74ff5f51 6709cc81 1d940a84 14ff9db8 3b7c57e1 --format jsonl -j 12 --full --max-output-chars 12000
```

Result comments arrived at `2026-05-31 23:24 Asia/Shanghai`. All rows are
expected FAIL with `diff=nan`, but stdout is valid.

Observed worker classes:

| class | hashes/runs | CNMON | cards | CPUs | product | BDF pattern |
| --- | --- | --- | ---: | ---: | --- | --- |
| A | `7cf7e7d0`, `1d940a84`, part of `14ff9db8` | `v5.10.22` | 8 | 112, SMT on | `MLU370-X4K` | `4f,50,53,57,9c,9d,a0,a4` |
| B | part of `74ff5f51`, `6709cc81`, `14ff9db8` | `v6.2.10` | 8 | 56, SMT off | `MLU370-X4` | `4f,50,53,57,9c,9d,a0,a4` |
| C | part of `6709cc81`, `3b7c57e1` | `v6.2.10` | 10 | 56, SMT off | `MLU370-X4` | `4f,50,53,57,9c,9d,a0,a1,a4,b1` |

Common CNRT properties:

```text
CNRT lib: 6.14.1
cluster=8, mcore=4
nram=786432, wram=1048576, sram=4194304
l2=2097152
gbus=384
totalMB=24576
firmware=v1.1.6
```

CPU/NUMA:

```text
v5.10.22/8-card class:
  Cpus_allowed_list: 0-111
  Thread(s) per core: 2
  NUMA node0: 0-27,56-83
  NUMA node1: 28-55,84-111

v6.2.10 classes:
  Cpus_allowed_list: 0-55
  Thread(s) per core: 1
  NUMA node0: 0-27
  NUMA node1: 28-55
```

Both forms expose two host NUMA nodes with distance `10/20`.

Probe latency:

```text
comment rows ranged roughly 182-1023 us
```

Do not interpret those as Linear performance; the timed region only allocates an
empty output tensor and then fails correctness.

## Safety Boundary

Only collect machine topology and runtime facts. Do not print secrets, tokens,
webhook payloads, environment dumps, repository private files, or evaluator
internal credentials.

After collecting enough stdout, restore `Linear.mlu` before attempting a real
085 solution.
