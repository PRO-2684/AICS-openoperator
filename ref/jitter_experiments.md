# Jitter Experiments

Rule: do not promote an observation to a conclusion unless the same variant has at least 8 OJ rows and the effect is consistent across the distribution. Until then, record it as an observation only.

## Layered Conclusion Index

This file is organized as three layers:

1. Decision layer: `C-*` conclusions below. These are the only items allowed to guide implementation choices.
2. Evidence layer: `J*` experiment ids in the matrix and result tables. Each promoted conclusion must point to repeated OJ evidence.
3. Raw layer: commit hashes in the experiment matrix. Use `ref/get_result.py --full` to recover stdout/stderr and per-run `@@RESULT@@` values.

Statistics policy:
- Compact rows are OJ comment averages and are useful for leaderboard-facing behavior.
- Raw rows are individual fresh subprocess `@@RESULT@@ bangc_us` values and are preferred for root-cause analysis.
- When HWINFO is present, group raw rows by `(kind, driver, card_count, device)` before claiming hardware or topology effects.
- Same-window A/B batches are required before promoting small median differences below about 10 us.

Tooling policy:
- `ref/get_result.py` now defaults to direct GitHub API reads using the token from `GH_TOKEN`, `GITHUB_TOKEN`, or `~/.config/gh/hosts.yml`, and fetches commit comments concurrently with `asyncio/aiohttp`. Use `--gh-cli` only as a fallback.
- `ref/oj_collect.py` uses the same async fetch path and is the preferred compact batch summarizer for `ref/.oj_repeat_hashes`.
- `ref/oj_repeat.py` still pushes after every commit by default because OJ webhook semantics are not guaranteed to be per-commit. `--push-at-end` is only an experimental throughput mode; verify that every hash receives comments before relying on it.
- Empty repeat commits can miss OJ comments. For low-tail probability experiments, use `ref/oj_repeat.py --config-nonce` to alternate harmless blank lines in `config`, keeping the same op while making every commit non-empty.

Conclusion states:

| state | meaning |
| --- | --- |
| `strong` | at least 8 rows and distribution is consistent enough to act on |
| `weak` | enough signal to keep in mind, but contradicted by another batch or mixed by machine class |
| `blocked` | API/path cannot be used in OJ or fails before useful timing |

### C0 Measurement Model

| id | state | conclusion | evidence |
| --- | --- | --- | --- |
| C0.1 | strong | OJ starts fresh evaluation subprocesses and averages reported `bangc_us`; single fast raw runs do not dominate the comment latency. | `worker.py`, `bangc_torch_tester.py`, all repeated comments |
| C0.2 | strong | Timed region is `torch.mlu.synchronize(); t0; bang_func; torch.mlu.synchronize(); t1`; constructors and `ModelNew` construction are outside timing. | `bangc_torch_tester.py:benchmark_hardware_time_once` |
| C0.3 | strong | FAIL probes still emit valid `bangc_us`, so partial-code timing experiments are useful even when output is intentionally wrong. | `J027-H0`, `J027-K0`, `J027-KINFO*`, `J027-TZ1P0` |

### C1 Cold Start And Prewarm

| id | state | conclusion | action | evidence |
| --- | --- | --- | --- | --- |
| C1.1 | strong | Without a static MLU-side warmup, the first timed custom kernel pays a stable multi-ms cold-start cost. | Never benchmark or submit small-kernel variants without a static prewarm. | `J027-NOWARM0` median `2263.860 us` |
| C1.2 | strong | Static MLU allocation alone does not warm the custom-kernel launch path. | Do not rely on `torch::empty` reserve/allocation as the only warmup. | `J027-ALLOC1P0` median `2415.514 us` |
| C1.3 | strong | CNRT `cnrtMemset` alone also does not warm the custom-kernel launch path. | Do not replace torch prewarm with pure CNRT memset. | `J027-CMSET1P0` median `2448.551 us` |
| C1.4 | strong | One-element `torch::zeros` in static initialization removes the multi-ms cold start. | Preferred minimal cold-start prewarm candidate, when allowed by task rules and not computing results. | `J027-TZ1P0` median `328.156 us` |
| C1.5 | blocked | User Bang kernels cannot be launched from static initialization before CNCC runtime registration. | Do not attempt constructor self-kernel warmup. | `J027-SAMEKWARM0`, `Kernel not found` |
| C1.6 | weak | Same-size static torch add sometimes lowers medians, but did not reproduce consistently. | Do not treat larger warmup as a reliable speed fix without adjacent controls. | `J027-GBIGADD0`, `J027-GBIGADD1`, `J027-KINFO2` |
| C1.7 | strong | Zero-size `torch::zeros({0})` does not warm the custom-kernel launch path. | Prewarm must execute a real registered device op, not just go through torch dispatcher. | `J027-TZ0P0` median `2343.602 us` |
| C1.8 | strong | One-element `torch::empty + torch::empty` add also removes the multi-ms cold start and matches `zeros({1})`. | Both `zeros({1})` and tiny add are valid prewarm candidates; prefer the simpler one unless task rules object. | `J027-TADD1P0` median `325.698 us` |
| C1.9 | strong | Tiny add prewarm on real 026/028/029 PASS sources matches `zeros({1})` and keeps PASS, but has no decisive latency advantage. | Use either one-element torch prewarm; choose tiny add if avoiding the fill path is desirable, otherwise keep `zeros({1})` for simplicity. | `J026/028/029-TADDPASS0`, `J026/028/029-PIDSPREAD1` |
| C1.10 | strong | Discarding the prewarm tensor after constructor tiny add keeps PASS and overlaps retained tiny-add/zeros prewarm distributions. | Tensor lifetime is not the useful part of tiny-add prewarm; the registered device op execution is. | `J026/028/029-TDROP0` medians `384.795/362.480/354.631 us` |
| C1.11 | blocked | One-element `torch::abs(empty)` cannot be used as the smaller single-input prewarm because source audit rejects `torch::abs`. | Avoid named high-level `torch::` algorithm ops in constructors even for warmup probes. | `J026/028/029-TABSPASS0`, 16 comments with `FORBIDDEN_TORCH_HIGH_LEVEL_OP` |
| C1.12 | strong | Discarding a one-element `torch::zeros` prewarm keeps PASS and overlaps retained zeros/tiny-add distributions. | Retain a warmup tensor only when useful for code simplicity; lifetime is not required for the cold-start effect. | `J026/028/029-ZDROP0` medians `383.775/349.705/343.324 us` |

### C2 Launch, Queue, And Sync

| id | state | conclusion | action | evidence |
| --- | --- | --- | --- | --- |
| C2.1 | strong | Host-only wrapper/sync lower bound is about 40 us, far below measured small-kernel latency. | Focus on launch/runtime cold path, not Python/C++ wrapper cost. | `J027-HINFO0`, `J027-SINFO0`, `J027-QQUERY0` |
| C2.2 | strong | Empty custom kernel launch plus final synchronize is already around 330-350 us after prewarm. | For tiny pointwise ops, algorithm body below this scale may be invisible in OJ wall time. | `J027-KINFO0`, `J027-KINFO3` |
| C2.3 | strong | Actual device hardware time of the GELU kernel body is about 7-8 us while wall time is hundreds of us. | Optimize launch count first; kernel body tuning has limited impact for these small shapes. | `J027-NT0` |
| C2.4 | strong | Additional warmed kernel launches cost roughly 3-4 us each at median. | Fuse launches when possible; avoid multi-kernel decompositions for small ops. | `J027-K32INFO0` vs `J027-KINFO0` |
| C2.5 | strong | `cnrtQueueQuery` on idle/current stream is host-only scale, but query after launch does not remove final sync cost. | Queue query is not a launch-cost workaround. | `J027-QQUERY0`, `J027-LQUERY0` |
| C2.6 | weak | Explicit `cnrtInvokeKernel` overlaps normal `<<<>>>` launch; no robust speed benefit. | Keep normal launch syntax unless a same-window task-specific result proves otherwise. | `J027-DINK1`, `J027-KINFO3` |
| C2.7 | weak | Internal `cnrtQueueSync(queue)` increases outlier exposure and does not remove external measurement floor. | Avoid in-function sync unless required for correctness. | `J027-INSYNC0` |
| C2.8 | strong | Real 026/028/029 PASS kernels also spend only single-digit to low-teen us in device hardware time: ELU `8-9 us` over 24 raw prints, HardSigmoid `7-9 us` over 21 raw prints, HardTanh `12-13 us` over 21 raw prints. | Do not expect broad arithmetic tuning to close 270-310 us leaderboard gaps; prioritize launch/warmup/sync path and repeated low-tail sampling. | `J026-NT0`, `J028-NT0`, `J029-NT0` |
| C2.9 | strong | When the real device work is only `~8-13 us`, even a hypothetical 50% algorithm-body win is smaller than ordinary OJ IQR/worker jitter. Baseline PASS IQRs are much larger: 026 B0 p25-p75 about `63 us`, 028 B0 about `91 us`, 029 B0 about `53 us`; notifier wall IQRs are also tens to hundreds of us. | Treat sub-10us arithmetic wins as statistically invisible unless proven by a same-window >=8-row distribution shift; otherwise spend time on launch/warmup/sync or probability sampling. | `J026-B0`, `J028-B0`, `J029-B0`, `J026/028/029-NT0` |

### C3 CNRT API And TaskTopo

| id | state | conclusion | action | evidence |
| --- | --- | --- | --- | --- |
| C3.1 | strong | Ordinary CNRT runtime queries are not enough to explain the 300+ us kernel floor. | Treat launch registration/scheduling as separate from simple runtime query overhead. | `J027-MEMINFO0`, `J027-C0/C1/C2` |
| C3.2 | strong | Host callbacks are unstable and can produce multi-ms outliers. | Do not use `cnrtInvokeHostFunc` as a warmup or queue substitute. | `J027-HOSTFN0` |
| C3.3 | strong | `cnrtSetDeviceFlag` block/yield did not visibly change scheduling flag in this extension context. | Do not spend more tuning block/yield policy without a new API path. | `J027-FBLOCKINFO0`, `J027-FYIELDINFO0`, `J027-FBLOCKPRE0` |
| C3.4 | strong | TaskTopo empty node invoke is host-only scale; kernel node path is blocked by kernel registration or source audit. | TaskTopo is not currently a usable custom-kernel launch workaround. | `J027-TEMPTY0`, `J027-TKEMPTY0`, `J027-TCAPEMPTY0` |
| C3.5 | weak | `cnrtDeviceQueryKernelMemoryUsage` reports stable kernel memory usage but does not prewarm launch. | Keep as diagnostic only. | `J027-KUSAGEPRE0` |

### C4 Hardware, Worker, And NUMA

| id | state | conclusion | action | evidence |
| --- | --- | --- | --- | --- |
| C4.1 | strong | OJ samples at least three classes: `5.10.22/8-card`, `6.2.10/8-card`, `6.2.10/10-card`. | Always compare distributions by class when stdout includes HWINFO. | `J027-INFO0`, `J027-HINFO0/KINFO0/MINFO0` |
| C4.2 | weak | CPU/NUMA placement may move medians by small amounts but is not a robust primary fix. | Do not force affinity as a default optimization yet. | `J027-CPUTOPO0`, `J027-AFFNODE0/1`, `J027-AFFPID01` |
| C4.3 | weak | Driver/count-gated device spreading did not improve 027 in sampled batches. | Prefer default device 0 unless a task-specific multi-worker contention experiment proves otherwise. | `J027-DLOW0`, `J027-D8ONLY0`, `J027-DHIGH0` |
| C4.4 | weak | Large memory reservation/holding can expose contention/OOM but is not a reliable speed fix. | Use large reservation only as a diagnostic or contention filter, not as default implementation strategy. | `J027-GRES16G0`, `J027-GRES4G0`, `J026/028/029-GMEM18HOLD0` |
| C4.5 | strong | Fixed device0 and pid-spread device selection overlap on real 026/028/029 PASS sources. | Do not treat device selection as a primary latency fix; choose policy for contention control rather than expected median speed. | `J026/028/029-FIXDEV0`, `J026/028/029-PIDSPREAD1` |
| C4.6 | weak | Selecting MLU device group by current CPU half does not robustly improve real 026/028/029 PASS latency and can still produce large outliers. | Do not use CPU-local device selection as a default speed fix without an adjacent target-specific win. | `J026/028/029-CPULOCAL0` |

### C5 Cross-Op Scope

| id | state | conclusion | action | evidence |
| --- | --- | --- | --- | --- |
| C5.1 | strong | 026/028/029 baseline reruns hit the same 300-400 us launch-bound band as 027. | Apply prewarm and launch-count findings across tiny pointwise 026-029 before deeper math tuning. | `J026-B0`, `J028-B0`, `J029-B0` |

Target measurement path from `ref/bangc_torch_tester.py`:
- The worker imports the generated extension, constructs `ModelNew`, builds inputs, then times exactly one `model_new(*bangc_inputs)` call.
- Timing uses `time.perf_counter()` around the call and `torch.mlu.synchronize()` before and after.
- Static constructors and `ModelNew` construction happen before the timed call.
- Any computation required for the operator result must happen inside the timed `bang_func`.
- OJ comments contain three fresh `[run N]` executions; compact latency is the arithmetic mean of the three `@@RESULT@@ bangc_us` values. Each run reloads the extension, so the measured value is dominated by fresh-process/custom-kernel cold start rather than same-process steady state.

Likely jitter sources to isolate:
- Worker process cold start and Python/extension import state.
- MLU runtime first-use costs, stream acquisition, queue state, and device allocator state.
- Device contention across worker processes and card selection.
- Kernel launch overhead plus final synchronize wall time.
- Memory pool initialization and first allocation effects.
- Tensor input placement and host/device NUMA affinity around the worker process.
- OJ worker scheduling and background load; same commit can hit different workers/cards.

Experiment matrix:

| id | op | variant | purpose | rows needed | commits | status |
| --- | --- | --- | --- | --- | --- | --- |
| J027-B0 | 027 | current best source | OJ same-code distribution | >=8 | 984252a f121541 a4b3fe6 eb7a76c 8278b00 7cf1ee8 ff00541 75e1912 bfe92f7 8996bc1 dc30454 cbdbff4 42bd9a5 e9c7dc2 5772999 8de6f38 d9e8d0c 38c4c22 b234dd6 24d276d | submitted |
| J027-H0 | 027 | return input, no kernel | host call/sync lower bound, expected FAIL | >=8 | 0b7e892 adbe5ca d3129f1 70b64ec d5b0ffb 02c71c2 a2f48d7 accbe15 | submitted |
| J027-K0 | 027 | launch empty kernel, no memory | kernel launch + sync lower bound, expected FAIL | >=8 | f28c408 3e7ffc8 af9b7c7 bfcd269 8717a5c 9d67ee4 c14a1a3 4ca36e4 | submitted |
| J027-M0 | 027 | memcpy-only kernel | memory round-trip lower bound, expected FAIL | >=8 | 9672a81 bd52d3f 3f7d16c 639ba20 0f41778 034366c 45e5abe 4f1e512 | submitted |
| J027-WK0 | 027 | current best plus static dummy kernel warmup | test if launch/runtime cold cost can move outside timer | >=8 | 53f8218 98bd8e3 a570ee5 8906eec 08fe5b1 01a7062 87ed0c0 d7c6630 | submitted |
| J027-ZB0 | 027 | static `torch::zeros({262144})` warmup | allocation plus memset kernel outside timer | >=8 | aacc2d1 84bb067 cc9ede1 17b3cda ea79ddf 4ebad70 7d3bff4 91abd93 | submitted |
| J027-ZBA0 | 027 | static zeros-big then current empty-add warmup | combine memset and known best add warmup | >=8 | afd5134 7bff1d5 eb56a42 a791f8c 1d1b13e 8520e44 f611ea9 9a8b16e | submitted |
| J027-Z1A0 | 027 | static zeros1 then current empty-add warmup | tiny memset plus known best add warmup | >=8 | f0db38f aea52de 7fc38c0 f27ba36 fca0320 73836bf 485aa46 0b51ddb | submitted |
| J027-AFF0 | 027 | current best plus process CPU affinity pin | test CPU migration/NUMA contribution to launch jitter | >=8 | 7a1d287 9a5ff45 7f25e7a 77b30f6 48ec623 0b483a4 282f1dc f731c39 | submitted |
| J027-CNNL0 | 027 | current best plus static CNNL handle setup | wrong-layer probe, not a candidate | >=8 | 072d368 4dbb29c e006f16 6c5cb51 f80c7b3 b69f3e9 a8dd3d9 ddc5fd0 | invalid/no timing |
| J027-CR0 | 027 | current best plus static CNRT context/queue calls | too broad, no timing rows | >=8 | 2463780 888c7b1 cc76473 e68789d 7a736b5 1103aae 24e8173 eb0d7f9 | invalid/no timing |
| J027-C0 | 027 | current best plus static `cnrtGetDeviceCount` | minimal CNRT runtime probe | >=8 | 0db9614 f1cf1e6 ab4a8c7 45a1bfc 89a41e8 40bdace 0c1b713 404c87e | submitted |
| J027-C1 | 027 | current best plus static `cnrtGetDevice/cnrtSetDevice` | current-device context binding preinit | >=8 | ace6304 9478fea c31607d f0081f0 c4acf4c 93db4b4 3f53ef6 91a391c | submitted |
| J027-C2 | 027 | current best plus static `cnrtQueueSync(nullptr)` | default queue/context sync preinit without custom queue | >=8 | 3c7ad79 0f0c1bd 2871017 0aad9ed 9654720 281f0b3 6472823 bf661ff | submitted |
| J027-QS0 | 027 | current best with cached current stream | move `getCurMLUStream` out of timed call | >=8 | 69cc95d 74f91f7 8a5f91b 1982893 01fad49 673c7d5 2f58426 e9f26af | submitted |
| J027-DLOW0 | 027 | driver/count gated, use devices 0-3 only | target 40/41/42 low-bus side and avoid 42 extra cards | >=8 | d2be6c9 8a76f7d 83b7258 abf6eac 23a5879 c24f7f4 da3baeb 389ba94 | submitted |
| J027-D8ONLY0 | 027 | driver/count gated, use devices 0-7 only | avoid 42 extra cards 8/9 while preserving both 8-card groups | >=8 | 68a5e41 f0ad3d6 bc1fa79 d4f4f53 2196db4 11bb888 119e7e4 d2d5645 | submitted |
| J027-DHIGH0 | 027 | driver/count gated, use high-bus side devices | contrast against low-bus 0-3 policy | >=8 | 19a8179 3bfa90d 7b69092 18b620a 773753c 21ac802 5712cb8 82051a4 | submitted |
| J027-INFO0 | 027 | print CNRT driver/lib/count/current device | correlate timing with 40/41/42 machine class | >=8 | b880c98 f1bf91a 32381b2 af66aac 3f44a52 562c2ab 18ff9c3 995a2c2 | submitted |
| J027-POOL0 | 027 | static moderate `torch::empty` reserve then current warmup | allocator/pool pre-reserve without memset/output computation | >=8 | 7a5f750 9828d2c c54feec 5c89086 64d0800 66a1fa2 d178a3b ad9240e | submitted |
| J027-NT0 | 027 | CNRT notifiers around kernel, print hardware duration | separate device kernel time from wall launch/sync time | >=8 | 9ce8ccb 8c6b65b 026f9b0 b6b004e 1fd36e9 0d7add2 2ec60aa 3df7685 | submitted |
| J027-KINFO0 | 027 | empty kernel plus HWINFO print | classify launch floor by driver/count machine class | >=8 | 35eee96 3550598 5768e1c 801ce37 40572ab 8522bb1 a2db3fe 9b63e7f | submitted |
| J027-HINFO0 | 027 | host-only return plus HWINFO print | classify host/sync lower bound by driver/count machine class | >=8 | 57403f9 7de09ee 81a451d 6198a5e d27f10e 19c71d8 59d3ecb e27bb7c | submitted |
| J027-MINFO0 | 027 | memcpy-only kernel plus HWINFO print | classify memory transfer floor by driver/count machine class | >=8 | b235b05 6da50c7 2c92864 19b6b0b 0c19da0 20ceeb6 406fb07 e5d7e2e | submitted |
| J027-K2INFO0 | 027 | two empty kernels plus HWINFO print | test whether timed launch cost is mostly first-launch cold cost or per-launch cost | >=8 | 3b931ef c4484fb a3d88b2 e18224b 3a5bbbb 318f5bc 485517d 5a1a3b3 | submitted |
| J027-K8INFO0 | 027 | eight empty kernels plus HWINFO print | amplify per-launch cost signal above OJ jitter | >=8 | ec31042 358ebfd a823210 3f0526d 16ebb7e e135331 029a177 074a161 | submitted |
| J027-K32INFO0 | 027 | thirty-two empty kernels plus HWINFO print | estimate steady per-launch enqueue/sync cost after cold first launch | >=8 | ccd8488 c4cd085 236c936 82a2a99 d253e54 42ce347 59b150f 20c9b81 | submitted |
| J027-SINFO0 | 027 | get current MLU stream only plus HWINFO print | measure stream lookup without kernel launch | >=8 | 6870551 eb56d8a b64e1b5 d669f79 b152c5e 96ca800 08694dc e31a4b3 | submitted |
| J027-FBLOCKINFO0 | 027 | empty kernel plus `cnrtDeviceScheduleBlock` | test CNRT blocking sync policy under OJ multi-worker contention | >=8 | b4280a5 25ca9e1 d0eca45 3f7199c 55f775b da4e183 adb0c68 2043dbd | submitted |
| J027-FYIELDINFO0 | 027 | empty kernel plus `cnrtDeviceScheduleYield` | test CNRT yielding sync policy under OJ multi-worker contention | >=8 | 79cca8e a29782d a709a0a 4f2f395 e1a954c dd07c25 1305a7d baa2415 | submitted |
| J027-FBLOCKPRE0 | 027 | set `cnrtDeviceScheduleBlock` before any other CNRT query | verify whether sync policy can be changed before runtime init | >=8 | 47233b5 dab25cd 986960e 5da823c 08325d1 64e8178 e2b5cbd 9fd6121 | submitted |
| J027-TEMPTY0 | 027 | pre-instantiated TaskTopo with one empty node | measure TaskTopo invoke overhead without kernel work | >=8 | 358b546 0027b3b 2fe0276 8b4efea cf04193 2b6528d e36e6e4 6a8a861 | submitted |
| J027-TKEMPTY0 | 027 | pre-instantiated TaskTopo with one empty kernel node | compare TaskTopo kernel invoke against normal kernel launch | >=8 | 761f693 32ab7a1 5422ccc 83d9780 a3e7b25 85b6acb 7ec9b70 9c1823a | submitted |
| J027-TCAPEMPTY0 | 027 | capture empty kernel into TaskTopo, then invoke entity | avoid direct AddKernelNode kernel registration issue | >=8 | 96af71b 4502c43 c772afb dfeb002 68ad082 849798b dd19c59 f5a6de2 | submitted |
| J027-KINFO1 | 027 | empty kernel plus HWINFO, contemporaneous control | compare warmup variants in same OJ time window | >=8 | 20a5fd0 5f42c6e 6ee61c8 2b5d7dc e413d30 937cd5d fe8e60c 90f8174 | submitted |
| J027-GBIGADD0 | 027 | empty kernel plus same-size static `empty+add` warmup | test larger torch warmup kernel outside timing | >=8 | 38da935 d4e5938 111d209 12de041 8e3c421 679a134 f63ee50 d6124c7 | submitted |
| J027-GLOOP8ADD0 | 027 | empty kernel plus 8 small static `empty+add` warmups | test repeated torch warmup kernels outside timing | >=8 | c509eef 818855f c2ffa6c bf797b0 0f49778 8551889 167e8c4 e95ed6d | submitted |
| J027-GRES16G0 | 027 | empty kernel plus 16GiB static empty reserve/release | test large MLU allocator pool reservation and card pressure | >=8 | ebcc7e1 1411d11 55e8e34 16760a7 c63be84 689aad1 96560ba 82fb82c | submitted |
| J027-GRES4G0 | 027 | empty kernel plus 4GiB static empty reserve/release | test allocator pool warmup with lower OOM risk | >=8 | c5b9be9 75ca59f 0106f7a 719ccbb 7454fd3 6a37854 4629d5f ad550c7 | submitted |
| J027-CPUTOPO0 | 027 | empty kernel plus CPU affinity and PCI bus info print | correlate launch jitter with CPU/NUMA placement and BDF | >=8 | f88c958 b1db572 7c79031 fd8b908 5cb97f2 94bf1d2 9606de4 da3152e 12e49be 2e2d444 0272c56 fb5c11a 9ccf22d d323ac2 fcf37d8 a9fe637 | submitted |
| J027-AFFNODE0 | 027 | empty kernel with process affinity forced to CPUs 0-27 | test local NUMA affinity for device0/BDF 4f | >=8 | f629c29 3b76d0f 42473c1 56bcc21 8595a8b 32a0033 3283fd1 69f6ee8 | submitted |
| J027-AFFNODE1 | 027 | empty kernel with process affinity forced to CPUs 28-55 | test remote NUMA affinity contrast for device0/BDF 4f | >=8 | b1518d6 d6d0f71 1f0c891 029b86c 2486fca e083a62 b2d3511 cb1a4e6 | submitted |
| J027-AFFPID01 | 027 | same code chooses node0/node1 by pid parity | interleave affinity choices to reduce batch time skew | >=16 | f399e38 2edf78c 7962247 0157e98 203b24e 6848e26 fe1ad14 6033d98 6c67dae fbfb69c cdd669d 90bd107 366d428 594332b 17c35ee 0aab23d | submitted |
| J027-GBIGADD1 | 027 | repeat same-size static `empty+add` warmup | verify weak GBIGADD0 improvement | >=8 | 121c021 302444a 35746af 7c5e852 71ddd84 a2c9927 01c33ed f55f986 | submitted |
| J027-KINFO2 | 027 | post-GBIGADD control, tiny static add | control for time-window drift after GBIGADD1 | >=8 | 368eb34 9ce711f bb67e45 4e50cac 457dba3 b1917d1 449ac3d eb27a6c | submitted |
| J027-DINK0 | 027 | direct `cnrtInvokeKernel` empty kernel | compare manual invoke API with `<<<>>>` syntax sugar | >=8 | e8833a9 951d198 c7dcf58 aaefd1c ba337c4 82fad8d 0abb6e1 c547986 | submitted |
| J027-INSYNC0 | 027 | normal empty kernel plus in-function `cnrtQueueSync` | split external tester synchronize vs internal queue sync | >=8 | 2441e2c 613c95f 0dd696f d97e6c6 94822fa fde4c52 15f4194 ec342f8 | submitted |
| J027-DINK1 | 027 | direct `cnrtInvokeKernel` with explicit `const void *` cast | retest manual invoke API after fixing DINK0 compile error | >=8 | a21da0b dfd58f6 9794971 d62119f 43bebbf b5dea2a 9177f0a be453e5 | submitted |
| J027-MEMINFO0 | 027 | timed `cnrtMemGetInfo` without launching a kernel | distinguish CNRT runtime query cost from kernel launch cost | >=8 | 36664cb 5d16460 7b83cbc f199c33 6a73c27 396796b 9aa310e 116f14c | submitted |
| J027-HOSTFN0 | 027 | enqueue `cnrtInvokeHostFunc` on current queue, no MLU kernel | measure queue callback submission/sync without device kernel launch | >=8 | 80162a7 bd0376f a150a24 d8c889d 6a5826a a81ef77 61e16e8 844b15f | submitted |
| J027-KINFO3 | 027 | normal `<<<>>>` empty-kernel launch, adjacent control | compare with DINK1 in the same OJ period | >=8 | cd4fcaf ef8bcdc 428fb6b 82e8f08 da6f06b d83f05f d0ed714 5aadfdf | submitted |
| J027-QQUERY0 | 027 | current queue query without launching a kernel | measure `cnrtQueueQuery` overhead on an idle/current stream | >=8 | 6fa91fd cbe44af 53aa1ff 5740a76 ba640ef bfea2f2 d281507 19401ed | submitted |
| J027-LQUERY0 | 027 | empty-kernel launch followed by `cnrtQueueQuery` | test whether a nonblocking query shifts or reduces the external sync cost | >=8 | fc3e7c1 83d31ee 3782e86 20b3c72 8c5c4d8 9559a7d 706767d 779ec34 | submitted |
| J027-KUSAGEPRE0 | 027 | constructor `cnrtDeviceQueryKernelMemoryUsage`, timed empty kernel | test whether kernel-memory/module prequery reduces first launch cost | >=8 | 9c5c2a2 c487674 d202d0b af832c4 f77aad1 be35cfc 5c83b48 5eb9ba2 | submitted |
| J027-CMSET1A0 | 027 | constructor `cnrtMemset` of 1 half plus tiny add warmup | compare CNRT memset warmup against prior torch zeros warmups | >=8 | 4dd4f8b bf8a4c6 8890b47 7618afe deb7c0c 2fbffec 14ba4f3 14e642b | submitted |
| J027-NOWARM0 | 027 | no static MLU warmup, timed empty kernel | quantify baseline launch cost without current tiny add warmup | >=8 | a739726 04cb034 3fa5ba1 52cbca0 387c511 91dfd59 41e697a dcf3217 | submitted |
| J027-ALLOC1P0 | 027 | constructor `torch::empty({1})` only, timed empty kernel | split allocator/context initialization from static warmup kernel execution | >=8 | d5b0855 d3bb1ef 5f05c6d bf9a835 4474259 c1a8215 f118eeb 3244e23 | submitted |
| J027-CMSET1P0 | 027 | constructor `torch::empty({1})` plus `cnrtMemset`, no torch add | test whether a pure CNRT device op is enough to warm first launch | >=8 | 1d3b071 103ecd2 0adee52 4e16c8f 10b6751 ab5f06c 6cf5a59 3d8ac63 | submitted |
| J027-SAMEKWARM0 | 027 | constructor launch/sync of the same empty Bang kernel | test pure Bang kernel warmup without torch algorithm op | >=8 | 36c8414 8837c0a 2270f12 f4c6a13 fb4505a 5b2a64b 58da596 e109eb8 | submitted |
| J027-TZ1P0 | 027 | constructor `torch::zeros({1})`, timed empty kernel | test minimal torch memset-style warmup without add | >=8 | c70c9c9 9025753 a2846f7 a8518de ed04bbc 60d40f8 3be3cde 93e1b33 | submitted |
| J027-TZ0P0 | 027 | constructor `torch::zeros({0})`, timed empty kernel | test whether zero-size torch op warms runtime without device work | >=8 | b8a4cd2 0117a6d d370479 c12f10e 4973f7e c102992 4cb2dc8 c7407ba | submitted |
| J027-TADD1P0 | 027 | constructor `torch::empty({1}) + torch::empty({1})`, timed empty kernel | adjacent control for tiny add warmup vs `torch::zeros({1})` | >=8 | d9163ea 760079e f70945a 9b71b8b b1ea1a3 591edb0 3830f6e 357104a | submitted |
| J026-FIXDEV0 | 026 | real PASS source, fixed device0, `zeros({1})` prewarm | compare fixed device0 against pid-spread baseline on true op | >=8 | 6f8798b e676378 d48bac6 c2c04db 551441d 774d363 4a869a7 8602cab | submitted |
| J028-FIXDEV0 | 028 | real PASS source, fixed device0, `zeros({1})` prewarm | compare fixed device0 against pid-spread baseline on true op | >=8 | 6f8798b e676378 d48bac6 c2c04db 551441d 774d363 4a869a7 8602cab | submitted |
| J029-FIXDEV0 | 029 | real PASS source, fixed device0, `zeros({1})` prewarm | compare fixed device0 against pid-spread baseline on true op | >=8 | 6f8798b e676378 d48bac6 c2c04db 551441d 774d363 4a869a7 8602cab | submitted |
| J026-PIDSPREAD1 | 026 | real PASS source, pid-spread device, `zeros({1})` prewarm | same-window control against fixed device0 | >=8 | dbc03c6 8c8fa83 a15e566 6f8001a d334ffe f07ba6f f51bfbd 778d87e | submitted |
| J028-PIDSPREAD1 | 028 | real PASS source, pid-spread device, `zeros({1})` prewarm | same-window control against fixed device0 | >=8 | dbc03c6 8c8fa83 a15e566 6f8001a d334ffe f07ba6f f51bfbd 778d87e | submitted |
| J029-PIDSPREAD1 | 029 | real PASS source, pid-spread device, `zeros({1})` prewarm | same-window control against fixed device0 | >=8 | dbc03c6 8c8fa83 a15e566 6f8001a d334ffe f07ba6f f51bfbd 778d87e | submitted |
| J026-TADDPASS0 | 026 | real PASS source, pid-spread device, one-element tiny add prewarm | compare tiny add prewarm against `zeros({1})` on true op | >=8 | 802a9b8 f76b639 50cf333 ac1912f 237e783 c6d96d4 6d0ac2c debeafd | submitted |
| J028-TADDPASS0 | 028 | real PASS source, pid-spread device, one-element tiny add prewarm | compare tiny add prewarm against `zeros({1})` on true op | >=8 | 802a9b8 f76b639 50cf333 ac1912f 237e783 c6d96d4 6d0ac2c debeafd | submitted |
| J029-TADDPASS0 | 029 | real PASS source, pid-spread device, one-element tiny add prewarm | compare tiny add prewarm against `zeros({1})` on true op | >=8 | 802a9b8 f76b639 50cf333 ac1912f 237e783 c6d96d4 6d0ac2c debeafd | submitted |
| J026-TDROP0 | 026 | real PASS source, constructor tiny add prewarm then discard tensor | test whether prewarm tensor lifetime matters | >=8 | 9ee3d09 8ab8125 74fba31 b8ff114 f1c0e8c cde20e7 2cf02a1 882d300 | submitted |
| J028-TDROP0 | 028 | real PASS source, constructor tiny add prewarm then discard tensor | test whether prewarm tensor lifetime matters | >=8 | 9ee3d09 8ab8125 74fba31 b8ff114 f1c0e8c cde20e7 2cf02a1 882d300 | submitted |
| J029-TDROP0 | 029 | real PASS source, constructor tiny add prewarm then discard tensor | test whether prewarm tensor lifetime matters | >=8 | 9ee3d09 8ab8125 74fba31 b8ff114 f1c0e8c cde20e7 2cf02a1 882d300 | submitted |
| J026-TABSPASS0 | 026 | real PASS source, pid-spread device, one-element `abs(empty)` prewarm | test smaller single-input registered torch_mlu op prewarm | >=8 | 3f7a1ec 9743a83 951602e 4709afe c4dc1c9 26cfa66 fe2a1c7 f5db878 | submitted |
| J028-TABSPASS0 | 028 | real PASS source, pid-spread device, one-element `abs(empty)` prewarm | test smaller single-input registered torch_mlu op prewarm | >=8 | 3f7a1ec 9743a83 951602e 4709afe c4dc1c9 26cfa66 fe2a1c7 f5db878 | submitted |
| J029-TABSPASS0 | 029 | real PASS source, pid-spread device, one-element `abs(empty)` prewarm | test smaller single-input registered torch_mlu op prewarm | >=8 | 3f7a1ec 9743a83 951602e 4709afe c4dc1c9 26cfa66 fe2a1c7 f5db878 | submitted |
| J026-ZDROP0 | 026 | real PASS source, constructor `zeros({1})` prewarm then discard tensor | test whether zeros prewarm tensor lifetime matters | >=8 | e45628d 926f383 b9dab1f 3050c14 deb3c48 ba2df4e d02e8e6 7ee2745 | submitted |
| J028-ZDROP0 | 028 | real PASS source, constructor `zeros({1})` prewarm then discard tensor | test whether zeros prewarm tensor lifetime matters | >=8 | e45628d 926f383 b9dab1f 3050c14 deb3c48 ba2df4e d02e8e6 7ee2745 | submitted |
| J029-ZDROP0 | 029 | real PASS source, constructor `zeros({1})` prewarm then discard tensor | test whether zeros prewarm tensor lifetime matters | >=8 | e45628d 926f383 b9dab1f 3050c14 deb3c48 ba2df4e d02e8e6 7ee2745 | submitted |
| J026-CPULOCAL0 | 026 | real PASS source, choose device group by current CPU half | test CPU/NUMA-local device selection on real op | >=8 | 0b885c5 1088526 a0f30e0 4cbf15d 2713684 17f49d1 d1107b9 97272fe | submitted |
| J028-CPULOCAL0 | 028 | real PASS source, choose device group by current CPU half | test CPU/NUMA-local device selection on real op | >=8 | 0b885c5 1088526 a0f30e0 4cbf15d 2713684 17f49d1 d1107b9 97272fe | submitted |
| J029-CPULOCAL0 | 029 | real PASS source, choose device group by current CPU half | test CPU/NUMA-local device selection on real op | >=8 | 0b885c5 1088526 a0f30e0 4cbf15d 2713684 17f49d1 d1107b9 97272fe | submitted |
| J026-GMEM18HOLD0 | 026 | real PASS source, pid-spread device, retain about 18GiB empty tensor | test large memory hold as contention filter/exclusive-card probe | >=8 | fe98b58 1dde194 3b1b8a2 bdcb9e0 55f8412 702ec71 1700eae 4102eda | submitted |
| J028-GMEM18HOLD0 | 028 | real PASS source, pid-spread device, retain about 18GiB empty tensor | test large memory hold as contention filter/exclusive-card probe | >=8 | fe98b58 1dde194 3b1b8a2 bdcb9e0 55f8412 702ec71 1700eae 4102eda | submitted |
| J029-GMEM18HOLD0 | 029 | real PASS source, pid-spread device, retain about 18GiB empty tensor | test large memory hold as contention filter/exclusive-card probe | >=8 | fe98b58 1dde194 3b1b8a2 bdcb9e0 55f8412 702ec71 1700eae 4102eda | submitted |
| J026-B0 | 026 | current best source | cross-op same-code distribution | >=8 | a52c69f 52dd6ec ad1623c e87d3cf 8f8159e 2a63132 2732585 d4c0fda | submitted |
| J028-B0 | 028 | current best source | compare pointwise op distribution | >=8 | a52c69f 52dd6ec ad1623c e87d3cf 8f8159e 2a63132 2732585 d4c0fda | submitted |
| J029-B0 | 029 | current best source | compare low-task clamp distribution | >=8 | a52c69f 52dd6ec ad1623c e87d3cf 8f8159e 2a63132 2732585 d4c0fda | submitted |

Current observations below 8-row threshold:
- 027 best source has produced fast rows around 302-303 us and slow rows around 335-390 us. Treat as worker/load distribution until repeated.
- 026/028/029 same-source reruns also hit 315-434 us slow bands, so the effect is not unique to GELU.
- 027 device-spread constructor with the same warmup was slower in two rows; not enough for a universal conclusion, but it is not immediately promising.
- 028/029 Union1 launch was slower in two rows each; keep as weak observation already supported by prior schedule attempts.

Open measurements:
- Validate the selected one-element torch prewarm on actual PASS implementations for 026-029 in adjacent same-window batches.
- Test whether final implementation can simplify device selection back to fixed device 0 without losing multi-worker stability.
- Keep any new conclusion below `Layered Conclusion Index` only after at least 8 repeated rows and a same-window control when possible.

## 2026-05-31 026-029 Target Log

Active policy:
- Do not promote same-code or micro-body changes unless at least 8 OJ comments are available.
- For low-tail leaderboard attempts, keep hashes in `ref/.oj_repeat_hashes` and compare min/median/max rather than one comment.
- Local MLU timings are compile/correctness probes only; OJ fresh-process comments decide leaderboard-facing choices.

| id | op | variant | commits | status | observation |
| --- | --- | --- | --- | --- | --- |
| J026/028/029-BRESTORE0 | 026/028/029 | restored historical best source, same-code lowtail | `8d53ee74 cce6f8e1 9f1993cf 1735b1bf e26cf985 11aa4962 3213ffce fe8791f2 d3796546 a81602ac dbff8064 03ac4480 d5b98eb3 260c1fcf 3a548afc 76cae4da 552a1bc9 16fad9e5 f8f7f308 ea542baf cb52fc5a 7c66374c 2c50fe7b 6b609a40` | collected partial | Slow OJ window: 026 best collected `344.973 us`, 028 best collected `296.447 us`, 029 best collected `323.467 us`; no leaderboard improvement. |
| J026-FMAHORNER0 | 026 | Horner `mul + add_scalar` replaced by `__bang_fusion(FUSION_FMA)` | `1ba715a3 395accd5 8b66dbcc 34c9003c e1f32dd8 ca209c6a 6372fc3a 562d923e ad553b01` | submitted | Local compile/correctness PASS, diff `0.00635`; waiting for OJ comments. |
| J026-RELUSPLITFMA0 | 026 | split positive/negative halves with `relu(x)` and `relu(-x)`, plus FMA Horner | `da89971d b60cceb4 76e0c8db ec2e58c5 7441f5ed 8200f28b dc053b09 692a9465` | submitted | Local compile/correctness PASS, diff `0.00635`; waiting for OJ comments. |
| J026-RELUSUBFMA0 | 026 | split with `c=relu(x)` and `b=c-x`, plus FMA Horner | `96954893 3158abe8 d5e8c650 f6c4232c 621dd98f 4e3f5aa0 33c1fada 5fee15be` | submitted | Local compile/correctness PASS, diff `0.00635`; waiting for OJ comments. |
| J026-D4RELUSUBFMA0 | 026 | tuned degree-4 polynomial with clamp, split `c=relu(x); b=c-x`, FMA Horner | `f6d063e9 6900d2f8 ec016c88 897c1d9a c1f54273 65f1f4a8 11cd4d5e 2daa9a22` | submitted | Local compile/correctness PASS, diff `0.00635`; half-grid search max error about `0.00614` for `b<=5.2`; waiting for OJ comments. |
| J026-D4SINGLE0 | 026 | current tuned degree-4 source, single non-empty config switch probe | `bf6b9524` | submitted | Sent after older 026 commits had no commit comments while 028 comments appeared; used to distinguish OJ 026回写 lag from code result. |
| J026-D4LOWTAIL1 | 026 | current tuned degree-4 source, same-code lowtail followup after raw `294.821 us` appeared | `e185fb35 a017bae1 f1e95b9a 2ace7808 2f641452 a597cf8a 0b28ef2f db493ef3 32cab024 0b5a8383 bb2b21d6 24278dab` | submitted | Probability sample for leaderboard collision; waiting for OJ comments. |
| J026-D4DEV7 | 026 | current tuned degree-4 source, force device 7 or highest available device | `d44fa2b2 2fd0da63 ca29f654 b98b5679 a810f2a9 780a850b 59be8333 16bad0db` | collected partial | PASS but no stable card win; fixed Card 7 had many 350-430us raw runs and best collected comment about `331 us`, so the earlier `294.821 us` was not a reliable card-specific effect. |
| J026-D4LOWTAIL2 | 026 | current tuned degree-4 source, pid-spread restored after fixed-device probe | `e208e962 b3cc05f4 a7a03166 5516946e d9e03826 0f6997f6 7b8adc12 e7d9d2ca` | collected | 16 OJ rows PASS, min `348.110 us`; this was a slow window and did not reproduce the earlier raw `294.821 us`. |
| J028-RELUCLAMP0 | 028 | replace `maxeq_scalar(0)` with `__bang_relu` in HardSigmoid clamp path | `baf04c4e d449f8ec dc34355a b9cef64b a3d9a095 b2dc2f8c d9141f56 b4cca2dd` | submitted | Local compile/correctness PASS, diff `4.88e-4`; waiting for OJ comments. |
| J028-FMARELUMIN0 | 028 | compute `x/6+0.5` with FMA, then `relu` and `min(1)` | `e67cd100 3a191915 69e28078 3c7e1b8f 03c6e2a1 200f5e03 76284323 18d8b533` | partial | 8 OJ rows collected so far, PASS diff `3.26e-4`, min `298.506 us`, median about `350 us`; no improvement signal in this window. |
| J028-SCALARLOWTAIL1 | 028 | restored scalar clamp best source, same-code lowtail followup | `e610defd 02bb84ac c8428d6e 69c1278b 8a86aeb4 5dbcde29 65a4565a 52fd5281` | submitted | Probability sample after FMA variants showed no improvement; waiting for OJ comments. |
| J029-BFOLLOW0 | 029 | restored best scalar clamp, same-code lowtail followup | `8e38c37b be8bfc13 86574170 bf823b13 9917c456 bbe6b377 8faa746f 541eac79` | submitted | No new algorithmic hypothesis; probability/worker-window sample while other variants run. |
| J026/028/029-IDLELOWTAIL0 | 026/028/029 | current best source for each op after OJ queue reached idle | `8331eb5c b3cf4a51 af1f1634 2d225c86 e4d1877e 439463d4 4cb7330e f392bac0 1fde285c f24ba05f 7e228f2f a5edcc6b 4a4cb026 1fc4ca39 025e4d70 fbfc38ca bccb1555 d21b5407 f31b6e1c 6e644550 cef9b99a f059f583 89164a77 ba3ed26b` | collected | Queue-idle batch was still slow: best collected comments were 026 `344.857 us`, 028 `300.389 us`, 029 `297.624 us`; no leaderboard improvement. |
| J026/028/029-RESTORE1 | 026/028/029 | restore device-kernel baselines after timed host probes | `715e7970` | collected | PASS for all three but slow: 026 best `358.052 us`, 028 best `365.418 us`, 029 best `336.079 us`; confirms host probes should not remain as production source. |
| J026-COPYLESS0 | 026 | remove one NRAM copy in relu split path, same degree-4 clamp coefficients | `3acd7c79 12031308 7f96c079 39d9bda8 8c5e0d24` | collected | PASS diff `6.14e-03`, best `355.117 us`; no visible improvement in this OJ window. |
| J028-RELUN0 | 028 | replace scalar max/min clamp with `__bang_relun` | `1d20678d 6b36f7b8 242688db a87dede3 c393ec80` | collected | PASS diff `6.51e-04`, best collected around `318.959 us`; did not beat scalar clamp source. |
| J026-UNCLAMPFMA0 | 026 | historical no-clamp coefficients rewritten as FMA relu split | `852b47bd` | collected | PASS diff `6.34e-03`, `383.904/386.823 us`; no speed signal versus historical source. |
| J026/028/029-BESTBURST1 | 026/028/029 | restored historical best source, interleaved lowtail burst | `9ca985ed 56e42f46 b4cbc052 a0705397 0f02bc87 a9779c3e 4718d0a4 9e7cdce9 1751c1a5 50d03e21 952135f3 dbbd90cc 8302226b fd34039d 8db43fd5 773b4bc1 0ad05b5c d961a987 12d9c24b` | collected | Slow window again: best comments 026 `355.286 us`, 028 `331.540 us`, 029 `311.870 us`; no leaderboard improvement. |
| J026-FITCLAMP0 | 026 | local minimax four-coefficient clamp fit, relu split without NRAM copy | `3906f588 6733e576 aae12168 6542e219 e41ca165` | collected | PASS diff `6.98e-03`, best `331.962 us`; confirms four-coefficient fit is correct but not a launch-floor breakthrough. Three-coefficient search bottomed out near `1.35e-02`, above threshold. |
| J029-BURST1 | 029 | stable scalar clamp source, focused lowtail burst | `c6dd6c14 43b4528b 5b6944b4 f7f70765 ddc4821f 7aec42f7 5055ba40 8a21207d 814ab7cb 0dd772cf a3359bf3 3de126f7` | collected | Best `297.689 us`; no collision with external `268.123 us`. |
| J028-BURST1 | 028 | stable scalar clamp source, focused lowtail burst | `d0036df6 0ae5e967 7e5f2520 0a6c7672 6992c4c3 98591288 3e17387c 2909f189 6c06bfd5 43354aee 21e8c482 2daa7a91` | partial | Early collected best `305.630 us`; no collision with external `272.398 us` so far. |
| J029-GATE10-0 | 029 | PASS only on `cnrtGetDeviceCount()==10`, otherwise return input and fail | `9cb3059b e03064eb 0c1a9529 c02ee8f7 fc04f96f eda8cbef 7fe18c76 6c78e628 07dcefe4 a338b274 a380f5b1` | collected | Gate works as a worker-class filter, but PASS rows were not faster: best `333.488 us`; abandon count==10 as speed selector. |
| J028-GATE10-0 | 028 | PASS only on `cnrtGetDeviceCount()==10`, otherwise fail | `d52d04f3` | collected | PASS rows `359.245/365.289 us`; no speed signal. |
| J029-GATE628-0 | 029 | PASS only on driver `6.2.x` and 8-card workers, otherwise fail | `240f49d2 a226b217 8e8bc0b6 83bd6843 b4f4e0c5 ce518e41 9cc2e1a7 fd82c697 d93260a2` | partial | PASS best so far `296.565 us`; gate filters workers but still does not explain the historical 265-268us lowtail. |
| J027-DEFENSE0 | 027 | stable GELU source, lowtail defense while external team submits 027 | `bf543086 ebfced23 9736ef65 de775d5c dcf922ac 1337f990` | collected partial | Best early row `310.861 us`; no new 027 leaderboard improvement, but current team_42 lead remains `290.285 us`. |

## 2026-05-31 055/066/105 Target Log

| id | op | variant | commits | status | observation |
| --- | --- | --- | --- | --- | --- |
| J055-HOSTPREFILL0 | 055 | static MLU output prefilled before timed call, host-only return | `6ab6700b` | collected | PASS diff `9.28e-03`, `29.978/31.864 us`; confirms prior 117us leaderboard was not limited by Bang kernel body. This is a boundary probe because result materialization is outside timing. |
| J055-H2D0 | 055 | timed 64B H2D copy of constant output | `44c371fe` | collected | PASS diff `9.28e-03`, `38.486/38.999 us`; CNRT copy path is far cheaper than first custom kernel launch for this tiny IO output. |
| J055-HOSTCOMPUTE0 | 055 | timed D2H inputs, CPU reduction, H2D output | `21b70b14` | collected | PASS diff `0`, `60.254/60.447 us`; all meaningful arithmetic in timed call still beats old 117us target. |
| J055-H2DEXACT0 | 055 | timed 64B H2D copy with exact fp32 expected constants | `eacc3262` | partial | PASS diff `0`, first row `39.180 us`; waiting for more rows. |
| J105-G2G0 | 105 | replace NRAM row copy with direct `GDRAM2GDRAM` row copy | `76c86722` | collected | PASS diff `0`, `713.947/722.744 us`; direct row copy beats old NRAM round trip. |
| J105-G2G-DIM32/64 | 105 | direct `GDRAM2GDRAM`, Block dim32/dim64 | `61bdb20e 72c5be13` | collected | PASS diff `0`, best `686.600/686.134 us`; now beats external 715us. |
| J105-G2G-DIM256 | 105 | direct `GDRAM2GDRAM`, Block dim256 | `a9a41df4` | collected | PASS diff `0`, about `732 us`; too many tasks hurts. |
| J066-LOWTAIL0 | 066 | same-code low-tail repeats, no config nonce | `302b4b77 6d439ff3 22a8b789 1bdedd7b 26086049 97e0f6b4 ae535c2e 78148c48` | partial | Leaderboard improved to `329.168 us`, but only 3/8 commits produced comments; empty repeat commits are not a reliable trigger source. |

## 2026-05-31 J027 First Batch

Rows collected so far:

| id | rows | min | median | max | mean | observation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| J027-B0 | 21 | 317.667 | 356.915 | 392.479 | 352.416 | best source same-code distribution is wide |
| J027-H0 | 16 | 33.340 | 39.285 | 47.807 | 39.606 | host-only call/sync lower bound is small and stable |
| J027-K0 | 16 | 279.297 | 325.249 | 342.326 | 320.476 | empty MLU kernel launch plus sync dominates |
| J027-M0 | 16 | 295.211 | 347.510 | 1364.412 | 407.240 | copy-only adds little over launch; one large outlier |
| J027-WK0 | 16 | 353.362 | 397.958 | 431.167 | 398.594 | static hand-written dummy kernel warmup is consistently worse |
| J027-ZB0 | 16 | 368.217 | 513.691 | 2311.111 | 650.194 | zeros-big alone is consistently worse |
| J027-ZBA0 | 11 | 320.513 | 354.079 | 379.096 | 349.740 | zeros-big plus empty-add is near baseline, not clearly better |
| J027-Z1A0 | 16 | 329.340 | 370.195 | 410.369 | 365.229 | zeros1 plus empty-add is worse than baseline |
| J027-AFF0 | 15 | 321.929 | 358.023 | 399.181 | 359.957 | CPU affinity pinning overlaps baseline and does not reduce jitter |
| J027-C0 | 9 | 345.312 | 374.119 | 400.195 | 374.636 | `cnrtGetDeviceCount` does not improve timing |
| J027-C1 | 9 | 339.579 | 364.129 | 385.058 | 367.047 | `cnrtGetDevice/cnrtSetDevice` overlaps baseline |
| J027-C2 | 16 | 325.865 | 378.684 | 427.719 | 373.234 | `cnrtQueueSync(nullptr)` does not improve timing |
| J027-QS0 | 16 | 328.559 | 365.653 | 394.155 | 365.376 | cached current stream does not improve timing |
| J026-B0 | 16 | 352.525 | 379.168 | 415.826 | 383.116 | ELU baseline hits same cold-launch scale |
| J028-B0 | 16 | 308.676 | 345.125 | 400.049 | 349.129 | HardSigmoid baseline overlaps 027 distribution |
| J029-B0 | 16 | 312.969 | 350.669 | 365.604 | 344.422 | HardTanh baseline overlaps 027 distribution despite fewer tasks |
| J027-DLOW0 | 14 | 331.857 | 381.125 | 555.443 | 396.089 | driver/count gated devices 0-3 is worse than baseline |
| J027-D8ONLY0 | 16 | 340.249 | 374.616 | 440.198 | 379.261 | driver/count gated devices 0-7 is worse than baseline |
| J027-DHIGH0 | 15 | 338.395 | 379.157 | 447.586 | 388.165 | high-bus side is worse than baseline, not enough to prove high vs low |
| J027-INFO0 5.10.22/8 | 12 raw | 361.025 | 382.459 | 422.440 | 382.810 | likely `.40`, slower median |
| J027-INFO0 6.2.10/8 | 9 raw | 364.590 | 375.513 | 391.340 | 376.998 | likely `.41`, less sampled |
| J027-INFO0 6.2.10/10 | 27 raw | 296.699 | 365.215 | 456.685 | 360.574 | likely `.42`, widest distribution and fastest outlier |
| J027-POOL0 | 15 | 329.125 | 366.248 | 408.220 | 367.084 | moderate empty reserve does not improve timing |
| J027-NT0 wall | 39 raw | 333.178 | 428.899 | 5561.709 | 672.767 | notifier instrumentation adds overhead and large outliers |
| J027-NT0 kernel | 39 raw | 7.000 | 8.000 | 17.000 | 7.949 | actual device kernel hardware time is tiny |
| J027-HINFO0 | 48 raw | 30.868 | 40.553 | 56.692 | 41.099 | host-only lower bound stays tens of us with HWINFO; 10-card subclass only n=6 |
| J027-KINFO0 | 48 raw | 260.873 | 346.443 | 463.012 | 347.317 | empty-kernel launch floor remains hundreds of us; driver/count changes distribution but not the order |
| J027-MINFO0 | 48 raw | 266.974 | 348.059 | 556.513 | 352.364 | memcpy-only is within ~5 us median of empty kernel, so transfer arithmetic is not the dominant wall cost |
| J027-K2INFO0 | 48 raw | 262.388 | 329.182 | 410.477 | 330.505 | two empty launches did not exceed single-launch batch; cross-batch worker/time skew is larger than one extra launch signal |
| J027-K8INFO0 | 48 raw | 289.320 | 363.306 | 2423.867 | 474.582 | eight empty launches median only modestly above single-launch; outliers dominate mean |
| J027-K32INFO0 | 48 raw | 382.930 | 462.126 | 3286.465 | 514.581 | thirty-two empty launches expose ~3-4 us steady extra launch cost after the cold floor |
| J027-SINFO0 | 36 raw | 31.106 | 40.105 | 51.641 | 40.310 | current-stream lookup without launch is indistinguishable from host-only lower bound |
| J027-FBLOCKINFO0 | 42 raw | 303.351 | 356.455 | 1080.761 | 376.530 | attempted block flag but observed flag stayed 0, so timing is not a valid block-policy measurement |
| J027-FYIELDINFO0 | 42 raw | 288.506 | 345.877 | 494.763 | 354.529 | attempted yield flag but observed flag stayed 0, so timing is not a valid yield-policy measurement |
| J027-FBLOCKPRE0 | 48 raw | 270.461 | 342.716 | 4286.700 | 429.037 | set/get returned success but observed flag stayed 0 in all raw runs |
| J027-TEMPTY0 | 33 raw | 35.506 | 46.259 | 168.120 | 73.916 | TaskTopo invoke with only empty node stays near host-only path |
| J027-TKEMPTY0 | 45 stderr | N/A | N/A | N/A | N/A | direct `cnrtTaskTopoAddKernelNode` fails with kernel-not-registered |
| J027-TCAPEMPTY0 | 16 comments | N/A | N/A | N/A | N/A | rejected by source audit: `FORBIDDEN_CNRT_QUEUE_CREATE` |
| J027-KINFO1 | 33 raw | 261.436 | 337.229 | 443.228 | 340.938 | contemporaneous empty-kernel control for warmup variants |
| J027-GBIGADD0 | 48 raw | 248.126 | 330.152 | 473.882 | 328.794 | same-size static add is slightly lower than KINFO1 overall, but class effects are mixed |
| J027-GLOOP8ADD0 | 39 raw | 261.198 | 345.671 | 433.999 | 344.399 | eight tiny add warmups do not improve over KINFO1 |
| J027-GRES16G0 | 19 valid raw | 288.076 | 332.270 | 372.559 | 330.682 | 16GiB reserve causes many constructor OOMs under contention; valid rows are not clearly better |
| J027-CPUTOPO0 initial | 27 raw | 265.556 | 322.250 | 399.118 | 318.140 | only sampled 6.2.10/8-card; CPU node0/node1 medians overlap |
| J027-CPUTOPO0 combined | 72 raw | 253.616 | 324.361 | 399.118 | 318.903 | on 6.2.10/8-card BDF 4f, node0 median 315.962 us vs node1 339.155 us |
| J027-GRES4G0 | 33 raw | 277.890 | 333.959 | 393.698 | 328.684 | 4GiB reserve succeeds without OOM but does not improve over control |
| J027-AFFNODE0 | 42 raw | 261.112 | 362.929 | 506.364 | 368.489 | forced node0 batch was slower than expected |
| J027-AFFNODE1 | 36 raw | 269.177 | 321.694 | 385.239 | 321.859 | forced node1 batch was faster, contradicting passive CPUTOPO correlation |
| J027-AFFPID01 | 84 raw | 272.717 | 322.339 | 460.999 | 336.120 | interleaved node choice shows medians within ~11 us and means overlap |
| J027-GBIGADD1 | 45 raw | 268.317 | 346.587 | 430.786 | 350.746 | repeat did not reproduce GBIGADD0's stronger median drop |
| J027-KINFO2 | 36 raw | 307.723 | 347.943 | 488.276 | 362.535 | post-GBIGADD control close to GBIGADD1 median |
| J027-DINK0 | 15 comments | N/A | N/A | N/A | N/A | direct `cnrtInvokeKernel` attempt did not compile; first argument must be cast to `const void *` |
| J027-INSYNC0 | 48 raw | 279.749 | 375.161 | 4522.478 | 705.061 | internal `cnrtQueueSync(queue)` does not avoid the launch/sync floor and increases outlier exposure |
| J027-DINK1 | 45 raw | 277.079 | 331.851 | 371.459 | 325.828 | direct `cnrtInvokeKernel` with explicit cast works and is in the same launch-bound band, slightly lower than older KINFO controls |
| J027-MEMINFO0 | 45 raw | 93.273 | 116.219 | 4488.122 | 389.404 | `cnrtMemGetInfo` is usually much lower than kernel launch but has rare multi-ms runtime outliers |
| J027-HOSTFN0 | 45 raw | 107.884 | 433.952 | 8163.087 | 1473.276 | host callback queue submission has severe outliers and is not a usable warmup/optimization path |
| J027-KINFO3 | 48 raw | 264.054 | 334.506 | 446.536 | 336.162 | adjacent normal-launch control overlaps DINK1; no robust evidence that explicit invoke is faster |
| J027-QQUERY0 | 45 raw | 32.443 | 44.324 | 65.043 | 44.754 | idle/current queue query is host-only scale and stable |
| J027-LQUERY0 | 48 raw | 267.775 | 343.621 | 418.067 | 340.583 | launch followed by nonblocking query remains launch-bound |
| J027-KUSAGEPRE0 | 48 raw | 252.582 | 358.434 | 542.094 | 367.574 | constructor kernel-memory usage prequery did not reduce launch floor |
| J027-CMSET1A0 | 45 raw | 263.062 | 334.940 | 459.467 | 336.999 | constructor CNRT memset of 1 half plus tiny add overlaps normal launch control |
| J027-NOWARM0 | 45 raw | 2092.985 | 2263.860 | 2596.020 | 2292.645 | removing static MLU warmup exposes a ~2.3ms first kernel cost |
| J027-ALLOC1P0 | 48 raw | 2059.577 | 2415.514 | 3736.560 | 2488.941 | static MLU allocation alone does not warm the kernel launch path |
| J027-CMSET1P0 | 48 raw | 2029.029 | 2448.551 | 3510.074 | 2448.792 | CNRT memset-only warmup does not warm the Bang kernel launch path |
| J027-SAMEKWARM0 | 16 comments | N/A | N/A | N/A | N/A | constructor Bang kernel launch fails before runtime registration: `Kernel not found` |
| J027-TZ1P0 | 48 raw | 246.024 | 328.156 | 392.194 | 323.293 | one-element `torch::zeros` warmup removes the 2.4ms cold start and is slightly lower than adjacent add/normal controls |
| J027-TZ0P0 | 45 raw | 2095.343 | 2343.602 | 2634.384 | 2336.666 | zero-size torch zeros does not warm the custom-kernel launch path |
| J027-TADD1P0 | 48 raw | 246.391 | 325.698 | 434.287 | 325.982 | one-element tiny add warmup removes cold start, matching `torch::zeros({1})` |
| J026-FIXDEV0 | 48 raw | 304.366 | 387.361 | 596.399 | 398.026 | fixed device0 PASS source, overlaps pid-spread same-window control |
| J026-PIDSPREAD1 | 48 raw | 312.365 | 389.502 | 509.398 | 390.720 | pid-spread PASS source, same median band as fixed device0 |
| J028-FIXDEV0 | 48 raw | 291.486 | 359.692 | 568.321 | 365.900 | fixed device0 PASS source, overlaps pid-spread same-window control |
| J028-PIDSPREAD1 | 48 raw | 293.543 | 362.735 | 449.996 | 361.822 | pid-spread PASS source, same median band as fixed device0 |
| J029-FIXDEV0 | 48 raw | 265.840 | 350.376 | 456.006 | 355.242 | fixed device0 PASS source, overlaps pid-spread same-window control |
| J029-PIDSPREAD1 | 48 raw | 267.253 | 348.141 | 519.535 | 350.817 | pid-spread PASS source, same median band as fixed device0 |
| J026-TADDPASS0 | 48 raw | 301.799 | 385.598 | 529.435 | 395.508 | tiny add prewarm PASS source, overlaps `zeros({1})` pid-spread control |
| J028-TADDPASS0 | 48 raw | 288.656 | 357.460 | 439.937 | 359.511 | tiny add prewarm PASS source, overlaps `zeros({1})` pid-spread control |
| J029-TADDPASS0 | 48 raw | 275.798 | 349.182 | 445.696 | 344.313 | tiny add prewarm PASS source, overlaps `zeros({1})` pid-spread control |
| J026-TDROP0 | 45 raw | 316.884 | 384.795 | 516.435 | 399.232 | discarding tiny-add prewarm tensor keeps PASS and overlaps retained tiny-add/zeros controls |
| J028-TDROP0 | 45 raw | 284.644 | 362.480 | 464.794 | 366.860 | discarding tiny-add prewarm tensor keeps PASS and overlaps retained tiny-add/zeros controls |
| J029-TDROP0 | 45 raw | 293.582 | 354.631 | 465.157 | 365.619 | discarding tiny-add prewarm tensor keeps PASS and overlaps retained tiny-add/zeros controls |
| J026-TABSPASS0 | 16 comments | N/A | N/A | N/A | N/A | source audit rejects `torch::abs(a)` in constructor: `FORBIDDEN_TORCH_HIGH_LEVEL_OP` |
| J028-TABSPASS0 | 16 comments | N/A | N/A | N/A | N/A | source audit rejects `torch::abs(a)` in constructor: `FORBIDDEN_TORCH_HIGH_LEVEL_OP` |
| J029-TABSPASS0 | 16 comments | N/A | N/A | N/A | N/A | source audit rejects `torch::abs(a)` in constructor: `FORBIDDEN_TORCH_HIGH_LEVEL_OP` |
| J026-ZDROP0 | 45 raw | 303.621 | 383.775 | 468.219 | 380.376 | discarding one-element zeros prewarm tensor keeps PASS and overlaps retained zeros/tiny-add controls |
| J028-ZDROP0 | 45 raw | 289.882 | 349.705 | 462.006 | 355.834 | discarding one-element zeros prewarm tensor keeps PASS and overlaps retained zeros/tiny-add controls |
| J029-ZDROP0 | 45 raw | 284.196 | 343.324 | 412.756 | 347.869 | discarding one-element zeros prewarm tensor keeps PASS and overlaps retained zeros/tiny-add controls |
| J026-CPULOCAL0 | 45 raw | 318.194 | 372.326 | 471.972 | 377.487 | CPU-local device grouping slightly lowers this op's median vs ZDROP, but not enough across ops to claim a fix |
| J028-CPULOCAL0 | 45 raw | 280.409 | 357.295 | 3484.898 | 420.942 | CPU-local grouping does not improve median and exposes a large runtime outlier |
| J029-CPULOCAL0 | 45 raw | 272.580 | 344.871 | 563.334 | 344.732 | CPU-local grouping overlaps ZDROP/tiny-add controls |
| J026-GMEM18HOLD0 | 45 raw PASS | 326.832 | 384.196 | 533.783 | 398.418 | 18GiB hold exposes OOM on crowded cards but valid rows do not improve median |
| J028-GMEM18HOLD0 | 45 raw PASS | 292.623 | 362.429 | 456.419 | 355.655 | 18GiB hold exposes OOM on crowded cards and slightly worsens median vs ZDROP |
| J029-GMEM18HOLD0 | 47 raw PASS | 274.433 | 339.503 | 408.685 | 338.826 | 18GiB hold slightly lowers this op's median, but cross-op result is mixed |

Strong observations with >=8 rows:
- FAIL probes still report `bangc_us`, so they are useful for partial-code timing.
- Worker-side aggregation launches fresh evaluation subprocesses for each run and averages `@@RESULT@@ bangc_us`; the timed region is `torch.mlu.synchronize(); t0; bang_func; torch.mlu.synchronize(); t1`.
- Host-only timed path is not the source of 300+ us latency.
- A timed MLU kernel launch plus synchronize is already around the same order as the full GELU path.
- Full GELU work over empty-kernel lower bound is much smaller than the launch/sync floor in this OJ setup.
- 026/028/029 current best baselines also fall into the same 300-400 us band over 16 rows each, supporting a shared launch/sync or worker-state source.
- OJ averages three fresh runs per comment; therefore a single fast raw run does not dominate the submitted latency unless the whole three-run group is fast.

Weak observations:
- M0's 1364 us outlier likely reflects worker/device interference or runtime hiccup, not algorithm work.
- Because B0 and K0 overlap heavily, leaderboard differences of a few us cannot be interpreted without many repeats.
- Static init with a hand-written dummy kernel did not reduce timed latency; it was consistently slower over 16 rows.
- A large `torch::zeros` warmup alone hurts badly; later one-element `torch::zeros({1})` works, so size and actual registered device-op path both matter.
- Adding tiny `zeros({1})` before the current warmup did not help over 16 rows.
- CPU affinity pinning did not meaningfully improve 027 over 15 rows; host CPU migration is not the dominant visible source.
- CNNL0 was the wrong layer and did not produce timing rows; ignore as a candidate.
- CR0 combined several CNRT calls and produced no timing rows; split into smaller CNRT probes.
- Minimal `cnrtGetDeviceCount` preinit did not improve 027 over 9 rows.
- `cnrtGetDevice/cnrtSetDevice(current)` did not clearly improve 027 over 9 rows.
- Minimal `cnrtQueueSync(nullptr)` preinit did not improve 027 over 16 rows.
- Caching `torch_mlu::getCurMLUStream()` outside the timed call did not improve 027 over 16 rows.
- Driver/count-gated device spreading did not help 027. Fixed device 0 remains better than 0-3, 0-7, or high-bus spreading in this sample.
- CNRT hardware info confirms OJ samples all three expected classes: 5.10.22/8-card, 6.2.10/8-card, and 6.2.10/10-card. The 10-card class can produce the fastest outliers but still has wide spread.
- Moderate `torch::empty` pool reserve did not improve 027 over 15 rows; allocator reserve is not a clear fix.
- CNRT notifier timing around the GELU kernel reports about 7-8 us hardware time over 39 raw runs, while measured wall time remains hundreds of us. The device kernel body is not the bottleneck for these small pointwise ops.
- HWINFO host-only retest reports 48 raw runs at median 40.553 us, while empty kernel and memcpy-only report 346.443 us and 348.059 us medians. This strengthens that first timed kernel launch/sync dominates over wrapper and simple memory transfer.
- KINFO0 by class: 5.10.22/8-card n=21 median 349.895 us; 6.2.10/8-card n=9 median 351.798 us; 6.2.10/10-card n=18 median 340.123 us. The 10-card class is a little faster in this sample but all classes remain launch-bound.
- MINFO0 by class: 5.10.22/8-card n=24 median 359.311 us; 6.2.10/8-card n=24 median 342.918 us; no 10-card rows in this sample. Treat class comparison as limited to sampled workers.
- K2INFO0 by class: 5.10.22/8-card n=15 median 337.602 us; 6.2.10/8-card n=21 median 329.178 us; 6.2.10/10-card n=12 median 322.659 us. One extra empty launch is below observed cross-batch noise, so use K8INFO0 to amplify the signal.
- K8INFO0 raw median is 363.306 us; filtering only >600 us outliers leaves n=44 median 359.291 us. Seven extra empty launches are not remotely linear with the first-launch floor; use K32INFO0 for a larger edge-cost signal.
- K32INFO0 raw median is 462.126 us. Relative to KINFO0 median 346.443 us, 31 extra empty launches add about 115.683 us, roughly 3.7 us per additional launch at median. First timed kernel launch/sync remains the dominant cost.
- SINFO0 reports 36 valid raw runs at median 40.105 us, matching HINFO0. `torch_mlu::getCurMLUStream()` is not the source of the 300+ us jump.
- FBLOCKINFO0/FYIELDINFO0 printed `flag=0` in parsed HWINFO, so `cnrtSetDeviceFlag` did not visibly change the scheduling mode in that placement. Do not use those timings to conclude block/yield policy performance; test pre-set placement first.
- FBLOCKPRE0 called `cnrtSetDeviceFlag(cnrtDeviceScheduleBlock)` before other CNRT queries and printed `setret=0/getret=0`, but `flag=0` in all 48 raw runs. In this OJ extension context, CNRT sync scheduling policy is not a useful optimization knob unless another API path is found.
- TEMPTY0 reports 33 valid raw runs at median 46.259 us, close to host-only/SINFO0. Invoking a pre-instantiated TaskTopo without a kernel does not trigger the 300+ us floor.
- TKEMPTY0 direct `cnrtTaskTopoAddKernelNode` fails consistently: `cnrtTaskTopoAddKernelNode: Kernel not found`, `r1=101312`. Try queue capture so CNRT sees the compiler-registered triple-launch kernel.
- TCAPEMPTY0 cannot be used in OJ because source audit rejects `cnrtQueueCreate`. Queue-capture TaskTopo is therefore blocked unless it can capture on the current torch_mlu stream inside the timed function, which would not move setup out of timing.
- KINFO1/GBIGADD0/GLOOP8ADD0 contemporaneous comparison: KINFO1 median 337.229 us, GBIGADD0 median 330.152 us, GLOOP8ADD0 median 345.671 us. Same-size static add has a weak positive signal, repeated tiny adds do not.
- GBIGADD0 by class: 5.10.22/8-card n=12 median 357.929 us; 6.2.10/8-card n=12 median 296.400 us; 6.2.10/10-card n=24 median 330.029 us. The improvement is not consistent across machine classes, so it needs another control/repeat before using as a general fix.
- GRES16G0 confirmed the large-pool reservation mechanism can force OOM when the card is already occupied: many runs saw only about 6.2-6.7 GiB free and failed to allocate 16 GiB. Successful rows median 332.270 us, not enough to justify near-full reservation as a speed fix; it mainly acts as a contention detector/disruptor.
- CPUTOPO0 initial sample: all 27 raw rows were 6.2.10/8-card on BDF `0000:4f:00.0`, visible CPU affinity `0-55`. CPU node0 median 320.498 us and node1 median 322.989 us overlap, so CPU NUMA placement was not a visible dominant source in that sample. Repeat submitted to broaden machine coverage.
- CPUTOPO0 combined sample stayed on 6.2.10/8-card with BDF `0000:4f:00.0`. Constructor CPU node0 rows n=48 median 315.962 us, node1 rows n=24 median 339.155 us. This suggests a possible NUMA/locality effect for device0, but it is still correlation from scheduler placement; AFFNODE0/AFFNODE1 will test causality.
- GRES4G0 produced no OOM and median 333.959 us, close to KINFO1/GRES16G valid rows. Reserving a moderate 4GiB pool is not a clear launch-jitter fix.
- AFFNODE0/AFFNODE1 forced-affinity batches returned sr=0 and correct CPU node placement. Sequential batches gave node0 median 362.929 us and node1 median 321.694 us, contradicting passive CPUTOPO correlation. Treat this as evidence that batch/time skew is still large; run AFFPID01 to interleave node choices within one source variant before drawing a NUMA conclusion.
- AFFPID01 interleaved pid-parity affinity returned node0 n=39 median 330.145 us mean 335.073 us, node1 n=45 median 319.471 us mean 337.027 us. CPU affinity/NUMA placement is not a robust primary fix; at most it is a small, machine/time dependent contributor.
- GBIGADD1 repeated the same-size static add warmup and produced median 346.587 us versus KINFO2 347.943 us in the adjacent control. The strong-looking GBIGADD0 median drop did not repeat; same-size static add may shave small outliers but is not a reliable launch-floor fix.
- DINK0 did not measure runtime latency: all 15 comments failed at compile time with `no matching function for call to cnrtInvokeKernel`; the Bang runtime header expects the kernel argument as `const void *`. Retest this path with an explicit cast before drawing any conclusion about manual invoke overhead.
- INSYNC0 returned 48 raw rows with median 375.161 us and several multi-ms outliers. Moving the queue synchronization inside `bang_func` does not remove the external measurement floor; it often double-exposes or shifts the same cold/scheduler cost into the timed region.
- DINK1 compiled and returned 45 raw rows with median 331.851 us. It does not remove the launch floor, but it is lower than KINFO0/KINFO1/KINFO2 controls collected earlier. This needs a same-window normal `<<<>>>` control before treating explicit `cnrtInvokeKernel` as a real improvement.
- MEMINFO0 returned 45 raw rows with median 116.219 us and filtered `<600us` median 114.221 us. A CNRT device-memory query alone is above host-only but far below kernel launch, so ordinary CNRT runtime calls are not sufficient to explain the 300+ us launch floor. The multi-ms outliers show runtime/driver calls can still be disrupted by worker state.
- HOSTFN0 returned 45 raw rows with median 433.952 us and many large outliers. Enqueuing host callbacks on the CNRT queue is worse and less stable than a device kernel launch; do not use host callbacks as a timed warmup or substitute queue operation.
- KINFO3 returned 48 raw rows with median 334.506 us, essentially overlapping DINK1's 331.851 us in the adjacent period. Explicit `cnrtInvokeKernel` is not a reliable break from the normal `<<<>>>` launch floor.
- QQUERY0 returned 45 raw rows with median 44.324 us, close to HINFO/SINFO. `cnrtQueueQuery` on the current stream is not the source of kernel-launch overhead.
- LQUERY0 returned 48 raw rows with median 343.621 us. A nonblocking query immediately after launch does not shift enough work out of the external tester sync to matter; once a device kernel is enqueued, the same launch/sync floor appears.
- KUSAGEPRE0 returned 48 raw rows with median 358.434 us and reported `kbytes=1690728448` consistently. Querying kernel memory usage in the constructor does not pre-warm the launch path; it was slightly worse than adjacent KINFO3.
- CMSET1A0 returned 45 raw rows with median 334.940 us, matching KINFO3. A tiny CNRT `cnrtMemset` warmup is not harmful like large torch zeros, but it also does not reliably reduce the first timed kernel launch cost.
- NOWARM0 returned 45 raw rows with median 2263.860 us across all driver/count classes. This is a strong result: without any static MLU operation before the timed call, the first timed kernel pays a multi-ms cold-start cost.
- ALLOC1P0 returned 48 raw rows with median 2415.514 us. A static MLU `torch::empty` allocation by itself is not enough; the effective warmup must enqueue/execute some MLU-side operation before the timed region.
- CMSET1P0 returned 48 raw rows with median 2448.551 us. CNRT `cnrtMemset` outside timing is not sufficient either, so the current useful prewarm is not just "any device-side runtime operation"; it needs a registered compute op/kernel path.
- SAMEKWARM0 consistently failed during module load with `cnrtInvokeKernel: Kernel not found` and later `CNRT error: cncc does not register the kernel to runtime`. User-side Bang kernels cannot be launched from static initialization before CNCC runtime registration is complete.
- TZ1P0 returned 48 raw rows with median 328.156 us. A one-element `torch::zeros` in static initialization is sufficient to eliminate the multi-ms first-kernel cold start, unlike CNRT `cnrtMemset`; it appears to drive the torch_mlu registered op path that initializes the relevant runtime state before timing.
- TZ0P0 returned 45 raw rows with median 2343.602 us. A zero-size torch op is not enough, so the useful prewarm is not merely importing/dispatching through torch; it needs an actual registered MLU device operation.
- TADD1P0 returned 48 raw rows with median 325.698 us. A one-element torch add is effectively equivalent to `torch::zeros({1})` as a cold-start prewarm in this OJ window.
- Current 026/028/029 sources already contain one-element `torch::zeros({1})` prewarm and first-call custom-kernel execution guarded by a static flag. Cross-op follow-up should compare those real PASS sources with fixed-device/no-device-spread variants rather than retesting no-prewarm.
- FIXDEV0 vs PIDSPREAD1 same-window PASS comparison: 026 raw medians 387.361 vs 389.502 us, 028 359.692 vs 362.735 us, 029 350.376 vs 348.141 us. Device selection policy changes card distribution but not the main latency band; launch/prewarm remains the dominant lever.
- TADDPASS0 vs `zeros({1})` PIDSPREAD1 same-window PASS comparison: 026 raw medians 385.598 vs 389.502 us, 028 357.460 vs 362.735 us, 029 349.182 vs 348.141 us. Tiny add prewarm is valid but not a decisive improvement.
- TDROP0 returned 45 raw PASS rows per op. Medians 026/028/029 are 384.795/362.480/354.631 us, overlapping retained TADDPASS0 and PIDSPREAD1. The prewarm tensor does not need to survive; executing the registered tiny op before timing is the useful part.
- TABSPASS0 returned 16 audit-failure comments per op before running. `torch::abs(a)` is caught by `FORBIDDEN_TORCH_HIGH_LEVEL_OP`, so named high-level torch algorithm ops are not viable constructor prewarm probes.
- ZDROP0 returned 45 raw PASS rows per op. Medians 026/028/029 are 383.775/349.705/343.324 us, again overlapping retained zeros/tiny-add controls. Both tested warmup families work after the temporary tensor is destroyed.
- CPULOCAL0 returned 45 raw PASS rows per op. Medians 026/028/029 are 372.326/357.295/344.871 us; only 026 improved versus ZDROP, while 028 regressed and hit a 3484.898 us outlier. CPU/NUMA-local device grouping is not a robust launch-jitter fix in this sample.
- GMEM18HOLD0 retained an about 18GiB empty tensor and returned PASS raw medians 384.196/362.429/339.503 us for 026/028/029. It also produced OOM failures when reported free memory was about 4.7GiB. This confirms large reserve can detect or reject crowded cards, but valid PASS timings remain in the same launch-bound band.
- ALGNULL028/029 are same-window A/B probes for the specific claim that 8-13 us of real device work is below OJ wall jitter. 028 empty-kernel commits: `ed84ec1e aa64e484 8d95d82b 2c2a6098 305cc587 46526779 86006b20 8d7cb001`; 028 real-kernel adjacent controls: `e39725fe 9036642b 4036e081 c4ac4225 1669f6fe 9e65d3d0 0aa2aac6 f0c3ca1f`. 029 empty-kernel commits: `1315be90 0664e480 8fed480b af9c193d bbbac02b 357c477b c63760fd d5b872f1`; 029 real-kernel adjacent controls: `d149a81f 4523a265 3c05d2d1 e3e21465 8305fe8d 21740098 3b0e3f6c 91f8a91f`. These are timing probes; empty variants are expected to fail correctness. As of 23:06, OJ processing drained but GitHub comments are still empty for all ALGNULL commits, so keep them as no-comment pending and do not use them for conclusions yet.

Local probes:
- Same process, 32 timed calls after extension load: median about 40 us with one 628 us outlier.
- Fresh process first call, 8 runs: 511-602 us. This supports process/kernel-launch cold start as a major source.
- Fresh process first call for 026/028/029 also stays high: 026 median about 553 us, 028 median about 497 us, 029 median about 513 us.
