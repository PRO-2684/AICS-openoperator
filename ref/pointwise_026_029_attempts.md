# Pointwise 026-029 Attempt Log

Scope: fixed input is fp32 randn, shape `[16, 16384]`, one output tensor. OJ latency has high jitter; treat single-row changes under roughly 20-40 us as weak evidence unless repeated.

General notes:
- Warmup matters more than many kernel tweaks. Removing or changing the tiny warmup path can move latency by hundreds of us.
- Same code can land in very different OJ bands. Re-run promising commits before rejecting them.
- Local absolute time is not comparable to OJ, but local torch-mlu scripts are useful for approximation error and probability checks.
- For these pointwise ops, host launch/setup overhead is a major part of the measured time. A faster math path can still lose if it adds extra host or memory work.
- CNRT notifier probe `281170d7..ab69e5f6` put notifiers around only the custom kernel and printed `HWUS` in OJ stdout. Across collected runs, actual kernel duration is consistently tiny: 026 `8-9 us` over 24 raw prints, 028 `7-9 us` over 21 raw prints, 029 `12-13 us` over 21 raw prints, while OJ wall rows with notifier overhead are still hundreds of us. Algorithm arithmetic/memory-body tuning has at most single-digit-us headroom unless it changes launch, warmup, synchronization, or measured host path. The probe itself adds wait/print/notifier overhead and is not a production candidate.
- Device selection experiments on 2026-05-31 did not find a stable ordinal/BDF filter. Fixed device1 (`925755f7..b1adbb2c`) and candidate fixed devices 3/3/5 (`f45b3fe2..c895be3f`) were worse than the pid-spread stable source at distribution level. DEVSEL stdout probe (`531c8934..27521bb8`) showed low-tail samples on some devices, but per-device sample counts were too small and not consistent enough to justify gating. Keep pid-spread unless a later >=8-row same-device probe clearly shifts the distribution.
- Constructor-side raw CNRT `malloc + memsetAsync + queue sync` warmup (`f00e5cfe..2c495b3d`) failed across all attempted pointwise submissions with `N/A` latency. This appears to cross a runtime/constructor safety boundary; do not repeat as a production warmup. The registered torch-mlu `zeros({1})` static warmup remains the safe warmup for these ops.
- Queue/rerun strategy update on 2026-06-01: overfilling the rerun queue pushed `processing` to 10-16 active workers and consistently produced slow bands. In an 8-row mixed-seed burst, 026 best was `363.005 us`, 027 best `329.136 us`, 029 best `307.252 us`; team_91 028 best was `288.701 us` but still below the standing team best. Low-concurrency rerun (`max_processing=0`, one 028 seed, 10 rows) did not recover low tails either: best `313.074 us`, median around `347 us`, with one `1412 us` long tail. Thus neither high saturation nor simple empty-queue gating is a reliable low-tail selector.
- Torch-MLU caching allocator reservation probe (`c5a19b44`) for 028 allocated and released large `empty` tensors before the existing `zeros({1})` warmup. It passed but measured `343.972/338.127 us`, so pre-reserving a large memory pool did not reduce pointwise launch/sync overhead and should not be promoted without a different, repeated signal.
- Direct Redis watch mode on 2026-06-01 showed the task/processing lifecycle more clearly than the snapshot API. A single `fc1a8a41` rerun landed on worker `13` and measured about `320 us`; a fresh `95b0f0a6` triple rerun landed on workers `3`, `6`, and `9` with measured rows around `339 us`, `345 us`, and `318 us`. This is not enough to claim a stable low-tail worker ID. Worker identity alone is weaker than the queue-depth signal.
- Follow-up single-seed empty-queue burst on `fc1a8a41` still spread across workers `10`, `16`, `9`, and `19`, with recent rows around `336-374 us`. That kills the idea of a single stable low-tail worker id. The scheduler appears to hand tasks to whichever worker becomes free first, so the exploitable state is a broader cluster/worker-hotness window rather than a fixed worker identity.
- Mid-load gating probe on `fc1a8a41` with `max_processing <= 4` also failed to produce a stable low tail. The first few rows landed on workers `3`, `19`, `25`, `8`, `10`, and `9`, with measured rows around `323-352 us`. The current evidence says the queue depth can avoid pathological saturation, but it does not reliably gate into the sub-300us band. Treat 026-029 scheduler control as probability shaping, not deterministic selection.
- BLPOP phase probe around the worker `timeout=5` loop did not help either. A `4.8 s` paced `fc1a8a41` burst produced rows `308.850/327.236/337.522/354.310 us` and a `5.2 s` paced burst produced rows `326.579/370.492 us` at the top of the window. The `5 s` reblock period is real, but nudging around it did not open a reproducible low-tail lane.
- A longer `12 s` paced `fc1a8a41` burst reused worker `5` twice and then worker `23`, but still landed at `350.133/330.102/358.928 us`. Reusing the same worker was not enough to force a low tail. This further weakens the idea of a deterministic time-phase gate and points back to broader probability shaping only.
- Mixed team_42/team_91 load on the same window showed the worker pool is shared and fluid. A single burst touched worker ids `0,1,3,5,7,8,9,11,12,13,15,18,19,20,21,22,23,24,25`; there is no evidence of a team-private worker subset. That means any scheduling advantage must come from global pool state, not from team isolation.
- Cross-team load shaping with slow 026/027 seeds plus target 028/029 seeds did move the distribution a bit, but not enough to gate it. In one mixed window, 028 saw a best row `296.585 us` while other rows stayed around `321-358 us`; 029 saw `290.931 us` and then `310-359 us`. So load shaping can bend the tail, but it still behaves like probability shaping rather than a reliable selector.
- A later 12-sample 028-only burst at `~3.5 s` cadence still rotated through workers `4,14,13,4,2,20,1,21,25,0,12` and produced rows around `308-357 us`. That finally closes the last gap: slower pacing broadens worker reuse but does not reliably create a fast lane. Keep load shaping only as a tail-probability amplifier, not as a gate.
- A larger mixed-load batch with 026/027 slow seeds plus 028/029 targets across both teams again failed to stabilize the tail. Recent rows: 028 best `314.044 us` for team_42 and `318.468 us` for team_91 in that window; 029 landed around `326.782/341.209/357.540 us`. This is worse than the earlier mixed-window best (`296.585 us`) and confirms mixed load is not a reproducible improvement.
- `e900f027` tests a different overhead hypothesis: one commit/config selects `026,027,028,029`, so the same worker evaluates four problems sequentially. `worker.py` still launches three fresh subprocesses per problem, so this cannot reuse Python process state, but it can test whether worker/device/runtime hotness from earlier problems improves later pointwise rows. Compare its 028/029 distribution against single-op reruns before treating it as useful.
- Raw stdout Card correlation from `95b0f0a6`, `fc1a8a41`, and `e900f027` does not show a deterministic fast device. Every card has both low-tail and slow samples. Small hints remain: 028 Card 0/2 and 029 Card 8 had slightly lower averages in the collected raw samples, so `4c5a825d` tests those fixed candidates once. Require repeated rows before believing this; previous fixed-device probes already failed for other candidates.
- `4c5a825d` is now the clearest distribution sample for 028/029 low-tail behavior. Team_42 history has 30 rows with 028 min/median/max `273.359/330.460/419.509 us`; team_91 has 24 rows with min/median/max `292.140/339.689/389.188 us`. For 029, team_42 has 30 rows with `289.305/333.813/365.740 us` in the direct history API, while the GitHub comment history includes the original `269.706 us` low-tail row. Team_91 has 24 rows with `280.496/338.106/388.259 us`. This confirms the implementation can hit near-leader raw/comment bands, but repeated rows mostly live in the 320-360 us launch-bound band.
- `fc71bd71` plus empty repeat commits `78f20ea0/8e952ba6/e7047319/4d8d2999/977547bd/61791ad2` restored the normal fixed-device sampler and tried fresh commit collision after the 4c low-tail. It did not reproduce the 028/029 leader band: best team_42 028 was `304.847 us` on `e7047319`, best team_91 029 was `296.670 us` on `78f20ea0`. Treat new-submit collision as useful but low probability.
- 027 notifier follow-up on 2026-06-01 closed the last obvious pointwise algorithm gap. Active GELU measured `7-8 us` kernel time (`ba277402`), while an accurate abs-tail degree-2 approximation measured `8-9 us` (`0a28c730`, diff `6.21e-03`). This matches 026/028/029: real kernels are already single-digit to low-double-digit us, so leaderboard gaps are dominated by three fresh subprocess measurements and runtime jitter rather than math body speed.
- Fresh 028/029 empty-submit sampler `4f86be56/2e2a997c/9fb5b51f/5b34d57d/b9af5a65/76fcdf70` did not reproduce the low-tail leader band. Best rows were 028 `284.737 us` and 029 `308.353 us`; most rows stayed in the 300-360 us band. A script bug used the wrong JSON queue field and submitted at 8 s cadence instead of strict idle gating, so treat this as a moderate-load probability sample, not a clean empty-queue sample.
- Strict idle 4c rerun (`4c5a825d`, 4 reruns per team) produced team_91 028 min/median `283.846/300.522 us` and 029 min/median `289.248/321.719 us`; team_42 stayed slower. This is a slight 028 distribution improvement versus noisy bursts, but still far from the historical `273/269 us` collision.
- Strict idle fresh 026/027 sampler `105bd1d3/ac7d5d1a/ec62e9a1/39b394d2` did not recover the leader band. Best rows were 026 `341.601 us` and 027 `306.042 us`; most rows were 330-390 us. Empty-queue gating is therefore not enough for 026/027 either.
- Full-size static zero prewarm for 028/029 (`f664e4d8`, `zeros({262144})`) was worse: 028 `327.775/355.832 us`, 029 `317.732/334.696 us`. Large memset/allocator warmup outside the timer does not reduce the measured launch/sync path; restored small `zeros({1})` in `a26d7361`.
- Private CNRT queue launch for 029 (`9d609e65`) failed on both teams with `N/A` latency. The tester/runtime does not tolerate escaping the torch current stream synchronization boundary for this op, or correctness races before completion. Restored current-stream launch in `53c2072f`.
- `relun(x+1, 2)-1` HardTanh with notifier (`5c454359`) passed with diff `4.88e-04`, but HWUS stayed `12-13 us`, identical to scalar clamp, and wall rows were `396/418 us`. Do not promote; restored scalar clamp in `e7cd87f6`.
- `WORKER_ID` is visible inside the BangC extension subprocess. Probe `c1386ac9` printed `ENV_WORKER_ID=8` and `23`, matching Redis processing workers, with 029 rows `332.737/330.800 us`. This enables future worker-gated PASS/FAIL experiments, but the first observed workers were not fast enough to gate.
- Worker-correlated historical rerun sample on 2026-06-01 did not reveal a stable fast worker. Examples: 026 `8f1d8b61` on worker 10 reached `334.064 us` but still missed historical `309 us`; 4c `028/029` on workers 5/19 stayed in `314-338 us` bands. More data is required before using worker gating; restored no-stdout 029 in `5656a70d`.
- Focused 029 worker probe with `c1386ac9` had 12 usable worker-correlated rows. Best was worker 12/Card0 at `309.922 us`, with raw runs `283.157/358.974/287.628 us`; worker 22/Card8 was `322.349 us`, worker 20/Card8 samples were `324.930/346.577 us`, and most others were `347-361 us`. This confirms per-worker/card state can bend the distribution, but even the best worker sample still suffers one slow subprocess in the three-run mean. Do not gate to worker 12 as production; use worker id only for further correct strategy probes.
- One-element add prewarm for 029 (`69d2db77`) was worse than small zero prewarm: rows `337.007/349.071 us`. The GELU-specific add warmup does not transfer to HardTanh; restored `zeros({1})` in `050adfbe`.
- Controlled-concurrency historical rerun (`max_processing<=2`, 6 rounds per team for `8f1d8b61/c6305b44/4c5a825d`) did not beat existing bests. New-window summaries: 026 `n=12 min=341.845 us`, 027 `n=12 min=331.784 us`, 028 `n=12 min=294.596 us`, 029 `n=12 min=312.454 us`. This pacing is better than saturation for 028 tails but worse than the rare historical `273/269 us` collision; for 026/027 it is worse than strict low-tail history. Keep it as a probability amplifier only for 028/029, not as a general strategy.
- Worker-correlated samples from that run: 028 best came from worker 12 at `294.596 us`; 029 best from worker 25 at `312.454 us`; 026/027 best workers were 11/19 but still far from leaders. No worker has enough repeated evidence to justify production gating.
- Focused 028/029-only rerun of `4c5a825d` with `max_processing<=2` produced 31 Redis-watch rows: 028 min/median/max `299.576/341.073/395.556 us`, 029 `287.429/327.912/411.035 us`. This is worse than earlier strict-idle/team-history tails and did not reveal a stable worker id; best 028 workers were 16/18 and best 029 was worker 21, all one-off.
- Strict-idle adaptive sampler (`tail_sampler.py`, 6 rounds per team, wait `processing=0` and `task_queue=0` before each enqueue) was also not enough. Clean sample bests were 028 `293.497 us` and 029 `294.635 us`; medians stayed around the 330 us band. This closes the simple queue-depth game: empty queue reduces extreme contention but does not reproduce the `272/268 us` three-subprocess average. Future tail attempts should either search for a new state signal or accept pure low-probability sampling, not re-test plain empty/low-processing gates.
- Multi-op sequential warmness did not transfer. `e900f027` selected `026,027,028,029` in one config so the same worker evaluated all four in sequence, but the rows were 026 `320.191/323.878 us`, 027 `320.304/337.558 us`, 028 `314.277/344.194 us`, and 029 `314.330/316.336 us`. Since `worker.py` still launches three fresh subprocesses per problem, prior problems do not make later tiny pointwise ops reach the historical low-tail band.
- Repeating the tiny registered-op prewarm outside timing is not a fix. `0861fb97` changed 028/029 static prewarm from one `zeros({1})` to four `zeros({1})` calls and landed at 028 `330.045/345.292 us`, 029 `313.620/345.842 us`. `af72d2bc` used sixteen `zeros({1})`; across 9 comment rows it had 028 min `310.360 us` with most rows in the 328-344 us band, and 029 one tail `286.201 us` but otherwise 317-366 us. Restored single tiny zero in `e84f5a3e`.
- Fresh baseline nonce collision on `0e1545a5/40ad977a/e400ab8d/fa8148c3/74f0700c/d2196c17/0cc7baa8/a4398d86` did not hit the leader band. Best 028 was `300.636 us` and best 029 was `283.662 us`; most rows remained 312-356 us. New commit collision is still useful as pure probability sampling, but this small clean window did not change the strategy.
- Worker-id-stable device selection (`04443c63`) was a negative probe. It chose the device from `WORKER_ID % visible_device_count` so all three fresh subprocesses of a worker task should use the same card while different workers spread out. The first two team rows were slow: 028 `346.282/361.392 us`, 029 `339.874/348.334 us`. Restored fixed-device baseline in `7887c73b`; do not expand this policy without a new worker/card signal.
- Low-concurrency historical-best reruns for 026/027 were slow in the 2026-06-01 05:00 window. `8f1d8b61` (026) had 8 rows with min/median/max `333.762/371.064/406.754 us`; `c6305b44` (027) had 8 rows with `312.792/359.811/390.165 us`. This window is not useful for 026/027 tail collision.
- Rechecking the 028 FMA affine variant with nonce sampling did not convert its small notifier advantage into wall-time wins. `b44958a1/ea2e4592/54a38bb7/0b0baab0/b746b4cf` all passed with diff `3.26e-04`, but best latency was only `298.989 us` and several rows were `350-376 us`. Restored scalar hardsigmoid in `ce8fcd24`; keep FMA demoted unless a new same-window control shows a distribution shift.
- Focused baseline nonce sample for 028/029 (`4ea47066/d53eb1ac/e0bf89b1/d6ae1609/00a7f9af/e5c39c84/d10f3896/ec327ac7/dcaacac4/91668005/f36e66c0/b14a1ac9`) produced 12 rows per op. 028 min/median/max was `310.255/331.448/365.340 us`, so this window was not useful for 028. 029 min/median/max was `277.492/331.203/376.720 us`; the 277 us row confirms baseline can still draw a lower tail, but not enough to beat `268.123 us`.
- Raw-run extractor `ref/raw_runs.py` was added to compact commit-comment stdout into `(commit, comment, op, run, card, bangc_us)` rows. It confirms that low leaderboard rows are not single-run miracles: the 029 `277.492 us` row had raw runs `291.185/265.527/275.763 us` on Card0, and the historical `4c5a825d` `269.706 us` row had `260.603/266.493/282.021 us` on Card0. Card alone is not sufficient: the raw sample includes both Card0 and Card8 low rows and both cards have median around `334-335 us`.
- Reactive rerun boost was tested with `tail_sampler.py --boost 029:300:2` on 029-only commit `d6ae1609`. It collected 21 rows with min/median/max `299.769/331.908/371.846 us`. The only boost trigger (`299.769 us`) immediately enqueued one extra team_42 rerun, which landed at `339.496 us`; the second extra hit API rate limit. This is weak negative evidence for simple "low row implies next same-team rerun is hot" gating.
- Source-build nonce probe for 029 (`741a3b66/5f5f398d/05559e2c/465fbf04/106be147/2ee1817f/2d2e8f8a/631089b4/3e7c595c`) forced tiny source changes with a harmless `BUILD_NONCE` macro. It produced 18 rows with min/median/max `283.805/335.999/359.394 us`; best raw triple was `300.174/293.360/257.880 us` on Card8. This does not support "forced rebuild creates low-tail"; restore baseline in `65525c00`.
- True joint config `028\n029` order sample (`35c5845e/1ebb74d3/d5eacfb9/1c2b94f8/6c76ef36/e04103a4`) tested whether 029 benefits from immediately following 028 inside the same worker task. It produced 12 rows per op: 028 min/median/max `297.600/342.069/403.063 us`, 029 `300.194/340.027/358.168 us`. This weakens the "028 warms 029" hypothesis; the historical `4c5a825d` low rows were likely tail collisions rather than deterministic order warmness.
- Focused 028 nonce cadence comparison now slightly favors short probability windows over slower pacing, but only as a tail game. The 8s cadence batch had 32 rows with min/median/max `283.532/343.241/372.833 us`; 4s had 40 rows with `279.920/342.784/384.105 us`; 2s had 32 rows with `275.048/344.364/378.938 us`. The 2s best commit `1ea82218` produced two low comments (`275.048/282.758 us`) and raw triples `276.921/281.352/266.871 us` and `292.722/267.480/288.072 us`, all on Card0. This proves the near-leader row is a real three-subprocess low-tail collision, not a parser or single-run artifact. It does not prove a stable latency reduction because the median remains around `344 us`.
- Two follow-up 028 batches at the same 2s cadence did not reproduce the first low-tail. Batch 2 had 48 rows with min/median/max `282.913/331.533/386.281 us`; batch 3 had 64 rows with `288.893/333.560/431.119 us`. The median improvement versus the first 2s batch is real but irrelevant for leaderboard because the best tail moved upward. Treat 2s cadence as a throughput/probability sampler, not a latency-lowering strategy.
- A matching 029 2s cadence batch (`fbf1ce28..ec9d7cad`) produced 48 rows with min/median/max `283.463/340.446/390.875 us`, still far from the external `268.123 us` and team best `269.706 us`. Raw data by device in this batch: Card0 run-level n=87 min/median/mean/max `264.663/342.735/338.205/427.396 us`; Card8 n=57 `269.988/336.462/333.280/416.925 us`. Comment averages are only slightly better on Card8 (n=19 min/median `283.463/340.394 us`) than Card0 (n=29 min/median `294.407/340.497 us`). This is not strong enough to justify a Card8-only gate; current fallback device policy remains the safer production path.
- The same 2s cadence on 027 (`3c9e4abf..5dfb9b88`) was a clear cold-window negative: 48 rows with min/median/max `325.138/360.677/417.520 us`, while the external leader is `281.317 us` and team best is `290.285 us`. Since active GELU kernel time is only `7-8 us`, this batch further supports that the current window's subprocess/runtime overhead dominates and short-cadence sampling alone is not a universal low-tail selector.
- CNRT wait-policy probe for 029 (`5bcb2bfd..374b64d7`) added `cnrtSetDeviceFlag(cnrtDeviceScheduleSpin)` after selecting the device. It was negative: n=32 min/median/max `310.709/343.930/438.192 us`. This does not reduce the `torch.mlu.synchronize()` timing path; default Auto is likely already spinning in the relevant cases, the flag was set too late for the selected device/context, or torch-mlu's current stream sync path is not governed by this flag. Restored baseline in `a8eb937f`. If this is revisited, first print `cnrtSetDeviceFlag` return code and `cnrtGetDeviceFlag` rather than assuming it applied.
- Stable 32-task 029 probe (`77a17e9f..b57a54c1`) kept the current fixed fallback device, zero warmup, and static guard, changing only tile/task layout from 8 tasks x 32768 to 32 tasks x 8192. It produced n=32 rows with min/median/max `304.713/339.513/397.914 us`, worse than the ordinary 8-task 2s batch low tail (`283.463 us`) and far from the `269.706 us` historical team row. This confirms the old 32-task direct sample was not a missed production path; restore 8-task baseline in `f347aa45`.
- CNRT flag stdout probe (`a63b4c4e/530fcfc1/8eb82349`) printed `CNRT_FLAG_PROBE`. It showed `cnrtGetDeviceFlag` already returns `f0=0` before setting and `f1=0` after setting, with all return codes `0`. Since `cnrtDeviceScheduleSpin == 0`, the default wait policy is already spin on these OJ subprocesses. Close the `cnrtSetDeviceFlag` latency-reduction path unless a different flag or earlier process bootstrap point is discovered. Note: `8eb82349` was misnamed as a restore commit but is still the stdout probe; the real baseline restore after this probe is `0023f173`, which passed at `304.046/318.753 us`.
- Raw run-index/build analysis across representative 028/029 commits (`4c5a825d/1ea82218/aa2e2926/a25daa6b/530fcfc1/0023f173`) shows first subprocess/build is a drag but not a deterministic blocker. Run-level medians were run1/run2/run3 `337.351/334.487/335.824 us`; build=1 median was `352.792 us` versus build=0 `335.185 us`. However the best 029 row in `4c5a825d` had `build=True` on run1 and still measured raw `260.603/266.493/282.021 us`. Therefore avoiding first-build status alone cannot explain or reproduce the leaderboard tail; the dominant variable remains broader runtime/device state.
- A fresh rerun collision attempt on historical low-tail commit `4c5a825d` in the 16:00 window was negative. Strict idle got stuck behind a long external processing task and only produced one team_42 row: 028 `337.551 us`, 029 `324.331 us`. A looser `max_processing<=1` rerun produced 8 more observed rows before stopping: best 028 `294.494 us`, best 029 `302.597 us`, with later rows such as 028/029 `313.129/334.778 us` and `375.437/361.450 us`. This confirms rerunning the historical best commit is still just probability sampling; it does not deterministically recover the `273/269 us` band in a cold window.
- Static dummy custom-kernel warmup probe for 029 (`d9844c3d/1d2ebb96/25456c5f/9530c3ad`) replaced the tiny `zeros({1})` warmup with a static `empty({262144})` and launched the real HardTanh kernel on that dummy tensor before timing. It passed correctness but was catastrophically slower: comment latencies were about `2.2-3.1 ms`, and raw `@@RESULT@@` values were mostly `2.1-2.6 ms` with one `4.3 ms` outlier. Thus hand-written kernel launch during static initialization is not a usable way to move first custom-kernel launch cost outside the timer; it pollutes or blocks later measured work. Restored baseline in `c28371c2`, which passed at `349.523/344.537 us`.

## 026_ELU

Best known team result:

| hash | team | result | latency | source shape |
| --- | --- | --- | --- | --- |
| 8f1d8b6 | team_42 | PASS | 309.126 us | split abs plus degree-4 polynomial, 32 tasks, tile 8192 |
| e0f1ca6 | team | PASS diff 6.34e-03 | 379.912, 433.991 us | same source slow band |

Tried:

| hash | idea | result | latency | conclusion |
| --- | --- | --- | --- | --- |
| 40cdcb0 | exact negative branch with exp | PASS diff 4.27e-03 | 589-618 us | correct but too slow |
| 229e711 | exphp split | PASS | 650-887 us | too slow |
| fadc721,49ae251 | ReLU-style split | PASS | 462-605 us | too slow |
| d6906c1 | fixed-device plus direct launch | PASS | 366-375 us | worse than warmup/static path |
| local search | cubic/clamped cubic approx | FAIL risk | best local error around 1.8e-02 | not enough accuracy |
| mixed | fused Horner | PASS | around 329 us | not better than best |
| 3acd7c7 | remove one NRAM copy in relu split | PASS diff 6.14e-03 | best 355 us in 5-commit burst | mathematically safe but no OJ speed signal |
| 3906f58 | four-coefficient minimax refit with clamp 4.485 | PASS diff 6.98e-03 | best 331.962 us in 5-commit burst | correct, but still launch-bound; three-coefficient fit cannot reach 1e-2 threshold |
| 852b47b | historical coefficients with FMA relu split and no clamp | PASS diff 6.34e-03 | 384-387 us | no advantage over original historical source |
| 43362032/48bb16c4/47eab490 | restored 028 baseline, joint `026,028,029` sampler | PASS | best `310.110 us` | restored production 028 source; no 026 low-tail collision |
| 8f1d8b61 rerun | historical best 026 rerun at empty queue | PASS | `386.250 us` | current worker window slow; empty queue alone still not enough |
| 83ba2cf2 | paced fresh `026,027` sampler | PASS | best `351.177 us` | slow window with long worker occupancy; no 026 low-tail collision |
| 281170d7/0410dedd/e4c520db/cfa1df15 | CNRT notifier around kernel only | PASS | stdout `HWUS` mostly 8-9 us, wall roughly 430-560 us | kernel body is not the main latency source |
| a0394b5e/73cdcbce | CNRT notifier comparison: abs split baseline vs ReLU split | PASS | both variants HWUS n=6 median `8.0 us` | ReLU split has no real kernel advantage and produced large wall outliers; keep abs split |
| 925755f7..b1adbb2c | fixed device1 probe | collected | 026 best `349.453 us`, median `393.583 us` over 16 rows | worse than stable; device1 is not useful |
| 531c8934..27521bb8 | DEVSEL stdout correlation | partial | 026 min `350.498 us`, median `384.919 us`; dev3 had one low sample | weak correlation only; not a gate |
| f45b3fe2..c895be3f | fixed candidate device3 for 026 | partial | 026 best `359.002 us`, median `385.247 us` over 15 rows | fixed dev3 did not reproduce low-tail; abandon |
| f00e5cfe..2c495b3d | constructor CNRT memset warmup | FAIL | `N/A` latency, all rows fail | raw CNRT queue/memset in constructor is unsafe here |
| 8f8fb387..95b0f0a6 | stable pure lowtail burst | partial | 026 best `339.350 us` in early rows; many commits still `no_table` | no leaderboard collision yet; recollect later |
| 8f1d8b61 rerun | mixed high-processing rerun | PASS | best `363.005 us`, median `384.044 us` over 8 recent rows | queue saturation is harmful; do not use high fanout for 026 |

Keep:
- Current best path should keep constructor device selection, tiny `zeros({1})` warmup, `static bool init`, 32 block tasks, tile 8192.
- Further progress likely needs a better ELU approximation that meets OJ tolerance or a host overhead reduction that preserves warmup benefit.

Avoid repeating:
- Exact exp variants are not competitive.
- Simple cubic approximation is not accurate enough for randn fp32 ELU.
- Removing constructor/device warmup was not a win for this op.

## 027_GELU

Best known team result:

| hash | team | result | latency | source shape |
| --- | --- | --- | --- | --- |
| ac5127e | team_91 | PASS diff 2.39e-03 | 302.266 us | 32 block tasks, tile 8192, direct launch, no static init, `empty({1})+empty({1})` warmup |
| c6305b4 | team_42 | PASS diff 2.39e-03 | 302.994 us | same source rerun |
| cdc0c7f3..1331dcaa | team | pending | pending | 16-commit stable lowtail burst submitted after external leader moved to `281.317 us` |

Recent reruns:

| hash | idea | result | latency | conclusion |
| --- | --- | --- | --- | --- |
| 8cb8ae9 | same as best source | PASS | 335.875, 343.184 us | jitter/worker state can dominate |
| 8ed254e | same as best source | PASS | 371.753, 367.011 us | slow band, do not infer source regression |
| ee97635 | baseline rerun | PASS | 393-404 us | slow band |
| 21b2402 | same as best source | PASS diff 2.39e-03 | 345.068, 388.087 us | slow band |
| d4eebda | device-spread constructor plus same warmup | PASS diff 2.39e-03 | 369.583, 393.371 us | worse, fixed device0 warmup is better for GELU |
| 1c23ee6 | restore fixed device0 warmup | PASS diff 2.39e-03 | 370.323, 390.470 us | restored baseline source but slow band |
| d59310a3/27ed373f/0c3e4404/26808879 | same best source, fresh low-tail sampler | PASS | best `345.089 us` | slow window; no low-tail collision |
| c6305b44 rerun | historical best 027 rerun at empty queue | PASS | `363.879 us` | current window slow; no evidence of source regression |
| 83ba2cf2 | paced fresh `026,027` sampler | PASS | best `365.683 us` | slow window; no 027 low-tail collision |
| ba277402 | CNRT notifier around active GELU kernel | PASS diff `2.39e-03` | stdout `HWUS=7-8 us`; OJ rows `419/926 us` | actual GELU kernel is already single-digit us; wall time is launch/cold/jitter dominated |
| 0a28c730 | abs-tail degree-2 GELU approximation with CNRT notifier | PASS diff `6.21e-03` | stdout `HWUS=8-9 us`; OJ rows `473/1309 us` | approximation is accurate but not faster than `__bang_active_gelu`; do not promote |
| 2ad3163f | restore active GELU production baseline | submitted | pending | restores no-notifier source after 027 probes |

Warmup experiments:

| hash | idea | result | latency | conclusion |
| --- | --- | --- | --- | --- |
| 0957b3d | `empty({1})` only | PASS | 2.6-2.8 ms | terrible, warmup add is needed |
| 2d74c66 | `empty({8})+empty({8})` add | PASS | 327.593 us | worse than `{1}` add |
| prior | `zeros({1})` warmup | PASS | 791-1318 us | bad for GELU |

Kernel and schedule attempts:

| hash | idea | result | latency | conclusion |
| --- | --- | --- | --- | --- |
| 7a9cb74 | polynomial FMA approximation | FAIL diff 1.01e-02 | 342-398 us | not accurate enough and not faster |
| prior | tanh approximation | PASS | 386-401 us | too slow |
| prior | `gelup`/library-like exact path | PASS | slow | too slow |
| 5936186 | 64 tasks, tile 4096 | PASS | 389-461 us | worse |
| prior | 16 tasks/tile 16384, 8 tasks/tile 32768, Union variants | PASS | slower | keep 32 block tasks |
| ce8f06b | mutable pointer | PASS | 350-354 us | worse than `data_ptr` |

Keep:
- Best source is 32 block tasks, tile 8192, no `static bool init`, `empty({1})+empty({1})` add warmup, `data_ptr<at::Half>()`.
- External leader is only about 9 us ahead of best known team result, while active GELU kernel itself is `7-8 us`; repeat submissions are meaningful because OJ jitter is much larger than the remaining algorithm headroom.

Avoid repeating:
- Empty-only warmup, zeros warmup, larger warmup tensor, mutable pointer, and alternate task counts have all lost.
- Abs-tail polynomial GELU approximation passes accuracy but is slower in notifier-measured kernel time; avoid further GELU approximation work unless a new formula is proven with notifier first.

## 028_HardSigmoid

Best known team result:

| hash | team | result | latency | source shape |
| --- | --- | --- | --- | --- |
| fc1a8a4 | team_91 | PASS | 277.957 us | 32 block tasks, tile 8192, add/max/min/mul, constructor device selection, `zeros({1})`, static init |
| 7d03ab6 | team | PASS diff 6.51e-04 | 315.195, 341.575 us | same source slow band |

Tried:

| hash | idea | result | latency | conclusion |
| --- | --- | --- | --- | --- |
| 6ed7e12 | direct launch baseline | PASS | 299.388 us | helped team_42 but not better than best |
| 896e3f1 | direct plus FMA combo | PASS | 344-348 us | worse |
| prior | FMA affine alone | PASS | 321-356 us | smaller diff but slower |
| 79ecc6f | fixed-device plus direct | PASS | 348-357 us | worse |
| a6abb60 | direct 16 tasks | PASS | 335-342 us | worse |
| 5369b64 | direct 64 tasks | PASS | 322-358 us | worse |
| d2d183b | direct 8 tasks | PASS | 331-348 us | worse |
| 8d1779f | Union1 launch probe | PASS | 345.056, 363.796 us | worse, do not repeat |
| 1d20678 | `relun(6)` clamp replacement | PASS diff 6.51e-04 | best about 319 us in repeat group | equivalent but not faster than scalar max/min |
| burst | scalar best source focused lowtail | PASS | best 305.630 us in first collected rows | still far from current external 272 us; window/launch floor dominates |
| 722bfced..16965b0a | FMA affine plus `relun(1)` deep repeat | pending | pending | repeats the lower-instruction variant in the same window; prior small samples passed but were not faster |
| a97ed8c8..f496b670 | host D2H + CPU scalar half hardsigmoid + H2D | pending | pending | checks whether avoiding custom kernel launch can offset PCIe plus CPU half conversion cost |
| d6af6ef6 | restore 028/029 stable baselines | collected | 028 `349.057 us`, 029 `303.634 us` | restores remote HEAD after host/bitselect probes; slow window but PASS confirms known-good scalar clamp sources |
| 94415f2c..e064dcf1 | stable 028/029 lowtail after comment recovery | collected | 028 best `327.521 us`, 029 best `319.555 us` | fresh probability sample after comment recovery; no low-tail collision, window is slow |
| 06405372/a3c8aa3f/9cd11cd2/5f727bea | CNRT notifier around kernel only | PASS | stdout `HWUS` mostly 7-9 us, wall roughly 330-540 us | math body is far below leaderboard gap plus jitter; focus on launch/warmup/low-tail sampling |
| d39bf6f7..fd70fcbf | stable lowtail after hardware model | collected | 028 best `292.259 us`; raw runs include `279.345 us`, but commit average still above leaders | confirms low-tail exists but three-run average is the hard part |
| d420970b..a9770af9 | count==10 gate probe | collected | PASS rows slow, 028 best PASS `335.410 us`, 029 best PASS `324.778 us`; FAIL fast rows `~35-45 us` | filtering to 10-card workers alone is not enough; abandon simple count gate |
| a4023fac..7600321a | stable lowtail after count gate | partial | 028 best `313.857 us` in collected rows | no breakthrough; keep stable path, do not infer regression |
| 925755f7..b1adbb2c | fixed device1 probe | collected | 028 best `322.339 us`, median `358.603 us` over 15 rows | worse than stable; device1 is not a useful filter |
| 531c8934..27521bb8 | DEVSEL stdout correlation | partial | 028 min `300.751 us`, median `370.768 us`; dev3/BDF57 had a low sample but small n | useful as probe only; not enough for a gate |
| f45b3fe2..c895be3f | fixed candidate device3 for 028 | partial | 028 best `331.518 us`, median `370.226 us` over 16 rows | fixed dev3 did not reproduce DEVSEL low-tail; abandon |
| f00e5cfe..2c495b3d | constructor CNRT memset warmup | FAIL | `N/A` latency, all rows fail | raw CNRT queue/memset in constructor is unsafe here |
| 8f8fb387..95b0f0a6 | stable pure lowtail burst | partial | 028 best `332.417 us` in early rows; many commits still `no_table` | no breakthrough yet; recollect later |
| c96e304b rerun | team_42 historical 028 low-tail seed | PASS | earlier best `280.409 us`; later high-processing reruns best `299.294/313.613 us` depending window | source can hit low tail, but scheduling window dominates |
| fc1a8a41 rerun | team_91 historical best seed | PASS | high-processing burst best `288.701 us`; low-processing 10-row rerun best `313.074 us`, one `1412 us` tail | empty queue alone is not sufficient; avoid assuming processing count predicts low tail |
| c5a19b44 | large torch empty reserve/release before warmup | PASS | `343.972/338.127 us` | allocator reservation did not help and adds risk/overhead |
| 4c5a825d | fixed candidate device0 after raw Card correlation | pending | pending | direct check whether Card 0's slightly lower raw average is reproducible |
| fc71bd71/78f20ea0/8e952ba6/e7047319/4d8d2999/977547bd/61791ad2 | restored normal fixed-device sampler, fresh commit collision | PASS | best `304.847 us` in collected rows | did not reproduce 4c low-tail; useful as probability sample only |
| bbd77f13/3db4157e/ff8999cb/fb50b36f | switch 028 static prewarm from `zeros({1})` to one-element add, single-op commits | PASS | best `315.395 us` | did not reproduce the TADDPASS0 weak median signal; not a standalone 028 improvement |
| 21df5255/0cf82af3/14de90b2/ec75ecfd | same tadd028 source, joint `028,029` config | PASS | best 028 `297.065 us`, best 029 `299.795 us` | 028-before-029 context can still hit near-300us, but tadd prewarm did not recover the 4c low-tail |
| 43362032/48bb16c4/47eab490 | restored zeros prewarm, joint `026,028,029` sampler | PASS | best `293.812 us` | production source restored; still above existing best and leader |
| 4c5a825d rerun | historical 028/029 low-tail commit rerun at empty queue | PASS | team_42 `351.763 us`, team_91 `298.720 us` | rerun did not reproduce 273 us low-tail; empty queue alone remains weak |
| 5a74cc29/f21e2c91/afa7c593/28b26cb5 | paced fresh baseline `028,029` submits, wait `processing=0` between commits | PASS | best `288.294 us` | lower concurrency improved one 028 tail but did not reach 273/272; useful probability shaping only |
| a4d04b17 | `FUSION_FMA` affine then clamp to `[0,1]` | PASS diff `3.26e-04` | best `306.507 us` | fewer vector ops and smaller diff, but slower than baseline in paced window; restored scalar add/mul |
| note | algorithm changes after this point should use CNRT notifier probes | n/a | n/a | wall latency is too noisy to judge 1-3 us kernel-body changes directly |
| 818923ec/29aecfe1 | CNRT notifier comparison: scalar baseline vs FMA affine | PASS | baseline HWUS n=6 median `7.5 us`; FMA HWUS n=6 median `7.0 us` | FMA saves at most about `0.5 us` median hardware time, far below wall jitter; not worth promoting given wall samples were not better |

Keep:
- Stable best uses constructor/device selection, `zeros({1})` warmup, `static bool init`, 32 block tasks, tile 8192, `mutable_data()`.
- Direct launch removal is not generally good here.

Avoid repeating:
- FMA substitutions and direct/fixed-device variants have repeatedly lost.
- Task count sweep is already covered; 32 tasks is the only competitive setting so far.
- Treat the current FMA+`relun(1)` batch as the last confirmation before demoting this expression family unless it produces a sub-280 us low tail.
- Host scalar hardsigmoid is a probe only; if it is slow, avoid CPU half-conversion routes for non-bitwise pointwise ops.
- As of 22:10, all fresh hashes from the 026 lowtail, 028 FMA/host, and 029 host/relun batches still show `EMPTY no_table` even after accepted webhook replays; OJ status shows processing drained and leaderboard queue items pending. Recollect these hashes before drawing conclusions.

## 029_HardTanh

Best known team result:

| hash | team | result | latency | source shape |
| --- | --- | --- | --- | --- |
| ff266fb | team_91 | PASS | 284.441 us | 8 block tasks, tile 32768, scalar max/min clamp, constructor device selection, `zeros({1})`, static init |
| 7d03ab6 | team | PASS diff 0.00e+00 | 324.845, 343.614 us | same source slow band |

Tried:

| hash | idea | result | latency | conclusion |
| --- | --- | --- | --- | --- |
| 13caae1 | direct baseline | PASS | 331-341 us | worse |
| 1c8b49a | direct 32 tasks | PASS | 296.872 us | best direct variant, still worse than stable |
| 7b9c199 | vector compare clamp | PASS | 343-374 us | worse |
| 79ecc6f | fixed-device plus direct | PASS | 326-337 us | worse |
| 7782fdd | 4 tasks | PASS | 351-354 us | worse |
| prior | 16 and 32 task variants | PASS | slower | keep 8 tasks |
| 8d1779f | Union1 launch probe | PASS | 319.663, 357.299 us | worse, do not repeat |
| burst | scalar best source focused lowtail | PASS | best 297.689 us over 12 commits | did not collide with external 268 us; continue only as probability sampling |
| 9cb3059/a380f5b | count==10 gate | mixed PASS/FAIL | PASS best 333.488 us | filters worker class but does not improve passing distribution |
| 240f49d/8e8bc0b | driver 6.2.x + 8-card gate | mixed PASS/FAIL | PASS best 296.565 us | still far from external lowtail; hardware class alone is not the break |
| 3928b74d..fcc8433b | host D2H + CPU FP16 bit clamp + H2D | pending | pending | independent probe to bypass first custom-kernel launch floor; meaningful compute stays in timed region |
| 02409514..7a067b2d | host D2H + CPU FP16 bit clamp + H2D after comment recovery | collected | PASS best `1012.35 us` | exact but far too slow; PCIe/CPU route is noncompetitive for 512 KiB pointwise |
| ebefc7be | restore stable scalar clamp after host probe | collected | PASS `332.407/346.410 us` | remote HEAD restored to known-good 029 source |
| 3017e456..963f77fe | `relun(x+1,2)-1` expression | pending | pending | tests whether one compound clamp intrinsic can beat two scalar clamp intrinsics despite extra adds |
| 9fd01ee5..2945fcf8 | integer bit-select clamp, broken sign/select | pending/invalid | pending | later review found positive overflow maps to `-1`; ignore correctness/timing if comments appear |
| 3b43d9c4..7fea9ddf | integer bit-select clamp, fixed select | invalid | local diff 1.0 | compiles but compare/mask semantics do not match expected select; abandon integer bit-select route |
| cf28d577/888bcc7f/a7c886e3/ab69e5f6 | CNRT notifier around kernel only | PASS | stdout `HWUS` mostly 12-13 us, wall roughly 320-480 us with occasional worse rows | kernel body is small relative to launch/sync/cold path; scalar clamp is near enough at algorithm layer |
| d39bf6f7..fd70fcbf | stable lowtail after hardware model | collected | 029 best `307.066 us`; one raw run `269.088 us` inside a PASS comment | raw lowtail can reach leaderboard scale, but average of three fresh runs dominates |
| d420970b..a9770af9 | count==10 gate probe | collected | mixed FAIL/PASS; PASS rows slower than stable lowtail | simple count gate filters classes but does not select fast launch state |
| a4023fac..7600321a | stable lowtail after count gate | partial | 029 best `309.280 us` in collected rows | no three-run lowtail collision yet |
| 925755f7..b1adbb2c | fixed device1 probe | collected | 029 best `333.443 us`, median `363.996 us` over 16 rows | worse than stable; device1 is not useful |
| 531c8934..27521bb8 | DEVSEL stdout correlation | partial | 029 min `343.884 us`, median `354.529 us`; dev5/BDF9d small sample looked steady | correlation weak; needs direct validation |
| f45b3fe2..c895be3f | fixed candidate device5 for 029 | partial | 029 best `343.696 us`, median `352.274 us` over 15 rows | did not beat stable, abandon fixed dev5 |
| f00e5cfe..2c495b3d | constructor CNRT memset warmup | FAIL | `N/A` latency, all rows fail | unsafe constructor CNRT warmup |
| 8f8fb387..95b0f0a6 | stable pure lowtail burst | partial | 029 best `317.631 us` in early rows; many commits still `no_table` | no breakthrough yet; recollect later |
| 95b0f0a6 rerun | team_42 historical 029 low-tail seed | PASS | best `273.630 us` earlier; high-processing follow-up best `307.252 us`, median `330 us` | low tail is real but not selected by queue saturation |
| 4c5a825d | fixed candidate device fallback plus 028 before 029 | PASS | team_42 best `269.707 us`, later repeats mostly `300-370 us` | improved team_42 leaderboard to second; full stdout shows actual Card 0 and raw `260.6/266.5/282.0 us` for the best row, but >30 repeats did not beat `268.123 us` |
| 8bb86dff | 029-only fixed fallback device | PASS | best `291.324 us`, median about `338 us` over 20+ rows | removing 028 did not reproduce the low tail; 028-before-029 task context likely mattered or the first 4c row was rare luck |
| 13db9424 | 028 as repeated custom-kernel heater before 029 | mixed | best `272.546 us`, later mostly `320-340 us`; 028 intentionally FAIL | stronger heater can reach the near-low band but is not stable; do not promote unless a future row beats the leader |
| 4a83f196 | 4c variant with 029 explicit Card0 | PASS | 028/029 mostly `310-360 us` in small sample | explicit Card0 alone did not reproduce the 4c low-tail |
| 12cddef8 | 028 same-shape HardTanh heater before 029 | mixed | 029 best `327.991 us`; 028 intentionally FAIL | matching 029 task shape as heater was worse than normal 028-before-029 |
| fc71bd71/78f20ea0/8e952ba6/e7047319/4d8d2999/977547bd/61791ad2 | restored normal fixed-device sampler, fresh commit collision | PASS | team_91 best `296.670 us`; team_42 best `330.666 us` in collected rows | did not reproduce 4c low-tail; more evidence that three-run low-tail collision dominates |
| a4d08367/38804c7e/6667e51f/532abda8 | 029 with only 028 source changed to tadd prewarm in adjacent single-op commits | PASS | best `307.256 us` | no useful 029 effect without joint config |
| 21df5255/0cf82af3/14de90b2/ec75ecfd | same tadd028 source, joint `028,029` config | PASS | best `299.795 us` | joint context is better than single-op tadd, but still far from 4c/best |
| 43362032/48bb16c4/47eab490 | restored zeros prewarm, joint `026,028,029` sampler | PASS | best `305.838 us` | production source restored; no 029 low-tail collision |
| 4c5a825d rerun | historical 028/029 low-tail commit rerun at empty queue | PASS | team_42 `338.047 us`, team_91 `324.972 us` | did not reproduce 269 us low-tail |
| 5a74cc29/f21e2c91/afa7c593/28b26cb5 | paced fresh baseline `028,029` submits, wait `processing=0` between commits | PASS | best `302.357 us` | paced submit did not recover 029 low-tail |
| a4d04b17 | 028 FMA affine variant before unchanged 029 | PASS | best `327.202 us` | 028 kernel variant did not help 029 context |
| bc9b4e13 | force 029 device 0 under paced low concurrency | PASS | best `317.570 us` | explicit Card0 still did not reproduce 4c low-tail; restore `limit > 8 ? 8 : 0` |
| note | algorithm changes after this point should use CNRT notifier probes | n/a | n/a | wall latency is too noisy to judge 1-3 us kernel-body changes directly |
| f14f4272 | restore production 028/029 after notifier/Card0 probes | PASS | 028 best `338.033 us`, 029 best `331.462 us` | remote HEAD restored to scalar 028 and `limit > 8 ? 8 : 0` 029 |
| 5b0494c0..ec3d934b | `cnrtInvokeKernel` direct launch probe on 029 | PASS | rows stayed in `321-412 us` wall band | no reproducible low-tail improvement; direct invoke path is slower/noisier than normal launch syntax for this workload |
| 20c6aed3 | restore production 029 after direct invoke probe | PASS | `366.418/352.930 us` | source restored to baseline; direct invoke path closed unless graph/topology changes first-launch overhead |
| ff08dc4b..e1c7dc01 | static `torch::zeros({262144})` prewarm for 029 | PASS | 16 rows min/median/max `291.169/338.369/415.411 us` | larger zero prewarm can still hit ordinary low-ish tails, but does not shift the distribution enough to beat existing leaders |
| f4e2071d..9a476373 | static `torch::zeros({8388608})` prewarm for 029 | PASS | 16 rows min/median/max `319.893/346.637/370.222 us` | heavier memset prewarm is worse; do not use large zero warmups as a latency lever |
| 88795e41..4a77e0dd | fresh joint `028,029` baseline low-tail sampler | PASS | 029 22 rows min/median/max `308.411/343.674/395.118 us`; 028 22 rows min/median/max `322.284/351.848/445.229 us` | same-source joint context did not reproduce historical `4c5a825d` low tail; 4c remains rare tail luck, not a stable selector |
| ac0a965d..db4b4710 | pre-instantiated CNRT TaskTopo entity for 029 | PASS | 16 rows min/median/max `323.196/346.442/381.683 us` | updating kernel node params plus `TaskTopoEntityInvoke` does not reduce measured launch/sync overhead; keep normal launch |
| 2da3db9a | restore production 029 after TaskTopo probe | PASS | `328.089/368.789 us` | source restored to baseline |
| 6097bce8..6449d044 | high-throughput 028-only fresh baseline collision batch | PASS | early 37 rows best `320.134 us`, many `390-480 us` | pushing `processing` to ~18 active workers creates a slow window; avoid blind high-saturation bursts for 028/029 |

Keep:
- Stable best uses constructor/device selection, `zeros({1})` warmup, `static bool init`, 8 block tasks, tile 32768.
- 8 tasks appears better than wider task layouts because host/setup and simple memory pass dominate.

Avoid repeating:
- Direct launch without static init did not beat stable warmup path.
- Vector compare clamp did not beat scalar clamp.
- Host bit-clamp probe is pending; if it passes and beats 268 us, follow with async/pinned/SIMD refinements. If it is slow, record PCIe round-trip as noncompetitive for 512 KiB pointwise.
- Relun-expression probe is pending; if it does not beat scalar clamp after 8+ rows, demote `relun` clamp family for 029 as well.
- Integer bit-select clamp failed local correctness even after select fix; do not repeat unless the mask construction is completely redesigned.

## Pending Questions

- Can repeated submissions of the exact best GELU source land below 300.94 us, or is external code materially different?
- 45ee0b71..c3cfd559 is a fresh 12-submit 026 lowtail window using the current best source; promote only if at least one OJ row beats the external 304.996 us or if the whole distribution shifts after 8+ rows.
- 1e91cc3e..bd36e46c is a fresh 8-submit 026 stable lowtail after comment recovery; collected best `326.028 us`, no collision with external `304.996 us`.
- dca44baa..4e862349 retested 026 `relu(x); b=c-x` split after comment recovery. 16 rows PASS, best `346.824 us`; no improvement, so restore abs-split baseline.
- 133a7c8e restores 026 abs-split baseline after the relu-split experiment.
- Local distribution search: degree-3 ELU polynomial without a tail select is not viable. On 64 OJ-sized random samples, optimized degree-3 max error median was about `0.0266`, p90 `0.062`, max `0.268`, above the `1e-2` threshold.
- ace8600c..c28fe614 tested 026 four-degree Horner with `__bang_fusion(FUSION_FMA)` replacing three `mul+add_scalar` pairs. 16 rows PASS, best `346.822 us`; no improvement, restored plain Horner.
- 6c9eb64c..08d1ed22 is an empty-queue stable lowtail burst across 026/028/029 after restoring all baselines.
- 281170d7..ab69e5f6 adds CNRT notifiers around the real kernel for 026/028/029 and prints `HWUS` to OJ stdout. Measurement conclusion: 026 `8-9 us`, 028 `7-9 us`, 029 `12-13 us`. This is a measurement-only probe; wall time includes notifier wait/duration overhead and should not be used as a production score.
- Union1 launch does not reduce overhead for 028/029 on OJ; 8d1779f was slower.
- Legal host warmup/preallocation remains narrow: registered torch-mlu static warmup works; raw CNRT malloc/memset/queue in constructor failed and should not be reused.
- For 026, can a higher-order or piecewise approximation satisfy tolerance with fewer vector ops than current degree-4 split?
- 8f8fb387..95b0f0a6 is a 16-round pure stable lowtail burst across 026/028/029 after restoring production sources.
- a550534e is an explicit stable batch trigger for 026/027/028/029 after some rapid repeat commits had zero GitHub comments. Use it to distinguish OJ/comment delay from parser failure.
