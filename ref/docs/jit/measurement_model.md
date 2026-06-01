# OJ Measurement Model

## Worker Pipeline

The worker handles one commit task as follows:

1. Download the submitted repo snapshot.
2. Read `config` and select matching problems.
3. For each problem, assemble the `.mlu` source into the wrapper template.
4. Run `bangc_torch_tester.py` in a fresh subprocess.
5. Repeat the subprocess run `EVAL_RUNS` times, currently `3`.
6. Average the three `bangc_us` values into the commit-comment latency.

Important implication: each comment row is an average of three fresh Python
processes. The extension is reloaded in every raw run, so first-use runtime and
first custom-kernel launch costs appear repeatedly.

## Timed Region

`bangc_torch_tester.py::benchmark_hardware_time_once` times exactly:

```text
torch.mlu.synchronize()
t0 = time.perf_counter()
output = fn(*inputs)
torch.mlu.synchronize()
t1 = time.perf_counter()
```

Included in `bangc_us`:

- host wrapper call into `ModelNew.forward`
- `bang_func` body
- custom kernel launch/enqueue
- final global MLU synchronize
- any runtime work forced by that synchronize
- all required operator computation if the implementation is valid

Outside `bangc_us`:

- importing the reference file
- compiling/loading the extension
- static C++ constructors
- `ModelNew` construction
- input generation and input transfer
- reference model timing when `ref_times.json` is used
- correctness computation after the timed call

Therefore constructor warmup and metadata setup can affect timing without being
counted, but computing the operator result outside the timed region is invalid.

## Comment Aggregation

The worker stores stdout/stderr for every fresh run as:

```text
[run 1]
...
[run 2]
...
[run 3]
...
```

The table latency is the arithmetic mean of the three raw `@@RESULT@@ bangc_us`
values. A single fast raw run does not improve the commit row unless the other
two runs are also fast.

## Queue State Observations

The Redis `processing` hash is useful for seeing active worker concurrency, but
it is not a direct predictor of fast rows. Recent 026-029 reruns showed:

- High fanout that drives `processing` to roughly `10-16` workers consistently
  lands in slow bands for tiny pointwise ops.
- Waiting for an empty task queue and `processing == 0` before enqueueing does
  not by itself recreate the 028/029 low tails.
- Low tails appear tied to a broader worker/card/runtime state that is not
  exposed by the public snapshot fields.

Use queue state as a guardrail to avoid self-inflicted saturation, not as a
complete scheduler oracle.

## Worker-ID Correlation

Direct Redis watch output can expose the worker id for an active task. In recent
028/029 probes, a single worker id was not enough to predict the low tail:

- `fc1a8a41` on worker `13` landed in the `~320 us` band.
- `95b0f0a6` on workers `3`, `6`, and `9` landed around `339 us`, `345 us`,
  and `318 us`.

A later empty-queue burst of `fc1a8a41` still spread across workers `10`, `16`,
`9`, and `19`, with recent rows around `336-374 us`. This rules out a single
stable worker-id gate for these ops.

A mid-load gate (`processing <= 4`) also did not reliably select the low tail.
The first rows of a fresh `fc1a8a41` burst landed on workers `3`, `19`, `25`,
`8`, `10`, and `9`, with measured rows around `323-352 us`. Queue depth is a
useful pressure dial, but it is not a deterministic selector.

The worker main loop uses `BLPOP(..., timeout=5)`, but pacing reruns around
that timeout did not produce a stable fast lane. A `4.8 s` paced 028 burst
stayed around `309-354 us`; a `5.2 s` paced burst started around `327/370 us`.
The five-second blocking cadence may affect which workers wake first, but it is
not enough by itself to align with fast hardware state.

A longer `12 s` paced burst reused worker `5` twice and then worker `23`, but
still landed at `350.133/330.102/358.928 us`. Reusing the same worker was not
sufficient to reach the low tail, so even the long-cycle phase is only a weak
probability shaper.

Mixed team_42/team_91 load in the same window touched worker ids `0,1,3,5,7,8,
9,11,12,13,15,18,19,20,21,22,23,24,25`, which strongly suggests a shared and
fluid worker pool rather than team-private worker buckets.

Cross-team load shaping with slower 026/027 seeds plus target 028/029 seeds can
move the tail a little, but not enough to gate it. In one mixed window, 028 hit
`296.585 us` once while the surrounding rows stayed around `321-358 us`; 029
reached `290.931 us` but also stayed mostly above `310 us`. So load shaping is
a probability amplifier, not a deterministic lane selector.

A later `3.5 s` cadence 028-only burst also just broadened worker reuse
(`4,14,13,4,2,20,1,21,25,0,12`) while staying around `308-357 us`. Even with
more regular pacing, the shared pool still does not produce a consistent fast
lane.

A larger mixed-load rerun with 026/027 slow seeds plus 028/029 targets across
both teams was not reproducible either. In that later window, 028 only reached
`314.044/318.468 us` for the two teams and 029 stayed around
`326.782-357.540 us`.

So worker id is useful for post-hoc forensics, but it is not yet a usable
submission gate by itself.

## Using Stdout Probes

CNRT notifiers around a kernel provide device hardware duration, but adding
notifiers, waits, duration queries, and printing changes the wall-time path.
Use notifier probes only to separate hardware kernel time from wall time. Do not
use their OJ latency as a production score.

FAIL probes can still be useful when they return `@@RESULT@@`; the worker
records `bangc_us` even if accuracy fails. This is valid for timing partial
paths such as host-only return, empty kernel, or memcpy-only kernels.

Constructor stdout probes can be useful for non-sensitive machine facts such as
driver version, card count, BDF, CPU affinity, and NUMA layout. Keep them
diagnostic-only and remove them after collection. Do not use constructor shell
probes to search for or print credentials, webhook secrets, repository tokens,
or other sensitive files.
