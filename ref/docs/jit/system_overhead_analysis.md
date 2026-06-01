# OJ System Overhead Analysis

This note explains why small custom kernels on the OJ have a `300us+` wall-time
floor even when device hardware work is only `~10us`.

## Measurement Path

The worker evaluates each selected problem by launching fresh tester processes:

```text
worker.py
  evaluate_one
    repeat EVAL_RUNS times, currently 3:
      subprocess.run(python3 bangc_torch_tester.py ...)
    comment latency = mean(raw bangc_us values)
```

Inside each tester process the timed region is:

```text
torch.mlu.synchronize()
t0 = time.perf_counter()
output = model_new(*bangc_inputs)
torch.mlu.synchronize()
t1 = time.perf_counter()
```

Consequences:

- every raw timing is a fresh Python process and fresh extension load;
- C++ static constructors and `ModelNew` construction are outside timing;
- the first timed `bang_func` call still pays custom-kernel launch and final
  global sync work;
- the commit-comment row is the mean of three raw runs, so a single lucky raw
  run is usually diluted.

## Cost Stack

Current repeated OJ probes show this approximate stack:

| component | median scale | evidence | meaning |
| --- | ---: | --- | --- |
| host-only return or stream lookup | `~40us` | `J027-HINFO0`, `J027-SINFO0` | Python/C++ wrapper is not the main floor. |
| idle `cnrtQueueQuery` | `~44us` | `J027-QQUERY0` | current queue query is host-only scale. |
| `cnrtMemGetInfo` | `~116us` | `J027-MEMINFO0` | runtime calls can jitter, but are still below launch. |
| empty custom kernel + external sync | `~334-346us` | `J027-KINFO0`, `J027-KINFO3` | first timed custom launch dominates tiny ops. |
| memcpy-only custom kernel | `~348us` | `J027-MINFO0` | one small GDRAM pass is barely above launch floor. |
| extra warmed launches | `~3-4us` each | `J027-K32INFO0` vs `KINFO0` | first launch is special; later launches are small. |
| no registered-op static warmup | `~2264us` | `J027-NOWARM0` | without prewarm, cold start is multi-ms. |

For 026/028/029, CNRT notifier probes around only the real custom kernel gave:

| op | hardware duration | raw prints |
| --- | ---: | ---: |
| 026 ELU | `8-9us` | 24 |
| 028 HardSigmoid | `7-9us` | 21 |
| 029 HardTanh | `12-13us` | 21 |

The real work is therefore below normal worker/launch jitter. A hypothetical
50% algorithm-body speedup saves only `4-7us`, while baseline PASS distributions
move by tens of microseconds inside one experiment window.

## Why Algorithm Micro-Tuning Disappears

For a tiny pointwise op, wall time can be modeled as:

```text
wall = host_wrapper
     + runtime/stream path
     + first timed custom-kernel launch
     + final synchronize and scheduler wait
     + kernel_hardware_time
     + contention/jitter
```

When `kernel_hardware_time ~= 10us`, changing it by a few microseconds barely
changes `wall`. The launch/sync term and contention term are much larger.

This is visible in three independent ways:

1. Empty custom kernels are already hundreds of microseconds.
2. Memcpy-only kernels overlap empty-kernel medians.
3. Real 026/028/029 hardware duration is only single-digit to low-teen us while
   their PASS wall distributions have p25-p75 bands far larger than that.

Decision rule:

```text
Do not promote a sub-10us arithmetic/body optimization unless a same-window
batch of at least 8 OJ rows shows a distribution-level shift.
```

## Jitter Sources

Likely contributors, ordered by evidence strength:

1. Missing or ineffective static registered-op warmup.
2. Fresh-process first timed custom-kernel launch and final `torch.mlu.synchronize`.
3. Worker class: `5.10.22/8-card`, `6.2.10/8-card`, `6.2.10/10-card`.
4. Device contention and OJ scheduling window.
5. Runtime query / memory pool state / occasional sync outliers.
6. CPU affinity and host NUMA placement, small and inconsistent.
7. Device kernel arithmetic/body for `~10us` pointwise kernels.

## Host Runtime Breakdown Probe

The 2026-06-01 host probe `b01ae226` kept correctness invalid on purpose but
returned the input tensor, so the worker still emitted a full timing row while
printing a host segment trace from inside `bang_func`.

Across 10 raw runs extracted from the OJ comments, the printed segment medians
were:

| segment | median us | min us | max us |
| --- | ---: | ---: | ---: |
| `torch_mlu::getCurMLUStream()` | `0.390` | `0.306` | `0.613` |
| `cnrtQueueSync(current_queue)` on an empty queue | `0.946` | `0.813` | `1.352` |

Observed affinity/context labels in the same sample:

- worker ids spanned `3,4,5,6,7,10,18,22,24`
- current CPU ids spanned `2,19,21,27,35,44,52,71,91`
- affinity counts were only `56` or `112`, matching the known worker-class split
- device id printed as `0` in all sampled runs

Interpretation:

- `getCurMLUStream` plus an empty `cnrtQueueSync` explain only about
  `1-1.5us` of host work inside the operator body.
- Therefore the `35-40us` no-op floor seen in `e84c999d` / `2346be54` is not
  dominated by those CNRT calls.
- The probe's own `fprintf`/logging and extra sync step lift the raw
  `bangc_us` rows into the `~51-63us` band, so its table rows should not be
  used as the base floor; use the segment numbers above instead.
- The remaining gap is mostly upstream: Python entry, C++/PyBind dispatch,
  `ModelNew.forward` wrapper work, and the outer `torch.mlu.synchronize()`
  calls that bracket the timer in `bangc_torch_tester.py`.
- The measured CPU affinity counts change with worker class, but the current CPU
  ids do not map cleanly to faster rows. So CPU affinity is a distribution
  shaper, not a single-step gate.

Practical use:

- Keep using `torch.mlu.synchronize`-bracketed timings for leaderboard work.
- Use host-segment probes only to decide where micro-optimizations are futile.
- For 026-029, the only plausible leverage left is to change the warm launch
  distribution or the worker/queue state, not to shave another microsecond off
  `getCurMLUStream` or empty `cnrtQueueSync`.

The `085_Linear` stdout probe sharpened the worker classes:

```text
5.10.22 / 8-card:
  112 visible CPUs, SMT on, MLU370-X4K, BDF 4f/50/53/57/9c/9d/a0/a4.

6.2.10 / 8-card:
  56 visible CPUs, SMT off, MLU370-X4, same 8-card BDF pattern.

6.2.10 / 10-card:
  56 visible CPUs, SMT off, MLU370-X4, BDF 4f/50/53/57/9c/9d/a0/a1/a4/b1.
```

All sampled cards report the same core execution shape through CNRT
(`8x4` cores, `768KiB` NRAM, `1MiB` WRAM), so the observed small-kernel variance
is more plausibly launch/runtime/host scheduling class than a different MLU core
geometry.

## What Can Reduce It

Reliable:

- run a real one-element registered torch-mlu op in static initialization;
- keep tiny ops to one custom kernel;
- avoid in-function syncs and host callbacks;
- use repeated submissions and compare medians/quartiles, not single rows.

Useful only as diagnostics:

- CNRT notifiers around the kernel;
- `cnrtMemGetInfo`, BDF, device count, driver/lib version prints;
- large memory reservation, because it exposes contention/OOM but does not
  consistently improve valid rows.

Weak or blocked:

- user Bang kernel launch from constructor: blocked by kernel registration;
- TaskTopo kernel-node path: blocked by registration/source audit in tested paths;
- `cnrtSetDeviceFlag` block/yield: did not visibly change sync mode;
- CPU affinity and NUMA pinning: effects are small and contradicted across windows;
- explicit `cnrtInvokeKernel`: overlaps normal `<<<>>>`.
