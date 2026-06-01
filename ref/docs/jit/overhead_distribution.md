# OJ Overhead Distribution

Numbers below are from repeated OJ experiments recorded in
`ref/jitter_experiments.md`. Treat exact values as current-window evidence, not
absolute hardware constants.

## Cost Layers

| layer | typical evidence | median scale | interpretation |
| --- | --- | ---: | --- |
| host-only wrapper/sync | `J027-HINFO0`, `J027-SINFO0` | `~40 us` | Python/C++ wrapper and stream lookup are not the 300us floor. |
| idle queue query | `J027-QQUERY0` | `~44 us` | current stream query is host-only scale. |
| CNRT memory info query | `J027-MEMINFO0` | `~116 us` | runtime queries can be above host-only and have outliers, but still below kernel launch. |
| empty custom kernel launch + final sync | `J027-KINFO0/KINFO3` | `~334-346 us` | first timed custom-kernel launch/sync dominates tiny kernels. |
| memcpy-only custom kernel | `J027-MINFO0` | `~348 us` | one GDRAM round trip over the tested tiny shape adds little over launch floor. |
| real tiny pointwise kernel hardware | `J026/028/029-NT0` | `7-13 us` | arithmetic/memory body is small relative to wall time. |
| no static MLU warmup | `J027-NOWARM0` | `~2264 us` | first-use cold start without registered-op prewarm is multi-ms. |

## Pointwise 026-029 Kernel Time

CNRT notifier probes around only the real custom kernels measured:

| op | hardware duration | raw prints |
| --- | ---: | ---: |
| 026 ELU | `8-9 us` | 24 |
| 028 HardSigmoid | `7-9 us` | 21 |
| 029 HardTanh | `12-13 us` | 21 |

The same families show OJ wall rows in the hundreds of microseconds. A
hypothetical 50% algorithm-body win would save only about `4-7 us`, which is
smaller than ordinary OJ distribution movement.

Baseline PASS distributions:

| id | rows | min | median | max | p25-p75 implication |
| --- | ---: | ---: | ---: | ---: | --- |
| `J026-B0` | 16 | `352.525` | `379.168` | `415.826` | tens of us |
| `J028-B0` | 16 | `308.676` | `345.125` | `400.049` | tens to about 90us |
| `J029-B0` | 16 | `312.969` | `350.669` | `365.604` | tens of us |

Decision rule: for these shapes, sub-10us arithmetic changes are statistically
invisible unless a same-window, at least 8-row batch shows a distribution-level
shift.

## First Launch Versus Later Launches

`J027-K32INFO0` compared many empty launches against single empty-launch
controls. The first timed custom-kernel launch plus final sync is the large
cost. Extra warmed launches add roughly `3-4 us` each at median.

Implications:

- Avoid multi-kernel decompositions for small ops.
- Fusing two tiny kernels can save only the extra warmed-launch cost if the
  first launch is still required.
- Removing the only custom kernel is the real break, but only valid when all
  meaningful operator computation remains inside timing or is metadata-only by
  operator semantics.

## Worker And Hardware Classes

Stdout HWINFO probes show at least three OJ classes:

| class | observed behavior |
| --- | --- |
| driver `5.10.22`, 8 cards, 112 CPUs | likely `.40`; product `MLU370-X4K`; SMT visible in container. |
| driver `6.2.10`, 8 cards, 56 CPUs | likely `.41`; product `MLU370-X4`; SMT not visible. |
| driver `6.2.10`, 10 cards, 56 CPUs | likely `.42`; product `MLU370-X4`; non-uniform 4+6 BDF layout. |

All classes remain launch-bound. Driver/card class shifts medians by tens of us,
which is much larger than a 5us arithmetic-body gain.

The `085_Linear` stdout probe confirms all sampled classes share the same core
MLU shape reported by CNRT: 8 clusters, 4 mcores/cluster, 768KiB NRAM, 1MiB
WRAM, 4MiB SRAM, 2MiB L2, 384-bit memory bus, and about 24GiB memory per card.

## Outliers

Observed outlier sources:

- worker/device contention
- MLU runtime calls such as `cnrtMemGetInfo`
- host callback queue operations
- internal `cnrtQueueSync`
- large allocator reservation OOMs or contention
- comment/result pipeline delays unrelated to runtime latency

Use medians, quartiles, and same-window controls. A single best or worst row is
not enough evidence.
