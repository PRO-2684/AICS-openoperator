# JIT And OJ Timing Notes

This directory records OJ timing, cold-start, and jitter findings that affect
operator optimization decisions.

Primary files:

- `measurement_model.md`: worker/tester timing path, what is inside the timed
  region, and how commit comments aggregate runs.
- `overhead_distribution.md`: measured base costs, jitter bands, and which
  costs dominate small kernels.
- `mitigation_playbook.md`: practical methods that helped, did not help, or are
  blocked by the OJ/runtime constraints.
- `system_overhead_analysis.md`: end-to-end cause model for why launch-bound
  kernels have hundreds of microseconds of wall time.
- `linear_stdout_probe.md`: op 085 stdout probe plan and hashes for collecting
  OJ hardware topology.
- `worker_hardware_mapping.md`: `WORKER_ID` to visible CNRT hardware class
  mapping, evidence levels, and safe use rules for `wc` labels.
- `code_profile_028.md`: additive/subtractive 028 probes separating no-op,
  tiny torch-mlu warmup, empty custom-kernel launch, and warmed empty launch.
- `webhook_tail_batch.py`: low-output tail sampler that uses repeated webhook
  seeds plus Redis leaderboard rows instead of waiting on GitHub comments.

Evidence source of truth:

- `ref/jitter_experiments.md` keeps the raw experiment ids, commit hashes, and
  promoted conclusions.
- `ref/pointwise_026_029_attempts.md` keeps task-specific attempts for
  026/028/029.

Current core conclusion:

For tiny pointwise kernels whose CNRT notifier hardware duration is only about
`7-13 us`, algorithm-body micro-optimization is usually below OJ wall-time
jitter. Require same-window, at least 8-row distribution evidence before using
sub-10us arithmetic changes as an optimization direction.

Tooling note:

- Local HTTP/SOCKS proxy environment variables can break high-frequency
  `requests` calls with `Missing dependencies for SOCKS support`. The active
  OJ helper scripts use `requests.Session(trust_env=False)` for webhook/rerun
  requests so sampling does not depend on shell proxy state.
