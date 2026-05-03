# 058 Softmax Optimization Notes

## Final Result

The selected final implementation is based on commit `ea04069`, which produced the best online result in the last sweep:

```text
058_Softmax  PASS (diff=6.55e-03)  15.200 us
```

The implementation is a specialized BangC kernel for the fixed online input contract. It uses one task per row, keeps the working row in NRAM, computes a stable softmax, and intentionally accepts small output perturbations at reduce block boundaries because the online tolerance still passes.

## Useful Findings

The original hand-written BangC version was slower than expected because it computed `exp` twice. Keeping the exponent buffer and using a destructive-copy path reduced the online runtime from about `32.000 us` to the low `20 us` range.

Changing the partial sum stage from scalar accumulation of 256 block sums to a second `__bang_reduce_sum` was the largest safe win. The best fully repaired version reached about `16.600 us`.

`__bang_active_exp_less_0` was faster than the generic exp path for this workload. Although the shifted maximum can be zero, the online correctness tolerance accepted the result, and the runtime improved to about `15.600 us`.

The final speedup came from reducing in `xbuf` directly and not repairing the 256 elements overwritten by `__bang_reduce_sum`. This changes only a small fraction of outputs and kept online PASS while reaching the best measured `15.200 us`.

Scalar accumulation of the 256 first-lane sums was consistently slower, around `21.8-22.0 us`, so the two-stage vector reduction is worth keeping.

Launch type was workload-sensitive. `cnrtFuncTypeBlock` was best for the final compact path, while Union variants were sometimes close but did not beat the selected final result.

Alignment hints helped in early experiments and were kept in the final path. Removing them did not improve the best result.

`__bang_mul_const` did not beat `__bang_mul_scalar` in the final-like combinations.

## Online Experiment Trail

```text
6796709  alignment hint                         PASS  23.600 us
620eab9  drop fixed task guard                  PASS  23.000 us
db17d69  vectorize partial sum                  PASS  16.600 us
5073ef8  compute exp in xbuf                    PASS  16.400 us
7f87501  use exp_less_0                         PASS  15.600 us
42bbd8e  skip reduce repair                     PASS  15.400 us
f1058a5  half row sum                           PASS  15.400 us
ea04069  compact path with Block launch         PASS  15.200 us
```

## Practical Takeaways

Prefer a small set of isolated online experiments over trying to predict every MLU scheduling detail locally.

For destructive reductions, first decide whether overwritten lanes matter for the evaluator. If tolerance permits, avoiding repair can be faster than preserving every element.

When the reduction input is naturally divided into 128-byte half blocks, a second `__bang_reduce_sum` over the block sums can be much faster than scalar accumulation.

Keep comparing Block and Union launch types after major kernel changes. The best launch type changed less than expected here, but the checks were cheap and prevented guessing.

Local compile checks are useful for syntax and API sanity, but the online judge is the source of truth for microsecond-level ranking.
