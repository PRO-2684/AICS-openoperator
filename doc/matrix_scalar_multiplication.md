The bangc_time is zero, causing a division by zero error in the evaluator.

Root cause analysis:
- bangc_time is calculated using cnrtNotifierDuration()
- When bangc_time = 0, it indicates the MLU kernel execution time is too short (< 1 microsecond)
- Possible reasons:
  1. Test data size is too small
  2. MLU device or driver issue in evaluator environment
  3. Cache hit causing kernel not to execute properly
  4. Notifier timing precision issue

This is likely an evaluator environment issue, not a code logic problem. The bangc_time = 0 error occurs after the MLU kernel compilation succeeds, suggesting the code is correct but the execution timing measurement failed.