Latest commit a5dfb1588f8cdde32838feb51897b8e869b79546 is not passing for 005_average_pooling_2d, but the failure is in the evaluator’s own reference path, not in our bang_func.

From the 2026-04-24 13:14:01Z comment for team 91:

- task 005_average_pooling_2d failed before bangc_out was checked
- the traceback shows torch_out = model(*inputs) failing inside /d1/ref/005_average_pooling_2d.py
- the actual error is RuntimeError: CNNL error: CNNL_STATUS_NOT_SUPPORTED

That means the organizer’s reference AvgPool2d on MLU with their full-size input is unsupported locally in the evaluator. Our operator is
not the blocking point there, so there is no code-only fix in average_pooling_2d.mlu that can make this commit pass while the evaluator
itself fails on torch_out.

The second comment at 2026-04-24 13:15:06Z for team 42 is just another rsync timeout, so it also doesn’t reflect our code.

Practical conclusion:

- 005 is currently evaluator-blocked
- changing our BangC implementation will not fix this specific remote failure until the organizers change the reference workload or
evaluator behavior
