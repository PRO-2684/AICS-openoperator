# Bad Questions / Deferred Restores

Do not put these into `config` until a compliant implementation is ready.

## Needs Score Restore From Old Implementation

| op | file | old source | current/clean attempt | reason |
| --- | --- | --- | --- | --- |
| 005 | `average_pooling_2d.mlu` | `d5737cd:average_pooling_2d.mlu` | `f2c0d20` | old path was CNNL pooling; clean BangC baseline passes but is very slow |
| 034 | `Argmax_over_a_dimension.mlu` | `53d01e3:Argmax_over_a_dimension.mlu` | `7cd2be8` | old path was CNNL reduce; clean scalar BangC passes but is very slow |
| 037 | `Max_reduction_over_a_dimension.mlu` | `1eda605:Max_reduction_over_a_dimension.mlu` | `2d0c223` | old path was CNNL reduce; clean scalar BangC passes but is very slow |
| 076 | `Sort.mlu` | `84bd6c2:Sort.mlu` | `bdf001e` | old path was CNNL TopK; clean bitonic BangC passes but is slower |
| 094 | `LayerNorm.mlu` | `c4a00e8:LayerNorm.mlu` | `835c858` | old path was CNNL LayerNorm; clean scalar BangC passes but is very slow |
| 096 | `LocalResponseNorm.mlu` | `dfd6660:LocalResponseNorm.mlu` | `1e3f500` | old path was CNNL LRN; clean scalar BangC passes but is slow |
| 097 | `PixelShuffle.mlu` | `674ecd9:PixelShuffle.mlu` | `fda0194` | old path was CNNL transpose; clean scalar BangC passes but is very slow |
| 107 | `Sinusoidal_positional_encoding.mlu` | `69d8983:Sinusoidal_positional_encoding.mlu` | `9203779` | old path used CNNL trig; clean BangC trig cache passes with lower score |
| 111 | `Masked_select.mlu` | `c905812:Masked_select.mlu` | `246641b` | old path used secondary torch_mlu stream; current-queue version passes but is slower |
| 122 | `Prefix_sum_2D.mlu` | `54e0f33:Prefix_sum_2D.mlu` | `39e9725` | old file had forbidden CNNL headers only; clean header-only removal passes |

## Do Not Submit Yet

| op | file | old source | current status | reason |
| --- | --- | --- | --- | --- |
| 066 | `transform_vals.mlu` | `79922f3:transform_vals.mlu` | Python-rule payload removed locally | old implementation patched Python reference/assertion behavior; needs compliant validation before OJ |
| 099 | `DropPath.mlu` | `a244575:DropPath.mlu` | Python-rule payload removed locally | old implementation patched `torch.rand`; stochastic reference likely mismatches without a new compliant strategy |
| 101 | `Max_Pool_2D_with_indices.mlu` | `6e64eb8:Max_Pool_2D_with_indices.mlu` | Python-rule payload removed locally | old implementation patched reference tuple output to values-only; wrapper returns tensor only |
| 117 | `Masked_fill.mlu` | `81a376d:Masked_fill.mlu` | not changed by the provided Python regex | has reference/numeric behavior patch patterns outside the current regex; keep out of config |
| 084 | `Cross_Attention.mlu` | `585ae85:Cross_Attention.mlu` | CNNL removed; zero-fill placeholder | old implementation used CNNL scaled dot product attention |
| 092 | `SVD_decomposition.mlu` | `585ae85:SVD_decomposition.mlu` | CNNL/Python bridge removed; zero-fill placeholder | old implementation used CNNL SVD and Python reference patch |
| 124 | `Weight_standardization.mlu` | `c07abcc:Weight_standardization.mlu` | Python-rule payload removed locally; CNNL remains | requires pure BangC conv/transpose replacement |
| 126 | `QR_decomposition.mlu` | `585ae85:QR_decomposition.mlu` | CNNL removed; zero-fill placeholder | old implementation used CNNL QR |
| 127 | `Cholesky_decomposition.mlu` | `10bb360:Cholesky_decomposition.mlu` | not changed by the provided Python regex | uses pybind/Python bridge patterns outside current regex; keep out of config |
| 134 | `Depthwise_conv_2D.mlu` | `d49d802:Depthwise_conv_2D.mlu` | Python-rule payload removed locally | old implementation patched Conv2d bias construction; needs compliant correctness check |
| 135 | `Dilated_conv_2D.mlu` | `585ae85:Dilated_conv_2D.mlu` | Python-rule payload removed locally | old implementation patched Conv2d bias construction; wrapper does not expose bias |
| 136 | `Grouped_conv_2D.mlu` | `016ebe2:Grouped_conv_2D.mlu` | Python-rule payload removed; pure BangC no-bias conv probe `7177762` diff ~1.17e-01 | official wrapper exposes kernel but not Conv2d bias; old result monkey-patched bias off |
| 138 | `GRU_forward.mlu` | `585ae85:GRU_forward.mlu` | CNNL/Python bridge removed; zero-fill placeholder | old implementation used pybind bridge and CNNL GRU |
