# AICS-OpenOperator

## 🔗 Related Links

- [OpenOperator](https://openoperator.cn) | [Monitor](http://43.143.241.66:13000)
- [BangC Tutorial](https://gitservice.cstcloud.cn/kcxain/BangcTutorial/)
- [南京智能计算中心算力平台](https://paas.extrotec.com:30443/)
- [Start Kit](https://github.com/kernel-competition-bot/openoperator-start-kit)

## 🚀 Quick Start

### Setup

1. Clone this repo
2. Update README.md for the problems you want to solve
    - Change "Author" to your GitHub username
    - Modify "AC" column in the table to "🟡"
    - Each one MUST select at least 2 "basic" problems
3. Add your solutions to existing `mlu` files
4. Test by running `./check.sh <Path-to.mlu>`
5. Update `config` file to include the indices of the problems you want to evaluate
    - Helper script `update_config.py` is available for updating the config file
    - You can run `python scripts/update_config.py`
    - It will check for staged solution files (`*.mlu`), update and stage the `config` file for you
6. Commit and push to `main` branch
7. When you received the bot's comment:
    - If your solution is accepted, update the "AC" column in the table to "🟢" and add a link to the commit
    - If you think it's not your problem, or there's something special:
        - Create a document under `doc/<problem_name>.md` to explain the situation
        - Link it in the "AC" column with 🟡
    - Otherwise, continue debugging and resubmit

<details><summary>For repo owner</summary>

To update templates and reference implementations from data source ([`problems.json`](https://openoperator.cn/problems.json)), run:

```bash
python scripts/update_problems.py
```

</details>

### Quick reference

- Check your solution: `./check.sh <Path-to.mlu>`
- Submit: `git add . && python scripts/update_config.py && git commit -m "feat: Add LeakyReLU implementation"`

## ❓ Problems

| Index | Problem | Author | AC |
| ----- | ------- | ------ | -- |
| 001 | [LeakyReLU](LeakyReLU.mlu) | [@PRO-2684](https://github.com/PRO-2684/) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d5737cdb054fbfa00a1e663d75c277871cfa7cc9) |
| 002 | [matrix_scalar_multiplication](matrix_scalar_multiplication.mlu) | [@sunweihao28](https://github.com/sunweihao28/) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d5737cdb054fbfa00a1e663d75c277871cfa7cc9) |
| 003 | [LogSoftmax](LogSoftmax.mlu) | [@sunweihao28](https://github.com/sunweihao28/) | 🔴 |
| 004 | [batched_matrix_multiplication](batched_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/08b0cc5) |
| 005 | [average_pooling_2d](average_pooling_2d.mlu) | [@PRO-2684](https://github.com/PRO-2684/) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d5737cdb054fbfa00a1e663d75c277871cfa7cc9) |
| 006 | [conv_depthwise_2D_square_input_square_kernel](conv_depthwise_2D_square_input_square_kernel.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b76c49f) |
| 007 | [conv_depthwise_separable_2D](conv_depthwise_separable_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f99b4d3) |
| 008 | [conv_pointwise_2D](conv_pointwise_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6b0a073) |
| 009 | [conv_standard_1D](conv_standard_1D.mlu) | N/A | 🔴 |
| 010 | [conv_standard_2D__square_input__square_kernel](conv_standard_2D__square_input__square_kernel.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/fb3103a) |
| 011 | [conv_transposed_1D](conv_transposed_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/284a301) |
| 012 | [conv_transposed_2D__asymmetric_input__square_kernel](conv_transposed_2D__asymmetric_input__square_kernel.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b71b89780d039527c797972e5c3a7179c21e33c3) |
| 013 | [3D_tensor_matrix_multiplication](3D_tensor_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/68e7544) |
| 014 | [4D_tensor_matrix_multiplication](4D_tensor_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ac6e6b1) |
| 015 | [Matmul_for_lower_triangular_matrices](Matmul_for_lower_triangular_matrices.mlu) | N/A | 🔴 |
| 016 | [Matmul_for_symmetric_matrices](Matmul_for_symmetric_matrices.mlu) | N/A | 🔴 |
| 017 | [Matmul_for_upper_triangular_matrices](Matmul_for_upper_triangular_matrices.mlu) | N/A | 🔴 |
| 018 | [Matmul_with_diagonal_matrices_](Matmul_with_diagonal_matrices_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/68e1e75) |
| 019 | [Matmul_with_irregular_shapes_](Matmul_with_irregular_shapes_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6351f12) |
| 020 | [Matmul_with_large_K_dimension_](Matmul_with_large_K_dimension_.mlu) | N/A | 🔴 |
| 021 | [Matmul_with_small_K_dimension_](Matmul_with_small_K_dimension_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bd18e4d) |
| 022 | [Matmul_with_transposed_both](Matmul_with_transposed_both.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/06470bc) |
| 023 | [Matrix_vector_multiplication_](Matrix_vector_multiplication_.mlu) | N/A | 🔴 |
| 024 | [Square_matrix_multiplication_](Square_matrix_multiplication_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bd18e4d) |
| 025 | [fused_matmul_fwd](fused_matmul_fwd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e8213e875de7cd972cf9f5035ef806922fc521d6) |
| 026 | [ELU](ELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a74eba229576bf70536a2b1cd5dcb6bb087dc6ac) |
| 027 | [GELU](GELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a74eba229576bf70536a2b1cd5dcb6bb087dc6ac) |
| 028 | [HardSigmoid](HardSigmoid.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/484f12fd3b34ef23698a8a39bf61948b93dbd03e) |
| 029 | [HardTanh](HardTanh.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ac43faf8cdc2ebfa312f8189d2ca5884c07ebe07) |
| 030 | [MinGPTNewGelu](MinGPTNewGelu.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/814029b) |
| 031 | [Softplus](Softplus.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/91b03a0) |
| 032 | [Softsign](Softsign.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2a96ace) |
| 033 | [Swish](Swish.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ac43faf8cdc2ebfa312f8189d2ca5884c07ebe07) |
| 034 | [Argmax_over_a_dimension](Argmax_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f5be7926396c4de26361c702c76e5aa06b22e63c) |
| 035 | [L1Norm_](L1Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/294939c) |
| 036 | [L2Norm_](L2Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/11a99a49f7de6985b2e5960304ab67ec8df6bef4) |
| 037 | [Max_reduction_over_a_dimension](Max_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7dfbde7) |
| 038 | [Product_reduction_over_a_dimension](Product_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 039 | [BatchNorm](BatchNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/61e4819) |
| 040 | [CosineSimilarityLoss](CosineSimilarityLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4618fe6) |
| 041 | [CrossEntropyLoss](CrossEntropyLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/929d759) |
| 042 | [FrobeniusNorm_](FrobeniusNorm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4959045) |
| 043 | [GroupNorm_](GroupNorm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/02436e1) |
| 044 | [HingeLoss](HingeLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/fcde4ebc3da8bdca80a43d5a305f1ef0db0e1ea9) |
| 045 | [HuberLoss](HuberLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/dc27df2) |
| 046 | [InstanceNorm](InstanceNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6e490cb) |
| 047 | [Max_Pooling_1D](Max_Pooling_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b54a64c) |
| 048 | [Max_Pooling_2D](Max_Pooling_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c2166a5) |
| 049 | [Max_Pooling_3D](Max_Pooling_3D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d9c335a) |
| 050 | [cumprod](cumprod.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ae8dbd8) |
| 051 | [cumsum](cumsum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5c4ebff) |
| 052 | [cumsum_exclusive](cumsum_exclusive.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/83382b3) |
| 053 | [cumsum_reverse](cumsum_reverse.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b0dd977) |
| 054 | [binned_gather](binned_gather.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d719b18) |
| 055 | [binned_scatter](binned_scatter.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d719b18) |
| 056 | [gather](gather.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/3c19c5d) |
| 057 | [rotary_pos_emb_fusion](rotary_pos_emb_fusion.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b5e16ee) |
| 058 | [Softmax](Softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e0152e4) |
| 059 | [hstu_mask_var_len_fwd](hstu_mask_var_len_fwd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6021dcf) |
| 060 | [cross_entropy](cross_entropy.mlu) | N/A | 🔴 |
| 061 | [adaptiveaveragepool](adaptiveaveragepool.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/559b5fa) |
| 062 | [compute_agg](compute_agg.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/9753fcd) |
| 063 | [calc_block_sums](calc_block_sums.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/9383663) |
| 064 | [adaptive_average_pool_nhwc](adaptive_average_pool_nhwc.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6a44d3a) |
| 065 | [lpmax_cleanup](lpmax_cleanup.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2b78a0b) |
| 066 | [transform_vals](transform_vals.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/79922f384d13008e49f64356b20e9e153742bdba) |
| 067 | [renormRowsL1](renormRowsL1.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/cfb29f0) |
| 068 | [upsample_bilinear2d_out_frame](upsample_bilinear2d_out_frame.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d679ca9) |
| 069 | [lstm_cell_forward](lstm_cell_forward.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ebf6dc7) |
| 070 | [Sqrt](Sqrt.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/12910514b2c8a3b84745ef8f80da6607ed18562e) |
| 071 | [Cos](Cos.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/484f12fd3b34ef23698a8a39bf61948b93dbd03e) |
| 072 | [ElementwiseAdd](ElementwiseAdd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 073 | [Power](Power.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/645c8cfba51eb53a547f3a31a4fec7c6a9b27696) |
| 074 | [Std_reduction_over_dim](Std_reduction_over_dim.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/40319901ca05684cc5d64114fbc99ba0adfc9596) |   
| 075 | [TopK](TopK.mlu) | N/A | 🔴 |
| 076 | [Sort](Sort.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a02f636) |
| 077 | [LogSumExp_over_dim](LogSumExp_over_dim.mlu) | N/A | 🔴 |
| 078 | [CumMin](CumMin.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/9a777c5) |   
| 079 | [Global_sum](Global_sum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/40319901ca05684cc5d64114fbc99ba0adfc9596) |   
| 080 | [All_over_dim](All_over_dim.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e80b7722eee7cea415aa88475513260221940a96) |
| 081 | [Sum_2D_reduction](Sum_2D_reduction.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 082 | [Scaled_Dot_Product_Attention](Scaled_Dot_Product_Attention.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d188bb5) |
| 083 | [Causal_Self_Attention](Causal_Self_Attention.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2bda4e6) |
| 084 | [Cross_Attention](Cross_Attention.mlu) | N/A | 🔴 |
| 085 | [Linear](Linear.mlu) | N/A | 🔴 |
| 086 | [Batch_outer_product](Batch_outer_product.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7a1519e) |
| 087 | [Hadamard_product](Hadamard_product.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 088 | [Matrix_trace](Matrix_trace.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/591ffedc0fbb7528b8e252a2d94c663eb19484ec) |
| 089 | [Diagonal_extraction](Diagonal_extraction.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f429319) |
| 090 | [Einsum_4D_contract](Einsum_4D_contract.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/11829f7) |
| 091 | [Matrix_inverse](Matrix_inverse.mlu) | N/A | 🔴 |
| 092 | [SVD_decomposition](SVD_decomposition.mlu) | N/A | 🔴 |
| 093 | [Batch_triangular_solve](Batch_triangular_solve.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6fb0cf5) |
| 094 | [LayerNorm](LayerNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/68d9c0a) |
| 095 | [Add_RMSNorm](Add_RMSNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c184df8) |
| 096 | [LocalResponseNorm](LocalResponseNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a51eccac04a84d9890f26378fd087b08757c90e8) |
| 097 | [PixelShuffle](PixelShuffle.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/27b8f1a13e2c7751c4b68a790c275ce7cdf0d470) |
| 098 | [Attention_with_temperature](Attention_with_temperature.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5962bed) |
| 099 | [DropPath](DropPath.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a244575) |
| 100 | [Adaptive_Max_Pool_2D](Adaptive_Max_Pool_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0fb6230) |
| 101 | [Max_Pool_2D_with_indices](Max_Pool_2D_with_indices.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2080e7a) |
| 102 | [Adaptive_Average_Pool_3D](Adaptive_Average_Pool_3D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7e26c82) |
| 103 | [MSE_Loss](MSE_Loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d2d2e4d) |
| 104 | [KL_Divergence_Loss](KL_Divergence_Loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/44113e7) |
| 105 | [Embedding_lookup](Embedding_lookup.mlu) | N/A | 🔴 |
| 106 | [Embedding_bag_mean](Embedding_bag_mean.mlu) | N/A | 🔴 |
| 107 | [Sinusoidal_positional_encoding](Sinusoidal_positional_encoding.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/69d89838b33a63c71fee6f860051f5c3c333bb4f) |
| 108 | [ALiBi_attention_bias](ALiBi_attention_bias.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/205da22) |
| 109 | [Scatter_add](Scatter_add.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/985dae3) |
| 110 | [Gather_rows](Gather_rows.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/cbad3f0) |
| 111 | [Masked_select](Masked_select.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e2a9009) |
| 112 | [Index_put](Index_put.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/fda51e0) |
| 113 | [Upsample_nearest](Upsample_nearest.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/df90531) |
| 114 | [Pad_constant](Pad_constant.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/3a042f3) |
| 115 | [Unfold](Unfold.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d91221e) |
| 116 | [Grid_sample](Grid_sample.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7ea3553) |
| 117 | [Masked_fill](Masked_fill.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/81a376d) |
| 118 | [Where_conditional](Where_conditional.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/63b972a) |
| 119 | [One_hot_encoding](One_hot_encoding.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/301490c) |
| 120 | [Masked_softmax](Masked_softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2f82d18) |
| 121 | [Scaled_masked_softmax](Scaled_masked_softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/94c3e95) |
| 122 | [Prefix_sum_2D](Prefix_sum_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/54e0f33785b794cab3eb830f5d0718765b1947ca) |
| 123 | [Masked_cumsum](Masked_cumsum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7231b36) |
| 124 | [Weight_standardization](Weight_standardization.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c07abcc) |
| 125 | [Conditional_LayerNorm](Conditional_LayerNorm.mlu) | N/A | 🔴 |
| 126 | [QR_decomposition](QR_decomposition.mlu) | N/A | 🔴 |
| 127 | [Cholesky_decomposition](Cholesky_decomposition.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/10bb360) |
| 128 | [Batch_norm_1D](Batch_norm_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0eaf36f3db1f823ae6b99e52f9e5d6208c213218) |
| 129 | [Attention_score_with_bias](Attention_score_with_bias.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b17f8f6) |
| 130 | [Attention_kv_cache](Attention_kv_cache.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ad76d1) |
| 131 | [Relative_position_encoding](Relative_position_encoding.mlu) | N/A | 🔴 |
| 132 | [Sparse_embedding](Sparse_embedding.mlu) | N/A | 🔴 |
| 133 | [Triplet_loss](Triplet_loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/99356a7) |
| 134 | [Depthwise_conv_2D](Depthwise_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d49d802) |
| 135 | [Dilated_conv_2D](Dilated_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/822d04b) |
| 136 | [Grouped_conv_2D](Grouped_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/016ebe2) |
| 137 | [LSTM_forward](LSTM_forward.mlu) | N/A | 🔴 |
| 138 | [GRU_forward](GRU_forward.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/238eb0d) |
| 139 | [Sparse_attention_mask](Sparse_attention_mask.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1865287) |
