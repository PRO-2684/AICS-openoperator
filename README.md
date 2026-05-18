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
| 004 | [batched_matrix_multiplication](batched_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1014e45) |
| 005 | [average_pooling_2d](average_pooling_2d.mlu) | [@PRO-2684](https://github.com/PRO-2684/) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d5737cdb054fbfa00a1e663d75c277871cfa7cc9) |
| 006 | [conv_depthwise_2D_square_input_square_kernel](conv_depthwise_2D_square_input_square_kernel.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1511c2f) |
| 007 | [conv_depthwise_separable_2D](conv_depthwise_separable_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f99b4d3) |
| 008 | [conv_pointwise_2D](conv_pointwise_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/302f7d8) |
| 009 | [conv_standard_1D](conv_standard_1D.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bdbed09) |
| 010 | [conv_standard_2D__square_input__square_kernel](conv_standard_2D__square_input__square_kernel.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f40ff2a) |
| 011 | [conv_transposed_1D](conv_transposed_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/284a301) |
| 012 | [conv_transposed_2D__asymmetric_input__square_kernel](conv_transposed_2D__asymmetric_input__square_kernel.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b71b89780d039527c797972e5c3a7179c21e33c3) |
| 013 | [3D_tensor_matrix_multiplication](3D_tensor_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 014 | [4D_tensor_matrix_multiplication](4D_tensor_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 015 | [Matmul_for_lower_triangular_matrices](Matmul_for_lower_triangular_matrices.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 016 | [Matmul_for_symmetric_matrices](Matmul_for_symmetric_matrices.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 017 | [Matmul_for_upper_triangular_matrices](Matmul_for_upper_triangular_matrices.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 018 | [Matmul_with_diagonal_matrices_](Matmul_with_diagonal_matrices_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 019 | [Matmul_with_irregular_shapes_](Matmul_with_irregular_shapes_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e481e10) |
| 020 | [Matmul_with_large_K_dimension_](Matmul_with_large_K_dimension_.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 021 | [Matmul_with_small_K_dimension_](Matmul_with_small_K_dimension_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 022 | [Matmul_with_transposed_both](Matmul_with_transposed_both.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 023 | [Matrix_vector_multiplication_](Matrix_vector_multiplication_.mlu) | N/A | 🔴 |
| 024 | [Square_matrix_multiplication_](Square_matrix_multiplication_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 025 | [fused_matmul_fwd](fused_matmul_fwd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4dbf765) |
| 026 | [ELU](ELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a74eba229576bf70536a2b1cd5dcb6bb087dc6ac) |
| 027 | [GELU](GELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a74eba229576bf70536a2b1cd5dcb6bb087dc6ac) |
| 028 | [HardSigmoid](HardSigmoid.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/484f12fd3b34ef23698a8a39bf61948b93dbd03e) |
| 029 | [HardTanh](HardTanh.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ac43faf8cdc2ebfa312f8189d2ca5884c07ebe07) |
| 030 | [MinGPTNewGelu](MinGPTNewGelu.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/814029b) |
| 031 | [Softplus](Softplus.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/91b03a0) |
| 032 | [Softsign](Softsign.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2a96ace) |
| 033 | [Swish](Swish.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/85a903b) |
| 034 | [Argmax_over_a_dimension](Argmax_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/aeb17f8) |
| 035 | [L1Norm_](L1Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/41c8d19) |
| 036 | [L2Norm_](L2Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ff273aa) |
| 037 | [Max_reduction_over_a_dimension](Max_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/41c8d19) |
| 038 | [Product_reduction_over_a_dimension](Product_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1c1d5f3) |
| 039 | [BatchNorm](BatchNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1014e45) |
| 040 | [CosineSimilarityLoss](CosineSimilarityLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 041 | [CrossEntropyLoss](CrossEntropyLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6e8b7a0) |
| 042 | [FrobeniusNorm_](FrobeniusNorm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 043 | [GroupNorm_](GroupNorm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 044 | [HingeLoss](HingeLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 045 | [HuberLoss](HuberLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 046 | [InstanceNorm](InstanceNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/df596fd) |
| 047 | [Max_Pooling_1D](Max_Pooling_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 048 | [Max_Pooling_2D](Max_Pooling_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/da11ef7) |
| 049 | [Max_Pooling_3D](Max_Pooling_3D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 050 | [cumprod](cumprod.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a550096) |
| 051 | [cumsum](cumsum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5c4ebff) |
| 052 | [cumsum_exclusive](cumsum_exclusive.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/beb1864) |
| 053 | [cumsum_reverse](cumsum_reverse.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1e22a70) |
| 054 | [binned_gather](binned_gather.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 055 | [binned_scatter](binned_scatter.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 056 | [gather](gather.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/aeb17f8) |
| 057 | [rotary_pos_emb_fusion](rotary_pos_emb_fusion.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 058 | [Softmax](Softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2625c32df5d199042c1356fd08c5889a61bde62f) |
| 059 | [hstu_mask_var_len_fwd](hstu_mask_var_len_fwd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ee82ef4) |
| 060 | [cross_entropy](cross_entropy.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/91b840d) |
| 061 | [adaptiveaveragepool](adaptiveaveragepool.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/beb1864) |
| 062 | [compute_agg](compute_agg.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 063 | [calc_block_sums](calc_block_sums.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 064 | [adaptive_average_pool_nhwc](adaptive_average_pool_nhwc.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/beb1864) |
| 065 | [lpmax_cleanup](lpmax_cleanup.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 066 | [transform_vals](transform_vals.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/79922f384d13008e49f64356b20e9e153742bdba) |
| 067 | [renormRowsL1](renormRowsL1.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 068 | [upsample_bilinear2d_out_frame](upsample_bilinear2d_out_frame.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 069 | [lstm_cell_forward](lstm_cell_forward.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4dbf765) |
| 070 | [Sqrt](Sqrt.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/12910514b2c8a3b84745ef8f80da6607ed18562e) |
| 071 | [Cos](Cos.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/484f12fd3b34ef23698a8a39bf61948b93dbd03e) |
| 072 | [ElementwiseAdd](ElementwiseAdd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bd429aa) |
| 073 | [Power](Power.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/645c8cfba51eb53a547f3a31a4fec7c6a9b27696) |
| 074 | [Std_reduction_over_dim](Std_reduction_over_dim.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/34f921e) |   
| 075 | [TopK](TopK.mlu) | N/A | 🔴 |
| 076 | [Sort](Sort.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/beb1864) |
| 077 | [LogSumExp_over_dim](LogSumExp_over_dim.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 078 | [CumMin](CumMin.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0cbdb6f) |   
| 079 | [Global_sum](Global_sum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/40319901ca05684cc5d64114fbc99ba0adfc9596) |   
| 080 | [All_over_dim](All_over_dim.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/41c8d19) |
| 081 | [Sum_2D_reduction](Sum_2D_reduction.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0cbdb6f) |
| 082 | [Scaled_Dot_Product_Attention](Scaled_Dot_Product_Attention.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 083 | [Causal_Self_Attention](Causal_Self_Attention.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 084 | [Cross_Attention](Cross_Attention.mlu) | N/A | 🔴 |
| 085 | [Linear](Linear.mlu) | N/A | 🔴 |
| 086 | [Batch_outer_product](Batch_outer_product.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 087 | [Hadamard_product](Hadamard_product.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/41c8d19) |
| 088 | [Matrix_trace](Matrix_trace.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/41c8d19) |
| 089 | [Diagonal_extraction](Diagonal_extraction.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/41c8d19) |
| 090 | [Einsum_4D_contract](Einsum_4D_contract.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0db5581) |
| 091 | [Matrix_inverse](Matrix_inverse.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/eab3516bde077f38056d46ed62cfd344a2f22f84) |
| 092 | [SVD_decomposition](SVD_decomposition.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/42b76bf) |
| 093 | [Batch_triangular_solve](Batch_triangular_solve.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 094 | [LayerNorm](LayerNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b08da20) |
| 095 | [Add_RMSNorm](Add_RMSNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b08da20) |
| 096 | [LocalResponseNorm](LocalResponseNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f74f987) |
| 097 | [PixelShuffle](PixelShuffle.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b08da20) |
| 098 | [Attention_with_temperature](Attention_with_temperature.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b4cdc0e) |
| 099 | [DropPath](DropPath.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a244575) |
| 100 | [Adaptive_Max_Pool_2D](Adaptive_Max_Pool_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b08da20) |
| 101 | [Max_Pool_2D_with_indices](Max_Pool_2D_with_indices.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/da11ef7) |
| 102 | [Adaptive_Average_Pool_3D](Adaptive_Average_Pool_3D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b08da20) |
| 103 | [MSE_Loss](MSE_Loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c29f962) |
| 104 | [KL_Divergence_Loss](KL_Divergence_Loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/610a54d) |
| 105 | [Embedding_lookup](Embedding_lookup.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d231f15) |
| 106 | [Embedding_bag_mean](Embedding_bag_mean.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d231f15) |
| 107 | [Sinusoidal_positional_encoding](Sinusoidal_positional_encoding.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b08da20) |
| 108 | [ALiBi_attention_bias](ALiBi_attention_bias.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/205da22) |
| 109 | [Scatter_add](Scatter_add.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/985dae3) |
| 110 | [Gather_rows](Gather_rows.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0db5581) |
| 111 | [Masked_select](Masked_select.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/aeb17f8) |
| 112 | [Index_put](Index_put.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0cbdb6f) |
| 113 | [Upsample_nearest](Upsample_nearest.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/beb1864) |
| 114 | [Pad_constant](Pad_constant.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/41c8d19) |
| 115 | [Unfold](Unfold.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 116 | [Grid_sample](Grid_sample.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 117 | [Masked_fill](Masked_fill.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/81a376d) |
| 118 | [Where_conditional](Where_conditional.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/beb1864) |
| 119 | [One_hot_encoding](One_hot_encoding.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/beb1864) |
| 120 | [Masked_softmax](Masked_softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/291527d) |
| 121 | [Scaled_masked_softmax](Scaled_masked_softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0db5581) |
| 122 | [Prefix_sum_2D](Prefix_sum_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 123 | [Masked_cumsum](Masked_cumsum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 124 | [Weight_standardization](Weight_standardization.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c07abcc) |
| 125 | [Conditional_LayerNorm](Conditional_LayerNorm.mlu) | N/A | 🔴 |
| 126 | [QR_decomposition](QR_decomposition.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f74f987) |
| 127 | [Cholesky_decomposition](Cholesky_decomposition.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/842c19a) |
| 128 | [Batch_norm_1D](Batch_norm_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0db5581) |
| 129 | [Attention_score_with_bias](Attention_score_with_bias.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 130 | [Attention_kv_cache](Attention_kv_cache.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/3cacc10) |
| 131 | [Relative_position_encoding](Relative_position_encoding.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d231f15) |
| 132 | [Sparse_embedding](Sparse_embedding.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d231f15) |
| 133 | [Triplet_loss](Triplet_loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ae1268) |
| 134 | [Depthwise_conv_2D](Depthwise_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d49d802) |
| 135 | [Dilated_conv_2D](Dilated_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f4283d0) |
| 136 | [Grouped_conv_2D](Grouped_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/016ebe2) |
| 137 | [LSTM_forward](LSTM_forward.mlu) | N/A | 🔴 |
| 138 | [GRU_forward](GRU_forward.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/238eb0d) |
| 139 | [Sparse_attention_mask](Sparse_attention_mask.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d69140d) |
