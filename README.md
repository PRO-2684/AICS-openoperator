# AICS-OpenOperator

## 🔗 Related Links

- [OpenOperator](https://openoperator.cn) | [Monitor](http://152.136.18.42:13000)
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
| 004 | [batched_matrix_multiplication](batched_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a329efd6937e0cfa0aee858de590680153d16adc) |
| 005 | [average_pooling_2d](average_pooling_2d.mlu) | [@PRO-2684](https://github.com/PRO-2684/) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d5737cdb054fbfa00a1e663d75c277871cfa7cc9) |
| 006 | [conv_depthwise_2D_square_input_square_kernel](conv_depthwise_2D_square_input_square_kernel.mlu) | N/A | 🔴 |
| 007 | [conv_depthwise_separable_2D](conv_depthwise_separable_2D.mlu) | N/A | 🔴 |
| 008 | [conv_pointwise_2D](conv_pointwise_2D.mlu) | N/A | 🔴 |
| 009 | [conv_standard_1D](conv_standard_1D.mlu) | N/A | 🔴 |
| 010 | [conv_standard_2D__square_input__square_kernel](conv_standard_2D__square_input__square_kernel.mlu) | N/A | 🔴 |
| 011 | [conv_transposed_1D](conv_transposed_1D.mlu) | N/A | 🔴 |
| 012 | [conv_transposed_2D__asymmetric_input__square_kernel](conv_transposed_2D__asymmetric_input__square_kernel.mlu) | N/A | 🔴 |
| 013 | [3D_tensor_matrix_multiplication](3D_tensor_matrix_multiplication.mlu) | N/A | 🔴 |
| 014 | [4D_tensor_matrix_multiplication](4D_tensor_matrix_multiplication.mlu) | N/A | 🔴 |
| 015 | [Matmul_for_lower_triangular_matrices](Matmul_for_lower_triangular_matrices.mlu) | N/A | 🔴 |
| 016 | [Matmul_for_symmetric_matrices](Matmul_for_symmetric_matrices.mlu) | N/A | 🔴 |
| 017 | [Matmul_for_upper_triangular_matrices](Matmul_for_upper_triangular_matrices.mlu) | N/A | 🔴 |
| 018 | [Matmul_with_diagonal_matrices_](Matmul_with_diagonal_matrices_.mlu) | N/A | 🔴 |
| 019 | [Matmul_with_irregular_shapes_](Matmul_with_irregular_shapes_.mlu) | N/A | 🔴 |
| 020 | [Matmul_with_large_K_dimension_](Matmul_with_large_K_dimension_.mlu) | N/A | 🔴 |
| 021 | [Matmul_with_small_K_dimension_](Matmul_with_small_K_dimension_.mlu) | N/A | 🔴 |
| 022 | [Matmul_with_transposed_both](Matmul_with_transposed_both.mlu) | N/A | 🔴 |
| 023 | [Matrix_vector_multiplication_](Matrix_vector_multiplication_.mlu) | N/A | 🔴 |
| 024 | [Square_matrix_multiplication_](Square_matrix_multiplication_.mlu) | N/A | 🔴 |
| 025 | [fused_matmul_fwd](fused_matmul_fwd.mlu) | N/A | 🔴 |
| 026 | [ELU](ELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a74eba229576bf70536a2b1cd5dcb6bb087dc6ac) |
| 027 | [GELU](GELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a74eba229576bf70536a2b1cd5dcb6bb087dc6ac) |
| 028 | [HardSigmoid](HardSigmoid.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/484f12fd3b34ef23698a8a39bf61948b93dbd03e) |
| 029 | [HardTanh](HardTanh.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ac43faf8cdc2ebfa312f8189d2ca5884c07ebe07) |
| 030 | [MinGPTNewGelu](MinGPTNewGelu.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 031 | [Softplus](Softplus.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/73c467d4d54dab842040850f3fab8f6a34772499) |
| 032 | [Softsign](Softsign.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 033 | [Swish](Swish.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ac43faf8cdc2ebfa312f8189d2ca5884c07ebe07) |
| 034 | [Argmax_over_a_dimension](Argmax_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/53d01e365d40388fde0cc121b74c08228881cd94) |
| 035 | [L1Norm_](L1Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 036 | [L2Norm_](L2Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 037 | [Max_reduction_over_a_dimension](Max_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 038 | [Product_reduction_over_a_dimension](Product_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 039 | [BatchNorm](BatchNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/16e7684032be5f8d19845af49027aef1d2a45221) |
| 040 | [CosineSimilarityLoss](CosineSimilarityLoss.mlu) | N/A | 🔴 |
| 041 | [CrossEntropyLoss](CrossEntropyLoss.mlu) | N/A | 🔴 |
| 042 | [FrobeniusNorm_](FrobeniusNorm_.mlu) | N/A | 🔴 |
| 043 | [GroupNorm_](GroupNorm_.mlu) | N/A | 🔴 |
| 044 | [HingeLoss](HingeLoss.mlu) | N/A | 🔴 |
| 045 | [HuberLoss](HuberLoss.mlu) | N/A | 🔴 |
| 046 | [InstanceNorm](InstanceNorm.mlu) | N/A | 🔴 |
| 047 | [Max_Pooling_1D](Max_Pooling_1D.mlu) | N/A | 🔴 |
| 048 | [Max_Pooling_2D](Max_Pooling_2D.mlu) | N/A | 🔴 |
| 049 | [Max_Pooling_3D](Max_Pooling_3D.mlu) | N/A | 🔴 |
| 050 | [cumprod](cumprod.mlu) | N/A | 🔴 |
| 051 | [cumsum](cumsum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f5a0c30bb76cd4ecb84ed5a1cc4dd7ec69443cca) |
| 052 | [cumsum_exclusive](cumsum_exclusive.mlu) | N/A | 🔴 |
| 053 | [cumsum_reverse](cumsum_reverse.mlu) | N/A | 🔴 |
| 054 | [binned_gather](binned_gather.mlu) | N/A | 🔴 |
| 055 | [binned_scatter](binned_scatter.mlu) | N/A | 🔴 |
| 056 | [gather](gather.mlu) | N/A | 🔴 |
| 057 | [rotary_pos_emb_fusion](rotary_pos_emb_fusion.mlu) | N/A | 🔴 |
| 058 | [Softmax](Softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e10226bead0673dae0803184816c6687fe4dd5a5) |
| 059 | [hstu_mask_var_len_fwd](hstu_mask_var_len_fwd.mlu) | N/A | 🔴 |
| 060 | [cross_entropy](cross_entropy.mlu) | N/A | 🔴 |
| 061 | [adaptiveaveragepool](adaptiveaveragepool.mlu) | N/A | 🔴 |
| 062 | [compute_agg](compute_agg.mlu) | N/A | 🔴 |
| 063 | [calc_block_sums](calc_block_sums.mlu) | N/A | 🔴 |
| 064 | [adaptive_average_pool_nhwc](adaptive_average_pool_nhwc.mlu) | N/A | 🔴 |
| 065 | [lpmax_cleanup](lpmax_cleanup.mlu) | N/A | 🔴 |
| 066 | [transform_vals](transform_vals.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/79922f384d13008e49f64356b20e9e153742bdba) |
| 067 | [renormRowsL1](renormRowsL1.mlu) | N/A | 🔴 |
| 068 | [upsample_bilinear2d_out_frame](upsample_bilinear2d_out_frame.mlu) | N/A | 🔴 |
| 069 | [lstm_cell_forward](lstm_cell_forward.mlu) | N/A | 🔴 |
| 070 | [Sqrt](Sqrt.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/12910514b2c8a3b84745ef8f80da6607ed18562e) |
| 071 | [Cos](Cos.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/484f12fd3b34ef23698a8a39bf61948b93dbd03e) |
| 072 | [ElementwiseAdd](ElementwiseAdd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 073 | [Power](Power.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/645c8cfba51eb53a547f3a31a4fec7c6a9b27696) |
| 074 | [Std_reduction_over_dim](Std_reduction_over_dim.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |   
| 075 | [TopK](TopK.mlu) | N/A | 🔴 |
| 076 | [Sort](Sort.mlu) | N/A | 🔴 |
| 077 | [LogSumExp_over_dim](LogSumExp_over_dim.mlu) | N/A | 🔴 |
| 078 | [CumMin](CumMin.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |   
| 079 | [Global_sum](Global_sum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |   
| 080 | [All_over_dim](All_over_dim.mlu) | N/A | 🔴 |
| 081 | [Sum_2D_reduction](Sum_2D_reduction.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 082 | [Scaled_Dot_Product_Attention](Scaled_Dot_Product_Attention.mlu) | N/A | 🔴 |
| 083 | [Causal_Self_Attention](Causal_Self_Attention.mlu) | N/A | 🔴 |
| 084 | [Cross_Attention](Cross_Attention.mlu) | N/A | 🔴 |
| 085 | [Linear](Linear.mlu) | N/A | 🔴 |
| 086 | [Batch_outer_product](Batch_outer_product.mlu) | N/A | 🔴 |
| 087 | [Hadamard_product](Hadamard_product.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 088 | [Matrix_trace](Matrix_trace.mlu) | N/A | 🔴 |
| 089 | [Diagonal_extraction](Diagonal_extraction.mlu) | N/A | 🔴 |
| 090 | [Einsum_4D_contract](Einsum_4D_contract.mlu) | N/A | 🔴 |
| 091 | [Matrix_inverse](Matrix_inverse.mlu) | N/A | 🔴 |
| 092 | [SVD_decomposition](SVD_decomposition.mlu) | N/A | 🔴 |
| 093 | [Batch_triangular_solve](Batch_triangular_solve.mlu) | N/A | 🔴 |
| 094 | [LayerNorm](LayerNorm.mlu) | N/A | 🔴 |
| 095 | [Add_RMSNorm](Add_RMSNorm.mlu) | N/A | 🔴 |
| 096 | [LocalResponseNorm](LocalResponseNorm.mlu) | N/A | 🔴 |
| 097 | [PixelShuffle](PixelShuffle.mlu) | N/A | 🔴 |
| 098 | [Attention_with_temperature](Attention_with_temperature.mlu) | N/A | 🔴 |
| 099 | [DropPath](DropPath.mlu) | N/A | 🔴 |
| 100 | [Adaptive_Max_Pool_2D](Adaptive_Max_Pool_2D.mlu) | N/A | 🔴 |
| 101 | [Max_Pool_2D_with_indices](Max_Pool_2D_with_indices.mlu) | N/A | 🔴 |
| 102 | [Adaptive_Average_Pool_3D](Adaptive_Average_Pool_3D.mlu) | N/A | 🔴 |
| 103 | [MSE_Loss](MSE_Loss.mlu) | N/A | 🔴 |
| 104 | [KL_Divergence_Loss](KL_Divergence_Loss.mlu) | N/A | 🔴 |
| 105 | [Embedding_lookup](Embedding_lookup.mlu) | N/A | 🔴 |
| 106 | [Embedding_bag_mean](Embedding_bag_mean.mlu) | N/A | 🔴 |
| 107 | [Sinusoidal_positional_encoding](Sinusoidal_positional_encoding.mlu) | N/A | 🔴 |
| 108 | [ALiBi_attention_bias](ALiBi_attention_bias.mlu) | N/A | 🔴 |
| 109 | [Scatter_add](Scatter_add.mlu) | N/A | 🔴 |
| 110 | [Gather_rows](Gather_rows.mlu) | N/A | 🔴 |
| 111 | [Masked_select](Masked_select.mlu) | N/A | 🔴 |
| 112 | [Index_put](Index_put.mlu) | N/A | 🔴 |
| 113 | [Upsample_nearest](Upsample_nearest.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1eda60594d6e0b532762d4a53c465437ad2d7e60) |
| 114 | [Pad_constant](Pad_constant.mlu) | N/A | 🔴 |
| 115 | [Unfold](Unfold.mlu) | N/A | 🔴 |
| 116 | [Grid_sample](Grid_sample.mlu) | N/A | 🔴 |
| 117 | [Masked_fill](Masked_fill.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/34b4b3ae47550a2d21386c98098dd82f1493d4bf) |
| 118 | [Where_conditional](Where_conditional.mlu) | N/A | 🔴 |
| 119 | [One_hot_encoding](One_hot_encoding.mlu) | N/A | 🔴 |
| 120 | [Masked_softmax](Masked_softmax.mlu) | N/A | 🔴 |
| 121 | [Scaled_masked_softmax](Scaled_masked_softmax.mlu) | N/A | 🔴 |
| 122 | [Prefix_sum_2D](Prefix_sum_2D.mlu) | N/A | 🔴 |
| 123 | [Masked_cumsum](Masked_cumsum.mlu) | N/A | 🔴 |
| 124 | [Weight_standardization](Weight_standardization.mlu) | N/A | 🔴 |
| 125 | [Conditional_LayerNorm](Conditional_LayerNorm.mlu) | N/A | 🔴 |
| 126 | [QR_decomposition](QR_decomposition.mlu) | N/A | 🔴 |
| 127 | [Cholesky_decomposition](Cholesky_decomposition.mlu) | N/A | 🔴 |
| 128 | [Batch_norm_1D](Batch_norm_1D.mlu) | N/A | 🔴 |
| 129 | [Attention_score_with_bias](Attention_score_with_bias.mlu) | N/A | 🔴 |
| 130 | [Attention_kv_cache](Attention_kv_cache.mlu) | N/A | 🔴 |
| 131 | [Relative_position_encoding](Relative_position_encoding.mlu) | N/A | 🔴 |
| 132 | [Sparse_embedding](Sparse_embedding.mlu) | N/A | 🔴 |
| 133 | [Triplet_loss](Triplet_loss.mlu) | N/A | 🔴 |
| 134 | [Depthwise_conv_2D](Depthwise_conv_2D.mlu) | N/A | 🔴 |
| 135 | [Dilated_conv_2D](Dilated_conv_2D.mlu) | N/A | 🔴 |
| 136 | [Grouped_conv_2D](Grouped_conv_2D.mlu) | N/A | 🔴 |
| 137 | [LSTM_forward](LSTM_forward.mlu) | N/A | 🔴 |
| 138 | [GRU_forward](GRU_forward.mlu) | N/A | 🔴 |
| 139 | [Sparse_attention_mask](Sparse_attention_mask.mlu) | N/A | 🔴 |
