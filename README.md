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
| 004 | [batched_matrix_multiplication](batched_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/15a23520) |
| 005 | [average_pooling_2d](average_pooling_2d.mlu) | [@PRO-2684](https://github.com/PRO-2684/) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d5737cdb054fbfa00a1e663d75c277871cfa7cc9) |
| 006 | [conv_depthwise_2D_square_input_square_kernel](conv_depthwise_2D_square_input_square_kernel.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1511c2f) |
| 007 | [conv_depthwise_separable_2D](conv_depthwise_separable_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/924a94e) |
| 008 | [conv_pointwise_2D](conv_pointwise_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/da43c81) |
| 009 | [conv_standard_1D](conv_standard_1D.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bdbed09) |
| 010 | [conv_standard_2D__square_input__square_kernel](conv_standard_2D__square_input__square_kernel.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f40ff2a) |
| 011 | [conv_transposed_1D](conv_transposed_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4b669c6) |
| 012 | [conv_transposed_2D__asymmetric_input__square_kernel](conv_transposed_2D__asymmetric_input__square_kernel.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b71b89780d039527c797972e5c3a7179c21e33c3) |
| 013 | [3D_tensor_matrix_multiplication](3D_tensor_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/73f9bdf) |
| 014 | [4D_tensor_matrix_multiplication](4D_tensor_matrix_multiplication.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 015 | [Matmul_for_lower_triangular_matrices](Matmul_for_lower_triangular_matrices.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7996024f) |
| 016 | [Matmul_for_symmetric_matrices](Matmul_for_symmetric_matrices.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/9e98b3b) |
| 017 | [Matmul_for_upper_triangular_matrices](Matmul_for_upper_triangular_matrices.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6453f15) |
| 018 | [Matmul_with_diagonal_matrices_](Matmul_with_diagonal_matrices_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/744f888) |
| 019 | [Matmul_with_irregular_shapes_](Matmul_with_irregular_shapes_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/66132a5) |
| 020 | [Matmul_with_large_K_dimension_](Matmul_with_large_K_dimension_.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8efc880) |
| 021 | [Matmul_with_small_K_dimension_](Matmul_with_small_K_dimension_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8893dc1) |
| 022 | [Matmul_with_transposed_both](Matmul_with_transposed_both.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6453f15) |
| 023 | [Matrix_vector_multiplication_](Matrix_vector_multiplication_.mlu) | [@crispllk](https://github.com/crispllk) | 🔴 |
| 024 | [Square_matrix_multiplication_](Square_matrix_multiplication_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8c695bb) |
| 025 | [fused_matmul_fwd](fused_matmul_fwd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/cf6847e2d) |
| 026 | [ELU](ELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c564674) |
| 027 | [GELU](GELU.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/edebaba) |
| 028 | [HardSigmoid](HardSigmoid.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ba345597) |
| 029 | [HardTanh](HardTanh.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bd5f6b55) |
| 030 | [MinGPTNewGelu](MinGPTNewGelu.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7b1043a) |
| 031 | [Softplus](Softplus.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bd5f6b55) |
| 032 | [Softsign](Softsign.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/3fefce22) |
| 033 | [Swish](Swish.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/23e7636) |
| 034 | [Argmax_over_a_dimension](Argmax_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a7c167c) |
| 035 | [L1Norm_](L1Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/21b5b409c) |
| 036 | [L2Norm_](L2Norm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4e2fcd6) |
| 037 | [Max_reduction_over_a_dimension](Max_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/815b246) |
| 038 | [Product_reduction_over_a_dimension](Product_reduction_over_a_dimension.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4e2fcd6) |
| 039 | [BatchNorm](BatchNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1014e45) |
| 040 | [CosineSimilarityLoss](CosineSimilarityLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8bccaf2) |
| 041 | [CrossEntropyLoss](CrossEntropyLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6e8b7a0) |
| 042 | [FrobeniusNorm_](FrobeniusNorm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/3ee6bb306) |
| 043 | [GroupNorm_](GroupNorm_.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/98eaea5) |
| 044 | [HingeLoss](HingeLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/88f4c7f) |
| 045 | [HuberLoss](HuberLoss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/38210af) |
| 046 | [InstanceNorm](InstanceNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/88f4c7f) |
| 047 | [Max_Pooling_1D](Max_Pooling_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/debbccb) |
| 048 | [Max_Pooling_2D](Max_Pooling_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/17e3af2) |
| 049 | [Max_Pooling_3D](Max_Pooling_3D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/17e3af2) |
| 050 | [cumprod](cumprod.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/ebb6990) |
| 051 | [cumsum](cumsum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ebe7aa) |
| 052 | [cumsum_exclusive](cumsum_exclusive.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/17e3af2) |
| 053 | [cumsum_reverse](cumsum_reverse.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e0d1b05) |
| 054 | [binned_gather](binned_gather.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7ec83de7) |
| 055 | [binned_scatter](binned_scatter.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6ab6700b) |
| 056 | [gather](gather.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/daab964) |
| 057 | [rotary_pos_emb_fusion](rotary_pos_emb_fusion.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/fd5fd95) |
| 058 | [Softmax](Softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/1b939fa8a48a9dd4c7297bd52a458fa55d32c527) |
| 059 | [hstu_mask_var_len_fwd](hstu_mask_var_len_fwd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d2371de) |
| 060 | [cross_entropy](cross_entropy.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e946d26) |
| 061 | [adaptiveaveragepool](adaptiveaveragepool.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/924a94e) |
| 062 | [compute_agg](compute_agg.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b87b05b) |
| 063 | [calc_block_sums](calc_block_sums.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/bccff9b9) |
| 064 | [adaptive_average_pool_nhwc](adaptive_average_pool_nhwc.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/27f42001734b9139135baca4255257cfc9138fab) |
| 065 | [lpmax_cleanup](lpmax_cleanup.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4044916) |
| 066 | [transform_vals](transform_vals.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f76b13f) |
| 067 | [renormRowsL1](renormRowsL1.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/75c4605728f551c3e7ba864bf6a0b9effeb782da) |
| 068 | [upsample_bilinear2d_out_frame](upsample_bilinear2d_out_frame.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/0861bce) |
| 069 | [lstm_cell_forward](lstm_cell_forward.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/334d9a2) |
| 070 | [Sqrt](Sqrt.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7c05f0d) |
| 071 | [Cos](Cos.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a2099ea) |
| 072 | [ElementwiseAdd](ElementwiseAdd.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/dd922e7) |
| 073 | [Power](Power.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5ac33ea) |
| 074 | [Std_reduction_over_dim](Std_reduction_over_dim.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/28c3998) |
| 075 | [TopK](TopK.mlu) | [@crispllk](https://github.com/crispllk) | 🔴 |
| 076 | [Sort](Sort.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8cc27cd) |
| 077 | [LogSumExp_over_dim](LogSumExp_over_dim.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f7dcce8) |
| 078 | [CumMin](CumMin.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f370bf8) |   
| 079 | [Global_sum](Global_sum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/4ec3b52) |   
| 080 | [All_over_dim](All_over_dim.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5301980) |
| 081 | [Sum_2D_reduction](Sum_2D_reduction.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/3cd7500) |
| 082 | [Scaled_Dot_Product_Attention](Scaled_Dot_Product_Attention.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/734357c) |
| 083 | [Causal_Self_Attention](Causal_Self_Attention.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e5d2b74) |
| 084 | [Cross_Attention](Cross_Attention.mlu) | N/A | 🔴 |
| 085 | [Linear](Linear.mlu) | N/A | 🔴 |
| 086 | [Batch_outer_product](Batch_outer_product.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/924a94e) |
| 087 | [Hadamard_product](Hadamard_product.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/741d372) |
| 088 | [Matrix_trace](Matrix_trace.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e2b6718) |
| 089 | [Diagonal_extraction](Diagonal_extraction.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b231825) |
| 090 | [Einsum_4D_contract](Einsum_4D_contract.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c85a674) |
| 091 | [Matrix_inverse](Matrix_inverse.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/15a2472d) |
| 092 | [SVD_decomposition](SVD_decomposition.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b29f69c9) |
| 093 | [Batch_triangular_solve](Batch_triangular_solve.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a29e8c97) |
| 094 | [LayerNorm](LayerNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/cee95c5) |
| 095 | [Add_RMSNorm](Add_RMSNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/baa0abe) |
| 096 | [LocalResponseNorm](LocalResponseNorm.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f11e2d5) |
| 097 | [PixelShuffle](PixelShuffle.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a082769) |
| 098 | [Attention_with_temperature](Attention_with_temperature.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b241a68) |
| 099 | [DropPath](DropPath.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a244575) |
| 100 | [Adaptive_Max_Pool_2D](Adaptive_Max_Pool_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b08da20) |
| 101 | [Max_Pool_2D_with_indices](Max_Pool_2D_with_indices.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c90b272) |
| 102 | [Adaptive_Average_Pool_3D](Adaptive_Average_Pool_3D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/064d3780) |
| 103 | [MSE_Loss](MSE_Loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/5b13366) |
| 104 | [KL_Divergence_Loss](KL_Divergence_Loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/610a54d) |
| 105 | [Embedding_lookup](Embedding_lookup.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/72c5be13) |
| 106 | [Embedding_bag_mean](Embedding_bag_mean.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b207669) |
| 107 | [Sinusoidal_positional_encoding](Sinusoidal_positional_encoding.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a125bd7) |
| 108 | [ALiBi_attention_bias](ALiBi_attention_bias.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/892e795d) |
| 109 | [Scatter_add](Scatter_add.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/985dae3) |
| 110 | [Gather_rows](Gather_rows.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e56f33e) |
| 111 | [Masked_select](Masked_select.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/daab964) |
| 112 | [Index_put](Index_put.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/6fb35f62b1da1a5e57e26465618ac04c51bf0340) |
| 113 | [Upsample_nearest](Upsample_nearest.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b3730f2) |
| 114 | [Pad_constant](Pad_constant.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c90b272) |
| 115 | [Unfold](Unfold.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e946d26) |
| 116 | [Grid_sample](Grid_sample.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/e946d26) |
| 117 | [Masked_fill](Masked_fill.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/8f95ab2) |
| 118 | [Where_conditional](Where_conditional.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b3730f2) |
| 119 | [One_hot_encoding](One_hot_encoding.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/073e7a1) |
| 120 | [Masked_softmax](Masked_softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/f74030ec) |
| 121 | [Scaled_masked_softmax](Scaled_masked_softmax.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/963e664) |
| 122 | [Prefix_sum_2D](Prefix_sum_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/efc97b0) |
| 123 | [Masked_cumsum](Masked_cumsum.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a73b35f) |
| 124 | [Weight_standardization](Weight_standardization.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/c07abcc) |
| 125 | [Conditional_LayerNorm](Conditional_LayerNorm.mlu) | N/A | 🔴 |
| 126 | [QR_decomposition](QR_decomposition.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/15a23520) |
| 127 | [Cholesky_decomposition](Cholesky_decomposition.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/924a94e) |
| 128 | [Batch_norm_1D](Batch_norm_1D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/b2a842ba223a598a2cb088dec5010230e5e9ed90) |
| 129 | [Attention_score_with_bias](Attention_score_with_bias.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/515eb5b) |
| 130 | [Attention_kv_cache](Attention_kv_cache.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/2551555) |
| 131 | [Relative_position_encoding](Relative_position_encoding.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/cc483f5) |
| 132 | [Sparse_embedding](Sparse_embedding.mlu) | [@ZJtoast](https://github.com/ZJtoast) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/7ab3495) |
| 133 | [Triplet_loss](Triplet_loss.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a082769) |
| 134 | [Depthwise_conv_2D](Depthwise_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/d49d802) |
| 135 | [Dilated_conv_2D](Dilated_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/a7ad868) |
| 136 | [Grouped_conv_2D](Grouped_conv_2D.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/016ebe2) |
| 137 | [LSTM_forward](LSTM_forward.mlu) | N/A | 🔴 |
| 138 | [GRU_forward](GRU_forward.mlu) | N/A | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/238eb0d) |
| 139 | [Sparse_attention_mask](Sparse_attention_mask.mlu) | [@MosRat](https://github.com/MosRat) | [🟢](https://github.com/PRO-2684/AICS-openoperator/commit/58ff14f) |
