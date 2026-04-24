# AICS-OpenOperator

## 🔗 Related Links

- [OpenOperator](https://openoperator.cn) | [Monitor](http://152.136.18.42:13000)
- [BangC Tutorial](https://gitservice.cstcloud.cn/kcxain/BangcTutorial/)
- [南京智能计算中心算力平台](https://paas.extrotec.com:30443/)
- [Start Kit](https://github.com/kernel-competition-bot/openoperator-start-kit)

## 🚀 Quick Start

### Setup

1. Clone this repo
2. Navigate to the root directory of the repo and run `setup.sh` to:
    - Set up git hooks: This will automatically update `config` file to the list of changed files in each commit

### Testing

```shell
# Not working yet
./check.sh LeakyReLU.mlu
```

### Submission

```shell
# `config` file will be automatically updated to include indices for changed files when you commit
git add . && git commit -m "feat: Add LeakyReLU implementation"
```

## ❓ Problems

- [001_LeakyReLU](LeakyReLU.mlu)
- [002_matrix_scalar_multiplication](matrix_scalar_multiplication.mlu)
- [003_LogSoftmax](LogSoftmax.mlu)
- [004_batched_matrix_multiplication](batched_matrix_multiplication.mlu)
- [005_average_pooling_2d](average_pooling_2d.mlu)
- [006_conv_depthwise_2D_square_input_square_kernel](conv_depthwise_2D_square_input_square_kernel.mlu)
- [007_conv_depthwise_separable_2D](conv_depthwise_separable_2D.mlu)
- [008_conv_pointwise_2D](conv_pointwise_2D.mlu)
- [009_conv_standard_1D](conv_standard_1D.mlu)
- [010_conv_standard_2D__square_input__square_kernel](conv_standard_2D__square_input__square_kernel.mlu)
- [011_conv_transposed_1D](conv_transposed_1D.mlu)
- [012_conv_transposed_2D__asymmetric_input__square_kernel](conv_transposed_2D__asymmetric_input__square_kernel.mlu)
- [013_3D_tensor_matrix_multiplication](3D_tensor_matrix_multiplication.mlu)
- [014_4D_tensor_matrix_multiplication](4D_tensor_matrix_multiplication.mlu)
- [015_Matmul_for_lower_triangular_matrices](Matmul_for_lower_triangular_matrices.mlu)
- [016_Matmul_for_symmetric_matrices](Matmul_for_symmetric_matrices.mlu)
- [017_Matmul_for_upper_triangular_matrices](Matmul_for_upper_triangular_matrices.mlu)
- [018_Matmul_with_diagonal_matrices_](Matmul_with_diagonal_matrices_.mlu)
- [019_Matmul_with_irregular_shapes_](Matmul_with_irregular_shapes_.mlu)
- [020_Matmul_with_large_K_dimension_](Matmul_with_large_K_dimension_.mlu)
- [021_Matmul_with_small_K_dimension_](Matmul_with_small_K_dimension_.mlu)
- [022_Matmul_with_transposed_both](Matmul_with_transposed_both.mlu)
- [023_Matrix_vector_multiplication_](Matrix_vector_multiplication_.mlu)
- [024_Square_matrix_multiplication_](Square_matrix_multiplication_.mlu)
- [025_fused_matmul_fwd](fused_matmul_fwd.mlu)
- [026_ELU](ELU.mlu)
- [027_GELU](GELU.mlu)
- [028_HardSigmoid](HardSigmoid.mlu)
- [029_HardTanh](HardTanh.mlu)
- [030_MinGPTNewGelu](MinGPTNewGelu.mlu)
- [031_Softplus](Softplus.mlu)
- [032_Softsign](Softsign.mlu)
- [033_Swish](Swish.mlu)
- [034_Argmax_over_a_dimension](Argmax_over_a_dimension.mlu)
- [035_L1Norm_](L1Norm_.mlu)
- [036_L2Norm_](L2Norm_.mlu)
- [037_Max_reduction_over_a_dimension](Max_reduction_over_a_dimension.mlu)
- [038_Product_reduction_over_a_dimension](Product_reduction_over_a_dimension.mlu)
- [039_BatchNorm](BatchNorm.mlu)
- [040_CosineSimilarityLoss](CosineSimilarityLoss.mlu)
- [041_CrossEntropyLoss](CrossEntropyLoss.mlu)
- [042_FrobeniusNorm_](FrobeniusNorm_.mlu)
- [043_GroupNorm_](GroupNorm_.mlu)
- [044_HingeLoss](HingeLoss.mlu)
- [045_HuberLoss](HuberLoss.mlu)
- [046_InstanceNorm](InstanceNorm.mlu)
- [047_Max_Pooling_1D](Max_Pooling_1D.mlu)
- [048_Max_Pooling_2D](Max_Pooling_2D.mlu)
- [049_Max_Pooling_3D](Max_Pooling_3D.mlu)
- [050_cumprod](cumprod.mlu)
- [051_cumsum](cumsum.mlu)
- [052_cumsum_exclusive](cumsum_exclusive.mlu)
- [053_cumsum_reverse](cumsum_reverse.mlu)
- [054_binned_gather](binned_gather.mlu)
- [055_binned_scatter](binned_scatter.mlu)
- [056_gather](gather.mlu)
- [057_rotary_pos_emb_fusion](rotary_pos_emb_fusion.mlu)
- [058_Softmax](Softmax.mlu)
- [059_hstu_mask_var_len_fwd](hstu_mask_var_len_fwd.mlu)
- [060_cross_entropy](cross_entropy.mlu)
- [061_adaptiveaveragepool](adaptiveaveragepool.mlu)
- [062_compute_agg](compute_agg.mlu)
- [063_calc_block_sums](calc_block_sums.mlu)
- [064_adaptive_average_pool_nhwc](adaptive_average_pool_nhwc.mlu)
- [065_lpmax_cleanup](lpmax_cleanup.mlu)
- [066_transform_vals](transform_vals.mlu)
- [067_renormRowsL1](renormRowsL1.mlu)
- [068_upsample_bilinear2d_out_frame](upsample_bilinear2d_out_frame.mlu)
- [069_lstm_cell_forward](lstm_cell_forward.mlu)
- [070_Sqrt](Sqrt.mlu)
- [071_Cos](Cos.mlu)
- [072_ElementwiseAdd](ElementwiseAdd.mlu)
- [073_Power](Power.mlu)
- [074_Std_reduction_over_dim](Std_reduction_over_dim.mlu)
- [075_TopK](TopK.mlu)
- [076_Sort](Sort.mlu)
- [077_LogSumExp_over_dim](LogSumExp_over_dim.mlu)
- [078_CumMin](CumMin.mlu)
- [079_Global_sum](Global_sum.mlu)
- [080_All_over_dim](All_over_dim.mlu)
- [081_Sum_2D_reduction](Sum_2D_reduction.mlu)
- [082_Scaled_Dot_Product_Attention](Scaled_Dot_Product_Attention.mlu)
- [083_Causal_Self_Attention](Causal_Self_Attention.mlu)
- [084_Cross_Attention](Cross_Attention.mlu)
- [085_Linear](Linear.mlu)
- [086_Batch_outer_product](Batch_outer_product.mlu)
- [087_Hadamard_product](Hadamard_product.mlu)
- [088_Matrix_trace](Matrix_trace.mlu)
- [089_Diagonal_extraction](Diagonal_extraction.mlu)
- [090_Einsum_4D_contract](Einsum_4D_contract.mlu)
- [091_Matrix_inverse](Matrix_inverse.mlu)
- [092_SVD_decomposition](SVD_decomposition.mlu)
- [093_Batch_triangular_solve](Batch_triangular_solve.mlu)
- [094_LayerNorm](LayerNorm.mlu)
- [095_Add_RMSNorm](Add_RMSNorm.mlu)
- [096_LocalResponseNorm](LocalResponseNorm.mlu)
- [097_PixelShuffle](PixelShuffle.mlu)
- [098_Attention_with_temperature](Attention_with_temperature.mlu)
- [099_DropPath](DropPath.mlu)
- [100_Adaptive_Max_Pool_2D](Adaptive_Max_Pool_2D.mlu)
- [101_Max_Pool_2D_with_indices](Max_Pool_2D_with_indices.mlu)
- [102_Adaptive_Average_Pool_3D](Adaptive_Average_Pool_3D.mlu)
- [103_MSE_Loss](MSE_Loss.mlu)
- [104_KL_Divergence_Loss](KL_Divergence_Loss.mlu)
- [105_Embedding_lookup](Embedding_lookup.mlu)
- [106_Embedding_bag_mean](Embedding_bag_mean.mlu)
- [107_Sinusoidal_positional_encoding](Sinusoidal_positional_encoding.mlu)
- [108_ALiBi_attention_bias](ALiBi_attention_bias.mlu)
- [109_Scatter_add](Scatter_add.mlu)
- [110_Gather_rows](Gather_rows.mlu)
- [111_Masked_select](Masked_select.mlu)
- [112_Index_put](Index_put.mlu)
- [113_Upsample_nearest](Upsample_nearest.mlu)
- [114_Pad_constant](Pad_constant.mlu)
- [115_Unfold](Unfold.mlu)
- [116_Grid_sample](Grid_sample.mlu)
- [117_Masked_fill](Masked_fill.mlu)
- [118_Where_conditional](Where_conditional.mlu)
- [119_One_hot_encoding](One_hot_encoding.mlu)
- [120_Masked_softmax](Masked_softmax.mlu)
- [121_Scaled_masked_softmax](Scaled_masked_softmax.mlu)
- [122_Prefix_sum_2D](Prefix_sum_2D.mlu)
- [123_Masked_cumsum](Masked_cumsum.mlu)
- [124_Weight_standardization](Weight_standardization.mlu)
- [125_Conditional_LayerNorm](Conditional_LayerNorm.mlu)
- [126_QR_decomposition](QR_decomposition.mlu)
- [127_Cholesky_decomposition](Cholesky_decomposition.mlu)
- [128_Batch_norm_1D](Batch_norm_1D.mlu)
- [129_Attention_score_with_bias](Attention_score_with_bias.mlu)
- [130_Attention_kv_cache](Attention_kv_cache.mlu)
- [131_Relative_position_encoding](Relative_position_encoding.mlu)
- [132_Sparse_embedding](Sparse_embedding.mlu)
- [133_Triplet_loss](Triplet_loss.mlu)
- [134_Depthwise_conv_2D](Depthwise_conv_2D.mlu)
- [135_Dilated_conv_2D](Dilated_conv_2D.mlu)
- [136_Grouped_conv_2D](Grouped_conv_2D.mlu)
- [137_LSTM_forward](LSTM_forward.mlu)
- [138_GRU_forward](GRU_forward.mlu)
- [139_Sparse_attention_mask](Sparse_attention_mask.mlu)

## 提交说明

提交时仓库根目录需要包含`config`文件和题目的`mlu`代码文件。文件组织结构如下

```bash
.
├── config			# 配置文件，用于指定要评估的题目
├── LeakyReLU.mlu	# bangc代码文件，必须包含kernel函数定义和用于外部程序调用的函数定义
├── ...				# 其他题目的bangc代码文件
└── README.md		# 可选的代码说明
```

> [!NOTE]
>
> 通过`config`文件可以指定本次提交想要评估的题目范围
> config文件的每行代表一个题目，应按照题目序号的三位数字给出
> 例如，LeakyReLU的序号是001，为了评估LeakyReLU题目，config中必须包含一行`001`

> [!TIP]
>
> 每道题目的评估耗时预计不少于30s，评估系统评估完所有题目后才会返回结果，请合理安排评估请求，尽量不要一次性评估太多题目。

> [!CAUTION]
>
> 如果提交中不包含config文件，则会默认评估所有题目！

### 代码要求

1. 代码文件必须以题目名称命名，这是评估脚本能找到你代码的关键要求。
2. 代码中要覆盖头文件引用，核函数定义和用于外部调用的函数定义。
3. 用于外部调用的函数名必须设置为bang_func，bang_func的返回值为`torch::Tensor`，输入参数包含`torch::Tensor input`和参考代码中`__init__`部分定义的其他参数，请参考LeakyReLU示例进行理解。

## 题目&打分

题目按照类别分为`basic`，`easy`，`medium`，`hard`。其中`basic`是必做题，其他类为挑战题。

打分有两个指标：

- 算子结果必须与参考结果误差不大于1e-2，精度达标后性能评估结果才有效
- 性能分数按照`bangc`代码硬件时间相对于`torch`的执行时间赋值

## 最佳实践

1. 每次只评估少量题目
2. 尽量在调试服务器debug，远程评估时通过阅读commit评论中的报错进行debug
3. 系统只接收`main`分支的提交，所以请分时开发或者做好分支管理
4. github评论是执行结束第一时间更新的，排行榜是周期性更新的，且只会记录团队历史最好成绩
