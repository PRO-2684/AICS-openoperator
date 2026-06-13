# team25 算子代码证据报告

调查目标：`cotangents/openoperator-start-kit-team25`，本地克隆路径为 `/tmp/openoperator-start-kit-team25`。

源仓库：`/workspace/algorithm/AICS-openoperator`。

调查日期：2026-06-13。

## 结论摘要

Git 历史证据显示，`openoperator-start-kit-team25` 中存在多个 BangC `.mlu` 算子实现，它们先出现在本仓库的 MosRat/team 提交历史中，随后才出现在 team25 仓库中。

最硬的证据是 git blob 完全相同：同一份 `.mlu` 文件内容在两个仓库里具有完全一致的 git blob SHA。这是字节级完全相同，不是“算法思路相似”。完整扫描发现 19 个历史 `.mlu` blob 交集；排除共同初始模板等噪声后，保留 17 条“我方 MosRat 提交先出现，team25 后出现”的精确匹配。

此外，team25 当前 HEAD 的 90 个 `.mlu` 文件中，已有 39 个能匹配到更早的 MosRat 历史版本；其中多项相似度超过 `0.95`，体现为核心 BangC 常量、tiling、索引、`__memcpy`、`__bang_*` 调用和 launch 结构高度一致，只做了函数名、变量名、host 端 cache/warmup 包装等改写。

## 可疑导入窗口

team25 最可疑的导入集中在 2026-06-12 深夜：

| commit | 时间 | 作者 | 提交信息 | `.mlu` 影响 |
|---|---|---|---|---|
| `d58b818` | 2026-06-12 23:35:14 +08:00 | `github-actions[bot]` | `Sample late reference-like package` | 61 个文件，6986 行新增 |
| `43b0180` | 2026-06-12 23:43:22 +08:00 | `github-actions[bot]` | `Submit late improved candidates` | 117 个文件，86220 行新增，29379 行删除 |
| `e648dda` | 2026-06-12 23:52:21 +08:00 | `github-actions[bot]` | `Submit late operators` | 55 个文件，6372 行新增 |
| `324d86b` | 2026-06-12 23:55:24 +08:00 | `github-actions[bot]` | `Submit late operators` | 后续同夜批量提交 |
| `cf6962f` | 2026-06-12 23:57:39 +08:00 | `github-actions[bot]` | `Submit late operators` | 后续同夜批量提交 |
| `5eb882b` | 2026-06-13 00:00:55 +08:00 | `github-actions[bot]` | `Submit late operators` | 当前 team25 HEAD |

## 精确 blob 证据

以下是字节级完全相同的 `.mlu` git blob，并且本仓库 MosRat 版本早于 team25 版本：

| blob 前缀 | 文件 | 本仓库首次出现 | team25 首次出现 |
|---|---|---|---|
| `c989a2d68ed8` | `Product_reduction_over_a_dimension.mlu` | `f724b7e5`, 2026-05-06 01:14:03, MosRat | `bb31285b`, 2026-06-10 20:33:52, Cui Jiangren |
| `363461e73c50` | `Depthwise_conv_2D.mlu` | `0ccdf1e7`, 2026-05-12 00:48:34, MosRat | `1f1be09d`, 2026-05-18 03:05:49, cuijiangren |
| `deff056b9154` | `conv_transposed_2D__asymmetric_input__square_kernel.mlu` | `b71b8978`, 2026-05-14 19:10:34, MosRat | `8b31f82e`, 2026-05-18 02:30:26, cuijiangren |
| `1d4e2d2bf13f` | `Dilated_conv_2D.mlu` | `159d7291`, 2026-05-16 11:12:54, MosRat | `8b31f82e`, 2026-05-18 02:30:26, cuijiangren |
| `630be6840046` | `GRU_forward.mlu` | `15110872`, 2026-05-16 19:50:31, MosRat | `8b31f82e`, 2026-05-18 02:30:26, cuijiangren |
| `97c1bbcde0fe` | `BatchNorm.mlu` | `0dfe8e81`, 2026-05-17 15:12:25, MosRat | `8b31f82e`, 2026-05-18 02:30:26, cuijiangren |
| `5587b01caaf3` | `GELU.mlu` | `68f1b2dd`, 2026-05-18 22:55:29, MosRat | `e66ecc5e`, 2026-05-30 20:13:29, cuijiangren |
| `f5bb69f2126c` | `GELU.mlu` | `2975e940`, 2026-05-20 22:43:50, MosRat | `2a58931c`, 2026-05-22 13:11:28, cuijiangren |
| `56bd2ca90937` | `ELU.mlu` | `d30b3367`, 2026-05-20 22:49:40, MosRat | `2a58931c`, 2026-05-22 13:11:28, cuijiangren |
| `5a537470ba60` | `HardTanh.mlu` | `83a774aa`, 2026-05-20 22:58:01, MosRat | `2a58931c`, 2026-05-22 13:11:28, cuijiangren |
| `074e4ffcea1f` | `HardSigmoid.mlu` | `57dbf365`, 2026-05-20 23:00:23, MosRat | `2a58931c`, 2026-05-22 13:11:28, cuijiangren |
| `13e7a0486197` | `HardTanh.mlu` | `d93fdc7e`, 2026-05-22 18:16:07, MosRat | `182dc7b5`, 2026-05-22 18:39:42, cuijiangren |
| `292854fd5bd9` | `HardSigmoid.mlu` | `7a943cf8`, 2026-05-22 18:16:39, MosRat | `182dc7b5`, 2026-05-22 18:39:42, cuijiangren |
| `0c10aac838c6` | `GELU.mlu` | `ac638fd1`, 2026-05-31 05:51:56, MosRat | `24748a1f`, 2026-05-31 10:46:00, cuijiangren |
| `29c28970cd8e` | `GELU.mlu` | `81195f9f`, 2026-05-31 18:14:00, MosRat | `dd23efff`, 2026-05-31 20:11:33, cuijiangren |
| `5af811921331` | `ELU.mlu` | `ffe26425`, 2026-06-02 02:34:00, MosRat | `f7866174`, 2026-06-02 04:27:36, cuijiangren |
| `c89bd0db5baf` | `Masked_softmax.mlu` | `c1573f279`, 2026-06-08 15:31:57, MosRat | `d58b818`, 2026-06-12 23:35:14, `github-actions[bot]` |

其中 `Masked_softmax.mlu` 可以用最短命令复核：

```bash
git -C /workspace/algorithm/AICS-openoperator rev-parse c1573f279:Masked_softmax.mlu
git -C /tmp/openoperator-start-kit-team25 rev-parse HEAD:Masked_softmax.mlu
```

两者都输出：

```text
c89bd0db5bafcbc0d164048ddd4a817c37f5dc9b
```

## 当前 HEAD 高相似证据

对 team25 当前 HEAD 的 `.mlu` 文件做归一化 token 匹配，要求匹配对象必须是本仓库更早的 MosRat 版本。完整结果在 `head_similarity_matches.tsv`，共 39 行。代表性高分项如下：

| 相似度 | 文件 | team25 blob | 本仓库更早匹配 |
|---:|---|---|---|
| `0.9991` | `Masked_softmax.mlu` | `c89bd0db5baf` | `660fab05`, 2026-06-04 23:32:06, MosRat |
| `0.9904` | `Matmul_with_diagonal_matrices_.mlu` | `fd512bd24ce4` | `518ce474`, 2026-05-18 05:28:21, MosRat |
| `0.9893` | `Matmul_with_large_K_dimension_.mlu` | `51328557d431` | `af7f6074`, 2026-06-04 23:46:08, MosRat |
| `0.9877` | `Matmul_for_upper_triangular_matrices.mlu` | `9edbad90ea10` | `5cd0ca42`, 2026-06-04 23:40:53, MosRat |
| `0.9870` | `3D_tensor_matrix_multiplication.mlu` | `cd1c3ccaca70` | `660fab05`, 2026-06-04 23:32:06, MosRat |
| `0.9826` | `QR_decomposition.mlu` | `6dc1b9d8412d` | `45680c64`, 2026-06-11 03:43:33, MosRat |
| `0.9804` | `Grid_sample.mlu` | `ee04ff632a16` | `5fe8d26e`, 2026-06-08 18:32:28, MosRat |
| `0.9781` | `Matmul_with_irregular_shapes_.mlu` | `036ee2759820` | `af7f6074`, 2026-06-04 23:46:08, MosRat |
| `0.9779` | `conv_depthwise_separable_2D.mlu` | `eb00feccefdf` | `03db8a5d`, 2026-06-05 00:49:03, MosRat |
| `0.9765` | `Matmul_for_symmetric_matrices.mlu` | `19fb916f03aa` | `75976716`, 2026-06-05 01:51:42, MosRat |
| `0.9663` | `Prefix_sum_2D.mlu` | `f8349f370f32` | `5cd0ca42`, 2026-06-04 23:40:53, MosRat |
| `0.9566` | `conv_standard_2D__square_input__square_kernel.mlu` | `6c4d46878911` | `660fab05`, 2026-06-04 23:32:06, MosRat |

观察到的改写模式：team25 文件经常保留核心 BangC kernel 常量、tiling、索引计算、`__memcpy`、`__bang_*` 调用和 launch 结构，同时修改函数名、变量名，或添加 host 侧 device pinning、static tensor cache、warmup 代码。这类改写在 `Grid_sample.mlu`、`Matmul_with_irregular_shapes_.mlu`、`Matmul_with_large_K_dimension_.mlu` 等 diff 附件中可以直接看到。

## 附件说明

| 文件 | 作用 |
|---|---|
| `exact_blob_matches.tsv` | 完整精确 blob 交集表，共 19 行，并标记 17 条可疑精确匹配。 |
| `head_similarity_matches.tsv` | team25 当前 HEAD 与本仓库更早 MosRat 历史版本的相似度匹配，共 39 行。 |
| `raw/team25_suspicious_commit_stats.txt` | team25 可疑批量导入提交的原始 `git show --stat` 输出。 |
| `raw/masked_softmax_blob_proof.txt` | `Masked_softmax.mlu` 的最小 blob 复核证据。 |
| `raw/team25_mlu_log.tsv` | team25 `.mlu` 历史日志。 |
| `raw/ours_mlu_log.tsv` | 本仓库 `.mlu` 历史日志。 |
| `diffs/Grid_sample_ours_5fe8d26e7_vs_team25_HEAD.diff` | 高相似改写 diff 样例。 |
| `diffs/Matmul_with_irregular_shapes_ours_af7f6074b_vs_team25_HEAD.diff` | 高相似改写 diff 样例。 |
| `diffs/Matmul_with_large_K_dimension_ours_af7f6074b_vs_team25_HEAD.diff` | 高相似改写 diff 样例。 |
| `diffs/Masked_softmax_ours_c1573f279_vs_team25_HEAD.diff` | `Masked_softmax.mlu` 精确/近精确对比。 |

## 复现命令

```bash
git -C /tmp/openoperator-start-kit-team25 show --stat d58b818 43b0180 e648dda -- '*.mlu'

git -C /workspace/algorithm/AICS-openoperator rev-parse c1573f279:Masked_softmax.mlu
git -C /tmp/openoperator-start-kit-team25 rev-parse HEAD:Masked_softmax.mlu

diff -u --ignore-all-space \
  <(git -C /workspace/algorithm/AICS-openoperator show c1573f279:Masked_softmax.mlu) \
  <(git -C /tmp/openoperator-start-kit-team25 show HEAD:Masked_softmax.mlu)
```
