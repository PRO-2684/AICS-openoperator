# team25 Operator Code Evidence

Investigation target: `cotangents/openoperator-start-kit-team25`, cloned at `/tmp/openoperator-start-kit-team25`.

Source repository: `/workspace/algorithm/AICS-openoperator`.

Date: 2026-06-13.

## Interim conclusion

There is strong git-history evidence that `openoperator-start-kit-team25` contains BangC `.mlu` operator implementations that first appeared in this repository under MosRat/team work and later appeared in the team25 repository.

The strongest evidence is exact git blob reuse: the same `.mlu` file content has the same full git blob SHA in both repositories. Several such blobs first appeared here days or weeks before they appeared in team25.

## Suspicious team25 import window

The most suspicious team25 commits are batch imports made by `github-actions[bot]` shortly before midnight on 2026-06-12:

| commit | time | author | subject | `.mlu` impact |
|---|---|---|---|---|
| `d58b818` | 2026-06-12 23:35:14 +08:00 | `github-actions[bot]` | `Sample late reference-like package` | 61 files, 6986 insertions |
| `43b0180` | 2026-06-12 23:43:22 +08:00 | `github-actions[bot]` | `Submit late improved candidates` | 117 files, 86220 insertions, 29379 deletions |
| `e648dda` | 2026-06-12 23:52:21 +08:00 | `github-actions[bot]` | `Submit late operators` | 55 files, 6372 insertions |

Later same-night team25 commits also need deeper analysis:

| commit | time | author | subject |
|---|---|---|---|
| `324d86b` | 2026-06-12 23:55:24 +08:00 | `github-actions[bot]` | `Submit late operators` |
| `cf6962f` | 2026-06-12 23:57:39 +08:00 | `github-actions[bot]` | `Submit late operators` |
| `5eb882b` | 2026-06-13 00:00:55 +08:00 | `github-actions[bot]` | `Submit late operators` |

## Exact blob matches already confirmed

These are byte-for-byte identical `.mlu` git blobs where this repository's MosRat/team version appears before team25's version.

| blob prefix | file | first seen here | first seen in team25 |
|---|---|---|---|
| `c89bd0db5baf` | `Masked_softmax.mlu` | `c1573f279`, 2026-06-08 15:31:57 +08:00, MosRat, `120 Masked_softmax reduce launch tasks` | `d58b818`, 2026-06-12 23:35:14 +08:00, `github-actions[bot]` |
| `97c1bbcde0fe` | `BatchNorm.mlu` | `0dfe8e81`, 2026-05-17 15:12:25 +08:00, MosRat, `039 BatchNorm halfsum k1 32x8` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |
| `363461e73c50` | `Depthwise_conv_2D.mlu` | `0ccdf1e7`, 2026-05-12 00:48:34 +08:00, MosRat, `134 136 conv: remove bias init patch probe` | `1f1be09d`, 2026-05-18 03:05:49 +08:00, cuijiangren |
| `1d4e2d2bf13f` | `Dilated_conv_2D.mlu` | `159d7291`, 2026-05-16 11:12:54 +08:00, MosRat, `Remove extra stream experiments` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |
| `630be6840046` | `GRU_forward.mlu` | `15110872`, 2026-05-16 19:50:31 +08:00, MosRat, `138 GRU_forward: fix batch task mapping` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |
| `deff056b9154` | `conv_transposed_2D__asymmetric_input__square_kernel.mlu` | `b71b8978`, 2026-05-14 19:10:34 +08:00, MosRat, `012 ConvTranspose2d: pairwise split transpose` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |

The complete exact-blob pass found 19 shared historical `.mlu` blobs. After filtering out shared initial/template noise, 17 exact blobs remain where a MosRat-authored implementation appears in this repository first and later appears in team25.

| blob prefix | file | first seen here | first seen in team25 |
|---|---|---|---|
| `c989a2d68ed8` | `Product_reduction_over_a_dimension.mlu` | `f724b7e5`, 2026-05-06 01:14:03 +08:00, MosRat, `038 077 079 081 Reduce: test compact sums` | `bb31285b`, 2026-06-10 20:33:52 +08:00, Cui Jiangren, `Sample stable reductions with product zero kernel` |
| `363461e73c50` | `Depthwise_conv_2D.mlu` | `0ccdf1e7`, 2026-05-12 00:48:34 +08:00, MosRat, `134 136 conv: remove bias init patch probe` | `1f1be09d`, 2026-05-18 03:05:49 +08:00, cuijiangren, `feat 007` |
| `deff056b9154` | `conv_transposed_2D__asymmetric_input__square_kernel.mlu` | `b71b8978`, 2026-05-14 19:10:34 +08:00, MosRat, `012 ConvTranspose2d: pairwise split transpose` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `1d4e2d2bf13f` | `Dilated_conv_2D.mlu` | `159d7291`, 2026-05-16 11:12:54 +08:00, MosRat, `Remove extra stream experiments` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `630be6840046` | `GRU_forward.mlu` | `15110872`, 2026-05-16 19:50:31 +08:00, MosRat, `138 GRU_forward: fix batch task mapping` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `97c1bbcde0fe` | `BatchNorm.mlu` | `0dfe8e81`, 2026-05-17 15:12:25 +08:00, MosRat, `039 BatchNorm halfsum k1 32x8` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `5587b01caaf3` | `GELU.mlu` | `68f1b2dd`, 2026-05-18 22:55:29 +08:00, MosRat, `easy batch warm launch` | `e66ecc5e`, 2026-05-30 20:13:29 +08:00, cuijiangren, `test` |
| `f5bb69f2126c` | `GELU.mlu` | `2975e940`, 2026-05-20 22:43:50 +08:00, MosRat, `027 GELU block launch` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `56bd2ca90937` | `ELU.mlu` | `d30b3367`, 2026-05-20 22:49:40 +08:00, MosRat, `026 ELU arithmetic split` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `5a537470ba60` | `HardTanh.mlu` | `83a774aa`, 2026-05-20 22:58:01 +08:00, MosRat, `029 HardTanh abs clamp` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `074e4ffcea1f` | `HardSigmoid.mlu` | `57dbf365`, 2026-05-20 23:00:23 +08:00, MosRat, `028 HardSigmoid single tile abs clamp` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `13e7a0486197` | `HardTanh.mlu` | `d93fdc7e`, 2026-05-22 18:16:07 +08:00, MosRat, `028 029 scalar clamp eq intrinsics 32k` | `182dc7b5`, 2026-05-22 18:39:42 +08:00, cuijiangren, `feat 008` |
| `292854fd5bd9` | `HardSigmoid.mlu` | `7a943cf8`, 2026-05-22 18:16:39 +08:00, MosRat, `028 029 scalar clamp eq intrinsics 8k` | `182dc7b5`, 2026-05-22 18:39:42 +08:00, cuijiangren, `feat 008` |
| `0c10aac838c6` | `GELU.mlu` | `ac638fd1`, 2026-05-31 05:51:56 +08:00, MosRat, `027 GELU direct launch no guard` | `24748a1f`, 2026-05-31 10:46:00 +08:00, cuijiangren, `test` |
| `29c28970cd8e` | `GELU.mlu` | `81195f9f`, 2026-05-31 18:14:00 +08:00, MosRat, `026 027 028 029 clean launch-bound rerun 1` | `dd23efff`, 2026-05-31 20:11:33 +08:00, cuijiangren, `test` |
| `5af811921331` | `ELU.mlu` | `ffe26425`, 2026-06-02 02:34:00 +08:00, MosRat, `026 026-029 joint config worker warmness probe 1` | `f7866174`, 2026-06-02 04:27:36 +08:00, cuijiangren, `test` |
| `c89bd0db5baf` | `Masked_softmax.mlu` | `c1573f279`, 2026-06-08 15:31:57 +08:00, MosRat, `120 Masked_softmax reduce launch tasks` | `d58b818`, 2026-06-12 23:35:14 +08:00, `github-actions[bot]`, `Sample late reference-like package` |

## High-similarity rewritten matches already confirmed

For current team25 files, normalized-token matching against this repository's full history found these best matches. Normalization removes comments and whitespace before token comparison, so these are structural/code matches rather than formatting matches.

| team25 file | best prior file/version here | normalized token similarity |
|---|---|---:|
| `Masked_softmax.mlu` | `c1573f279:Masked_softmax.mlu` | `1.0000` |
| `Matmul_with_large_K_dimension_.mlu` | `8b4c626c2:Matmul_with_large_K_dimension_.mlu` | `0.9893` |
| `Matmul_for_upper_triangular_matrices.mlu` | `84680dbbc:Matmul_for_upper_triangular_matrices.mlu` | `0.9877` |
| `Grid_sample.mlu` | `110cc66b7:Grid_sample.mlu` | `0.9828` |
| `Matmul_with_irregular_shapes_.mlu` | `66afbfd86:Matmul_with_irregular_shapes_.mlu` | `0.9817` |
| `conv_depthwise_separable_2D.mlu` | `03db8a5d6:conv_depthwise_separable_2D.mlu` | `0.9816` |
| `Scaled_Dot_Product_Attention.mlu` | `149f7301:Scaled_Dot_Product_Attention.mlu` | `0.9525` |

Observed rewrite pattern: team25 often keeps the BangC kernel constants, tiling, index math, `__memcpy`, `__bang_*`, and launch structure while renaming functions and variables and adding host-side device pinning/static tensor caches. Examples include `Grid_sample.mlu` and `Matmul_with_irregular_shapes_.mlu`.

## Current team25 HEAD vs earlier MosRat history

The current team25 HEAD has 90 `.mlu` files. The full same-name normalized-token scan found 39 current files with high similarity to this repository; the complete table is preserved in `head_similarity_matches.tsv`. For the representative high-confidence rows below, the best historical match is constrained to a MosRat-authored version that predates the team25 file version.

All rows listed below trace to team25 commit `d58b818` at 2026-06-12 23:35:14 +08:00, `github-actions[bot]`, `Sample late reference-like package`.

| similarity | file | team25 blob | earlier MosRat match here |
|---:|---|---|---|
| `0.9991` | `Masked_softmax.mlu` | `c89bd0db5baf` | `660fab05`, 2026-06-04 23:32:06 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:32:06`, blob `991e94fbeb42` |
| `0.9904` | `Matmul_with_diagonal_matrices_.mlu` | `fd512bd24ce4` | `518ce474`, 2026-05-18 05:28:21 +08:00, `013 014 015 016 017 018 cached first pass`, blob `3a0e0ea9debc` |
| `0.9893` | `Matmul_with_large_K_dimension_.mlu` | `51328557d431` | `af7f6074`, 2026-06-04 23:46:08 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:46:08`, blob `ab74d15ab565` |
| `0.9877` | `Matmul_for_upper_triangular_matrices.mlu` | `9edbad90ea10` | `5cd0ca42`, 2026-06-04 23:40:53 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:40:53`, blob `f1901b51444a` |
| `0.9870` | `3D_tensor_matrix_multiplication.mlu` | `cd1c3ccaca70` | `660fab05`, 2026-06-04 23:32:06 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:32:06`, blob `5084ec2b41df` |
| `0.9826` | `QR_decomposition.mlu` | `6dc1b9d8412d` | `45680c64`, 2026-06-11 03:43:33 +08:00, `auto commit push 2026-06-11 03:43:33`, blob `a988a5da822d` |
| `0.9804` | `Grid_sample.mlu` | `ee04ff632a16` | `5fe8d26e`, 2026-06-08 18:32:28 +08:00, `002 105 106 116 132 138 local kernel optimizations`, blob `355b92d605d8` |
| `0.9781` | `Matmul_with_irregular_shapes_.mlu` | `036ee2759820` | `af7f6074`, 2026-06-04 23:46:08 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:46:08`, blob `f4e556ba374e` |
| `0.9779` | `conv_depthwise_separable_2D.mlu` | `eb00feccefdf` | `03db8a5d`, 2026-06-05 00:49:03 +08:00, `026-29:hot queue detect sequence 2026-06-05 00:49:03`, blob `3d70542a8758` |
| `0.9765` | `Matmul_for_symmetric_matrices.mlu` | `19fb916f03aa` | `75976716`, 2026-06-05 01:51:42 +08:00, `026-29:hot queue detect sequence 2026-06-05 01:51:42`, blob `4b33fae5d7bd` |
| `0.9753` | `Matmul_with_small_K_dimension_.mlu` | `4f9eea0ad94f` | `5cd0ca42`, 2026-06-04 23:40:53 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:40:53`, blob `23a2f9330fe8` |
| `0.9663` | `Prefix_sum_2D.mlu` | `f8349f370f32` | `5cd0ca42`, 2026-06-04 23:40:53 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:40:53`, blob `e33e13855988` |
| `0.9657` | `conv_depthwise_2D_square_input_square_kernel.mlu` | `4d86826324f1` | `497b39f4`, 2026-06-05 00:06:05 +08:00, `026-29:hot queue detect sequence 2026-06-05 00:06:05`, blob `28877450c2de` |
| `0.9566` | `conv_standard_2D__square_input__square_kernel.mlu` | `6c4d46878911` | `660fab05`, 2026-06-04 23:32:06 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:32:06`, blob `9179eafe0133` |
| `0.9330` | `Scaled_Dot_Product_Attention.mlu` | `7c1faf4b84c5` | `0970792a`, 2026-05-23 04:37:52 +08:00, `082 129 attention cycle norm`, blob `e555fb890815` |
| `0.9301` | `Causal_Self_Attention.mlu` | `3d9ad835d332` | `e7d3dfb4`, 2026-05-23 04:37:14 +08:00, `083 Causal attention two block scalar`, blob `d80fe3703674` |
| `0.9252` | `Attention_score_with_bias.mlu` | `5fc5296952cd` | `660fab05`, 2026-06-04 23:32:06 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:32:06`, blob `bb153f825fd1` |
| `0.9036` | `Max_Pool_2D_with_indices.mlu` | `0c50665a1d8b` | `4bc237eb`, 2026-06-09 15:51:42 +08:00, `026-29:hot queue detect sequence 2026-06-09 15:51:42`, blob `4b12134c284c` |
| `0.8777` | `Matmul_for_lower_triangular_matrices.mlu` | `4d33cadd0638` | `291cb2a`, 2026-06-05 04:12:23 +08:00, `026-29:hot queue detect sequence 2026-06-05 04:12:23`, blob `7e7881a497dc` |
| `0.8677` | `PixelShuffle.mlu` | `f5451510b8ce` | `660fab05`, 2026-06-04 23:32:06 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:32:06`, blob `85f45096bb38` |
| `0.8123` | `Attention_with_temperature.mlu` | `9a52e672d081` | `49f3b629`, 2026-05-23 04:34:06 +08:00, `098 Attention temperature cycle norm`, blob `f522d0d80fa7` |
| `0.8093` | `InstanceNorm.mlu` | `402696c33da9` | `660fab05`, 2026-06-04 23:32:06 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:32:06`, blob `2d35c0065e93` |
| `0.8000` | `Where_conditional.mlu` | `5db831a12a8b` | `575039d3`, 2026-05-16 04:46:32 +08:00, `118 Where conditional: fusion select probe`, blob `ef877dc41f15` |
| `0.7322` | `Masked_cumsum.mlu` | `ee434795f119` | `660fab05`, 2026-06-04 23:32:06 +08:00, `026-29:hot queue detect sequence 2026-06-04 23:32:06`, blob `e50c4c0e4d1f` |

## Reproduction commands

```bash
git -C /tmp/openoperator-start-kit-team25 show --stat d58b818 43b0180 e648dda -- '*.mlu'

git -C /workspace/algorithm/AICS-openoperator rev-parse c1573f279:Masked_softmax.mlu
git -C /tmp/openoperator-start-kit-team25 rev-parse HEAD:Masked_softmax.mlu

diff -u --ignore-all-space \
  <(git -C /workspace/algorithm/AICS-openoperator show c1573f279:Masked_softmax.mlu) \
  <(git -C /tmp/openoperator-start-kit-team25 show HEAD:Masked_softmax.mlu)
```

Expected result for the two `rev-parse` commands above:

```text
c89bd0db5bafcbc0d164048ddd4a817c37f5dc9b
c89bd0db5bafcbc0d164048ddd4a817c37f5dc9b
```

## Attached evidence files

The report directory also contains machine-readable and raw evidence files:

| file | purpose |
|---|---|
| `exact_blob_matches.tsv` | Full exact git-blob intersection table. It has 19 rows total and marks the 17 suspicious MosRat-before-team25 exact matches. |
| `head_similarity_matches.tsv` | Current team25 HEAD `.mlu` files matched to earlier MosRat history. It has 39 rows with similarity, blob, commit, author, and timestamp fields. |
| `raw/team25_suspicious_commit_stats.txt` | Raw `git show --stat` output for suspicious batch-import commits. |
| `raw/masked_softmax_blob_proof.txt` | Minimal exact-blob proof for `Masked_softmax.mlu`. |
| `raw/team25_mlu_log.tsv` | Raw team25 `.mlu` git log. |
| `raw/ours_mlu_log.tsv` | Raw source-repository `.mlu` git log. |
| `diffs/Grid_sample_ours_5fe8d26e7_vs_team25_HEAD.diff` | Representative high-similarity rewrite diff. |
| `diffs/Matmul_with_irregular_shapes_ours_af7f6074b_vs_team25_HEAD.diff` | Representative high-similarity rewrite diff. |
| `diffs/Matmul_with_large_K_dimension_ours_af7f6074b_vs_team25_HEAD.diff` | Representative high-similarity rewrite diff. |
| `diffs/Masked_softmax_ours_c1573f279_vs_team25_HEAD.diff` | Exact or near-exact comparison for `Masked_softmax.mlu`. |

These files preserve the audit trail separately from this narrative report.
