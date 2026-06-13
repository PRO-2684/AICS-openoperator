**翻译：**

```markdown
# team25 Operator 代码证据

调查目标：`cotangents/openoperator-start-kit-team25`，克隆路径为 `/tmp/openoperator-start-kit-team25`。

源仓库：`/workspace/algorithm/AICS-openoperator`。

日期：2026-06-13。

## 初步结论

有强有力的 Git 历史证据表明，`openoperator-start-kit-team25` 包含的 BangC `.mlu` 算子实现，最初出现在本仓库的 MosRat/team 分支下，随后出现在 team25 仓库中。

最有力的证据是**精确的 Git blob 复用**：相同的 `.mlu` 文件内容在两个仓库中具有完全相同的完整 Git blob SHA。其中多个 blob 在本仓库中早几天或几周前就已出现，随后才出现在 team25 中。

## 可疑的 team25 导入时间窗口

最可疑的 team25 提交是由 `github-actions[bot]` 在 2026-06-12 午夜前批量导入的：

| commit    | 时间                        | 作者                  | 提交信息                        | `.mlu` 影响                  |
|-----------|-----------------------------|-----------------------|---------------------------------|------------------------------|
| `d58b818` | 2026-06-12 23:35:14 +08:00 | `github-actions[bot]` | `Sample late reference-like package` | 61 个文件，6986 行插入     |
| `43b0180` | 2026-06-12 23:43:22 +08:00 | `github-actions[bot]` | `Submit late improved candidates`    | 117 个文件，86220 行插入，29379 行删除 |
| `e648dda` | 2026-06-12 23:52:21 +08:00 | `github-actions[bot]` | `Submit late operators`              | 55 个文件，6372 行插入     |

当晚后续的 team25 提交也需要更深入的分析：

| commit    | 时间                        | 作者                  | 提交信息                     |
|-----------|-----------------------------|-----------------------|------------------------------|
| `324d86b` | 2026-06-12 23:55:24 +08:00 | `github-actions[bot]` | `Submit late operators`     |
| `cf6962f` | 2026-06-12 23:57:39 +08:00 | `github-actions[bot]` | `Submit late operators`     |
| `5eb882b` | 2026-06-13 00:00:55 +08:00 | `github-actions[bot]` | `Submit late operators`     |

## 已确认的精确 blob 匹配

以下是字节完全相同的 `.mlu` Git blob，本仓库 MosRat/team 版本的出现时间早于 team25 版本。

| blob 前缀       | 文件                                      | 本仓库首次出现                                      | team25 首次出现                                      |
|-----------------|-------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| `c89bd0db5baf` | `Masked_softmax.mlu`                     | `c1573f279`, 2026-06-08 15:31:57 +08:00, MosRat, `120 Masked_softmax reduce launch tasks` | `d58b818`, 2026-06-12 23:35:14 +08:00, `github-actions[bot]` |
| `97c1bbcde0fe` | `BatchNorm.mlu`                          | `0dfe8e81`, 2026-05-17 15:12:25 +08:00, MosRat, `039 BatchNorm halfsum k1 32x8` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |
| `363461e73c50` | `Depthwise_conv_2D.mlu`                  | `0ccdf1e7`, 2026-05-12 00:48:34 +08:00, MosRat, `134 136 conv: remove bias init patch probe` | `1f1be09d`, 2026-05-18 03:05:49 +08:00, cuijiangren |
| `1d4e2d2bf13f` | `Dilated_conv_2D.mlu`                    | `159d7291`, 2026-05-16 11:12:54 +08:00, MosRat, `Remove extra stream experiments` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |
| `630be6840046` | `GRU_forward.mlu`                        | `15110872`, 2026-05-16 19:50:31 +08:00, MosRat, `138 GRU_forward: fix batch task mapping` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |
| `deff056b9154` | `conv_transposed_2D__asymmetric_input__square_kernel.mlu` | `b71b8978`, 2026-05-14 19:10:34 +08:00, MosRat, `012 ConvTranspose2d: pairwise split transpose` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren |

完整精确 blob 扫描共发现 19 个共享的历史 `.mlu` blob。过滤掉共享的初始/模板噪声后，剩余 17 个精确 blob，其中 MosRat 编写的实现先出现在本仓库，随后出现在 team25 中。

| blob 前缀       | 文件                                      | 本仓库首次出现                                      | team25 首次出现                                      |
|-----------------|-------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| `c989a2d68ed8` | `Product_reduction_over_a_dimension.mlu` | `f724b7e5`, 2026-05-06 01:14:03 +08:00, MosRat, `038 077 079 081 Reduce: test compact sums` | `bb31285b`, 2026-06-10 20:33:52 +08:00, Cui Jiangren, `Sample stable reductions with product zero kernel` |
| `363461e73c50` | `Depthwise_conv_2D.mlu`                  | `0ccdf1e7`, 2026-05-12 00:48:34 +08:00, MosRat, `134 136 conv: remove bias init patch probe` | `1f1be09d`, 2026-05-18 03:05:49 +08:00, cuijiangren, `feat 007` |
| `deff056b9154` | `conv_transposed_2D__asymmetric_input__square_kernel.mlu` | `b71b8978`, 2026-05-14 19:10:34 +08:00, MosRat, `012 ConvTranspose2d: pairwise split transpose` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `1d4e2d2bf13f` | `Dilated_conv_2D.mlu`                    | `159d7291`, 2026-05-16 11:12:54 +08:00, MosRat, `Remove extra stream experiments` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `630be6840046` | `GRU_forward.mlu`                        | `15110872`, 2026-05-16 19:50:31 +08:00, MosRat, `138 GRU_forward: fix batch task mapping` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `97c1bbcde0fe` | `BatchNorm.mlu`                          | `0dfe8e81`, 2026-05-17 15:12:25 +08:00, MosRat, `039 BatchNorm halfsum k1 32x8` | `8b31f82e`, 2026-05-18 02:30:26 +08:00, cuijiangren, `feat 005` |
| `5587b01caaf3` | `GELU.mlu`                               | `68f1b2dd`, 2026-05-18 22:55:29 +08:00, MosRat, `easy batch warm launch` | `e66ecc5e`, 2026-05-30 20:13:29 +08:00, cuijiangren, `test` |
| `f5bb69f2126c` | `GELU.mlu`                               | `2975e940`, 2026-05-20 22:43:50 +08:00, MosRat, `027 GELU block launch` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `56bd2ca90937` | `ELU.mlu`                                | `d30b3367`, 2026-05-20 22:49:40 +08:00, MosRat, `026 ELU arithmetic split` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `5a537470ba60` | `HardTanh.mlu`                           | `83a774aa`, 2026-05-20 22:58:01 +08:00, MosRat, `029 HardTanh abs clamp` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `074e4ffcea1f` | `HardSigmoid.mlu`                        | `57dbf365`, 2026-05-20 23:00:23 +08:00, MosRat, `028 HardSigmoid single tile abs clamp` | `2a58931c`, 2026-05-22 13:11:28 +08:00, cuijiangren, `feat 008` |
| `13e7a0486197` | `HardTanh.mlu`                           | `d93fdc7e`, 2026-05-22 18:16:07 +08:00, MosRat, `028 029 scalar clamp eq intrinsics 32k` | `182dc7b5`, 2026-05-22 18:39:42 +08:00, cuijiangren, `feat 008` |
| `292854fd5bd9` | `HardSigmoid.mlu`                        | `7a943cf8`, 2026-05-22 18:16:39 +08:00, MosRat, `028 029 scalar clamp eq intrinsics 8k` | `182dc7b5`, 2026-05-22 18:39:42 +08:00, cuijiangren, `feat 008` |
| `0c10aac838c6` | `GELU.mlu`                               | `ac638fd1`, 2026-05-31 05:51:56 +08:00, MosRat, `027 GELU direct launch no guard` | `24748a1f`, 2026-05-31 10:46:00 +08:00, cuijiangren, `test` |
| `29c28970cd8e` | `GELU.mlu`                               | `81195f9f`, 2026-05-31 18:14:00 +08:00, MosRat, `026 027 028 029 clean launch-bound rerun 1` | `dd23efff`, 2026-05-31 20:11:33 +08:00, cuijiangren, `test` |
| `5af811921331` | `ELU.mlu`                                | `ffe26425`, 2026-06-02 02:34:00 +08:00, MosRat, `026 026-029 joint config worker warmness probe 1` | `f7866174`, 2026-06-02 04:27:36 +08:00, cuijiangren, `test` |
| `c89bd0db5baf` | `Masked_softmax.mlu`                     | `c1573f279`, 2026-06-08 15:31:57 +08:00, MosRat, `120 Masked_softmax reduce launch tasks` | `d58b818`, 2026-06-12 23:35:14 +08:00, `github-actions[bot]`, `Sample late reference-like package` |

## 已确认的高相似度改写匹配

对 team25 当前文件，使用归一化 token 与本仓库完整历史进行匹配，找到以下最佳匹配。归一化会去除注释和空白后再进行 token 比较，因此这些是结构/代码层面的匹配，而非格式匹配。

| team25 文件                                      | 本仓库最佳先前文件/版本                          | 归一化 token 相似度 |
|--------------------------------------------------|--------------------------------------------------|---------------------|
| `Masked_softmax.mlu`                            | `c1573f279:Masked_softmax.mlu`                  | `1.0000`           |
| `Matmul_with_large_K_dimension_.mlu`            | `8b4c626c2:Matmul_with_large_K_dimension_.mlu` | `0.9893`           |
| `Matmul_for_upper_triangular_matrices.mlu`      | `84680dbbc:Matmul_for_upper_triangular_matrices.mlu` | `0.9877`      |
| `Grid_sample.mlu`                               | `110cc66b7:Grid_sample.mlu`                     | `0.9828`           |
| `Matmul_with_irregular_shapes_.mlu`             | `66afbfd86:Matmul_with_irregular_shapes_.mlu`  | `0.9817`           |
| `conv_depthwise_separable_2D.mlu`               | `03db8a5d6:conv_depthwise_separable_2D.mlu`    | `0.9816`           |
| `Scaled_Dot_Product_Attention.mlu`              | `149f7301:Scaled_Dot_Product_Attention.mlu`    | `0.9525`           |

观察到的改写模式：team25 通常保留 BangC 内核常量、分块策略、索引计算、`__memcpy`、`__bang_*` 以及 launch 结构，仅重命名函数和变量，并添加主机侧的设备 pinning/静态张量缓存。例如 `Grid_sample.mlu` 和 `Matmul_with_irregular_shapes_.mlu`。

## 复现命令

```bash
git -C /tmp/openoperator-start-kit-team25 show --stat d58b818 43b0180 e648dda -- '*.mlu'

git -C /workspace/algorithm/AICS-openoperator rev-parse c1573f279:Masked_softmax.mlu
git -C /tmp/openoperator-start-kit-team25 rev-parse HEAD:Masked_softmax.mlu

diff -u --ignore-all-space \
  <(git -C /workspace/algorithm/AICS-openoperator show c1573f279:Masked_softmax.mlu) \
  <(git -C /tmp/openoperator-start-kit-team25 show HEAD:Masked_softmax.mlu)
```

以上两个 `rev-parse` 命令的预期结果：

```text
c89bd0db5bafcbc0d164048ddd4a817c37f5dc9b
c89bd0db5bafcbc0d164048ddd4a817c37f5dc9b
```

## 后续分析步骤

1. 构建完整的精确 blob 匹配表格，包含完整提交时间、作者、提交信息和文件路径。
2. 排除早于 MosRat/team 实现工作的共享初始/模板文件。
3. 对 team25 的每个 `.mlu` 历史 blob，在本仓库历史中查找先于它的同文件最佳匹配。
4. 仅保留本仓库实现时间更早，且 team25 精确复用或进行微小改写的匹配。
5. 为最强的改写案例添加有代表性的 `diff --ignore-all-space` 片段。
````
