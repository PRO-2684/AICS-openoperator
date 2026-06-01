# Convolution 初学者教程

本目录包含多个卷积实现，从最小实现到大规模 Tiling 实现：

- **`conv_minmal.mlu`**：最小实现（5×5×64），适合初学者理解基础概念
- **`conv_2048_tiling.mlu`**：大规模实现（2048×2048×64），使用二维 Tiling 技术处理超大规模数据

本教程还包含一个运行脚本（`build_eval.sh`）和快速上手说明。

卷积（Convolution）是深度学习中最基础和最重要的算子之一，广泛应用于图像处理、计算机视觉和自然语言处理等领域。本教程将带你从零开始，理解如何在 MLU 上高效实现卷积操作，包括：

1. **基础实现**：最小规模的卷积实现，理解基本概念
2. **Tiling 技术**：处理大规模数据的核心技术
   - 一维 Tiling：在高度方向分块
   - 二维 Tiling：在高度和宽度方向同时分块

## 1. 最小实现（`conv_minmal.mlu`）

- 使用单个任务，通过 BangC API（`__memcpy`、`__bang_conv`）完成标准卷积操作
- **不做 tiling**：假设数据量较小（`IN_HEIGHT=5, IN_WIDTH=5, IN_CHANNEL=64, OUT_CHANNEL=64`），一次性将所有数据加载到 NRAM 中处理，然后写回 GDRAM
- **单组卷积**：使用标准卷积（`group_count=1`），不支持分组卷积
- Kernel 使用 NRAM 存储输入和输出数据，WRAM 存储卷积核（filter）
- **Filter Reshape**：在主机端将 filter 重新排列以满足 WRAM 16 字节对齐要求
- 主机端准备随机输入、分配设备内存、拷贝数据、执行 Kernel，然后拷贝回主机并与 CPU 参考结果比对
- 适合初学者理解 BangC 编译器的基本编译/执行流程和卷积算子的实现

### 核心数据布局

```40:78:06_conv/conv_minmal.mlu
__mlu_entry__ void ConvKernel(float* out_data,
                               const float* in_data,
                               const float* filter_data,
                               int in_channel,
                               int in_height,
                               int in_width,
                               int filter_height,
                               int filter_width,
                               int stride_height,
                               int stride_width,
                               int out_channel) {
  int out_height = (in_height - filter_height) / stride_height + 1;
  int out_width = (in_width - filter_width) / stride_width + 1;

  // NRAM 缓冲区：存储输入数据和输出数据
  __nram__ float nram_in_data[IN_DATA_NUM];
  __nram__ float nram_out_data[OUT_DATA_NUM];

  // WRAM 缓冲区：存储 filter（卷积核）
  __wram__ float wram_filter[FILTER_DATA_NUM];

  // 从 GDRAM 加载输入数据到 NRAM
  __memcpy(nram_in_data, in_data, IN_DATA_NUM * sizeof(float), GDRAM2NRAM);

  // 从 GDRAM 加载 filter 数据到 WRAM
  __memcpy(wram_filter, filter_data, FILTER_DATA_NUM * sizeof(float), GDRAM2WRAM);

  // 执行卷积操作
  __bang_conv(nram_out_data, nram_in_data, wram_filter,
              in_channel, in_height, in_width,
              filter_height, filter_width,
              stride_width, stride_height, out_channel);

  // 将结果从 NRAM 写回 GDRAM
  __memcpy(out_data, nram_out_data, OUT_DATA_NUM * sizeof(float), NRAM2GDRAM);
}
```

**数据布局说明**：
- **输入数据（CPU/GDRAM）**：`[IN_HEIGHT, IN_WIDTH, IN_CHANNEL]` - HWC 布局
- **Filter（卷积核，CPU/GDRAM）**：`[OUT_CHANNEL, FILTER_HEIGHT, FILTER_WIDTH, IN_CHANNEL]` - NHWC 布局
- **输出数据（CPU/GDRAM）**：`[OUT_HEIGHT, OUT_WIDTH, OUT_CHANNEL]` - HWC 布局

---

### __bang_conv API 参数详解

**参考文档**：[Cambricon BANG C 内置函数 - Bang Conv](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Artificial%20Intelligence%20Functions.html?highlight=conv#_CPPv411__bang_convPfPKfPKfjjjjjjjjjjjj)

**API 签名**：
```c
void __bang_conv(float *dst, const float *src, const float *filter,
                 int channel_input, int height, int width,
                 int filter_height, int filter_width,
                 int stride_width, int stride_height, int channel_output);
```

**参数说明**：
- `dst`: 输出数据（NRAM），布局：`[channel_output, out_height, out_width]` - **CHW 布局**
- `src`: 输入数据（NRAM），布局：`[channel_input, height, width]` - **CHW 布局**
- `filter`: 卷积核（WRAM），布局：`[channel_output, filter_height, filter_width, channel_input]` - NHWC 布局
- `channel_input`: 输入通道数（`IN_CHANNEL`）
- `height`, `width`: 输入特征图的高度和宽度（`IN_HEIGHT`, `IN_WIDTH`）
- `filter_height`, `filter_width`: 卷积核的高度和宽度（`FILTER_HEIGHT`, `FILTER_WIDTH`）
- `stride_width`, `stride_height`: 步长（`STRIDE_WIDTH`, `STRIDE_HEIGHT`）
- `channel_output`: 输出通道数（`OUT_CHANNEL`）

**重要提示**：
- `__bang_conv` 期望的输入/输出数据在 NRAM 中是 **CHW 布局**，而 CPU 端使用的是 **HWC 布局**
- 本教程中，数据在 GDRAM 中使用 HWC 布局，但在调用 `__bang_conv` 之前，需要理解并确保数据布局正确
- 如果使用 `__memcpy` 直接从 GDRAM 拷贝到 NRAM，需要注意布局转换（当前简化实现假设数据已在正确布局）

---

### 探索任务

参考文档：[Cambricon BANG C/C++ 编程指南 - 硬件实现](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/hardware_implementation/index.html)

---

#### 任务 1：数据对齐下限探索

**目标**：理解数据对齐要求，找到最小的合法数据规模

**步骤**：
1. 将数据规模设置为极小值：
   - `IN_HEIGHT=3, IN_WIDTH=3, IN_CHANNEL=4, OUT_CHANNEL=4`
   - `FILTER_HEIGHT=1, FILTER_WIDTH=1`（1x1 卷积）
2. 尝试编译和运行
3. 逐步调整参数，观察最小可运行的数据规模
4. 观察不同数据规模下的性能变化

**问题**：
- 是否存在最小数据规模限制？
- 对齐要求是什么？（WRAM 需 16 字节对齐）
- 为什么小数据规模下性能较低？
- 如何平衡数据规模和性能？

---

#### 任务 2：Filter Reshape 理解

**目标**：理解为什么需要对 Filter 进行 reshape，以及如何实现

**参考文档**：[Cambricon BANG C 编程指南 - __bang_conv](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Artificial%20Intelligence%20Functions.html?highlight=conv#_CPPv411__bang_convPfPKfPKfjjjjjjjjjjjj)

**Reshape 原理**：

```144:176:06_conv/conv_minmal.mlu
static void reshape_filter(const float* filter_original,
                           float* filter_reshaped,
                           int out_channel,
                           int filter_height,
                           int filter_width,
                           int in_channel) {
  int filter_size_per_out = filter_height * filter_width * in_channel;

  // 临时存储：按 out_channel 组织
  float tmp[OUT_CHANNEL][FILTER_DATA_NUM / OUT_CHANNEL];

  // 第一步：将原始 filter 按标准 NHWC 布局组织
  for (int oc = 0; oc < out_channel; ++oc) {
    for (int i = 0; i < filter_size_per_out; ++i) {
      tmp[oc][i] = filter_original[oc * filter_size_per_out + i];
    }
  }

  // 第二步：reshape 以满足 WRAM 16字节对齐
  // 重新排列 out_channel 的维度，使其按 16*4 的块组织
  int align_16 = out_channel / (16 * 4);
  int idx = 0;

  for (int i = 0; i < 16 * 4; i += 4) {
    for (int k = 0; k < align_16; ++k) {
      for (int m = 0; m < 4; ++m) {
        for (int j = 0; j < filter_size_per_out; ++j) {
          filter_reshaped[idx++] = tmp[i + k * 16 * 4 + m][j];
        }
      }
    }
  }
}
```

**实验**：
1. 尝试不进行 reshape，直接使用原始 filter
2. 观察是否出现错误或性能下降
3. 理解 16 字节对齐的含义（4 个 float 对齐）

**问题**：
- 为什么需要对 Filter 进行 reshape？
- WRAM 的 16 字节对齐要求是什么？
- 16*4 的块大小是如何确定的？
- reshape 后的 filter 布局是什么样子的？

---

## 2. 运行脚本（`build_eval.sh`）

### 2.1 环境变量配置

脚本自动设置以下环境变量：

- **`NEUWARE_HOME=/usr/local/neuware`**: 指定 Neuware SDK 的安装路径，Neuware 是寒武纪 MLU 的开发工具包
- **`LD_LIBRARY_PATH`**: 添加 Neuware 的库文件路径（`$NEUWARE_HOME/lib64`），确保运行时能找到 MLU 相关的动态链接库
- **`PATH`**: 添加 Neuware 的二进制工具路径（`$NEUWARE_HOME/bin`），使 `cncc` 编译器可以直接调用
- **`MLU_VISIBLE_DEVICES=0`**: 指定使用第 0 号 MLU 设备（在多卡环境下可以选择其他设备）
- **`TORCH_DEVICE_BACKEND_AUTOLOAD=0`**: 禁用 PyTorch 的设备后端自动加载，避免与 BangC 运行时冲突

### 2.2 使用方法

脚本接受一个参数：`.mlu` 源文件的文件名。

```bash
./build_eval.sh conv_minmal.mlu
```

脚本会：
1. 自动切换到脚本所在目录（`Experiments/06_conv`）
2. 使用 `cncc` 编译器编译 `.mlu` 文件，生成可执行文件
3. 执行生成的可执行文件并输出结果

### 2.3 编译参数说明

脚本使用的编译命令：
```bash
cncc "${MLU_SOURCE}" -o "${TARGET}" --bang-mlu-arch=mtp_592 -O3 -lm
```

- `--bang-mlu-arch=mtp_592`: 指定目标 MLU 架构为 mtp_592
- `-O3`: 最高级别的优化
- `-lm`: 链接数学库（虽然当前实现不使用 math 库，但保留以备后续扩展）

---

## 3. 建议的学习流程

### 第一阶段：理解基础概念

1. **学习最小实现**
   - 阅读 `conv_minmal.mlu` 的代码和注释
   - 运行 `./build_eval.sh conv_minmal.mlu`，观察输出结果
   - 理解数据布局（HWC、CHW、NHWC）
   - 理解卷积的计算流程

2. **完成探索任务**
   - 数据对齐下限探索
   - Filter Reshape 理解

3. **阅读硬件文档**
   - [硬件实现文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/hardware_implementation/index.html)
   - 理解 NRAM、WRAM、SRAM 的特点和容量限制

### 第二阶段：进阶优化

1. **理解 Filter Reshape 的必要性**
   - 为什么 WRAM 需要 16 字节对齐
   - 如何正确地 reshape filter
   - reshape 后的数据布局

2. **尝试不同的配置**
   - 修改卷积核大小（1x1, 3x3, 5x5）
   - 修改步长（1, 2, 4）
   - 修改输入/输出通道数
   - 观察性能和正确性

3. **理解性能瓶颈**
   - 内存访问 vs 计算
   - NRAM 容量限制
   - WRAM 对齐要求

### 第三阶段：Tiling 技术

1. **理解 Tiling 的必要性**
   - NRAM 容量限制（768KB）
   - 当数据规模超过 NRAM 容量时，需要分块处理
   - Tiling 是处理大规模数据的关键技术

2. **学习一维 Tiling**
   - 参考实现：`conv_large_tiling.mlu`（224×224×64）
   - 在高度维度进行分块
   - 理解任务分配和内存管理

3. **学习二维 Tiling**
   - 参考实现：`conv_2048_tiling.mlu`（2048×2048×64）
   - 同时在高度和宽度维度分块
   - 理解二维任务分配策略

4. **完成 Tiling 探索任务**
   - 理解不同 Tiling 策略的适用场景
   - 学习如何计算 NRAM 使用量
   - 掌握任务 ID 解码和边界处理

### 第四阶段：实践与探索

1. **修改和实验**
   - 尝试不同的数据规模和参数组合
   - 实现自己的优化方案
   - 比较不同实现的性能

2. **进阶主题**
   - 分组卷积（Grouped Convolution）
   - 深度可分离卷积（Depthwise Convolution）
   - 转置卷积（Transposed Convolution）
   - 三维 Tiling（通道维度分块）

3. **实际应用**
   - 将卷积算子集成到实际的神经网络中
   - 优化特定应用场景下的性能

---

## 4. Tiling 技术详解

当输入数据规模超过 NRAM 容量限制（768KB）时，我们需要使用 **Tiling（分块）** 技术将数据分成多个小块，每次处理一个块。Tiling 是处理大规模卷积操作的核心技术。

### 4.1 为什么需要 Tiling？

**问题场景**：
- NRAM 容量：768KB（约 196,608 个 float）
- 对于 224×224×64 的输入：224 × 224 × 64 × 4B = 12.8MB >> 768KB ❌
- 对于 1024×1024×64 的输入：1024 × 1024 × 64 × 4B = 256MB >> 768KB ❌

**解决方案**：将数据分块处理
- 每个任务只处理数据的一个子集（tile）
- 多个任务并行处理不同的 tile
- 最终合并所有结果

### 4.2 一维 Tiling（空间维度 - 高度方向）

**参考实现**：`conv_large_tiling.mlu`（224×224×64）

#### 核心思想

按输出行分配任务，每个任务处理固定数量的输出行：

```c
#define ROWS_PER_TASK 2  // 每个任务处理 2 行输出

// 任务分配
int out_row_start = taskId * ROWS_PER_TASK;
int out_row_end = out_row_start + ROWS_PER_TASK;
```

#### 输入数据范围计算

由于 3×3 卷积需要上下文，每个任务需要加载额外的输入行：

```c
// 输出 2 行，需要输入 4 行（2 + 3 - 1）
#define MAX_IN_ROWS (ROWS_PER_TASK + FILTER_HEIGHT - 1)  // 4

// 计算输入行范围
int in_row_start = out_row_start * stride_height;
int in_row_end = in_row_start + (actual_rows - 1) * stride_height + filter_height;
```

#### NRAM 内存使用

```c
// 输入 tile: 4 × 224 × 64 × 4B = 229,376 bytes (~224KB)
__nram__ float nram_in_data[MAX_IN_ROWS * IN_WIDTH * IN_CHANNEL];

// 输出 tile: 2 × 222 × 64 × 4B = 113,664 bytes (~111KB)
__nram__ float nram_out_data[ROWS_PER_TASK * OUT_WIDTH * OUT_CHANNEL];

// 总计: ~343KB < 768KB ✓
```

#### 数据加载（按行加载）

```c
// 从 GDRAM 加载输入数据 tile 到 NRAM
for (int row = 0; row < actual_in_rows; ++row) {
    int global_row = in_row_start + row;
    const float* src = in_data + global_row * in_width * in_channel;
    float* dst = nram_in_data + row * in_width * in_channel;
    __memcpy(dst, src, in_width * in_channel * sizeof(float), GDRAM2NRAM);
}
```

#### 任务并行度

- **总任务数**：`(OUT_HEIGHT + ROWS_PER_TASK - 1) / ROWS_PER_TASK`
- **224×224 示例**：222 行输出 ÷ 2 = 111 个任务

#### 适用场景

- ✅ 输入宽度较小（≤ 224）
- ✅ 输入高度较大（> 100）
- ✅ 单任务 NRAM 使用 < 768KB

### 4.3 二维 Tiling（空间维度 - 高度和宽度方向）

**参考实现**：`conv_2048_tiling.mlu`（2048×2048×64）

#### 为什么需要二维 Tiling？

对于 1024×1024 输入，如果只用一维 Tiling：
```
输入 tile: 4 × 1024 × 64 × 4B = 1,048,576 bytes > 768KB ❌
```

**解决方案**：同时在高度和宽度方向分块

#### 核心思想

按输出的 (行, 列块) 分配任务：

```c
#define ROWS_PER_TASK 2   // 每个任务处理 2 行输出
#define TILE_WIDTH 256    // 每个任务处理 256 列输出

// 计算宽度方向的 tile 数量
int num_width_tiles = (out_width + TILE_WIDTH - 1) / TILE_WIDTH;

// 从 taskId 解码出行任务 ID 和列任务 ID
int row_task_id = taskId / num_width_tiles;
int col_task_id = taskId % num_width_tiles;
```

#### 输出范围计算

```c
// 计算当前任务处理的输出行范围
int out_row_start = row_task_id * ROWS_PER_TASK;
int out_row_end = out_row_start + ROWS_PER_TASK;
int actual_rows = out_row_end - out_row_start;

// 计算当前任务处理的输出列范围
int out_col_start = col_task_id * TILE_WIDTH;
int out_col_end = out_col_start + TILE_WIDTH;
int actual_cols = out_col_end - out_col_start;
```

#### 输入范围计算

考虑卷积核的上下文需求：

```c
// 计算输入数据需要的行范围
int in_row_start = out_row_start * stride_height;
int in_row_end = in_row_start + (actual_rows - 1) * stride_height + filter_height;
int actual_in_rows = in_row_end - in_row_start;

// 计算输入数据需要的列范围
int in_col_start = out_col_start * stride_width;
int in_col_end = in_col_start + (actual_cols - 1) * stride_width + filter_width;
int actual_in_cols = in_col_end - in_col_start;
```

#### NRAM 内存使用

```c
#define MAX_IN_ROWS (ROWS_PER_TASK + FILTER_HEIGHT - 1)  // 4
#define MAX_IN_COLS (TILE_WIDTH + FILTER_WIDTH - 1)      // 258

// 输入 tile: 4 × 258 × 64 × 4B = 263,168 bytes (~257KB)
__nram__ float nram_in_data[MAX_IN_ROWS * MAX_IN_COLS * IN_CHANNEL];

// 输出 tile: 2 × 256 × 64 × 4B = 131,072 bytes (~128KB)
__nram__ float nram_out_data[ROWS_PER_TASK * TILE_WIDTH * OUT_CHANNEL];

// 总计: 394,240 bytes (~385KB, 51.3% of NRAM) ✓
```

#### 数据加载（二维切片）

```c
// 从 GDRAM 加载输入数据 tile 到 NRAM（二维切片）
for (int row = 0; row < actual_in_rows; ++row) {
    int global_row = in_row_start + row;
    // 从全局位置 [global_row, in_col_start:in_col_end, :] 加载
    const float* src = in_data + global_row * in_width * in_channel 
                      + in_col_start * in_channel;
    float* dst = nram_in_data + row * actual_in_cols * in_channel;
    __memcpy(dst, src, actual_in_cols * in_channel * sizeof(float), GDRAM2NRAM);
}
```

#### 数据写回（二维切片）

```c
// 将结果从 NRAM 写回 GDRAM（二维切片）
for (int row = 0; row < actual_rows; ++row) {
    int global_row = out_row_start + row;
    // 写回到全局位置 [global_row, out_col_start:out_col_end, :]
    float* dst = out_data + global_row * out_width * out_channel 
                + out_col_start * out_channel;
    const float* src = nram_out_data + row * actual_cols * out_channel;
    __memcpy(dst, src, actual_cols * out_channel * sizeof(float), NRAM2GDRAM);
}
```

#### 任务并行度

```
总任务数 = 行任务数 × 列任务数

1024×1024 示例：
- 行任务数: (1022 + 2 - 1) / 2 = 511
- 列任务数: (1022 + 256 - 1) / 256 = 4
- 总任务数: 511 × 4 = 2044

2048×2048 示例：
- 行任务数: (2046 + 2 - 1) / 2 = 1023
- 列任务数: (2046 + 256 - 1) / 256 = 8
- 总任务数: 1023 × 8 = 8184
```

#### 适用场景

- ✅ 输入宽度较大（> 256）
- ✅ 输入高度较大（> 100）
- ✅ 需要处理超大规模数据（1024×1024+）

### 4.4 Tiling 参数选择

#### 关键考虑因素

1. **NRAM 容量限制**：768KB
   - 需要计算输入 tile + 输出 tile 的总大小
   - 建议使用 < 70% 的 NRAM，留有余量

2. **任务粒度平衡**
   - 太小：任务调度开销大
   - 太大：可能超出 NRAM 限制

3. **并行度**
   - 更多任务 = 更高并行度
   - 但也要考虑任务调度开销

#### NRAM 使用量计算公式

```c
// 输入 tile 大小
input_tile_size = (ROWS_PER_TASK + FILTER_HEIGHT - 1) 
                × (TILE_WIDTH + FILTER_WIDTH - 1) 
                × IN_CHANNEL 
                × sizeof(float)

// 输出 tile 大小
output_tile_size = ROWS_PER_TASK 
                 × TILE_WIDTH 
                 × OUT_CHANNEL 
                 × sizeof(float)

// 总 NRAM 使用
total_nram = input_tile_size + output_tile_size
```

#### 参数调优经验

| 配置 | ROWS_PER_TASK | TILE_WIDTH | NRAM 使用 | 结果 |
|------|--------------|------------|-----------|------|
| 尝试 1 | 2 | 480 | ~571KB | ❌ 超限 |
| 尝试 2 | 2 | 360 | ~450KB | ❌ 编译失败 |
| **最终** | **2** | **256** | **~385KB** | **✅ 成功** |

**经验教训**：
- 编译器对 NRAM 使用有严格检查
- 需要预留足够余量（建议 < 70%）
- 保守的 tile 大小更可靠

### 4.5 Tiling 探索任务

#### 任务 3：理解一维 Tiling

**目标**：理解一维 Tiling 的实现原理

**步骤**：
1. 阅读 `conv_large_tiling.mlu`（如果存在）或参考 `conv_2048_tiling.mlu` 的实现思路
2. 理解任务分配：`out_row_start = taskId * ROWS_PER_TASK`
3. 理解输入范围计算：为什么需要 `+ FILTER_HEIGHT - 1`？
4. 计算 NRAM 使用量，验证是否 < 768KB

**问题**：
- 为什么每个任务需要加载 `ROWS_PER_TASK + FILTER_HEIGHT - 1` 行输入？
- 如何计算总任务数？
- 边界任务如何处理？

#### 任务 4：理解二维 Tiling

**目标**：理解二维 Tiling 的实现原理

**步骤**：
1. 阅读 `conv_2048_tiling.mlu` 的代码
2. 理解任务 ID 解码：`row_task_id = taskId / num_width_tiles`
3. 理解二维数据加载：如何从 GDRAM 加载一个矩形区域？
4. 理解二维数据写回：如何写回到正确的全局位置？

**问题**：
- 为什么需要二维 Tiling？一维 Tiling 的局限性是什么？
- 如何从 `taskId` 解码出 `(row_task_id, col_task_id)`？
- 输入 tile 的列范围如何计算？为什么需要 `+ FILTER_WIDTH - 1`？

#### 任务 5：Tiling 参数调优

**目标**：尝试不同的 Tiling 参数，观察性能变化

**实验**：
1. 修改 `ROWS_PER_TASK`（如 1, 3, 4），观察：
   - NRAM 使用量变化
   - 任务数量变化
   - 执行时间变化

2. 修改 `TILE_WIDTH`（如 128, 256, 512），观察：
   - NRAM 使用量变化
   - 任务数量变化
   - 执行时间变化

3. 找到最优的参数组合

**问题**：
- 更大的 tile 是否总是更好？
- 任务数量与性能的关系是什么？
- 如何平衡 NRAM 使用率和并行度？

### 4.6 性能对比

| 实现 | 输入规模 | Tiling 策略 | 任务数 | NRAM 使用 | 执行时间 |
|------|---------|------------|--------|----------|---------|
| conv_minmal | 5×5×64 | 无 | 1 | ~9KB | ~0.1ms |
| conv_large_tiling | 224×224×64 | 一维（行） | 111 | ~343KB | ~0.1ms |
| conv_2048_tiling | 2048×2048×64 | 二维（行×列） | 8184 | ~385KB | ~3.5ms |

**观察**：
- Tiling 使处理大规模数据成为可能
- 二维 Tiling 可以处理超大规模数据（1024×1024+）
- 任务并行度显著提升（从 1 到 8184）

### 4.7 进一步优化方向

1. **增大 Tile 大小**
   - 如果 NRAM 有余量，可以尝试增大 `ROWS_PER_TASK` 或 `TILE_WIDTH`
   - 减少任务数量，降低调度开销

2. **异步数据传输**
   - 使用 `__memcpy_async` 实现 IO-Compute 重叠
   - 在计算当前 tile 时预取下一个 tile

3. **三维 Tiling**
   - 对于更多通道（128+），可以在通道维度也进行分块
   - 实现通道维度的累加

4. **Union 任务类型**
   - 利用多 Cluster 并行
   - 可能进一步提升性能

---

## 常见问题

### Q1：为什么需要将 Filter 放在 WRAM 而不是 NRAM？

**A**：
- **内存共享**：Filter 在所有 batch 间共享，放在 WRAM 可以避免重复加载
- **容量考虑**：WRAM 容量（1024KB）大于 NRAM（768KB），可以容纳更大的 Filter
- **访问效率**：WRAM 的访问延迟与 NRAM 相当，不会降低性能
- **设计原则**：共享数据放在 WRAM，私有数据放在 NRAM

### Q2：为什么需要对 Filter 进行 reshape？

**A**：
- **对齐要求**：WRAM 需要 16 字节对齐（4 个 float）
- **性能优化**：对齐后的数据访问效率更高
- **硬件约束**：MLU 硬件对内存访问的对齐要求
- **Reshape 方法**：按照 16*4 的块重新排列 out_channel 维度

### Q3：如何计算卷积的输出尺寸？

**A**：
```
OUT_HEIGHT = (IN_HEIGHT - FILTER_HEIGHT) / STRIDE_HEIGHT + 1
OUT_WIDTH = (IN_WIDTH - FILTER_WIDTH) / STRIDE_WIDTH + 1
OUT_DATA_NUM = OUT_HEIGHT * OUT_WIDTH * OUT_CHANNEL
```

### Q4：为什么验证使用 CPU 参考实现而不是数学公式？

**A**：
- **确保算法逻辑正确**：CPU 参考实现可以验证 MLU 实现的正确性
- **检测边界条件问题**：边界条件（如 padding、stride）容易出错
- **验证数据布局**：确保数据布局（HWC、NHWC）正确
- **便于调试**：CPU 代码更容易调试和打印中间结果

### Q5：如何处理数据超过 NRAM 容量的情况？

**A**：
- **Tiling**：将数据分块，每次处理一个块
  - **一维 Tiling**：在高度方向分块（适用于宽度较小的情况）
  - **二维 Tiling**：在高度和宽度方向同时分块（适用于超大规模数据）
  - **三维 Tiling**：在高度、宽度和通道方向分块（适用于超多通道的情况）
- **分组卷积**：减少每组的数据量
- **多任务并行**：每个任务处理一部分数据
- **使用 SRAM**：将部分数据放在 SRAM 中

**详细说明**：参见 [4. Tiling 技术详解](#4-tiling-技术详解)

### Q6：`__bang_conv` 的输入/输出数据在 NRAM 中的布局是什么？

**A**：
- **输入数据**：`[channel_input, height, width]` - CHW 布局（**注意**：不同于 CPU 端的 HWC）
- **输出数据**：`[channel_output, out_height, out_width]` - CHW 布局
- **Filter 数据**：`[channel_output, filter_height, filter_width, channel_input]` - NHWC 布局

**注意**：`__bang_conv` 期望的数据布局可能与 CPU 参考实现不同，需要注意转换！

### Q7：如何选择一维 Tiling 还是二维 Tiling？

**A**：
- **一维 Tiling** 适用场景：
  - 输入宽度 ≤ 224
  - 输入高度较大（> 100）
  - 单任务 NRAM 使用 < 768KB
  - 示例：224×224×64

- **二维 Tiling** 适用场景：
  - 输入宽度 > 256
  - 输入高度较大（> 100）
  - 需要处理超大规模数据（1024×1024+）
  - 示例：1024×1024×64, 2048×2048×64

**判断方法**：
1. 计算一维 Tiling 的 NRAM 使用量
2. 如果 > 768KB，则需要二维 Tiling
3. 如果 < 768KB，一维 Tiling 可能更简单高效

### Q8：如何计算 Tiling 的 NRAM 使用量？

**A**：
```c
// 输入 tile 大小
input_tile_size = (ROWS_PER_TASK + FILTER_HEIGHT - 1) 
                × (TILE_WIDTH + FILTER_WIDTH - 1)  // 二维 Tiling
                × IN_CHANNEL 
                × sizeof(float)

// 输出 tile 大小
output_tile_size = ROWS_PER_TASK 
                 × TILE_WIDTH  // 二维 Tiling
                 × OUT_CHANNEL 
                 × sizeof(float)

// 总 NRAM 使用
total_nram = input_tile_size + output_tile_size

// 一维 Tiling 时，TILE_WIDTH = IN_WIDTH
```

**示例**（二维 Tiling，1024×1024×64）：
```
输入 tile: 4 × 258 × 64 × 4B = 263,168 bytes (~257KB)
输出 tile: 2 × 256 × 64 × 4B = 131,072 bytes (~128KB)
总计: 394,240 bytes (~385KB, 51.3% of NRAM) ✓
```

### Q9：为什么输入 tile 需要 `+ FILTER_HEIGHT - 1` 行？

**A**：
- 卷积操作需要上下文信息
- 对于 3×3 卷积，计算输出第 `i` 行需要输入的第 `i-1, i, i+1` 行
- 如果任务处理 `ROWS_PER_TASK` 行输出，需要 `ROWS_PER_TASK + FILTER_HEIGHT - 1` 行输入
- 例如：处理 2 行输出（3×3 卷积）需要 4 行输入（2 + 3 - 1）

**图示**：
```
输入:  [row0]  ← 需要这行（上下文）
       [row1]  ← 输出行 0
       [row2]  ← 输出行 1
       [row3]  ← 需要这行（上下文）
       
输出:  [out0]  ← 由 row0, row1, row2 计算
       [out1]  ← 由 row1, row2, row3 计算
```

### Q10：如何从 taskId 解码出二维 Tiling 的行列任务 ID？

**A**：
```c
// 计算宽度方向的 tile 数量
int num_width_tiles = (out_width + TILE_WIDTH - 1) / TILE_WIDTH;

// 从 taskId 解码出行任务 ID 和列任务 ID
int row_task_id = taskId / num_width_tiles;  // 整数除法
int col_task_id = taskId % num_width_tiles;  // 取余

// 计算输出范围
int out_row_start = row_task_id * ROWS_PER_TASK;
int out_col_start = col_task_id * TILE_WIDTH;
```

**示例**（1024×1024，TILE_WIDTH=256）：
```
num_width_tiles = (1022 + 256 - 1) / 256 = 4

taskId = 0:  row_task_id = 0/4 = 0,  col_task_id = 0%4 = 0  → (0, 0)
taskId = 1:  row_task_id = 1/4 = 0,  col_task_id = 1%4 = 1  → (0, 1)
taskId = 4:  row_task_id = 4/4 = 1,  col_task_id = 4%4 = 0  → (1, 0)
taskId = 7:  row_task_id = 7/4 = 1,  col_task_id = 7%4 = 3  → (1, 3)
```

---

## 参考资料

### 官方文档
- [Cambricon BANG C/C++ 编程指南](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/index.html)
- [硬件实现文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/hardware_implementation/index.html)
- [__bang_conv API 文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Artificial%20Intelligence%20Functions.html?highlight=conv#_CPPv411__bang_convPfPKfPKfjjjjjjjjjjjj)

### 推荐阅读
- Convolutional Neural Networks (LeCun et al., 2015)
- Efficient Convolutional Neural Networks for Mobile Vision Applications (MobileNet, Howard et al., 2017)
- Understanding and Optimizing Lightweight Low-Power Convolutional Neural Networks on MLU (寒武纪白皮书)

### 本目录相关文档
- [conv_large_implementation_summary.md](conv_large_implementation_summary.md) - 224×224 一维 Tiling 实现总结
- [conv_1024_summary.md](conv_1024_summary.md) - 1024×1024 二维 Tiling 实现总结
- [conv_2048_tiling.mlu](conv_2048_tiling.mlu) - 2048×2048 二维 Tiling 实现代码

---

**祝学习愉快！**
