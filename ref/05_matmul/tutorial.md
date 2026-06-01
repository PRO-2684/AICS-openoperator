# MatMul 初学者教程

本目录围绕矩阵乘法算子展示了五个学习阶段的实现程序、一个运行脚本和详细的说明。所有示例都在 `Experiments/05_matmul` 下，可以直接参考对应文件。

**学习路径**：从最基础的朴素实现开始，逐步引入向量化 API、专用算子 API，最后学习 Tiling 优化技术。

## 矩阵乘法基础知识

### 数学定义
矩阵乘法 `C = A * B` 的计算公式：
```
C[i][j] = sum(A[i][l] * B[l][j]) for l in [0, n)
```

其中：
- A 是 [M, N] 矩阵（M 行 N 列）
- B 是 [N, K] 矩阵（N 行 K 列）
- C 是 [M, K] 矩阵（M 行 K 列）
- 计算复杂度：O(M × N × K)

### 内存布局
本教程中所有矩阵都使用**行主序（row-major）**连续存储：
- 二维索引转一维：`matrix[i][j] = matrix[i * cols + j]`
- A 矩阵：`A[i][n + l]` 访问第 i 行第 l 列
- B 矩阵：`B[l][k + j]` 访问第 l 行第 j 列
- C 矩阵：`C[i][k + j]` 访问第 i 行第 j 列

---

## 1. 朴素实现（`matmul_00.mlu`）

**特点**：
- 使用与 CPU 相同的三重循环实现，不使用任何向量化指令
- 数据类型：float32 输入和输出
- 规模：64×64×64 小矩阵，一次性加载到 NRAM
- **不使用 tiling**：假设数据量小到可以放入 NRAM

### 核心实现
```c
// 使用朴素三重循环实现矩阵乘法
for (int i = 0; i < m; ++i) {
  for (int j = 0; j < k; ++j) {
    float sum = 0.0f;
    for (int l = 0; l < n; ++l) {
      sum += nram_A[i * n + l] * nram_B[l * k + j];
    }
    nram_C[i * k + j] = sum;
  }
}
```

### 设计思路
1. 将所有数据从 GDRAM 加载到 NRAM（一次性加载）
2. 在 NRAM 中使用三重循环计算矩阵乘法
3. 将结果写回 GDRAM

### 适用场景
- 理解矩阵乘法的基本计算过程
- 对比 CPU 和 MLU 的实现方式
- 作为后续优化的基准

### 探索任务

参考文档：[Cambricon BANG C/C++ 编程指南 - 硬件实现](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/hardware_implementation/index.html)

#### 1.1 NRAM 容量上限探索
- **任务**：逐步增大矩阵大小（如 128、256、512...），观察程序是否仍能正常编译和运行
- **观察点**：
  - NRAM 使用量如何计算？`NRAM_使用量 = (M×N + N×K + M×K) × sizeof(float)`
  - 编译错误会出现在多大矩阵规模？
  - 理论计算：64×64×64×3×4 = 3MB，实际会失败（NRAM 仅 768KB）
- **硬件背景**：NRAM（核内 RAM）容量有限（MLUv03 为 768KB），需要仔细规划内存使用
- **思考**：为什么三个矩阵的总大小超过了 NRAM 容量？如何通过 tiling 解决这个问题？

#### 1.2 对齐要求下限探索
- **任务**：逐步减小矩阵大小（如 32、16、8、4...），观察程序是否仍能正常编译和运行
- **观察点**：
  - 最小的矩阵维度是多少？
  - 是否存在对齐约束？
- **硬件背景**：MLU 硬件对向量运算有对齐要求（地址对齐到 64 字节，长度对齐到 128 字节）
- **思考**：对于标量循环，对齐约束的影响是什么？

#### 1.3 矩阵乘法计算过程理解
- **任务**：手动计算一个 2×2 矩阵乘法，理解计算过程
  ```
  A = [[1, 2],     B = [[5, 6],
       [3, 4]]          [7, 8]]
  C = A * B = ?
  ```
- **验证**：使用 CPU 参考实现验证你的计算
- **对比实验**：对比 CPU 实现的三重循环和 MLU 实现的三重循环，理解它们的异同
- **思考**：为什么矩阵乘法需要三重循环？能否用更少的循环实现？

#### 1.4 与向量运算的对比
- **任务**：对比 `vecadd_minimal.mlu` 和 `matmul_00.mlu` 的代码结构
- **观察点**：
  - 输入数量：VecAdd 需要 2 个向量，MatMul 需要 3 个矩阵
  - 计算复杂度：VecAdd 是 O(N)，MatMul 是 O(M×N×K)
  - 内存访问：VecAdd 需要访问 2 个输入和 1 个输出，MatMul 需要访问 3 个矩阵
  - 循环结构：VecAdd 没有循环（使用 __bang_add），MatMul 有三重循环
- **思考**：矩阵运算比向量运算复杂在哪里？为什么优化空间更大？

#### 1.5 内存访问模式分析
- **任务**：分析三重循环中的内存访问模式
- **观察点**：
  - A 矩阵：按行访问（`A[i * n + l]`），访问模式连续
  - B 矩阵：按列访问（`B[l * k + j]`），访问模式跳跃（步长为 k）
  - C 矩阵：按行写入（`C[i * k + j]`），写入模式连续
- **思考**：B 矩阵的访问模式是否不利于缓存？如何优化？

---

## 2. 向量化实现（`matmul_01.mlu`）

**特点**：
- 使用 `__bang_mul` 和 `__bang_add` 向量化 API
- 数据类型：float32 输入和输出
- 规模：64×64×64 小矩阵，一次性加载到 NRAM
- **不使用 tiling**：假设数据量小到可以放入 NRAM
- **算法**：外积（outer product）+ 累加方法

### 核心实现
```c
// 使用 __bang_mul 和 __bang_add 实现矩阵乘法
// 矩阵乘法 C = A * B 可以分解为：C = sum_k (A[:,k] ⊗ B[k,:])
for (int k_idx = 0; k_idx < n; ++k_idx) {
  // 构造 A[:,k_idx] 的广播矩阵 [M, K]
  for (int i = 0; i < m; ++i) {
    __bang_write_value(nram_a_vec + i * k, k, nram_A[i * n + k_idx]);
  }
  
  // 构造 B[k_idx,:] 的复制矩阵 [M, K]
  for (int i = 0; i < m; ++i) {
    __memcpy(nram_b_vec + i * k, nram_B + k_idx * k, k * sizeof(float), NRAM2NRAM);
  }
  
  // 计算外积并累加
  __bang_mul(nram_temp, nram_a_vec, nram_b_vec, m * k);
  __bang_add(nram_C, nram_C, nram_temp, m * k);
}
```

### 设计思路
1. **外积分解**：将矩阵乘法分解为外积的累加
   - `C = A * B = sum_k (A[:,k] ⊗ B[k,:])`
   - 其中 `A[:,k] ⊗ B[k,:]` 是外积运算，结果是一个 [M, K] 矩阵
2. **广播和复制**：将向量广播/复制为矩阵，以便使用向量化 API
   - `A[:,k]` 广播为 [M, K] 矩阵（每列相同）
   - `B[k,:]` 复制为 [M, K] 矩阵（每行相同）
3. **逐元素乘法和累加**：使用 `__bang_mul` 计算逐元素乘积，使用 `__bang_add` 累加到结果

### 适用场景
- 理解向量化 API 的使用方式
- 学习外积方法
- 理解向量化带来的性能提升

### 探索任务

#### 2.1 外积方法理解
- **任务**：手动计算一个 2×2 矩阵乘法的外积分解
  ```
  A = [[1, 2],     B = [[5, 6],
       [3, 4]]          [7, 8]]
  C = A * B = A[:,0] ⊗ B[0,:] + A[:,1] ⊗ B[1,:]
  ```
- **验证**：分解计算后验证结果是否正确
- **思考**：外积方法为什么可以用向量化 API 实现？相比三重循环有什么优势？

#### 2.2 向量化 API 的使用
- **任务**：对比 `matmul_00.mlu` 和 `matmul_01.mlu` 的核心计算部分
- **观察点**：
  - 三重循环 vs 二重循环（外层循环）
  - 标量乘法 vs `__bang_mul`（向量化乘法）
  - 标量加法 vs `__bang_add`（向量化加法）
- **思考**：向量化 API 能带来多大的性能提升？为什么？

#### 2.3 广播和复制的开销
- **任务**：分析外积方法的广播和复制开销
- **观察点**：
  - 每次循环需要广播 A 列向量：M 次 `__bang_write_value`
  - 每次循环需要复制 B 行向量：M 次 `__memcpy`
  - 广播和复制的总数据量：M × K × sizeof(float) × 2 × N
- **思考**：广播和复制的开销是否抵消了向量化的优势？如何优化？

#### 2.4 内存访问模式对比
- **任务**：对比 `matmul_00.mlu` 和 `matmul_01.mlu` 的内存访问模式
- **观察点**：
  - matmul_00：B 矩阵按列访问（步长为 k）
  - matmul_01：B 矩阵每次只访问一行（`B[k_idx, :]`），连续访问
- **思考**：matmul_01 是否优化了 B 矩阵的访问？为什么？

#### 2.5 与朴素实现的性能对比
- **任务**：运行两个版本，对比性能
- **观察点**：
  - 执行时间对比
  - 理论分析：matmul_01 使用了向量化 API，应该更快
- **思考**：向量化 API 的性能提升来自哪里？是否有其他因素影响性能？

---

## 3. 专用算子实现 - MatMul API（`matmul_02.mlu`）

**特点**：
- 使用 `__bang_matmul` 专用矩阵乘法 API
- 数据类型：int16_t 输入，float 输出（半精度输入，单精度输出）
- 规模：128×128×128 中等矩阵，一次性加载到 NRAM
- **不使用 tiling**：假设数据量小到可以放入 NRAM
- **布局要求**：右矩阵 B 必须是列主序布局

### 核心实现
```c
// 将右矩阵从行主序转换为列主序
for (int i = 0; i < k; ++i) {
  for (int j = 0; j < n; ++j) {
    nram_src1_col[j * k + i] = nram_src1[i * n + j];
  }
}

// 将列主序右矩阵复制到 WRAM
__memcpy(wram_src1, nram_src1_col, k * n * sizeof(int16_t), NRAM2WRAM);

// 使用 __bang_matmul 计算矩阵乘法
__bang_matmul(nram_dst, nram_src0, wram_src1, m, k, n, FIX_POSITION);
```

### 设计思路
1. **数据类型转换**：使用 int16_t 输入可以减少内存带宽需求
2. **布局转换**：`__bang_matmul` 要求右矩阵是列主序，需要进行转置
   - 行主序：`B[i][j] = B[i * n + j]`
   - 列主序：`B[j][i] = B[j * k + i]`
3. **WRAM 使用**：将右矩阵存储在 WRAM（写回 RAM），优化内存访问
4. **专用 API**：`__bang_matmul` 是专门为矩阵乘法优化的 API

### API 参数说明
```c
__bang_matmul(dst, src0, src1, m, k, n, fix_position)
```
- `dst`: 输出矩阵 [M, N]（float）
- `src0`: 左矩阵 [M, K]（行主序，int16_t）
- `src1`: 右矩阵 [K, N]（列主序，int16_t）
- `m, k, n`: 矩阵维度
- `fix_position`: 固定点位置（用于 int16_t 计算）

### 适用场景
- 学习专用矩阵乘法 API 的使用
- 理解数据类型和布局的影响
- 学习 WRAM 的使用

### 探索任务

#### 3.1 数据类型的影响
- **任务**：对比 `matmul_01.mlu`（float32）和 `matmul_02.mlu`（int16_t）
- **观察点**：
  - 内存使用量：int16_t 是 float32 的一半
  - 精度损失：int16_t 可能会损失精度
  - 性能差异：int16_t 可以减少内存带宽需求
- **思考**：什么情况下使用 int16_t？什么情况下必须使用 float32？

#### 3.2 列主序布局理解
- **任务**：理解行主序和列主序的区别
- **观察点**：
  - 行主序：`matrix[i][j] = matrix[i * n + j]`
  - 列主序：`matrix[j][i] = matrix[j * m + i]`
  - 转置操作：`B_col[j][i] = B_row[i][j]`
- **思考**：为什么 `__bang_matmul` 要求右矩阵是列主序？这有什么优势？

#### 3.3 WRAM 的使用
- **任务**：理解 WRAM（Write RAM）的作用
- **硬件背景**：
  - NRAM：核内高速 RAM，容量小（768KB），访问速度快
  - WRAM：写回 RAM，容量大，访问速度稍慢
  - SRAM：静态 RAM，容量小，访问速度快
- **思考**：为什么右矩阵使用 WRAM？为什么不能都用 NRAM？

#### 3.4 fix_position 参数理解
- **任务**：理解 fix_position 参数的作用
- **观察点**：
  - fix_position=0：不进行定点缩放
  - fix_position>0：进行定点缩放（类似乘以 2^fix_position）
- **思考**：什么时候需要使用 fix_position？如何选择合适的值？

#### 3.5 与向量化实现的对比
- **任务**：对比 `matmul_01.mlu` 和 `matmul_02.mlu`
- **观察点**：
  - 实现复杂度：matmul_02 更简单（直接调用 API）
  - 性能：matmul_02 应该更快（专用 API）
  - 灵活性：matmul_01 更灵活（可以自定义算法）
- **思考**：什么时候使用通用 API（如 __bang_mul），什么时候使用专用 API（如 __bang_matmul）？

---

## 4. Tiling 实现 - MatMul API（`matmul_03.mlu`）

**特点**：
- 使用 `__bang_matmul` 专用矩阵乘法 API
- 数据类型：int16_t 输入，float 输出
- 规模：1024×1024×1024 大矩阵
- **三层 Tiling**：支持大矩阵的分块计算
- **多任务并行**：使用多个任务并行计算

### 核心实现
```c
// 三层 Tiling 结构
// Level 1: 按 K 维度分块（输出列维度）
for (int tile_k = 0; tile_k < (k + TILE_SIZE - 1) / TILE_SIZE; tile_k++) {
  int k_start = tile_k * TILE_SIZE;
  int k_size = min(TILE_SIZE, k - k_start);
  
  // Level 2: 按 M 维度分块（输出行维度）+ 任务并行
  for (int loop = 0; loop < LOOPS; loop++) {
    int m_start = m_per_block * taskId + loop * m_per_loop;
    
    // Level 3: 按 N 维度分块（规约维度）
    for (int tile_n = 0; tile_n < (n + TILE_SIZE - 1) / TILE_SIZE; tile_n++) {
      int n_start = tile_n * TILE_SIZE;
      int n_size = min(TILE_SIZE, n - n_start);
      
      // 加载 A tile [M_PER_LOOP, n_size]
      __memcpy_async(nram_a, a + m_start * n + n_start, ...);
      
      // 加载 B tile [n_size, k_size]（列主序）
      // ... 布局转换代码 ...
      
      // 计算并累加：C_tile += A_tile * B_tile
      __bang_matmul(nram_c, nram_a, wram_b, M_PER_LOOP, n_size, k_size, FIX_POSITION);
    }
    
    // 写回结果
    __memcpy(dst + m_start * k + k_start, nram_c, ...);
  }
}
```

### 三层 Tiling 策略

#### Level 1：K 维度分块（输出列维度）
- 将输出矩阵 C 的列维度 K 切分为多个 tile
- 每个 tile 大小为 `TILE_SIZE`（默认 128）
- 减少每次迭代的内存访问量

#### Level 2：M 维度分块（输出行维度）+ 任务并行
- 将输出矩阵 C 的行维度 M 切分为多个任务块
- 每个任务块处理 `M_PER_BLOCK` 行（`M_PER_LOOP * LOOPS = 128` 行）
- 使用 `taskId` 分配任务，实现多任务并行
- 每个 task 处理 `m_per_block` 行

#### Level 3：N 维度分块（规约维度）
- 将规约维度 N 切分为多个 tile
- 每个 tile 大小为 `TILE_SIZE`
- 通过累加完成规约（`C_tile += A_tile * B_tile`）
- `__bang_matmul` 会自动累加到目标缓冲区

### 关键参数

```c
#define TILE_SIZE 128      // K 和 N 维度的 tile 大小
#define M_PER_LOOP 16     // 每次内层循环处理的 M 行数
#define LOOPS 8           // 外层循环的次数
#define M_PER_BLOCK (M_PER_LOOP * LOOPS) // 每个 task 处理的 M 行数（128）
```

### 适用场景
- 学习 Tiling 技术
- 学习多任务并行
- 理解如何处理大矩阵

### 探索任务

参考文档：[Cambricon BANG C/C++ 编程指南 - 任务映射](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/programming_model/index.html)

#### 4.1 三层 Tiling 策略理解
- **任务**：理解三层循环的作用
- **观察点**：
  - Level 1（K 维度）：为什么需要按 K 维度分块？减少了什么？
  - Level 2（M 维度）：为什么需要按 M 维度分块？如何实现任务并行？
  - Level 3（N 维度）：为什么需要按 N 维度分块？如何实现累加？
- **思考**：为什么需要三层循环？能否只用一层或两层？

#### 4.2 任务并行理解
- **任务**：理解任务并行的实现方式
- **观察点**：
  - Host 端：`cnrtDim3_t dim = {8, 1, 1};` 启动 8 个任务
  - Kernel 端：`taskId` 用于分配任务
  - 任务分配：`m_start = m_per_block * taskId + loop * m_per_loop`
- **思考**：如何计算任务数量？如何保证负载均衡？任务数量和硬件核心数的关系？

#### 4.3 Tiling 大小调优
- **任务**：尝试修改 tiling 参数，观察性能变化
- **实验**：
  - 修改 `TILE_SIZE`（如 64、256、512），观察性能和 NRAM 使用量
  - 修改 `M_PER_LOOP`（如 8、32、64），观察性能
  - 修改 `LOOPS`（如 4、16、32），观察任务粒度和性能
- **思考**：如何根据矩阵大小选择最优的 tiling 参数？这些参数之间有什么关系？

#### 4.4 内存访问优化
- **任务**：分析 Tiling 实现的内存访问模式
- **观察点**：
  - A 矩阵：按 tile 加载，每次只加载需要的部分
  - B 矩阵：按 tile 加载（列主序），存储在 WRAM
  - C 矩阵：按 tile 写回，减少写回次数
- **思考**：Tiling 如何减少内存访问？为什么 B 矩阵使用 WRAM？

#### 4.5 与非 Tiling 实现的对比
- **任务**：对比 `matmul_02.mlu` 和 `matmul_03.mlu`
- **观察点**：
  - matmul_02：一次性加载所有数据，适合小矩阵
  - matmul_03：分块加载，适合大矩阵
  - 实现复杂度：matmul_03 更复杂
  - 性能：大矩阵时 matmul_03 应该更快
- **思考**：什么时候需要使用 Tiling？Tiling 的开销是什么？

#### 4.6 任务类型调优
- **任务**：尝试将 `cnrtFuncTypeBlock` 改为 `cnrtFuncTypeUnion1`
- **观察点**：
  - 是否能正常编译和运行？
  - 如果出现 `CN_ERROR_INVALID_VALUE` 错误，检查对齐要求
  - 性能是否有提升或下降？
- **思考**：对于矩阵乘法这种计算密集型操作，Block 和 Union 哪种更合适？为什么？

---

## 5. 专用算子实现 - Conv API（`matmul_04.mlu`）

**特点**：
- 使用 `__bang_conv` 卷积 API 实现矩阵乘法
- 数据类型：float32 输入和输出
- 规模：64×64×64 小矩阵
- **不使用 tiling**：假设数据量小到可以放入 NRAM
- **卷积映射**：将矩阵乘法映射为卷积操作
- **全 float 支持**：支持纯 float32 的矩阵乘法（A、B、C 都是 float）

### 核心实现
```c
// 使用 __bang_conv 计算矩阵乘法
// Conv: out[k, m, 0] = sum_n src[n, m, 0] * kernel[n, k]
// Matmul: C[m, k] = sum_n A[m, n] * B[n, k]
// 映射关系：
//   - src[n, m, 0] = A[m, n]
//   - kernel[n, k] = B[n, k]
//   - out[k, m, 0] = C[m, k]
__bang_conv(nram_result, nram_A, wram_B,
           n, m, 1,           // channel_input=n, height=m, width=1
           1, 1, 1, 1, k);    // kernel_height=1, kernel_width=1, ..., channel_output=k
```

### 设计思路
1. **为什么使用 `__bang_conv` 而不是 `__bang_matmul`？**

   这是本实现的一个关键设计决策。查看 [`__bang_matmul` API 文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Matrix%20Multiplication%20Functions.html?highlight=matmul#_CPPv413__bang_matmulPfPKfPKfjjj)，我们会发现：

   - `__bang_matmul` 支持的数据类型组合有限，**不支持全 float32 的矩阵乘法**
   - 例如，对于 `src0=float, src1=float, dst=float` 的组合，API 文档中没有列出
   - 支持的组合包括：`int16×int16→float`、`int8×int8→float`、`half×half→float` 等，但缺少 `float×float→float`
   - 这意味着如果我们的输入 A、B 和输出 C 都是 float32 类型，**无法使用 `__bang_matmul`**

   对于常见的 `C = A * B + C`（矩阵乘法加偏置）场景：
   - A、B、C 都是 float32
   - `__bang_matmul` 不支持这种全 float 的组合
   - 因此需要使用 `__bang_conv` 来实现，它支持全 float32 类型

   **设计原则**：
   - 如果输入是 int16/half/int8 等低精度类型 → 优先使用 `__bang_matmul`（性能最优）
   - 如果输入是 float32 且需要保持精度 → 使用 `__bang_conv`（功能完整）
   - 根据实际应用的数据类型和精度要求选择合适的 API

2. **卷积映射**：将矩阵乘法映射为卷积操作
   - A 矩阵 [M, N] → 卷积输入特征图 [channel_input=N, height=M, width=1]
   - B 矩阵 [N, K] → 卷积核 [kernel_height=1, kernel_width=1, channel_output=K]
   - C 矩阵 [M, K] → 卷积输出 [channel_output=K, height=M, width=1]
2. **卷积公式**：`out[k, m, 0] = sum_n src[n, m, 0] * kernel[n, k]`
   - 对应矩阵乘法公式：`C[m, k] = sum_n A[m, n] * B[n, k]`
3. **输出转置**：卷积输出是 [K, M]（转置的），需要转置回 [M, K]

### API 参数说明
```c
__bang_conv(dst, src, kernel, 
            channel_input, height, width,
            kernel_height, kernel_width, stride_width, stride_height, 
            channel_output)
```
- `dst`: 输出特征图 [channel_output, height, width]
- `src`: 输入特征图 [channel_input, height, width]
- `kernel`: 卷积核 [kernel_height × kernel_width, channel_output, channel_input]
- `channel_input`: 输入通道数（对应 N）
- `height`: 特征图高度（对应 M）
- `width`: 特征图宽度（对应 1）
- `kernel_height, kernel_width`: 卷积核大小（都是 1）
- `stride_width, stride_height`: 步长（都是 1）
- `channel_output`: 输出通道数（对应 K）

### 适用场景
- 学习卷积 API 的使用
- 理解矩阵乘法和卷积的关系
- 探索不同的实现方式

### 探索任务

#### 5.1 矩阵乘法到卷积的映射
- **任务**：理解如何将矩阵乘法映射为卷积操作
- **观察点**：
  - A 矩阵 [M, N] → 卷积输入 [N, M, 1]
  - B 矩阵 [N, K] → 卷积核 [1×1, K, N]
  - C 矩阵 [M, K] → 卷积输出 [K, M, 1]
- **思考**：为什么 width=1？为什么 kernel 大小是 1×1？这种映射方式如何实现矩阵乘法？

#### 5.2 输出转置问题
- **任务**：理解卷积输出是转置的
- **观察点**：
  - 卷积输出：[K, M, 1]
  - 期望输出：[M, K]
  - 索引映射：`C[m, k] = out[k, m, 0]`
- **思考**：为什么卷积输出是转置的？如何处理这个转置？验证函数是否会自动处理？

#### 5.3 与 MatMul API 的对比
- **任务**：对比 `matmul_02.mlu`（`__bang_matmul`）和 `matmul_04.mlu`（`__bang_conv`）
- **观察点**：
  - matmul_02：专用矩阵乘法 API，直接计算
  - matmul_04：通用卷积 API，通过映射实现
  - 数据类型：matmul_02 使用 int16_t，matmul_04 使用 float32
  - 布局要求：matmul_02 要求右矩阵列主序，matmul_04 不要求
- **关键差异 - 数据类型支持**：
  - 查看 [`__bang_matmul` API 文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Matrix%20Multiplication%20Functions.html?highlight=matmul#_CPPv413__bang_matmulPfPKfPKfjjj)
  - `__bang_matmul` **不支持** `float×float→float` 的组合
  - 支持的组合包括：`int16×int16→float`、`int8×int8→float`、`half×half→float` 等
  - 这意味着如果 A、B、C 都是 float32，**必须使用 `__bang_conv`**
  - 常见场景：`C = A * B + C`（带偏置的矩阵乘法），A、B、C 都是 float
- **性能对比**：
  - `__bang_matmul`：专用硬件加速，性能通常最优（在支持的类型下）
  - `__bang_conv`：通用卷积操作，性能稍差，但功能更灵活
- **设计决策**：
  - 如果数据是 int16/half/int8 → 优先使用 `__bang_matmul`
  - 如果数据是 float32 → 使用 `__bang_conv`
  - 如果需要混合精度（如 float32 输入 + half 偏置）→ 可能需要拆分操作
- **思考**：
  - 两种实现方式各有什么优缺点？
  - 在实际项目中，如何根据数据类型选择 API？
  - 如果输入是 float32 但又想用 `__bang_matmul`，有什么解决方案？（提示：类型转换，但会损失精度）

#### 5.4 卷积 API 的理解
- **任务**：理解 `__bang_conv` API 的参数
- **观察点**：
  - channel_input: 输入通道数
  - height, width: 输入特征图尺寸
  - kernel_height, kernel_width: 卷积核大小
  - stride_width, stride_height: 步长
  - channel_output: 输出通道数
- **思考**：这些参数如何映射到矩阵乘法？能否用卷积实现更复杂的操作？

#### 5.5 性能对比
- **任务**：对比不同实现的性能
- **观察点**：
  - matmul_01（`__bang_mul` + `__bang_add`）
  - matmul_02（`__bang_matmul`）
  - matmul_04（`__bang_conv`）
- **思考**：哪种实现方式最快？为什么？性能差异来自哪里？

---

## 6. 运行脚本（`build_eval.sh`）

运行脚本提供了完整的编译和执行环境，包含必要的环境变量设置和编译命令。

### 6.1 环境变量配置

脚本开头设置了以下环境变量，这些是运行 BangC 程序所必需的：

- **`NEUWARE_HOME=/usr/local/neuware`**: 指定 Neuware SDK 的安装路径，Neuware 是寒武纪 MLU 的开发工具包
- **`LD_LIBRARY_PATH`**: 添加 Neuware 的库文件路径（`$NEUWARE_HOME/lib64`），确保运行时能找到 MLU 相关的动态链接库
- **`PATH`**: 添加 Neuware 的二进制工具路径（`$NEUWARE_HOME/bin`），使 `cncc` 编译器可以直接调用
- **`MLU_VISIBLE_DEVICES=0`**: 指定使用第 0 号 MLU 设备（在多卡环境下可以选择其他设备）
- **`TORCH_DEVICE_BACKEND_AUTOLOAD=0`**: 禁用 PyTorch 的设备后端自动加载，避免与 BangC 运行时冲突

### 6.2 使用方法

脚本接受一个参数：`.mlu` 源文件的文件名。

```bash
# 编译并运行不同版本的矩阵乘法
./build_eval.sh matmul_00.mlu
./build_eval.sh matmul_01.mlu
./build_eval.sh matmul_02.mlu
./build_eval.sh matmul_03.mlu
./build_eval.sh matmul_04.mlu
```

脚本会：
1. 自动切换到脚本所在目录（`Experiments/05_matmul`）
2. 使用 `cncc` 编译器编译 `.mlu` 文件，生成可执行文件
3. 执行生成的可执行文件并输出结果

### 6.3 编译参数说明

脚本使用的编译命令：
```bash
cncc "${MLU_SOURCE}" -o "${TARGET}" --bang-mlu-arch=mtp_592 -O3 -lm
```

- `--bang-mlu-arch=mtp_592`: 指定目标 MLU 架构为 mtp_592
- `-O3`: 最高级别的优化
- `-lm`: 链接数学库

---

## 7. 常见问题解答

### Q1: 为什么需要这么多不同版本的实现？

**A**: 每个版本都展示了不同的优化技术和设计思路：

- **matmul_00**: 最朴素的三重循环，作为理解基础
- **matmul_01**: 展示向量化 API 的使用（外积方法）
- **matmul_02**: 展示专用矩阵乘法 API 的使用
- **matmul_03**: 展示 Tiling 技术和多任务并行
- **matmul_04**: 展示如何用卷积 API 实现矩阵乘法

通过对比不同版本，可以理解各种优化技术的优缺点和适用场景。

### Q2: 什么时候使用 `__bang_matmul`，什么时候使用 `__bang_conv`？

**A**: 选择哪种 API 主要取决于**数据类型**和**精度要求**：

#### 数据类型支持

查看 [`__bang_matmul` API 文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Matrix%20Multiplication%20Functions.html?highlight=matmul#_CPPv413__bang_matmulPfPKfPKfjjj)，支持的数据类型组合包括：

- ✅ `int16 × int16 → float`
- ✅ `int8 × int8 → float`
- ✅ `half × half → float`
- ❌ **不支持** `float × float → float`

**关键问题**：`__bang_matmul` **不支持全 float32 的矩阵乘法**！

#### 常见场景：`C = A * B + C`

在深度学习中，矩阵乘法经常需要加上偏置（bias）：
- A [M, N]: float32
- B [N, K]: float32
- C [M, K]: float32（既是输出，也是偏置）
- 计算：`C = A * B + C`

这种场景下：
- ❌ `__bang_matmul`：不支持（A、B、C 都是 float32）
- ✅ `__bang_conv`：支持全 float32

#### 选择建议

| 场景 | 推荐使用 | 原因 |
|------|---------|------|
| 输入是 int16/half/int8 | `__bang_matmul` | 性能最优，硬件加速 |
| 输入是 float32 | `__bang_conv` | `__bang_matmul` 不支持 |
| 需要混合精度（如 float32 输入 + half 偏置） | 需要拆分操作 | 先用 `__bang_conv` 计算矩阵乘法，再用 `__bang_add` 加偏置 |
| 对性能要求极高 | `__bang_matmul`（如果支持数据类型） | 专用硬件加速 |
| 需要灵活性 | `__bang_conv` | 支持更多数据类型组合 |

#### 性能对比

- **`__bang_matmul`**：
  - 优点：专用的矩阵乘法硬件加速，性能通常最优
  - 缺点：数据类型支持有限，要求右矩阵列主序

- **`__bang_conv`**：
  - 优点：通用的卷积 API，支持全 float32，布局要求低
  - 缺点：性能可能稍差于专用 API

#### 实际应用建议

在实际项目中，建议：
1. **优先使用 `__bang_matmul`**：如果输入是 int16/half/int8，性能最优
2. **必要时使用 `__bang_conv`**：如果输入是 float32 且不能转换类型
3. **考虑类型转换的代价**：将 float32 转换为 int16 可以使用 `__bang_matmul`，但会损失精度
4. **查看官方文档**：使用前查看 [`__bang_matmul` API 文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Matrix%20Multiplication%20Functions.html?highlight=matmul#_CPPv413__bang_matmulPfPKfPKfjjj)，确认数据类型是否支持

### Q3: 为什么 Tiling 需要三层循环？

**A**: 三层循环分别对应矩阵乘法的三个维度：

- **Level 1（K 维度）**：减少每次迭代的内存访问量
- **Level 2（M 维度）**：实现任务并行，多任务同时计算不同的行块
- **Level 3（N 维度）**：实现累加，将部分和累加到最终结果

每层循环都有特定的优化目标，缺一不可。

### Q4: 为什么 B 矩阵使用 WRAM？

**A**: WRAM（Write RAM）有以下优势：

- 容量更大，可以存储更大的分块
- 与 NRAM 配合使用，优化内存层次
- 对于 B 矩阵这种需要频繁访问的数据，放在 WRAM 可以减少 NRAM 压力

### Q5: 如何选择最优的 tiling 参数？

**A**: 选择 tiling 参数需要考虑：

- **NRAM 容量**：确保所有 NRAM 缓冲区不超过 NRAM 容量
- **计算效率**：tile 太小会增加循环开销，太大会增加 NRAM 压力
- **任务粒度**：每个任务的工作量要适中，避免负载不均衡
- **硬件特性**：考虑硬件的并行能力和缓存特性

通常需要通过实验找到最优参数组合。

### Q6: 为什么有些版本使用 int16_t，有些使用 float32？

**A**: 数据类型的选择需要权衡：

- **int16_t**：
  - 优点：减少内存带宽需求，可能提升性能
  - 缺点：可能损失精度，需要考虑定点缩放（fix_position）
  - 适用：对精度要求不高、追求性能的场景

- **float32**：
  - 优点：精度高，不需要考虑定点缩放
  - 缺点：内存带宽需求大，可能影响性能
  - 适用：对精度要求高的场景

---

## 8. 学习资源

### 官方文档
- [Cambricon BANG C/C++ 编程指南 - 硬件实现](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/hardware_implementation/index.html)
- [Cambricon BANG C/C++ 编程指南 - 任务映射](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/programming_guide_1.7.0/programming_model/index.html)
- [BangC API 参考](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/api_reference/index.html)
- [__bang_matmul API 文档](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Matrix%20Multiplication%20Functions.html?highlight=matmul#_CPPv413__bang_matmulPfPKfPKfjjj)

## 9. 总结

本教程通过五个版本的矩阵乘法实现，展示了从基础到高级的完整学习路径：

1. **朴素实现**：理解基础，建立基准
2. **向量化实现**：学习 API 优化
3. **专用算子（MatMul）**：学习专用 API 和数据类型优化
4. **Tiling 实现**：学习内存优化和并行优化
5. **专用算子（Conv）**：学习不同的实现方式

通过循序渐进的学习，可以掌握 BangC 编程的核心技术和优化方法。建议按照学习流程逐步完成每个版本的探索任务，并通过对比不同版本的理解优化技术的应用场景和效果。

**关键要点**：
- 矩阵乘法是计算密集型操作，有多种优化方式
- API 层、算法层、内存层、并行层都可以进行优化
- Tiling 是处理大矩阵的关键技术
- 理解硬件特性（NRAM、WRAM、任务并行）是优化的基础
- 通过实验和对比，可以找到最优的实现方式
