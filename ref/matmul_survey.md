# Torch_MLU 矩阵乘法 (MatMul) 源码实现调研报告
源码位置 /torch/src/torch_mlu

## 1. 底层 API 调用逻辑及文件映射

在 `torch_mlu` 中，前台的 PyTorch 矩阵操作（如 `torch.matmul()`, `torch.mm()`, `torch.bmm()`, `torch.addmm()`, `torch.baddbmm()`）都被分发并映射到了内部的 **CNNL (Cambricon Neuware Neural-network Library)** 库来实现，并没有使用 BANGC 语言去手写基础的矩阵乘内核。

在 `torch_mlu/csrc/aten/operators/cnnl/internal/` 和相关算子文件内，针对不同的应用场景，主要调用了以下几个层级的 CNNL API 模型：

*   **`cnnlMatMul_v2`** 
    应用场景：基础的矩阵乘法和加法（如 `addmm` 操作）。
    **源码位置：** `torch_mlu/csrc/aten/operators/cnnl/internal/addmm_internal.cpp`
    *   在行号 `L43` 的注释中明确了在 MLU 端的调用名为 `cnnlMatMul_v2` （当含 batch 时使用 `cnnlBatchMatMulBCast_v2`）。
    *   实际执行处位于 **L161**，传入预先构造好的 `matmul_desc` 与 `matmul_algo`，完成基础的矩阵运算：
        ```cpp
        TORCH_CNNL_CHECK(cnnlMatMul_v2(
            handle,
            matmul_desc.desc(),
            matmul_algo.algo(), ...));
        ```

*   **`cnnlMatMulEx`**
    应用场景：扩展版本的矩阵算法，支持更复杂的操作融合（如带 Bias 或 Epilogue，将 `add` 和 `mm` 算子合并后执行）。
    **源码位置：** `torch_mlu/csrc/aten/operators/cnnl/internal/addmm_bias_internal.cpp`
    *   第 **L76-L79** 声明了 `CnnlMatmulExDescriptor` 描述符和相关前置结构。
    *   在第 **L195** 处完成最终的融合算法调用：
        ```cpp
        TORCH_CNNL_CHECK(cnnlMatMulEx(
            handle,
            matmul_desc.desc(),
            &alpha, ...));
        ```

## 2. 精度策略 (TF32 支持与默认值)

关于默认计算精度，`torch_mlu` 在矩阵运算中**默认使用最严格的高计算精度 (Highest，即关闭 TF32 加速，只用真实 FP32)**。该行为直接对齐了 PyTorch 上游的整体设计（自 PyTorch 1.12 的精度策略）。

### 检查与控制精度的源码逻辑

针对 MatMul 运算精度控制的全局函数实现在：
**源码位置：** `torch_mlu/csrc/utils/common.cpp`， 第 **L71 - L80** 中定义的 `Global::allowTF32CnMatMul()` :

```cpp
// L71 
bool Global::allowTF32CnMatMul() const {
  static const bool allow_tf32_cnmatmul_override = []() {
    const std::vector<std::string> TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE = {
        "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"};
    // L76: 允许通过环境变量改变这一默认配置，默认值为 false
    return getCvarBool(TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE, false);  
  }();
  // L78: 捕捉上游原生 PyTorch 维护的 matmul 精度级别
  auto float32_matmul_precision = at::globalContext().float32MatmulPrecision();
  
  // L79: 若环境未强行覆盖，只有当精度设置不等于 HIGHEST 时才开启 TF32
  return allow_tf32_cnmatmul_override ||
      float32_matmul_precision != at::Float32MatmulPrecision::HIGHEST;
}
```

### 结合到算子层的传递（以 `baddbmm` 为例）
**源码位置：** `torch_mlu/csrc/aten/operators/cnnl/baddbmm.cpp` 
*   **L170** 可以看到获取该精度允许标记位的组合逻辑：
    ```cpp
    bool allow_tf32 = !at::NoTF32Guard::should_disable_tf32() && 
                      torch_mlu::Global::instance().allowTF32CnMatMul();
    ```
随后这个 `allow_tf32` 的 bool 值将被向下透传，并挂载到底层的 CNNL 描述符属性上。

## 3. 实现形态总结：CNNL 算子库 vs 手写 BANGC

**结论：针对矩阵乘法相关算子，使用的是纯粹的 CNNL API 集成，不存在独立编写底层 BANGC ( MLU 手写 Kernel ) 的实现。**

1. 所有底层分发调用最终都落在 `CnnlMatmulDescriptor` (定义在 `aten/cnnl/cnnlOpDescriptors.h`)，随后直接将 `desc()` 句柄抛给 `cnnlMatMul_v2` / `cnnlMatMulEx` 执行。
2. 虽然在项目 `torch_mlu/csrc/aten/operators/bang/internal/` 目录下发现了若干 `.mlu`（寒武纪算子语言 BANGC 手写的模块），但其覆盖面主要是：
   * 细碎的 Fused Optimizer（如 `fused_adam_internal.mlu`, `fused_lamb_internal.mlu`）
   * AMP / 混合精度下的 Scale 更新（如 `amp_non_finite_and_unscale_internal.mlu`）
   * 并没有实现自定义的矩阵乘 GEMM / MatMul 结构。

综上，在 `torch_mlu` 当中关于矩阵乘法的调用栈非常清晰上浮：**PyTorch Tensor -> ATen 分发 -> 初始化 CNNL 描述符层 -> 设置基于 PyTorch Context 的精度标准 (默认 TF32 关) -> 移交寒武纪 CNNL 闭源库执行。**

## 4. Float16 (Half) 输入的混合精度与累加计算处理链条

针对用户传入两个 `torch.float16` 数据类型的 Tensor 的情况，`torch_mlu` 在底层做了专门的**精度提升（类型提升 / Promote）与累加器数据类型（Accumulate Type）设定**以防止计算溢出，其详细处理链条如下：

### 4.1 累加器精度映射 (Accumulate Type)
即使输入是 `Float16` (Half)，在执行底层的乘加累积操作以及使用 scaling factor（即 $\alpha$ 和 $\beta$ 参数）时，都会强制提升操作精度到 `Float32` (FP32) 进行运算保护。
**源码位置：** `torch_mlu/csrc/aten/utils/accumulate_type.h`
通过模板偏特化结构 `MLUAccumulateType` 和 `MLUOpMathType` 完成类型映射：
```cpp
// L103: Promote less or equal 16bit to 32bit.
template <>
struct MLUAccumulateType<at::Half> {
  using type = float;
};
```
当传入是 `at::Half` 时，底层的 math 计算精度会被推导为 C++ 原生的 `float`。

### 4.2 MatMul中的缩放参数处理
**源码位置：** `torch_mlu/csrc/aten/operators/cnnl/internal/addmm_internal.cpp`
在分发给 CNNL 前的预处理期间，标量因子使用上述的推导提升精度结构化：
```cpp
// L158
using math_type = MLUAccumulateType_t<scalar_t>;
auto alpha = alpha_.to<math_type>();
auto beta = beta_.to<math_type>();
```
这表明如果 `scalar_t` 是 `Half`，`math_type` 会变成 `float`，后续传递给 `cnnlMatMul_v2` 等 C 接口的 `alpha` 和 `beta` 指针，其内部均会按 32-bit float 解析。

### 4.3 输入张量的 Descriptor 映射
虽然标量及数学计算精度被推成了 FP32，但对矩阵自身的物理内存布局描述，系统仍会通过 `getCnnlDataType(at::Half)` 保留原始的张量大小。`CnnlTensorDescriptor` 会正确标记这是一个半精度输入张量发给 CNNL。因此：
> **Float16 最终的硬件执行流**：
> "FP16 输入 + FP32 内部乘加累加器 (Accumulate) + FP32 $\alpha/\beta$ 缩放 -> 输出"。
> 这一处理逻辑在内部库被严密封装并对齐了诸如 CUDA 平台上 Tensor Cores 处理 FP16 的常规数学计算流（即 pseudo/mixed precision math）。