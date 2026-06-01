# Torch_MLU 卷积算子 (Convolution) 源码实现与精度策略调研

## 1. 算子分发与函数分配链路 (Function Dispatch)

在 `torch_mlu` 中，针对卷积计算（Conv2d, Conv3d等），原生的 PyTorch API （如 `torch.nn.Conv2d`、`torch.nn.functional.conv2d`）均会在 ATen 层对应到底层的 `at::convolution` 或 `at::_convolution` 操作。

针对 MLU 硬件平台，函数的拦截与分配按照如下链条进行解析：

1. **自动代码生成注册 (Codegen Registry)**
   * **源码位置：** `torch_mlu/codegen/mlu_functions.yaml`
   * 构建编译时，自动代码生成引擎会捕获定义在该文件中的 `- func: _convolution` 和 `- func: convolution_backward`，将其与 ATen Dispatcher（分发器）绑定。
   
2. **顶层入口 (Top-level Wrapper)**
   * **源码位置：** `torch_mlu/csrc/aten/operators/cnnl/convolution.cpp`
   * ATen 分发至 MLU 后，调用统一入口函数 `cnnl__convolution`（第 44 行起）。
   * `cnnl__convolution` 承担了参数解析、维度的升降（如 1D 变 2D 卷积代理）、并根据是否为转置卷积决定分支。此方法最终调用：
     * `cnnl_convolution_normal_forward` (常规卷积)
     * `cnnl_convolution_transpose_forward` (转置卷积)

3. **CNNL 算子内部分发 (Internal API)**
   * **源码位置 (前向为例)：** `torch_mlu/csrc/aten/operators/cnnl/convolution_forward.cpp`
   * 进一步拆解 optional 数据并在底层分派入 `internal` 计算层函数 `cnnl_convolution_forward_internal`。

4. **CNNL 执行层 (Under-the-hood Kernel Launch)**
   * **源码位置：** `torch_mlu/csrc/aten/operators/cnnl/internal/convolution_forward_internal.cpp`
   * 这里负责配置底层的算子描述符 (`CnnlConvolutionDescriptor`)，调用 `cnnlGetConvolutionForwardWorkspaceSize` 计算和分配显存缓存，并最终执行系统闭源库函数 `cnnlConvolutionForward`。

## 2. Float16 (Half) 精度计算路径详解

以两个原生具有 `torch.float16` 格式属性信息的 Tensor 做 Convolution 运算为例，`torch_mlu` 的卷积对半精度的处理包含了细致的精度“安全网”（即内部自动启用伪混合计算（Pseudo-Mixed Compute）），核心逻辑同样位于最后一步分发的 internal 函数中。

**源码追踪：** `torch_mlu/csrc/aten/operators/cnnl/internal/convolution_forward_internal.cpp`（约 52～63 行）

### 2.1 运算累加器类型提升 (Compute Type Promotion)

程序会获取输出与输入张量映射到 CNNL 库层级的数据类型，如果侦测到使用 Float16，系统会将内部计算数据格式强拉至 Float32（FP32），确保累加不剧烈溢出：
```cpp
auto output_cnnl_type = getCnnlDataType(output.scalar_type());
auto input_scalar_type = input.scalar_type();
auto input_cnnl_type = getCnnlDataType(input_scalar_type);
auto weight_cnnl_type = getCnnlDataType(weight.scalar_type());

// 检查是否为半精度 或 BF16 类型
const bool promote_compute_dtype =
    (output_cnnl_type == CNNL_DTYPE_HALF ||
     output_cnnl_type == CNNL_DTYPE_BFLOAT16);

// 如果是半精度，底层引擎使用的 compute_dtype 累加器会被置为 FP32 
auto compute_dtype =
    promote_compute_dtype ? CNNL_DTYPE_FLOAT : output_cnnl_type;
```

### 2.2 防干扰机制 (TF32 Bypass)

一旦系统检测到张量输入标量类型**不为**常规的全精度 (`at::kFloat`)，框架强制覆写关闭 `allow_tf32` 的开关：
```cpp
// Modify allow_tf32 based on input tensor scalar type
if (input_scalar_type != at::kFloat)
  allow_tf32 = false;
```
这一步保证了即便系统环境变量（或者 PyTorch context）全局允许基于 TF32 做加速，针对原生 FP16 的卷积运算也不会遭受 TF32 下舍入切断带来的额外尾数损失。

### 2.3 下放给底层加速器

随后，被提升到 FP32 的精度意图会被挂载在 `CnnlConvolutionDescriptor`（卷积描述结构体）的 `compute_dtype` 属性上：
```cpp
conv_desc.set(
    input_dim,
    stride,
    padding,
    dilation,
    groups,
    compute_dtype,  // <--- 此时已被推演为了 CNNL_DTYPE_FLOAT (FP32)
    allow_tf32);    // <--- 此时被安全限制成了 false
```
并由此执行最终硬件底层 `cnnlConvolutionForward` 调用。

**最终形态总结**：与 MatMul 的处理链相一致，`torch.float16` 格式的输入/权重在通过 `Convolution` 计算时，底层 MLU 遵循 **"FP16 张量取值 + 强关闭 TF32 模式 + FP32 数学乘加器 (MAC / Accumulator)"** 的计算链路以兼顾高吞吐与容错高保真度。