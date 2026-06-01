> 所述源码位置：本机的 /torch/src/torch_mlu
```py
def forward(self, src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
return out
```

# `scatter_add_` 在 MLU 上的底层实现调研

经过对 `torch_mlu` 相关源码的调研，`scatter_add_` 算子在 MLU 上的底层依赖是 **CNNL**（并非 BANGC）实现的。

## 详细调用路径及源码位置

1. **ATen 算子分发入口 (ATen Dispatch)**
   在 Python 前端调用的 `out.scatter_add_` 经过 PyTorch 内部的 Dispatcher 路由到了对于 PrivateUse1 (MLU 设备) 注册的 Kernel。此处底层对应的方法名是 `scatter_add_stub`。

2. **MLU 后端 Kernel 注册**
   - **源码位置**：`torch_mlu/csrc/aten/operators/cnnl/scatter.cpp`
   - 在这个文件中，有 `REGISTER_PRIVATEUSE1_DISPATCH(scatter_add_stub, &scatter_add_mlu_kernel);` 完成了算子注册，核心执行逻辑交由 `scatter_add_mlu_kernel`。

3. **核心 Kernel (`scatter_add_mlu_kernel` & `scatter_reduce_mlu_kernel`)**
   - **源码位置**：`torch_mlu/csrc/aten/operators/cnnl/scatter.cpp`
   - `scatter_add_mlu_kernel` 接受输入后，统一被分发到了 `scatter_reduce_mlu_kernel` 中，并指定 `ReductionType::SUM` 作为规约方式。
   - `scatter_reduce_mlu_kernel` 中对 Tensor 格式进行了连续化处理，随后调用内部函数 `cnnl_scatter_internal`，并传入枚举模式 `CNNL_SCATTER_ADD`。

4. **内部接口与底层 API 调用 (`cnnl_scatter_internal`)**
   - **源码位置**：`torch_mlu/csrc/aten/operators/cnnl/internal/scatter_internal.cpp`
   - `cnnl_scatter_internal` 负责执行设备上的最终计算调用。函数里首先借助 `cnnlGetScatterWorkspaceSize` 计算所需的 Workspace 显存大小并进行分配。
   - 最后使用宏 `TORCH_CNNL_CHECK` 调用了底层的 **CNNL C API**：`cnnlScatter_v2`，传入参数和操作类型 `mode` (值为 `CNNL_SCATTER_ADD`)。

**结论**：`scatter_add_` 算子在 MLU 设备上的直接承载框架是 PyTorch ATen 到 CNNL API 的映射，通过 `cnnlScatter_v2` 来实现，属于基于 CNNL 库的实现，而非直接手写的 BANGC 核函数。
