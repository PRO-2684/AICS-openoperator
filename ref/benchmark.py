#!/usr/bin/env python3
# You can modify this script at will to meet your needs. If you feel that this script is not perfect enough, you can refactor it.
import torch
import torch_mlu
from pathlib import Path
import importlib.util
import sys
import time
from typing import List, Tuple


# blow code registers torch.mlu in torch_mlu.__init__.py:
# torch.utils.rename_privateuse1_backend("mlu")
# torch._register_device_module("mlu", torch_mlu.mlu)
torch.mlu = torch_mlu.mlu  # type: ignore[unresolved-attribute]  # ty:ignore[unresolved-attribute]

# like cuda, async compute device need sync before actual running time benchmarking.
torch.mlu.synchronize()  # ty:ignore[unresolved-attribute]


def load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def is_floating_dtype(dtype):
    return dtype is not None and torch.is_floating_point(torch.empty((), dtype=dtype))


def cast_tree(obj, dtype, device=None):
    if isinstance(obj, torch.Tensor):
        out = obj
        if should_cast_tensor(obj, dtype):
            out = out.to(dtype)
        if device is not None:
            out = out.to(device)
        return out.contiguous()
    if isinstance(obj, list):
        return [cast_tree(x, dtype, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(cast_tree(x, dtype, device) for x in obj)
    return obj


def should_cast_tensor(tensor, dtype):
    if dtype is None:
        return False
    return tensor.is_floating_point() or is_floating_dtype(dtype)


def tolerances(dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-4, 1e-4


def benchmark_fn(
    fn, *args, warmup=10, iters=50, sync=True
) -> Tuple[float, List[float]]:
    """返回 (平均时间(ms), 所有迭代时间列表)"""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
        if sync:
            torch.mlu.synchronize()  # ty:ignore[unresolved-attribute]

    times: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _output = fn(*args)
        if sync:
            torch.mlu.synchronize()  # ty:ignore[unresolved-attribute]
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_time = sum(times) / len(times)
    return avg_time, times


def main():
    if len(sys.argv) < 4:
        print("Usage: python benchmark.py <op_ext_path> <ref_path> [dtype] [opname]")
        sys.exit(1)

    op_ext_path = Path(sys.argv[1])
    ref_path = Path(sys.argv[2])
    dtype_str = sys.argv[3] if len(sys.argv) > 3 else "float32"
    op_name = sys.argv[4]

    # 支持的 dtype 转换
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.float32)
    print(f"Using dtype: {dtype}")

    assert op_ext_path.exists() and ref_path.exists(), "Paths do not exist"

    op_ext_lib = load_module_from_file(
        ("op" if not op_name else op_name), op_ext_path
    )  # you can change the "op" to any op name you compiled
    ref_mod = load_module_from_file("ref", ref_path)

    init_inputs = list(ref_mod.get_init_inputs())
    raw_inputs = list(ref_mod.get_inputs())

    # Reference model
    model = ref_mod.Model(*init_inputs)
    model.eval()

    if is_floating_dtype(dtype):
        model = model.to(dtype=dtype)

    mlu_inputs = cast_tree(raw_inputs, dtype=dtype, device="mlu")
    model_mlu = model.to("mlu")
    init_inputs_mlu = cast_tree(init_inputs, dtype=dtype, device="mlu")

    # ====================== Benchmark ======================
    print("\n" + "=" * 60)
    print("Benchmarking... (MLU)")
    print("=" * 60)

    # 1. Reference model (torch.nn.Module)
    def ref_forward():
        with torch.no_grad():
            return model_mlu(*mlu_inputs)

    ref_avg_ms, ref_times = benchmark_fn(ref_forward, warmup=20, iters=100)
    print(f"Reference Model (PyTorch) : {ref_avg_ms:8.3f} ms")

    # 2. Custom Bang operator
    def custom_forward():
        return op_ext_lib.bang_func(*mlu_inputs, *init_inputs_mlu)

    custom_avg_ms, custom_times = benchmark_fn(custom_forward, warmup=20, iters=100)
    print(f"Custom Bang Operator     : {custom_avg_ms:8.3f} ms")

    speedup = ref_avg_ms / custom_avg_ms
    print(f"Speedup (Bang vs Ref)    : {speedup:8.2f}x")

    # ====================== Correctness Check ======================
    print("\n" + "=" * 60)
    print("Correctness Check")
    print("=" * 60)

    with torch.no_grad():
        expect = ref_forward()
        actual = custom_forward()

    e = expect.float().cpu()
    a = actual.float().cpu()

    atol, rtol = tolerances(dtype)
    is_close = torch.allclose(e, a, rtol=rtol, atol=atol)

    print(f"Allclose (rtol={rtol}, atol={atol}): {is_close}")

    if not is_close:
        diff = (e - a).abs()
        print(f"Max diff : {diff.max().item():.6f}")
        print(f"Mean diff: {diff.mean().item():.6f}")
        flat_diff = diff.flatten()
        flat_expect = e.flatten()
        flat_actual = a.flatten()
        max_idx = int(flat_diff.argmax().item())
        print(
            "Max diff detail: "
            f"expect={flat_expect[max_idx].item():.8f}, "
            f"actual={flat_actual[max_idx].item():.8f}, "
            f"idx={max_idx}"
        )
        for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
            print(f"Diff q{q:g}: {torch.quantile(flat_diff, q).item():.6f}")
        for atol_probe in (1e-2, 1.1e-2, 1.25e-2, 1.5e-2, 2e-2):
            close = torch.allclose(e, a, rtol=rtol, atol=atol_probe)
            threshold = atol_probe + rtol * e.abs()
            over = (diff > threshold).sum().item()
            print(
                f"Allclose probe (rtol={rtol}, atol={atol_probe}): {close}, over={over}"
            )
        for rtol_probe in (1e-2, 1.25e-2, 1.5e-2, 2e-2):
            close = torch.allclose(e, a, rtol=rtol_probe, atol=atol)
            threshold = atol + rtol_probe * e.abs()
            over = (diff > threshold).sum().item()
            print(
                f"Allclose probe (rtol={rtol_probe}, atol={atol}): {close}, over={over}"
            )
        small = e.abs() < 0.2
        if small.any():
            print(
                "Small-output diff: "
                f"count={small.sum().item()}, "
                f"max={diff[small].max().item():.6f}, "
                f"mean={diff[small].mean().item():.6f}"
            )
        # 保存错误输出用于调试
        torch.save({"expect": e, "actual": a}, "diff_debug.pt")
        print("Saved diff to diff_debug.pt")

    print("\nBenchmark finished.")


if __name__ == "__main__":
    main()
