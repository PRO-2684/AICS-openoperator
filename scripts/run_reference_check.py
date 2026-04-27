#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path
import subprocess
import time

import torch
import torch_mlu  # noqa: F401

from problem_utils import REPO_ROOT, task_dtypes, task_for_source


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}

WARMUP_ITERS = 3
BENCH_ITERS = 10

def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def max_numel(items):
    total = 0
    for item in items:
        if isinstance(item, torch.Tensor):
            total = max(total, item.numel())
    return total


def build_binding_source(task, module_name):
    wrapper = task["description"]["cpp_wrapper"].strip()
    return f"""#include <torch/extension.h>

{wrapper}

PYBIND11_MODULE({module_name}, m) {{
  m.def("bang_func", &bang_func, "{task["base_name"]} bang_func");
}}
"""


def synchronize_mlu():
    torch.mlu.synchronize()


def benchmark(label, fn):
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            fn()
        synchronize_mlu()

        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            result = fn()
        synchronize_mlu()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / BENCH_ITERS

    print(f"[time] {label}: {elapsed_ms:.3f} ms avg over {BENCH_ITERS} runs")
    return result, elapsed_ms


def dtype_from_name(dtype_name):
    if dtype_name is None:
        return None
    if dtype_name not in DTYPE_MAP:
        raise SystemExit(f"unsupported dtype in problems.json: {dtype_name}")
    return DTYPE_MAP[dtype_name]


def is_floating_dtype(dtype):
    return dtype is not None and torch.is_floating_point(torch.empty((), dtype=dtype))


def should_cast_tensor(tensor, dtype):
    if dtype is None:
        return False
    return tensor.is_floating_point() or not is_floating_dtype(dtype)


def tolerances(dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-4, 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--wrapper-so", required=True)
    args = parser.parse_args()

    if not torch.mlu.is_available():
        raise SystemExit("MLU is not available, cannot run reference check")

    source = Path(args.source).resolve()
    obj_path = Path(args.object).resolve()
    wrapper_so = Path(args.wrapper_so).resolve()
    task = task_for_source(source.name)

    ref_path = REPO_ROOT / "reference-impl" / f"{task['base_name']}.py"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(task["reference_implementation"].rstrip() + "\n")

    module_name = wrapper_so.stem
    binding_cpp = obj_path.parent / f"{module_name}_binding.cpp"
    binding_obj = obj_path.parent / f"{module_name}_binding.o"
    binding_cpp.write_text(build_binding_source(task, module_name))

    compile_cmd = [
        "c++",
        "-c",
        str(binding_cpp),
        "-o",
        str(binding_obj),
        f"-DTORCH_EXTENSION_NAME={module_name}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        '-DPYBIND11_COMPILER_TYPE="_gcc"',
        '-DPYBIND11_STDLIB="_libstdcpp"',
        '-DPYBIND11_BUILD_ABI="_cxxabi1011"',
        "-isystem",
        "/torch/venv3/pytorch/lib/python3.10/site-packages/torch/include",
        "-isystem",
        "/torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include",
        "-isystem",
        "/torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/TH",
        "-isystem",
        "/torch/venv3/pytorch/lib/python3.10/site-packages/torch/include/THC",
        "-isystem",
        "/opt/py3.10/include/python3.10",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-fPIC",
        "-std=c++17",
    ]
    subprocess.run(compile_cmd, check=True)

    link_cmd = [
        "c++",
        str(binding_obj),
        str(obj_path),
        "-shared",
        "-o",
        str(wrapper_so),
        "-L/torch/venv3/pytorch/lib/python3.10/site-packages/torch/lib",
        "-L/torch/venv3/pytorch/lib/python3.10/site-packages/torch_mlu/csrc/lib",
        "-L/usr/local/neuware/lib64",
        "-lc10",
        "-ltorch_cpu",
        "-ltorch",
        "-ltorch_python",
        "-ltorch_mlu",
        "-ltorch_mlu_python",
        "-lcnrt",
        "-lbangc",
    ]
    subprocess.run(link_cmd, check=True)

    ref_mod = load_module_from_file(f"{task['base_name']}_reference", ref_path)
    ext_mod = load_module_from_file(module_name, wrapper_so)

    for dtype_name in task_dtypes(task):
        dtype = dtype_from_name(dtype_name)
        dtype_label = dtype_name or "default"
        print(f"[check] dtype={dtype_label}")

        init_inputs = list(ref_mod.get_init_inputs())
        raw_inputs = list(ref_mod.get_inputs())

        model = ref_mod.Model(*init_inputs)
        model.eval()
        if is_floating_dtype(dtype):
            model = model.to(dtype=dtype)

        param_count = sum(p.numel() for p in model.parameters())
        buffer_count = sum(b.numel() for b in model.buffers())
        if param_count or buffer_count:
            print(
                f"[skip] reference check skipped for {task['base_name']}: "
                f"stateful model with {param_count} parameters and {buffer_count} buffers"
            )
            return

        mlu_inputs = cast_tree(raw_inputs, dtype=dtype, device="mlu")
        model = model.to("mlu")

        expected, reference_ms = benchmark(
            f"reference[{dtype_label}]", lambda: model(*mlu_inputs)
        )
        actual, bang_ms = benchmark(
            f"bang_func[{dtype_label}]",
            lambda: ext_mod.bang_func(*mlu_inputs, *init_inputs),
        )

        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise SystemExit("reference checker currently supports tensor outputs only")

        compare_expected = expected.float().cpu()
        compare_actual = actual.float().cpu()

        atol, rtol = tolerances(dtype)
        if not torch.allclose(compare_actual, compare_expected, atol=atol, rtol=rtol):
            diff = (compare_actual - compare_expected).abs().max().item()
            raise SystemExit(
                f"reference check failed for {task['base_name']} dtype={dtype_label}, "
                f"max abs diff={diff}"
            )

        print(
            f"[ok] reference check passed for {task['base_name']} dtype={dtype_label} "
            f"(max_elements={max_numel(raw_inputs)}, atol={atol}, rtol={rtol})"
        )
        if bang_ms > 0:
            print(f"[time] speedup[{dtype_label}]: {reference_ms / bang_ms:.3f}x")


if __name__ == "__main__":
    main()
