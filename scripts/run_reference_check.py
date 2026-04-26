#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path
import subprocess

import torch
import torch_mlu  # noqa: F401

from problem_utils import REPO_ROOT, first_dtype, task_for_source


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

MAX_LOCAL_NUMEL = 1 << 24


def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cast_tree(obj, dtype, device=None):
    if isinstance(obj, torch.Tensor):
        out = obj
        if obj.is_floating_point() and dtype is not None:
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


def shrink_tensor(tensor):
    shape = list(tensor.shape)
    if tensor.numel() <= MAX_LOCAL_NUMEL:
        return tensor

    if tensor.dim() == 1:
        limits = [4096]
    elif tensor.dim() == 2:
        limits = [512, 512]
    elif tensor.dim() == 3:
        limits = [8, 256, 256]
    elif tensor.dim() == 4:
        limits = [4, 64, 128, 128]
    else:
        limits = [32] * tensor.dim()

    slices = tuple(slice(0, min(size, limits[i])) for i, size in enumerate(shape))
    shrunk = tensor[slices].contiguous()

    while shrunk.numel() > MAX_LOCAL_NUMEL:
        largest_dim = max(range(shrunk.dim()), key=lambda i: shrunk.shape[i])
        new_size = max(1, shrunk.shape[largest_dim] // 2)
        resize_slices = [slice(None)] * shrunk.dim()
        resize_slices[largest_dim] = slice(0, new_size)
        shrunk = shrunk[tuple(resize_slices)].contiguous()

    return shrunk


def shrink_inputs(items):
    shrunk = []
    changed = False
    for item in items:
        if isinstance(item, torch.Tensor):
            new_item = shrink_tensor(item)
            changed = changed or tuple(new_item.shape) != tuple(item.shape)
            shrunk.append(new_item)
        else:
            shrunk.append(item)
    return shrunk, changed


def build_binding_source(task, module_name):
    wrapper = task["description"]["cpp_wrapper"].strip()
    return f"""#include <torch/extension.h>

{wrapper}

PYBIND11_MODULE({module_name}, m) {{
  m.def("bang_func", &bang_func, "{task["base_name"]} bang_func");
}}
"""


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

    init_inputs = list(ref_mod.get_init_inputs())
    raw_inputs = list(ref_mod.get_inputs())
    raw_inputs, shrunk = shrink_inputs(raw_inputs)
    dtype = DTYPE_MAP.get(first_dtype(task))

    model = ref_mod.Model(*init_inputs)
    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)

    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(b.numel() for b in model.buffers())
    if param_count or buffer_count:
        print(
            f"[skip] reference check skipped for {task['base_name']}: "
            f"stateful model with {param_count} parameters and {buffer_count} buffers"
        )
        return

    if shrunk:
        print(
            f"[info] downscaled oversized reference inputs for {task['base_name']} "
            f"to fit local MLU limits"
        )

    cpu_inputs = cast_tree(raw_inputs, dtype=dtype, device=None)
    mlu_inputs = cast_tree(raw_inputs, dtype=dtype, device="mlu")

    with torch.no_grad():
        expected = model(*cpu_inputs)
        actual = ext_mod.bang_func(*mlu_inputs, *init_inputs)

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise SystemExit("reference checker currently supports tensor outputs only")

    compare_expected = expected.float().cpu()
    compare_actual = actual.float().cpu()

    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = atol
    if not torch.allclose(compare_actual, compare_expected, atol=atol, rtol=rtol):
        diff = (compare_actual - compare_expected).abs().max().item()
        raise SystemExit(
            f"reference check failed for {task['base_name']}, max abs diff={diff}"
        )

    print(
        f"[ok] reference check passed for {task['base_name']} "
        f"(max_elements={max_numel(raw_inputs)}, atol={atol}, rtol={rtol})"
    )


if __name__ == "__main__":
    main()
