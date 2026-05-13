#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch_mlu  # noqa: F401
from problem_utils import REPO_ROOT, task_dtypes, task_for_source
from torch_mlu.utils.cpp_extension import load_inline

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}

WARMUP_ITERS = 3
BENCH_ITERS = 10


def log(args, *xs):
    if not args.quiet:
        print(*xs, flush=True)


def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_floating_dtype(dtype):
    return dtype is not None and torch.is_floating_point(torch.empty((), dtype=dtype))


def dtype_from_name(dtype_name):
    if dtype_name is None:
        return None
    if dtype_name not in DTYPE_MAP:
        raise SystemExit(f"unsupported dtype in problems.json: {dtype_name}")
    return DTYPE_MAP[dtype_name]


def should_cast_tensor(tensor, dtype):
    if dtype is None:
        return False
    return tensor.is_floating_point() or not is_floating_dtype(dtype)


def cast_tree(obj, dtype, device=None):
    if isinstance(obj, torch.Tensor):
        out = obj
        if should_cast_tensor(out, dtype):
            out = out.to(dtype)
        if device is not None:
            out = out.to(device)
        return out.contiguous()

    if isinstance(obj, list):
        return [cast_tree(x, dtype, device) for x in obj]

    if isinstance(obj, tuple):
        return tuple(cast_tree(x, dtype, device) for x in obj)

    if isinstance(obj, dict):
        return {k: cast_tree(v, dtype, device) for k, v in obj.items()}

    return obj


def flatten_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]

    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(flatten_tensors(x))
        return out

    if isinstance(obj, dict):
        out = []
        for k in sorted(obj.keys()):
            out.extend(flatten_tensors(obj[k]))
        return out

    return []


def max_numel(obj):
    total = 0
    for tensor in flatten_tensors(obj):
        total = max(total, tensor.numel())
    return total


def tolerances(dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-4, 1e-4


def synchronize_mlu():
    torch.mlu.synchronize()  # ty:ignore[unresolved-attribute]


def benchmark(args, label, fn):
    with torch.no_grad():
        result = None

        for _ in range(WARMUP_ITERS):
            result = fn()
        synchronize_mlu()

        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            result = fn()
        synchronize_mlu()

        elapsed_ms = (time.perf_counter() - start) * 1000.0 / BENCH_ITERS

    log(args, f"[time] {label}: {elapsed_ms:.3f} ms avg over {BENCH_ITERS} runs")
    return result, elapsed_ms


def build_cpp_decl(task):
    wrapper = task["description"]["cpp_wrapper"].strip()
    return f"""#include <torch/extension.h>

{wrapper}
"""


def make_ext_name(task, source_path, source_text):
    h = hashlib.sha1()
    h.update(task["base_name"].encode())
    h.update(str(source_path).encode())
    h.update(source_text.encode())
    return f"{task['base_name']}_inline_{h.hexdigest()[:12]}"


def load_bang_extension(args, task, source_path):
    bang_source = Path(source_path).read_text()
    cpp_source = build_cpp_decl(task)
    ext_name = make_ext_name(task, source_path, bang_source)

    log(args, f"[build] extension={ext_name}")
    log(
        args,
        f"[build] torch_extensions_dir={os.environ.get('TORCH_EXTENSIONS_DIR', '')}",
    )

    return load_inline(
        name=ext_name,
        cpp_sources=cpp_source,
        bang_sources=bang_source,
        functions=["bang_func"],
        verbose=args.verbose,
        extra_cflags=[
            "-O3",
        ],
        extra_ldflags=[
            "-lcnrt",
            "-lbangc",
            "-lcnnl",
        ],
        extra_bang_cflags=[
            "-O3",
            "-lm",
            "--bang-arch=compute_30",
            "--no-neuware-version-check",
            "--neuware-path=/usr/local/neuware",
        ],
    )


def compare_outputs(actual, expected, dtype, task_name, dtype_label):
    actual_tensors = flatten_tensors(actual)
    expected_tensors = flatten_tensors(expected)

    if len(actual_tensors) != len(expected_tensors):
        raise SystemExit(
            f"reference check failed for {task_name} dtype={dtype_label}: "
            f"output tensor count mismatch, actual={len(actual_tensors)}, expected={len(expected_tensors)}"
        )

    if not actual_tensors:
        raise SystemExit("reference checker currently supports tensor outputs only")

    atol, rtol = tolerances(dtype)
    max_diff = 0.0

    for i, (actual_tensor, expected_tensor) in enumerate(
        zip(actual_tensors, expected_tensors)
    ):
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            raise SystemExit(
                f"reference check failed for {task_name} dtype={dtype_label}: "
                f"output[{i}] shape mismatch, actual={tuple(actual_tensor.shape)}, "
                f"expected={tuple(expected_tensor.shape)}"
            )

        actual_cpu = actual_tensor.float().cpu()
        expected_cpu = expected_tensor.float().cpu()

        diff = (actual_cpu - expected_cpu).abs().max().item()
        max_diff = max(max_diff, diff)

        if not torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol):
            raise SystemExit(
                f"reference check failed for {task_name} dtype={dtype_label}, "
                f"output[{i}] max_abs_diff={diff}, atol={atol}, rtol={rtol}"
            )

    return max_diff, atol, rtol


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)

    # Compatibility with the older shell script. These are intentionally unused.
    parser.add_argument("--object", default=None)
    parser.add_argument("--wrapper-so", default=None)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-jobs", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_jobs is not None:
        os.environ["MAX_JOBS"] = str(args.max_jobs)

    if args.verbose:
        args.quiet = False

    if not torch.mlu.is_available():  # ty:ignore[unresolved-attribute]
        raise SystemExit("MLU is not available, cannot run reference check")

    device_count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
    if args.device < 0 or args.device >= device_count:
        raise SystemExit(
            f"invalid MLU device {args.device}, device_count={device_count}"
        )

    torch.mlu.set_device(args.device)  # ty:ignore[unresolved-attribute]

    source = Path(args.source).resolve()
    task = task_for_source(source.name)

    ref_path = REPO_ROOT / "reference-impl" / f"{task['base_name']}.py"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(task["reference_implementation"].rstrip() + "\n")

    log(args, f"[check] task={task['base_name']} source={source}")
    log(args, f"[check] device=mlu:{args.device}")

    ext_mod = load_bang_extension(args, task, source)
    ref_mod = load_module_from_file(f"{task['base_name']}_reference", ref_path)

    result_parts = []

    for dtype_name in task_dtypes(task):
        dtype = dtype_from_name(dtype_name)
        dtype_label = dtype_name or "default"
        log(args, f"[check] dtype={dtype_label}")

        init_inputs = list(ref_mod.get_init_inputs())
        raw_inputs = list(ref_mod.get_inputs())

        model = ref_mod.Model(*init_inputs)
        model.eval()

        if is_floating_dtype(dtype):
            model = model.to(dtype=dtype)

        param_count = sum(p.numel() for p in model.parameters())
        buffer_count = sum(b.numel() for b in model.buffers())
        if param_count or buffer_count:
            raise SystemExit(
                f"reference check skipped for {task['base_name']}: "
                f"stateful model with {param_count} parameters and {buffer_count} buffers"
            )

        mlu_inputs = cast_tree(raw_inputs, dtype=dtype, device="mlu")
        model = model.to("mlu")

        expected, reference_ms = benchmark(
            args,
            f"reference[{dtype_label}]",
            lambda: model(*mlu_inputs),
        )

        actual, bang_ms = benchmark(
            args,
            f"bang_func[{dtype_label}]",
            lambda: ext_mod.bang_func(*mlu_inputs, *init_inputs),
        )

        max_diff, atol, rtol = compare_outputs(
            actual=actual,
            expected=expected,
            dtype=dtype,
            task_name=task["base_name"],
            dtype_label=dtype_label,
        )

        speedup = reference_ms / bang_ms if bang_ms > 0 else 0.0

        log(
            args,
            f"[ok] reference check passed for {task['base_name']} dtype={dtype_label} "
            f"(max_abs_diff={max_diff:.6g}, max_elements={max_numel(raw_inputs)}, "
            f"atol={atol}, rtol={rtol}, speedup={speedup:.3f}x)",
        )

        result_parts.append(
            f"dtype={dtype_label} diff={max_diff:.6g} "
            f"ref_ms={reference_ms:.3f} bang_ms={bang_ms:.3f} speedup={speedup:.3f}x"
        )

    print(f"PASS task={task['base_name']} " + " | ".join(result_parts), flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
