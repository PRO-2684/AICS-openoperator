#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch_mlu
from torch_mlu.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]


torch.mlu = torch_mlu.mlu  # type: ignore[attr-defined]


def load_py_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tasks() -> dict[str, dict[str, Any]]:
    data = json.loads((ROOT / "reference-impl/problems.json").read_text())
    return {str(t["id"]).zfill(3): t for t in data["tasks"]}


def read_readme_sources() -> dict[str, str]:
    out: dict[str, str] = {}
    pat = re.compile(r"\|\s*(\d{3})\s*\|\s*\[[^\]]+\]\(([^)]+\.mlu)\)")
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def resolve_paths(op: str, source: str | None, ref: str | None) -> tuple[Path, Path]:
    op = op.zfill(3)
    if source:
        src = Path(source)
    else:
        src_name = read_readme_sources().get(op)
        if not src_name:
            raise RuntimeError(f"cannot resolve source for op {op}")
        src = ROOT / src_name
    if ref:
        ref_path = Path(ref)
    else:
        matches = sorted((ROOT / "ref/ref_files").glob(f"{op}_*.py"))
        if not matches:
            raise RuntimeError(f"cannot resolve ref file for op {op}")
        ref_path = matches[0]
    return src.resolve(), ref_path.resolve()


def source_wrapper_signature(source: Path) -> str | None:
    text = source.read_text(encoding="utf-8")
    m = re.search(
        r"((?:torch::Tensor|std::vector<torch::Tensor>)\s+bang_func\s*\([^)]*\))\s*\{",
        text,
        flags=re.S,
    )
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip() + ";"


def wrapper_signature(op: str, source: Path | None = None) -> str:
    if source is not None:
        sig = source_wrapper_signature(source)
        if sig:
            return sig
    task = load_tasks()[op.zfill(3)]
    return task["description"]["cpp_wrapper"].rstrip(";") + ";"


def wrapper_params(op: str, source: Path | None = None) -> list[tuple[str, str]]:
    sig = wrapper_signature(op, source)
    m = re.search(r"bang_func\s*\((.*)\)\s*;", sig)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    out: list[tuple[str, str]] = []
    for part in body.split(","):
        part = part.strip()
        pieces = part.rsplit(" ", 1)
        if len(pieces) == 2:
            out.append((pieces[0].strip(), pieces[1].strip()))
    return out


def module_name_for(op: str, source: Path) -> str:
    stem = re.sub(r"\W+", "_", source.stem)
    return f"vp_{op.zfill(3)}_{stem}"


def build_extension(op: str, source: Path, build_dir: Path, verbose: bool):
    name = module_name_for(op, source)
    module_dir = build_dir / name
    module_dir.mkdir(parents=True, exist_ok=True)
    binding = module_dir / f"{name}_binding.cpp"
    binding.write_text(
        "#include <torch/extension.h>\n\n"
        f"{wrapper_signature(op, source)}\n\n"
        f"PYBIND11_MODULE({name}, m) {{\n"
        '  m.def("bang_func", &bang_func, "BangC torch extension entry");\n'
        "}\n",
        encoding="utf-8",
    )
    return load(
        name=name,
        sources=[str(source), str(binding)],
        build_directory=str(module_dir),
        verbose=verbose,
        with_bang=True,
        extra_cflags=["-O3"],
        extra_bang_cflags=["-O3"],
    )


def load_extension_from_so(path: Path, name: str):
    return load_py_module(name, path)


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in table:
        raise RuntimeError(f"unsupported dtype: {name}")
    return table[key]


def task_dtype(op: str) -> torch.dtype:
    dtypes = load_tasks()[op.zfill(3)]["description"].get("dtype") or ["float32"]
    return dtype_from_name(str(dtypes[0]))


def is_float_dtype(dtype: torch.dtype) -> bool:
    return torch.empty((), dtype=dtype).is_floating_point()


def cast_tree(obj: Any, dtype: torch.dtype | None, device: str | None = None):
    if isinstance(obj, torch.Tensor):
        y = obj
        if dtype is not None and (obj.is_floating_point() or is_float_dtype(dtype)):
            y = y.to(dtype)
        if device is not None:
            y = y.to(device)
        return y.contiguous()
    if isinstance(obj, list):
        return [cast_tree(x, dtype, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(cast_tree(x, dtype, device) for x in obj)
    return obj


def complete_init_inputs(model_cls: type, init_inputs: list[Any]) -> list[Any]:
    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.values())[1:]
    out = list(init_inputs)
    for p in params[len(out) :]:
        if p.default is inspect._empty:
            break
        out.append(p.default)
    return out


def wrapper_extra_inputs(
    op: str,
    source: Path,
    raw_inputs: list[Any],
    model: torch.nn.Module,
    model_init: list[Any],
) -> list[Any]:
    params = wrapper_params(op, source)
    extra = params[len(raw_inputs) :]
    init_left = list(model_init)
    model_tensors = list(model.parameters()) + list(model.buffers())
    fixed: list[Any] = []
    for typ, name in extra:
        val = None
        if "torch::Tensor" in typ:
            for i, candidate in enumerate(init_left):
                if isinstance(candidate, torch.Tensor):
                    val = init_left.pop(i)
                    break
            if val is None and model_tensors:
                val = model_tensors.pop(0).detach()
        elif init_left:
            val = init_left.pop(0)
        if val is None and name == "scale" and raw_inputs:
            val = math.sqrt(raw_inputs[0].shape[-1])
        elif val is None and name == "temperature":
            val = 1.0
        elif val is None and typ in {"double", "float"}:
            val = 1.0
        elif val is None and typ in {"int", "int64_t", "long"}:
            val = 0
        elif val is None and typ == "bool":
            val = False
        fixed.append(val)
    return fixed


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-4, 1e-4


def flatten_output(obj: Any, prefix: str = "out") -> list[tuple[str, Any]]:
    if isinstance(obj, (list, tuple)):
        rows: list[tuple[str, Any]] = []
        for i, x in enumerate(obj):
            rows.extend(flatten_output(x, f"{prefix}.{i}"))
        return rows
    return [(prefix, obj)]


def tensor_diff(name: str, expect: Any, actual: Any, atol: float, rtol: float) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name}
    if not isinstance(expect, torch.Tensor) or not isinstance(actual, torch.Tensor):
        row.update({"ok": expect == actual, "kind": "scalar"})
        return row
    e = expect.detach().float().cpu()
    a = actual.detach().float().cpu()
    row.update(
        {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "expect_dtype": str(expect.dtype),
            "ok": bool(torch.allclose(e, a, atol=atol, rtol=rtol)),
        }
    )
    if e.numel() and a.numel() and e.shape == a.shape:
        diff = (e - a).abs()
        row.update(
            {
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
            }
        )
    else:
        row.update({"max_abs_diff": None, "mean_abs_diff": None})
    return row


@contextlib.contextmanager
def capture_fd1():
    old = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            yield tmp
        finally:
            sys.stdout.flush()
            os.dup2(old, 1)
            os.close(old)


def call_and_capture(fn, *args):
    t0 = time.perf_counter()
    with capture_fd1() as tmp:
        out = fn(*args)
        torch.mlu.synchronize()
        tmp.flush()
        tmp.seek(0)
        text = tmp.read().decode("utf-8", errors="replace")
    wall_us = (time.perf_counter() - t0) * 1e6
    vals = [float(x) for x in re.findall(r"KERNEL_US op=\d+ us=([0-9.]+)", text)]
    return out, text, vals, wall_us


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "avg": statistics.fmean(vals),
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate ref correctness and summarize KERNEL_US.")
    ap.add_argument("op", help="problem id, e.g. 004")
    ap.add_argument("--source", help="override .mlu source path")
    ap.add_argument("--ref", help="override ref/ref_files path")
    ap.add_argument("--so", help="use an existing compiled extension .so")
    ap.add_argument("--build", action="store_true", help="compile source with torch_mlu cpp_extension.load")
    ap.add_argument("--build-dir", default=str(ROOT / "target/validate_profile_build"))
    ap.add_argument("--module-name", help="module name when loading --so")
    ap.add_argument("--dtype", help="override dtype; default from problems.json")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    args = ap.parse_args()

    op = args.op.zfill(3)
    source, ref_path = resolve_paths(op, args.source, args.ref)
    dtype = dtype_from_name(args.dtype) if args.dtype else task_dtype(op)
    atol, rtol = tolerances(dtype)
    if args.atol is not None:
        atol = args.atol
    if args.rtol is not None:
        rtol = args.rtol

    if args.so:
        ext = load_extension_from_so(Path(args.so).resolve(), args.module_name or f"vp_so_{op}")
    else:
        if not args.build:
            raise SystemExit("provide --so or pass --build")
        ext = build_extension(op, source, Path(args.build_dir), args.verbose_build)

    ref_mod = load_py_module(f"ref_{op}", ref_path)
    init_inputs = complete_init_inputs(ref_mod.Model, list(ref_mod.get_init_inputs()))
    raw_inputs = list(ref_mod.get_inputs())

    model = ref_mod.Model(*init_inputs).eval()
    if is_float_dtype(dtype):
        model = model.to(dtype=dtype)
    bang_extra_inputs = wrapper_extra_inputs(op, source, raw_inputs, model, init_inputs)
    model = model.to("mlu")
    mlu_inputs = cast_tree(raw_inputs, dtype, "mlu")
    mlu_init = cast_tree(bang_extra_inputs, dtype, "mlu")

    def ref_call():
        with torch.no_grad():
            return model(*mlu_inputs)

    def bang_call():
        with torch.no_grad():
            return ext.bang_func(*mlu_inputs, *mlu_init)

    torch.mlu.synchronize()
    expect = ref_call()
    actual, text, kernel_vals, wall_us = call_and_capture(bang_call)
    exp_flat = flatten_output(expect)
    act_flat = flatten_output(actual)
    if len(exp_flat) != len(act_flat):
        diffs = [{"name": "out", "ok": False, "error": f"arity {len(exp_flat)} != {len(act_flat)}"}]
    else:
        diffs = [
            tensor_diff(en, ev, av, atol, rtol)
            for (en, ev), (_an, av) in zip(exp_flat, act_flat)
        ]

    warm_kernel: list[float] = []
    warm_wall: list[float] = []
    for _ in range(args.warmup):
        _out, _text, vals, wu = call_and_capture(bang_call)
        warm_kernel.extend(vals)
        warm_wall.append(wu)

    kernel: list[float] = []
    wall: list[float] = []
    for _ in range(args.iters):
        _out, _text, vals, wu = call_and_capture(bang_call)
        kernel.extend(vals)
        wall.append(wu)

    result = {
        "op": op,
        "source": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "ref": str(ref_path.relative_to(ROOT) if ref_path.is_relative_to(ROOT) else ref_path),
        "dtype": str(dtype),
        "atol": atol,
        "rtol": rtol,
        "correct": all(bool(d.get("ok")) for d in diffs),
        "diffs": diffs,
        "first_kernel_us": kernel_vals,
        "first_wall_us": wall_us,
        "kernel_us": stats(kernel),
        "wall_us": stats(wall),
        "warmup_kernel_us": stats(warm_kernel),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"op={op} correct={result['correct']} dtype={result['dtype']} "
            f"kernel_us={result['kernel_us']}"
        )
        for d in diffs:
            print("diff", json.dumps(d, ensure_ascii=False, sort_keys=True))
    return 0 if result["correct"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
