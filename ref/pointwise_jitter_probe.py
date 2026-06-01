#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch_mlu  # noqa: F401
from torch_mlu.utils.cpp_extension import load_inline

ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_problems():
    data = json.loads((ROOT / "reference-impl" / "problems.json").read_text())
    out = {}
    for task in data["tasks"]:
        base = task["name"].split("_", 1)[1]
        out[base] = task
    return out


def task_for_source(source):
    stem = Path(source).stem
    return load_problems()[stem]


def ref_for_op(op):
    matches = sorted((ROOT / "ref" / "ref_files").glob(f"{op}_*.py"))
    if not matches:
        raise SystemExit(f"missing ref file for {op}")
    return matches[0]


def build_decl(task):
    wrapper = task["description"]["cpp_wrapper"].strip()
    return "#include <torch/extension.h>\n\n" + wrapper + "\n"


def ext_name(task, source_path, source_text):
    h = hashlib.sha1()
    h.update(task["name"].encode())
    h.update(str(source_path).encode())
    h.update(source_text.encode())
    return f"jitter_{task['name']}_{h.hexdigest()[:12]}".replace("-", "_")


def load_ext(source_path):
    task = task_for_source(source_path)
    text = Path(source_path).read_text()
    return load_inline(
        name=ext_name(task, source_path, text),
        cpp_sources=build_decl(task),
        bang_sources=text,
        functions=["bang_func"],
        verbose=False,
        extra_cflags=["-O3"],
        extra_ldflags=["-lcnrt", "-lbangc", "-lcnnl"],
        extra_bang_cflags=[
            "-O3",
            "-lm",
            "--bang-arch=compute_30",
            "--no-neuware-version-check",
            "--neuware-path=/usr/local/neuware",
        ],
    )


def force_fp16(values):
    out = []
    for v in values:
        if torch.is_tensor(v) and v.is_floating_point():
            out.append(v.to(torch.float16))
        else:
            out.append(v)
    return out


def to_mlu(values):
    return [v.mlu() if torch.is_tensor(v) else v for v in values]


def summarize(name, xs):
    xs = sorted(xs)
    return {
        "name": name,
        "n": len(xs),
        "min": xs[0],
        "median": statistics.median(xs),
        "mean": statistics.mean(xs),
        "max": xs[-1],
        "range": xs[-1] - xs[0],
        "first": xs[: min(5, len(xs))],
    }


def time_once(fn):
    torch.mlu.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.mlu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--same-process-calls", type=int, default=1)
    ap.add_argument("--prewarm", choices=["none", "empty1", "zeros1", "zeros_big"], default="none")
    args = ap.parse_args()

    torch.mlu.set_device(0)
    ext = load_ext(ROOT / args.source)
    ref = load_module(ref_for_op(args.op))
    init_inputs = force_fp16(ref.get_init_inputs())

    xs = []
    for _ in range(args.repeat):
        raw = force_fp16(ref.get_inputs())
        inputs = to_mlu(raw)

        if args.prewarm == "empty1":
            torch.empty((1,), device="mlu", dtype=torch.float16)
            torch.mlu.synchronize()
        elif args.prewarm == "zeros1":
            torch.zeros((1,), device="mlu", dtype=torch.float16)
            torch.mlu.synchronize()
        elif args.prewarm == "zeros_big":
            torch.zeros((16, 16384), device="mlu", dtype=torch.float16)
            torch.mlu.synchronize()

        for call in range(args.same_process_calls):
            us, _ = time_once(lambda: ext.bang_func(*inputs, *init_inputs))
            xs.append(us)

    print(json.dumps(summarize(f"{args.op}:{args.source}:{args.prewarm}", xs), sort_keys=True))


if __name__ == "__main__":
    main()
