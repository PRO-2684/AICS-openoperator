#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def expand_ops(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        for part in item.replace(",", " ").split():
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(f"{i:03d}" for i in range(int(a), int(b) + 1))
            else:
                out.append(part.zfill(3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run validate_profile.py for many ops.")
    ap.add_argument("ops", nargs="+", help="op ids/ranges, e.g. 001 082-083 090,130")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--eval-runs", type=int, default=1)
    ap.add_argument("--ref-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--atol", type=float)
    ap.add_argument("--rtol", type=float)
    ap.add_argument("--diff-topk", type=int, default=0)
    ap.add_argument("--build", action="store_true", default=True)
    ap.add_argument("--no-build", action="store_false", dest="build")
    ap.add_argument("--keep-going", action="store_true", default=True)
    ap.add_argument("--fail-fast", action="store_false", dest="keep_going")
    ap.add_argument("--summary", choices=["jsonl", "table"], default="table")
    args = ap.parse_args()

    rows = []
    failed = 0
    for op in expand_ops(args.ops):
        cmd = [
            sys.executable,
            str(ROOT / "ref/validate_profile.py"),
            op,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--eval-runs",
            str(args.eval_runs),
            "--ref-dtype",
            args.ref_dtype,
            "--json",
        ]
        if args.atol is not None:
            cmd.extend(["--atol", str(args.atol)])
        if args.rtol is not None:
            cmd.extend(["--rtol", str(args.rtol)])
        if args.diff_topk:
            cmd.extend(["--diff-topk", str(args.diff_topk)])
        if args.build:
            cmd.append("--build")
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        json_line = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_line = line
                break
        if json_line:
            row = json.loads(json_line)
        else:
            row = {
                "op": op,
                "correct": False,
                "error": (proc.stderr or proc.stdout)[-4000:],
            }
        row["returncode"] = proc.returncode
        rows.append(row)
        if proc.returncode != 0 or not row.get("correct"):
            failed += 1
            if not args.keep_going:
                break
        if args.summary == "jsonl":
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        else:
            ku = row.get("kernel_us") or {}
            diff = row.get("diffs") or [{}]
            max_diff = diff[0].get("max_abs_diff")
            print(
                f"{op} correct={row.get('correct')} "
                f"kavg={ku.get('avg')} kmin={ku.get('min')} "
                f"maxdiff={max_diff} rc={proc.returncode}",
                flush=True,
            )

    if args.summary == "jsonl":
        return 1 if failed else 0

    ok = sum(1 for r in rows if r.get("correct"))
    print(f"summary total={len(rows)} correct={ok} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
