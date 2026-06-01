#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from typing import Any

import get_result


def emit_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def parse_runs(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parts = re.split(r"\[run\s+(\d+)\]\n", text)
    for i in range(1, len(parts), 2):
        run = int(parts[i])
        body = parts[i + 1]
        m = re.search(r"@@RESULT@@({.+?})", body)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        card = None
        pid = None
        ts = None
        cm = re.search(r"\[([0-9-]+\s+[0-9:.]+)\]\[CNNL\]\[WARNING\]\[(\d+)\]\[Card:(\d+)\]", body)
        if cm:
            ts = cm.group(1)
            pid = int(cm.group(2))
            card = int(cm.group(3))
        elif (cm := re.search(r"\[Card:(\d+)\]", body)):
            card = int(cm.group(1))
        rows.append(
            {
                "run": run,
                "card": card,
                "pid": pid,
                "ts": ts,
                "build": "[1/2]" in body or "cncc -c" in body,
                "ninja_no_work": "ninja: no work to do" in body,
                "passed": bool(obj.get("passed")),
                "diff": obj.get("max_abs_diff"),
                "torch_us": obj.get("torch_us"),
                "bangc_us": obj.get("bangc_us"),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact raw @@RESULT@@ extractor.")
    ap.add_argument("commits", nargs="+")
    ap.add_argument("--repo")
    ap.add_argument("--no-resolve", action="store_true")
    ap.add_argument("--format", choices=["jsonl", "text"], default="text")
    args = ap.parse_args()

    ns = argparse.Namespace(
        no_resolve=args.no_resolve,
        verbose=True,
        full=True,
        max_output_chars=200000,
        token=None,
        gh_cli=False,
    )
    repo = get_result.repo_detect(args.repo)
    ns.token = get_result.gh_token()

    all_rows: list[dict[str, Any]] = []
    for commit in args.commits:
        r = get_result.fetch_one(repo, commit, ns)
        for ci, out in enumerate(r.get("outputs") or [], start=1):
            text = out.get("text") or ""
            problem = ""
            if "HardTanh" in text:
                problem = "029"
            elif "HardSigmoid" in text:
                problem = "028"
            elif "GELU" in text:
                problem = "027"
            elif "ELU" in text:
                problem = "026"
            vals = []
            for rr in parse_runs(text):
                row = {
                    "c": commit[:8],
                    "comment": ci,
                    "p": problem,
                    **rr,
                    "url": out.get("url"),
                }
                vals.append(float(rr["bangc_us"]))
                all_rows.append(row)
                if args.format == "jsonl":
                    print(emit_json(row))
                else:
                    print(
                        f"{row['c']}\tc{ci}\t{problem}\tr{row['run']}\t"
                        f"card={row['card']}\tpid={row['pid']}\t"
                        f"build={int(bool(row['build']))}\t{float(row['bangc_us']):.3f}us"
                    )
            if vals and args.format == "text":
                print(
                    f"{commit[:8]}\tc{ci}\t{problem}\tavg={statistics.mean(vals):.3f}us\t"
                    f"min={min(vals):.3f}us\tmax={max(vals):.3f}us"
                )

    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
