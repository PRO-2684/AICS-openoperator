#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LAT_CONST = {
    "026": 0.09486 * 304.996191,
    "027": 0.085999 * 281.317079,
    "028": 0.083176 * 272.397728,
    "029": 0.107052 * 268.123113,
}


def read_rows(paths: list[str]) -> list[dict[str, Any]]:
    proc: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("PROC "):
                obj = json.loads(line[5:])
                proc[obj.get("task", "")] = obj
            elif line.startswith("LB   "):
                obj = json.loads(line[5:])
                task = obj.get("task", "")
                po = proc.get(task, {})
                for score in obj.get("scores") or []:
                    if not score.get("ok"):
                        continue
                    p_id = str(score.get("p") or "")
                    us = score.get("us")
                    if us is None and score.get("s") and p_id in LAT_CONST:
                        us = LAT_CONST[p_id] / float(score["s"])
                    if us is None:
                        continue
                    rows.append(
                        {
                            "p": p_id,
                            "us": float(us),
                            "team": obj.get("team"),
                            "task": task,
                            "c": obj.get("c"),
                            "w": po.get("w"),
                            "repo": obj.get("repo"),
                        }
                    )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize compact redis_q watch logs.")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--best", type=int, default=8)
    ap.add_argument("--by-worker", action="store_true")
    args = ap.parse_args()

    rows = read_rows(args.logs)
    if not rows:
        print("NO_ROWS")
        return 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = r["p"] if not args.by_worker else f'{r["p"]}/w{r.get("w")}'
        groups[key].append(r)

    for key in sorted(groups):
        xs = sorted(groups[key], key=lambda r: r["us"])
        vals = [r["us"] for r in xs]
        print(
            f"{key}\tn={len(xs)}\tmin={min(vals):.3f}\t"
            f"med={statistics.median(vals):.3f}\tmax={max(vals):.3f}"
        )
        for r in xs[: args.best]:
            print(
                f"  {r['us']:.3f}us\tteam={r.get('team')}\tw={r.get('w')}\t"
                f"c={r.get('c')}\ttask={r.get('task')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
