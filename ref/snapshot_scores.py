#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from typing import Any


DEFAULT_CONST = {
    "026_ELU": 0.09486 * 304.996191,
    "027_GELU": 0.085999 * 281.317079,
    "028_HardSigmoid": 0.083176 * 272.397728,
    "029_HardTanh": 0.107052 * 268.123113,
}


def fetch_raw() -> dict[str, Any]:
    out = subprocess.check_output(
        ["python", "ref/get_oj_status.py", "--raw", "--format", "json"], text=True
    )
    return json.loads(out)["raw"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Poll leaderboard_queue scores.")
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--team", action="append", default=["team_91", "team_42"])
    ap.add_argument("--op", action="append", help="problem name filter, e.g. 027_GELU")
    args = ap.parse_args()

    teams = set(args.team)
    ops = set(args.op or [])
    seen: set[str] = set()
    end = time.time() + args.seconds

    while time.time() < end:
        raw = fetch_raw()
        now = dt.datetime.now().strftime("%H:%M:%S")
        proc = raw.get("processing") or {}
        if proc:
            workers = ",".join(str(v.get("worker_id", "")) for v in proc.values())
            print(f"{now}\tPROC\tn={len(proc)}\tworkers={workers}", flush=True)

        items = (raw.get("queues", {}).get("bangc:leaderboard_queue") or {}).get(
            "items"
        ) or []
        for it in items:
            if it.get("team_name") not in teams:
                continue
            key = json.dumps(it, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            for name, info in (it.get("scores") or {}).items():
                if ops and name not in ops:
                    continue
                score = float(info.get("score") or 0.0)
                lat = DEFAULT_CONST.get(name)
                lat_s = "" if not score or lat is None else f"\tlat~{lat / score:.3f}"
                print(
                    f"{now}\tLB\t{it.get('team_name')}\t{name}\t"
                    f"score={score:.9f}{lat_s}\tpassed={info.get('passed')}\t"
                    f"ts={it.get('timestamp','')}",
                    flush=True,
                )
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
