#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def qcounts() -> tuple[int, int]:
    p = run(
        [
            "python",
            "ref/get_oj_status.py",
            "--source",
            "redis",
            "--format",
            "json",
            "--limit",
            "1",
            "team_42",
            "team_91",
        ]
    )
    d = json.loads(p.stdout.strip().splitlines()[-1])
    q = d.get("q") or {}
    return int(q.get("processing", 0)), int(q.get("task", 0))


def wait_idle(max_processing: int, max_task: int, poll: float, timeout: float) -> None:
    end = time.time() + timeout
    while True:
        proc, task = qcounts()
        if proc <= max_processing and task <= max_task:
            return
        if time.time() >= end:
            print(
                json.dumps(
                    {"type": "wait_idle", "proc": proc, "task": task, "ok": 0},
                    separators=(",", ":"),
                ),
                flush=True,
            )
            return
        time.sleep(poll)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paced OJ empty-submit sampler.")
    ap.add_argument("-o", "--op", required=True)
    ap.add_argument("-n", "--count", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--idle-timeout", type=float, default=300.0)
    ap.add_argument("--max-processing", type=int, default=0)
    ap.add_argument("--max-task", type=int, default=0)
    ap.add_argument("--msg", default="paced sampler")
    ap.add_argument("--out", default="/tmp/oj_paced_hashes.txt")
    args = ap.parse_args()

    hashes: list[str] = []
    for i in range(args.count):
      wait_idle(args.max_processing, args.max_task, args.poll, args.idle_timeout)
      p = run(
          [
              "python",
              "ref/oj_git.py",
              "-o",
              args.op,
              "-m",
              f"{args.op} {args.msg} {i + 1}",
              "-a",
              "config",
              "--empty",
          ]
      )
      text = (p.stdout or p.stderr).strip()
      print(text, flush=True)
      try:
          obj = json.loads(text.splitlines()[-1])
          if obj.get("c"):
              hashes.append(obj["c"])
      except Exception:
          pass
      if args.sleep:
          time.sleep(args.sleep)

    Path(args.out).write_text("\n".join(hashes) + ("\n" if hashes else ""))
    print(
        json.dumps({"type": "hashes", "hashes": hashes}, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
