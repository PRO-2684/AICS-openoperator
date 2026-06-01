#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


BASE = "http://43.143.241.66:13000/api"
SECRET_FILE = Path(__file__).with_name("webhook_secrets.json")

LAT_CONST = {
    "026_ELU": 0.09486 * 304.996191,
    "027_GELU": 0.085999 * 281.317079,
    "028_HardSigmoid": 0.083176 * 272.397728,
    "029_HardTanh": 0.107052 * 268.123113,
}


def jdumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def secret_for(team: str) -> str:
    data = json.loads(SECRET_FILE.read_text(encoding="utf-8"))
    t = team.strip().replace("team_", "")
    for key in (f"team_{t}", f"team{t}", t):
        if data.get(key):
            return str(data[key])
    raise RuntimeError(f"missing secret for team_{t}")


def latency(name: str, score: Any) -> float | None:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    c = LAT_CONST.get(name)
    if not c or s <= 0:
        return None
    return c / s


def post_json(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    r = requests.post(f"{BASE}/{path}", json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"text": r.text[:500]}
    if r.status_code >= 400:
        raise RuntimeError(f"{path} http {r.status_code}: {data}")
    return data


def snapshot(timeout: float) -> dict[str, Any]:
    r = requests.get(f"{BASE}/snapshot", timeout=timeout)
    r.raise_for_status()
    return r.json()


def queue_counts(timeout: float) -> tuple[int, int]:
    try:
        import redis_q

        r = redis_q.connect()
        pipe = r.pipeline()
        pipe.hlen("bangc:processing")
        pipe.llen("bangc:task_queue")
        nproc, ntask = pipe.execute()
        return int(nproc), int(ntask)
    except Exception:
        data = snapshot(timeout)
        proc = data.get("processing") or {}
        queues = data.get("queues") or {}
        task_q = (queues.get("bangc:task_queue") or {}).get("items") or []
        return len(proc), len(task_q)


def wait_idle(args: argparse.Namespace) -> None:
    if args.max_processing is None:
        return
    end = time.time() + args.idle_timeout
    while True:
        nproc, ntask = queue_counts(args.timeout)
        if nproc <= args.max_processing and ntask <= args.max_task_queue:
            return
        if time.time() >= end:
            print(
                jdumps(
                    {
                        "type": "wait_idle",
                        "ok": 0,
                        "processing": nproc,
                        "task_queue": ntask,
                    }
                ),
                flush=True,
            )
            return
        time.sleep(args.idle_poll)


def cmd_rerun(args: argparse.Namespace) -> int:
    teams = [
        x.strip() for x in str(args.team).replace("team_", "").split(",") if x.strip()
    ]
    commits = [x.strip() for x in args.commits if x.strip()]
    for i in range(args.count):
        for team in teams:
            sec = secret_for(team)
            for sha in commits:
                wait_idle(args)
                payload = {"secret": sec, "commit_sha": sha}
                last_err: Exception | None = None
                data: dict[str, Any] = {}
                ok = False
                for attempt in range(1, args.retry + 1):
                    try:
                        data = post_json("rerun", payload, args.timeout)
                        ok = True
                        break
                    except Exception as e:
                        last_err = e
                        if attempt < args.retry:
                            time.sleep(args.retry_sleep)
                obj = {
                    "type": "rerun",
                    "ok": 1 if ok else 0,
                    "team": f"team_{team}",
                    "commit": sha[:8],
                    "i": i + 1,
                    "status": data.get("status")
                    or data.get("message")
                    or data.get("ok"),
                }
                if not ok:
                    obj["error"] = str(last_err)
                print(jdumps(obj), flush=True)
                if args.sleep:
                    time.sleep(args.sleep)
    return 0


def fetch_history(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = {"secret": secret_for(args.team), "error_only": args.error_only}
    data = post_json("team_history", payload, args.timeout)
    rows = data.get("results")
    if rows is None:
        rows = data.get("history") or []
    if args.commit:
        prefixes = tuple(args.commit)
        rows = [r for r in rows if (r.get("commit_sha") or "").startswith(prefixes)]
    return rows[: args.limit] if args.limit else rows


def cmd_history(args: argparse.Namespace) -> int:
    rows = fetch_history(args)
    if args.format == "json":
        print(jdumps(rows))
        return 0

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        sha = (r.get("commit_sha") or "")[:8]
        ts = r.get("timestamp", "")
        for name, info in (r.get("scores") or {}).items():
            if args.op and name not in set(args.op):
                continue
            score = info.get("score")
            lat = latency(name, score)
            if args.format == "jsonl":
                print(
                    jdumps(
                        {
                            "type": "history",
                            "team": r.get("team_name"),
                            "commit": sha,
                            "ts": ts,
                            "op": name,
                            "passed": info.get("passed"),
                            "score": score,
                            "latency_us": lat,
                            "error": r.get("error"),
                        }
                    )
                )
            else:
                lat_s = "" if lat is None else f"{lat:.3f}us"
                print(f"{ts}\t{sha}\t{name}\t{lat_s}\t{score}\t{info.get('passed')}")
            if lat is not None and info.get("passed"):
                grouped[(sha, name)].append(lat)

    if args.summary and args.format == "text":
        print("SUMMARY")
        for (sha, name), xs in sorted(
            grouped.items(), key=lambda kv: (kv[0][1], min(kv[1]))
        ):
            print(
                f"{sha}\t{name}\tn={len(xs)}\tmin={min(xs):.3f}us\t"
                f"med={statistics.median(xs):.3f}us\tmax={max(xs):.3f}us"
            )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Direct OJ helper for rerun/history APIs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rerun", help="enqueue reruns for existing full commit sha(s)")
    r.add_argument("commits", nargs="+")
    r.add_argument("--team", default="42")
    r.add_argument("-n", "--count", type=int, default=1)
    r.add_argument("--sleep", type=float, default=0.0)
    r.add_argument("--timeout", type=float, default=10.0)
    r.add_argument("--retry", type=int, default=3)
    r.add_argument("--retry-sleep", type=float, default=1.0)
    r.add_argument(
        "--max-processing",
        type=int,
        help="wait before enqueue until processing is at most this value",
    )
    r.add_argument("--max-task-queue", type=int, default=0)
    r.add_argument("--idle-poll", type=float, default=2.0)
    r.add_argument("--idle-timeout", type=float, default=300.0)
    r.set_defaults(func=cmd_rerun)

    h = sub.add_parser("history", help="print compact team history")
    h.add_argument("--team", default="42")
    h.add_argument("--commit", action="append", help="commit prefix filter")
    h.add_argument("--op", action="append", help="problem name filter")
    h.add_argument("--limit", type=int, default=50)
    h.add_argument("--error-only", action="store_true")
    h.add_argument("--summary", action="store_true")
    h.add_argument("--timeout", type=float, default=10.0)
    h.add_argument("--format", choices=["text", "jsonl", "json"], default="text")
    h.set_defaults(func=cmd_history)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
