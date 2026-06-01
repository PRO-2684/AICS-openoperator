#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-friendly OpenOperator queue/Redis summarizer.

Default tracks team_91 and team_42 for compatibility:
  python get_oj_status.py
  python get_oj_status.py team_42 team_91 --format text
  python get_oj_status.py team_42 --verbose
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import sys
from typing import Any, Dict, Iterable, List, Optional

import requests

DEFAULT_URL = "http://43.143.241.66:13000/api/snapshot"
DEFAULT_TEAMS = ["team_91", "team_42"]
Q_TASK = "bangc:task_queue"
Q_RESULT = "bangc:result_queue"
Q_LEADERBOARD = "bangc:leaderboard_queue"
Q_PROCESSING = "bangc:processing"

LAT_CONST = {
    "026_ELU": 0.09486 * 304.996191,
    "027_GELU": 0.085999 * 281.317079,
    "028_HardSigmoid": 0.083176 * 272.397728,
    "029_HardTanh": 0.107052 * 268.123113,
}


def jdumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def bj_time(t: str) -> str:
    return (
        dt.datetime.fromisoformat(t.replace("Z", "+00:00")) + dt.timedelta(hours=8)
    ).strftime("%m-%d %H:%M:%S")


def now_bj() -> str:
    return dt.datetime.now().strftime("%m-%d %H:%M:%S")


def parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def norm_team(x: str) -> str:
    x = str(x or "").strip()
    if x.isdigit():
        return f"team_{x}"
    return x


def short_sha(x: Any) -> str:
    s = str(x or "")
    return s[:8] if s else ""


def age_s(t: Any) -> Optional[int]:
    if not t:
        return None
    try:
        ts = dt.datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        return max(0, int((dt.datetime.now(dt.timezone.utc) - ts).total_seconds()))
    except Exception:
        return None


def score_latency(name: str, score: Any) -> Optional[float]:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    c = LAT_CONST.get(name)
    if not c or s <= 0:
        return None
    return c / s


def pick_fields(x: Dict[str, Any]) -> Dict[str, Any]:
    # Keep common useful fields but avoid dumping huge payloads by default.
    out: Dict[str, Any] = {}
    team = x.get("team_name") or x.get("team") or x.get("team_id")
    if team:
        out["team"] = norm_team(team)
    for dst, keys in {
        "task": ("task_id", "id", "job_id"),
        "w": ("worker_id",),
        "c": ("commit_sha", "commit", "after"),
        "repo": ("repo_full_name", "repo"),
        "op": ("problem_name", "problem_id"),
        "st": ("status",),
        "err": ("error",),
    }.items():
        for k in keys:
            if x.get(k) not in (None, ""):
                v = x.get(k)
                out[dst] = short_sha(v) if dst in ("c", "task") else v
                break
    t = x.get("started_at") or x.get("timestamp") or x.get("created_at")
    a = age_s(t)
    if a is not None:
        out["age"] = a
    scores = x.get("scores") or {}
    if isinstance(scores, dict) and scores:
        ss = []
        for name, info in scores.items():
            if not isinstance(info, dict):
                continue
            score = info.get("score")
            lat = score_latency(name, score)
            item: Dict[str, Any] = {
                "p": name.split("_", 1)[0],
                "ok": 1 if info.get("passed") else 0,
            }
            if score is not None:
                try:
                    item["s"] = round(float(score), 6)
                except Exception:
                    item["s"] = score
            if lat is not None:
                item["us"] = round(lat, 3)
            ss.append(item)
        if ss:
            out["scores"] = ss
    return out


def fetch_snapshot(url: str, timeout: float) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_redis(limit: int) -> Dict[str, Any]:
    import redis_q

    r = redis_q.connect()
    pipe = r.pipeline()
    pipe.llen(Q_TASK)
    pipe.llen(Q_RESULT)
    pipe.llen(Q_LEADERBOARD)
    pipe.lrange(Q_TASK, 0, max(0, limit - 1))
    pipe.lrange(Q_RESULT, 0, max(0, limit - 1))
    pipe.lrange(Q_LEADERBOARD, 0, max(0, limit - 1))
    pipe.hgetall(Q_PROCESSING)
    task_n, result_n, lb_n, task_raw, result_raw, lb_raw, proc_raw = pipe.execute()
    proc = {k: parse_json(v) for k, v in (proc_raw or {}).items()}
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "queues": {
            Q_TASK: {"length": task_n, "items": [parse_json(x) for x in task_raw]},
            Q_RESULT: {"length": result_n, "items": [parse_json(x) for x in result_raw]},
            Q_LEADERBOARD: {"length": lb_n, "items": [parse_json(x) for x in lb_raw]},
        },
        "processing": proc,
    }


def summarize(
    d: Dict[str, Any], teams: List[str], verbose: bool, raw: bool, source: str
) -> Dict[str, Any]:
    team_set = {norm_team(x) for x in teams}
    out: Dict[str, Any] = {
        "type": "queue",
        "status": "ok",
        "src": source,
        "time_bj": bj_time(d.get("timestamp", "1970-01-01T00:00:00Z"))
        if d.get("timestamp")
        else now_bj(),
        "teams": teams,
    }

    queues = d.get("queues") or {}
    task_items = (queues.get(Q_TASK) or {}).get("items") or []
    result_items = (queues.get(Q_RESULT) or {}).get("items") or []
    lb_items = (queues.get(Q_LEADERBOARD) or {}).get("items") or []
    proc = d.get("processing") or {}

    def is_hit(x: Dict[str, Any]) -> bool:
        return norm_team(x.get("team_name") or x.get("team_id") or x.get("team")) in team_set

    task_hits = [x for x in task_items if isinstance(x, dict) and is_hit(x)]
    result_hits = [x for x in result_items if isinstance(x, dict) and is_hit(x)]
    lb_hits = [x for x in lb_items if isinstance(x, dict) and is_hit(x)]
    proc_hits = [
        x
        for x in proc.values()
        if isinstance(x, dict) and is_hit(x)
    ]

    out["q"] = {
        "task": int((queues.get(Q_TASK) or {}).get("length", len(task_items)) or 0),
        "result": int((queues.get(Q_RESULT) or {}).get("length", len(result_items)) or 0),
        "leaderboard": int(
            (queues.get(Q_LEADERBOARD) or {}).get("length", len(lb_items)) or 0
        ),
        "processing": len(proc),
    }
    out["hits"] = {
        "task": len(task_hits),
        "result": len(result_hits),
        "leaderboard": len(lb_hits),
        "processing": len(proc_hits),
    }
    out["leaderboard"] = len(lb_hits)
    out["processing"] = len(proc_hits)

    if verbose:
        out["task_items"] = [pick_fields(x) for x in task_hits]
        out["result_items"] = [pick_fields(x) for x in result_hits]
        out["leaderboard_items"] = [pick_fields(x) for x in lb_hits]
        out["processing_items"] = [pick_fields(x) for x in proc_hits]

    if raw:
        out["raw"] = d
    return out


def print_text(r: Dict[str, Any]) -> None:
    if r.get("status") != "ok":
        print(f"STATUS: ERROR\nKIND: {r.get('kind', '')}\nMSG: {r.get('msg', '')}")
        return
    print(f"STATUS: OK")
    print(f"SRC: {r.get('src')}")
    print(f"TIME_BJ: {r.get('time_bj')}")
    print(f"TEAMS: {','.join(r.get('teams') or [])}")
    q = r.get("q") or {}
    h = r.get("hits") or {}
    print(
        "Q: "
        f"task={q.get('task', 0)} result={q.get('result', 0)} "
        f"lb={q.get('leaderboard', 0)} proc={q.get('processing', 0)}"
    )
    print(
        "HITS: "
        f"task={h.get('task', 0)} result={h.get('result', 0)} "
        f"lb={h.get('leaderboard', 0)} proc={h.get('processing', 0)}"
    )
    for key, title in (
        ("task_items", "TASK"),
        ("result_items", "RESULT"),
        ("leaderboard_items", "LB"),
        ("processing_items", "PROC"),
    ):
        for x in r.get(key) or []:
            print(f"{title}: " + jdumps(x))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Summarize OpenOperator OJ queue/processing status."
    )
    ap.add_argument(
        "teams", nargs="*", help="team names to track; default: team_91 team_42"
    )
    ap.add_argument("--url", default=DEFAULT_URL, help="snapshot API URL")
    ap.add_argument(
        "--source",
        choices=["auto", "redis", "snapshot"],
        default="auto",
        help="default: auto (redis first, snapshot fallback)",
    )
    ap.add_argument("--limit", type=int, default=20, help="redis list fetch limit")
    ap.add_argument(
        "--timeout", type=float, default=5.0, help="request timeout seconds; default: 5"
    )
    ap.add_argument(
        "--format",
        choices=["json", "jsonl", "text", "legacy"],
        default="json",
        help="default: json",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="include compact matching item summaries",
    )
    ap.add_argument("--raw", action="store_true", help="include raw source JSON")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    teams = args.teams or DEFAULT_TEAMS
    source = args.source
    try:
        if source in ("auto", "redis"):
            try:
                data = fetch_redis(args.limit)
                source = "redis"
            except Exception:
                if args.source == "redis":
                    raise
                data = fetch_snapshot(args.url, args.timeout)
                source = "snapshot"
        else:
            data = fetch_snapshot(args.url, args.timeout)
            source = "snapshot"
        result = summarize(data, teams, args.verbose, args.raw, source)
    except Exception as e:
        result = {
            "type": "queue",
            "status": "error",
            "kind": "fetch",
            "msg": str(e),
            "teams": teams,
        }

    if args.format == "jsonl":
        print(jdumps(result))
    elif args.format == "json":
        print(jdumps(result))
    elif args.format == "legacy" and result.get("status") == "ok":
        now = result.get("time_bj")
        print(f"{now}-leaderboard-{result.get('leaderboard')}")
        print(f"{now}-processing-{result.get('processing')}")
    else:
        print_text(result)

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
