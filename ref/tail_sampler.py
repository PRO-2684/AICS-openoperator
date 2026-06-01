#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import redis_q
from oj_api import LAT_CONST, secret_for


BASE = "http://43.143.241.66:13000/api"
LEADERBOARD = "bangc:leaderboard_queue"
PROCESSING = "bangc:processing"
TASK_Q = "bangc:task_queue"


def jdumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def short_team(x: str) -> str:
    t = str(x).strip().replace("team_", "")
    return f"team_{t}"


def score_us(p: str, score: Any) -> float | None:
    name = {
        "026": "026_ELU",
        "027": "027_GELU",
        "028": "028_HardSigmoid",
        "029": "029_HardTanh",
    }.get(str(p))
    if not name:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s <= 0:
        return None
    return LAT_CONST[name] / s


def post_rerun(team: str, sha: str, timeout: float) -> dict[str, Any]:
    payload = {"secret": secret_for(team), "commit_sha": sha}
    r = requests.post(f"{BASE}/rerun", json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"text": r.text[:300]}
    if r.status_code >= 400:
        raise RuntimeError(f"http {r.status_code}: {data}")
    return data


def counts(r) -> tuple[int, int]:
    pipe = r.pipeline()
    pipe.hlen(PROCESSING)
    pipe.llen(TASK_Q)
    nproc, ntask = pipe.execute()
    return int(nproc), int(ntask)


def wait_gate(r, max_processing: int, max_task_queue: int, poll: float, timeout: float) -> bool:
    end = time.time() + timeout
    while True:
        nproc, ntask = counts(r)
        if nproc <= max_processing and ntask <= max_task_queue:
            return True
        if time.time() >= end:
            return False
        time.sleep(poll)


def read_new_lb(r, seen: set[str], teams: set[str], commit: str) -> list[dict[str, Any]]:
    rows = []
    for raw in r.lrange(LEADERBOARD, 0, 199):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        team = short_team(obj.get("team_name") or obj.get("team_id") or "")
        if teams and team not in teams:
            continue
        sha = str(obj.get("commit_sha") or "")
        if commit and not sha.startswith(commit):
            continue
        key = f"{team}:{obj.get('timestamp')}:{sha}:{obj.get('task_id')}"
        if key in seen:
            continue
        seen.add(key)
        for name, info in (obj.get("scores") or {}).items():
            if not isinstance(info, dict) or not info.get("passed"):
                continue
            p = name.split("_", 1)[0]
            us = score_us(p, info.get("score"))
            if us is None:
                continue
            rows.append(
                {
                    "type": "row",
                    "team": team,
                    "task": str(obj.get("task_id") or "")[:8],
                    "c": sha[:8],
                    "p": p,
                    "us": round(us, 3),
                }
            )
    return rows


def prime_seen_lb(r, seen: set[str], teams: set[str], commit: str) -> None:
    for raw in r.lrange(LEADERBOARD, 0, 199):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        team = short_team(obj.get("team_name") or obj.get("team_id") or "")
        if teams and team not in teams:
            continue
        sha = str(obj.get("commit_sha") or "")
        if commit and not sha.startswith(commit):
            continue
        seen.add(f"{team}:{obj.get('timestamp')}:{sha}:{obj.get('task_id')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptive low-tail rerun sampler.")
    ap.add_argument("commit", help="full or short commit sha accepted by OJ rerun")
    ap.add_argument("--team", action="append", default=[], help="team id; repeatable; default 42")
    ap.add_argument("-n", "--count", type=int, default=8, help="reruns per team")
    ap.add_argument("--max-processing", type=int, default=0)
    ap.add_argument("--max-task-queue", type=int, default=0)
    ap.add_argument("--idle-poll", type=float, default=1.0)
    ap.add_argument("--idle-timeout", type=float, default=180.0)
    ap.add_argument("--between", type=float, default=0.0, help="sleep after each enqueue")
    ap.add_argument("--collect", type=float, default=45.0, help="seconds to collect after enqueue")
    ap.add_argument("--target", action="append", default=[], help="op:us, e.g. 028:272.4")
    ap.add_argument(
        "--boost",
        action="append",
        default=[],
        help="op:us:count; when a live row is below us, immediately enqueue count extra reruns for that team",
    )
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--out", help="optional jsonl log path")
    args = ap.parse_args()

    teams = [short_team(t) for t in (args.team or ["42"])]
    targets = {}
    for item in args.target:
        p, v = item.split(":", 1)
        targets[p] = float(v)
    boosts = {}
    for item in args.boost:
        p, v, n = item.split(":", 2)
        boosts[p] = (float(v), int(n))

    r = redis_q.connect()
    run_id = uuid.uuid4().hex[:8]
    out_f = open(args.out, "a", encoding="utf-8") if args.out else None
    seen: set[str] = set()
    prime_seen_lb(r, seen, set(teams), args.commit[:8])

    def emit(obj: dict[str, Any]) -> None:
        obj.setdefault("run", run_id)
        line = jdumps(obj)
        print(line, flush=True)
        if out_f:
            out_f.write(line + "\n")
            out_f.flush()

    best: dict[str, float] = {}
    try:
        for i in range(1, args.count + 1):
            for team in teams:
                ok_gate = wait_gate(
                    r,
                    args.max_processing,
                    args.max_task_queue,
                    args.idle_poll,
                    args.idle_timeout,
                )
                nproc, ntask = counts(r)
                emit(
                    {
                        "type": "gate",
                        "ok": 1 if ok_gate else 0,
                        "i": i,
                        "team": team,
                        "processing": nproc,
                        "task_queue": ntask,
                    }
                )
                try:
                    data = post_rerun(team, args.commit, args.timeout)
                    emit(
                        {
                            "type": "rerun",
                            "ok": 1,
                            "i": i,
                            "team": team,
                            "c": args.commit[:8],
                            "status": data.get("status") or data.get("message") or data.get("ok"),
                        }
                    )
                except Exception as e:
                    emit({"type": "rerun", "ok": 0, "i": i, "team": team, "error": str(e)})
                if args.between:
                    time.sleep(args.between)

                end = time.time() + args.collect
                while time.time() < end:
                    hit = False
                    for row in read_new_lb(r, seen, set(teams), args.commit[:8]):
                        p = row["p"]
                        us = float(row["us"])
                        if p not in best or us < best[p]:
                            best[p] = us
                            row["best"] = 1
                        emit(row)
                        if p in targets and us < targets[p]:
                            emit({"type": "hit", "p": p, "us": us, "target": targets[p]})
                            hit = True
                        if p in boosts:
                            threshold, extra = boosts[p]
                            if us < threshold and extra > 0:
                                emit(
                                    {
                                        "type": "boost",
                                        "p": p,
                                        "us": us,
                                        "threshold": threshold,
                                        "team": row.get("team"),
                                        "count": extra,
                                    }
                                )
                                for bi in range(extra):
                                    try:
                                        data = post_rerun(
                                            str(row.get("team", team)).replace("team_", ""),
                                            args.commit,
                                            args.timeout,
                                        )
                                        emit(
                                            {
                                                "type": "boost_rerun",
                                                "ok": 1,
                                                "p": p,
                                                "team": row.get("team"),
                                                "j": bi + 1,
                                                "status": data.get("status")
                                                or data.get("message")
                                                or data.get("ok"),
                                            }
                                        )
                                    except Exception as e:
                                        emit(
                                            {
                                                "type": "boost_rerun",
                                                "ok": 0,
                                                "p": p,
                                                "team": row.get("team"),
                                                "j": bi + 1,
                                                "error": str(e),
                                            }
                                        )
                    if hit:
                        return 0
                    time.sleep(max(0.5, args.idle_poll))
        emit({"type": "summary", "best": {k: round(v, 3) for k, v in sorted(best.items())}})
        return 0
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
