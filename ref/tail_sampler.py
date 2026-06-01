#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
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

WORKER_CLASS = {
    "16": "10x56",
    "17": "10x56",
    "18": "10x56",
    "19": "10x56",
    "20": "10x56",
    "21": "10x56",
    "22": "10x56",
    "23": "10x56",
    "24": "10x56",
    "25": "10x56",
    "8": "8x56",
    "9": "8x56",
    "10": "8x56",
    "11": "8x56",
    "12": "8x56",
    "13": "8x56",
    "14": "8x56",
    "15": "8x56",
    "0": "8x112",
    "1": "8x112",
    "2": "8x112",
    "3": "8x112",
    "4": "8x112",
    "5": "8x112",
    "6": "8x112",
    "7": "8x112",
}


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


def worker_class(w: Any) -> str | None:
    if w is None:
        return None
    return WORKER_CLASS.get(str(w), "unknown")


def annotate_worker(item: dict[str, Any]) -> dict[str, Any]:
    cls = worker_class(item.get("w"))
    if cls:
        item["wc"] = cls
    return item


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


def _active_processing_count(r, max_age: float | None) -> int:
    if max_age is None:
        return int(r.hlen(PROCESSING))
    now = time.time()
    n = 0
    for raw in r.hgetall(PROCESSING).values():
        try:
            obj = json.loads(raw)
            started = obj.get("started_at") or obj.get("timestamp")
            if not started:
                n += 1
                continue
            ts = datetime.fromisoformat(str(started).replace("Z", "+00:00")).timestamp()
            if now - ts <= max_age:
                n += 1
        except Exception:
            n += 1
    return n


def _active_workers(r, max_age: float | None, limit: int) -> list[dict[str, Any]]:
    rows = []
    now = time.time()
    for raw in r.hgetall(PROCESSING).values():
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        started = obj.get("started_at") or obj.get("timestamp")
        if max_age is not None and started:
            try:
                ts = datetime.fromisoformat(str(started).replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = None
            if ts is not None and now - ts > max_age:
                continue
        wid = str(obj.get("worker_id") or obj.get("w") or "")
        if not wid:
            continue
        age = obj.get("age")
        if age is None and started:
            try:
                age = int(now - datetime.fromisoformat(str(started).replace("Z", "+00:00")).timestamp())
            except Exception:
                age = None
        rows.append(annotate_worker({"w": wid, "age": age}))
    rows.sort(key=lambda x: (x["age"] is None, x["age"] if x["age"] is not None else 10**9, x["w"]))
    out = []
    seen = set()
    for item in rows:
        key = item["w"]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def counts(r, max_age: float | None = None) -> tuple[int, int]:
    pipe = r.pipeline()
    pipe.llen(TASK_Q)
    (ntask,) = pipe.execute()
    return _active_processing_count(r, max_age), int(ntask)


def update_task_workers(r, task_workers: dict[str, dict[str, Any]], max_age: float | None) -> None:
    now = time.time()
    for task_id, raw in r.hgetall(PROCESSING).items():
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        started = obj.get("started_at") or obj.get("timestamp")
        age = None
        if started:
            try:
                age = int(now - datetime.fromisoformat(str(started).replace("Z", "+00:00")).timestamp())
            except Exception:
                age = None
        if max_age is not None and age is not None and age > max_age:
            continue
        wid = obj.get("worker_id") or obj.get("w")
        if wid is None:
            continue
        task_workers[str(task_id)[:8]] = annotate_worker({
            "w": str(wid),
            "age": age,
            "team": short_team(obj.get("team_name") or obj.get("team_id") or ""),
            "c": str(obj.get("commit_sha") or "")[:8],
        })


def wait_gate(
    r,
    max_processing: int,
    max_task_queue: int,
    poll: float,
    timeout: float,
    processing_max_age: float | None,
    require_workers: set[str] | None,
) -> bool:
    end = time.time() + timeout
    while True:
        nproc, ntask = counts(r, processing_max_age)
        workers = _active_workers(r, processing_max_age, 64)
        worker_ids = {str(x.get("w")) for x in workers if x.get("w")}
        if (
            nproc <= max_processing
            and ntask <= max_task_queue
            and (not require_workers or worker_ids.intersection(require_workers))
        ):
            return True
        if time.time() >= end:
            return False
        time.sleep(poll)


def read_new_lb(
    r,
    seen: set[str],
    teams: set[str],
    commit: str,
    task_workers: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
        task_short = str(obj.get("task_id") or "")[:8]
        worker = (task_workers or {}).get(task_short)
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
                    "task": task_short,
                    "c": sha[:8],
                    "p": p,
                    "us": round(us, 3),
                    **(
                        {
                            "w": worker.get("w"),
                            "wc": worker.get("wc"),
                            "w_age": worker.get("age"),
                        }
                        if worker
                        else {}
                    ),
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
    ap.add_argument(
        "--processing-max-age",
        type=float,
        default=None,
        help="count only processing tasks younger than this many seconds; ignores stale worker records",
    )
    ap.add_argument("--between", type=float, default=0.0, help="sleep after each enqueue")
    ap.add_argument("--collect", type=float, default=45.0, help="seconds to collect after enqueue")
    ap.add_argument("--target", action="append", default=[], help="op:us, e.g. 028:272.4")
    ap.add_argument(
        "--boost",
        action="append",
        default=[],
        help="op:us:count; when a live row is below us, immediately enqueue count extra reruns for that team",
    )
    ap.add_argument("--boost-between", type=float, default=0.0, help="sleep between boost rerun API calls")
    ap.add_argument(
        "--emit-workers",
        action="store_true",
        help="include active worker snapshot on gate/boost events",
    )
    ap.add_argument(
        "--worker-snapshot-limit",
        type=int,
        default=6,
        help="max workers to emit when --emit-workers is set",
    )
    ap.add_argument(
        "--require-worker",
        action="append",
        default=[],
        help="only pass gate when an active worker id matches one of these ids; repeatable",
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
    require_workers = {str(x).strip() for x in (args.require_worker or []) if str(x).strip()}
    task_workers: dict[str, dict[str, Any]] = {}
    try:
        for i in range(1, args.count + 1):
            for team in teams:
                update_task_workers(r, task_workers, args.processing_max_age)
                ok_gate = wait_gate(
                    r,
                    args.max_processing,
                    args.max_task_queue,
                    args.idle_poll,
                    args.idle_timeout,
                    args.processing_max_age,
                    require_workers,
                )
                nproc, ntask = counts(r, args.processing_max_age)
                emit(
                    {
                        "type": "gate",
                        "ok": 1 if ok_gate else 0,
                        "i": i,
                        "team": team,
                        "processing": nproc,
                        "task_queue": ntask,
                        **(
                            {"workers": _active_workers(r, args.processing_max_age, args.worker_snapshot_limit)}
                            if args.emit_workers
                            else {}
                        ),
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
                    update_task_workers(r, task_workers, args.processing_max_age)
                    for row in read_new_lb(r, seen, set(teams), args.commit[:8], task_workers):
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
                                        **(
                                            {
                                                "workers": _active_workers(
                                                    r, args.processing_max_age, args.worker_snapshot_limit
                                                )
                                            }
                                            if args.emit_workers
                                            else {}
                                        ),
                                    }
                                )
                                for bi in range(extra):
                                    if bi and args.boost_between:
                                        time.sleep(args.boost_between)
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
