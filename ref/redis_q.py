#!/usr/bin/env python3
# redis_queue_dump.py

import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import redis


# ====== 你补全这里 ======
REDIS_HOST = "43.143.241.66"
REDIS_PORT = 16379
REDIS_PASSWORD = "nfiqwovbifq"
REDIS_DB = 0
# =======================

TASK_Q = "bangc:task_queue"
RESULT_Q = "bangc:result_queue"
PROCESSING = "bangc:processing"
LEADERBOARD = "bangc:leaderboard_queue"

LAT_CONST = {
    "026_ELU": 0.09486 * 304.996191,
    "027_GELU": 0.085999 * 281.317079,
    "028_HardSigmoid": 0.083176 * 272.397728,
    "029_HardTanh": 0.107052 * 268.123113,
}


def pretty_json(raw: str):
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except Exception:
        return raw


def compact_obj(obj):
    out = {}
    team = obj.get("team_name") or obj.get("team_id")
    if team:
        out["team"] = f"team_{team}" if str(team).isdigit() else team
    for dst, keys in {
        "task": ("task_id",),
        "w": ("worker_id",),
        "c": ("commit_sha",),
        "repo": ("repo_full_name",),
        "err": ("error",),
    }.items():
        for k in keys:
            if obj.get(k):
                v = str(obj[k])
                out[dst] = v[:8] if dst in ("task", "c") else v
                break
    scores = obj.get("scores") or {}
    if isinstance(scores, dict) and scores:
        out["scores"] = [
            {
                "p": name.split("_", 1)[0],
                "ok": 1 if isinstance(info, dict) and info.get("passed") else 0,
                "s": round(float(info.get("score") or 0.0), 6)
                if isinstance(info, dict)
                else 0.0,
                **(
                    {
                        "us": round(
                            LAT_CONST[name] / float(info.get("score") or 0.0),
                            3,
                        )
                    }
                    if isinstance(info, dict)
                    and name in LAT_CONST
                    and float(info.get("score") or 0.0) > 0.0
                    else {}
                ),
            }
            for name, info in scores.items()
        ]
    return out


def connect():
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD or None,
        db=REDIS_DB,
        decode_responses=True,
        socket_timeout=10,
        socket_connect_timeout=10,
    )
    r.ping()
    return r


def dump_list(r, key: str, limit: int, output_dir: Path | None = None, compact=False):
    total = r.llen(key)
    end = limit - 1 if limit > 0 else -1
    items = r.lrange(key, 0, end)

    print(f"\n[{key}] total={total}, dumped={len(items)}")

    for i, raw in enumerate(items):
        print("=" * 100)
        print(f"{key}[{i}]")
        if compact:
            try:
                print(json.dumps(compact_obj(json.loads(raw)), ensure_ascii=False))
            except Exception:
                print(raw[:500])
        else:
            print(pretty_json(raw))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{key.replace(':', '_')}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for raw in items:
                try:
                    obj = json.loads(raw)
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except Exception:
                    f.write(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
        print(f"saved: {out}")


def dump_processing(r, output_dir: Path | None = None, compact=False):
    data = r.hgetall(PROCESSING)
    print(f"\n[{PROCESSING}] total={len(data)}")

    rows = []
    for task_id, raw in data.items():
        print("=" * 100)
        print(f"{PROCESSING}[{task_id}]")
        if compact:
            try:
                print(json.dumps(compact_obj(json.loads(raw)), ensure_ascii=False))
            except Exception:
                print(raw[:500])
        else:
            print(pretty_json(raw))

        try:
            obj = json.loads(raw)
        except Exception:
            obj = {"raw": raw}
        obj["_task_id"] = task_id
        rows.append(obj)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{PROCESSING.replace(':', '_')}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for obj in rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"saved: {out}")


def consume_one_task(r):
    """
    危险：会从正式任务队列弹走一个任务。
    仅在你明确要抢任务/备份任务时使用。
    """
    raw = r.lpop(TASK_Q)
    if raw is None:
        print("no task")
        return

    print(f"CONSUMED one item from {TASK_Q}:")
    print(pretty_json(raw))


def compact_processing(obj):
    try:
        return json.dumps(compact_obj(obj), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return json.dumps({"raw": str(obj)[:300]}, ensure_ascii=False)


def compact_leaderboard(obj):
    return compact_processing(obj)


def watch(r, interval: float, limit: int, team_filter: set[str], duration: float | None):
    seen_proc: set[str] = set()
    seen_lb: set[str] = set()
    end = None if duration is None else time.time() + duration
    while True:
        if end is not None and time.time() >= end:
            break

        proc = r.hgetall(PROCESSING)
        lb = r.lrange(LEADERBOARD, 0, max(0, limit - 1))

        for task_id, raw in proc.items():
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"raw": raw}
            team = obj.get("team_name") or obj.get("team_id")
            team = f"team_{team}" if str(team).isdigit() else team
            if team_filter and team not in team_filter:
                continue
            k = f"{task_id}:{obj.get('commit_sha','')}:{obj.get('worker_id','')}"
            if k in seen_proc:
                continue
            seen_proc.add(k)
            print(f"PROC {compact_processing(obj)}", flush=True)

        for raw in lb:
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"raw": raw}
            team = obj.get("team_name") or obj.get("team_id")
            team = f"team_{team}" if str(team).isdigit() else team
            if team_filter and team not in team_filter:
                continue
            k = f"{obj.get('team_name','')}:{obj.get('timestamp','')}:{json.dumps(obj.get('scores',{}), sort_keys=True, ensure_ascii=False)}"
            if k in seen_lb:
                continue
            seen_lb.add(k)
            print(f"LB   {compact_leaderboard(obj)}", flush=True)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=20, help="每个 list 最多抓多少条；0 表示全部"
    )
    parser.add_argument("--task", action="store_true", help="抓 task_queue")
    parser.add_argument("--result", action="store_true", help="抓 result_queue")
    parser.add_argument("--processing", action="store_true", help="抓 processing hash")
    parser.add_argument("--leaderboard", action="store_true", help="抓 leaderboard_queue")
    parser.add_argument("--all", action="store_true", help="抓 task/result/leaderboard/processing")
    parser.add_argument("--compact", action="store_true", help="单行短字段输出，省 token")
    parser.add_argument("--watch", action="store_true", help="持续观察 processing/leaderboard 的新增项")
    parser.add_argument("--interval", type=float, default=1.0, help="watch 轮询间隔秒")
    parser.add_argument("--duration", type=float, default=0.0, help="watch 持续秒数，0 表示一直运行")
    parser.add_argument("--team", action="append", default=[], help="watch 关注的 team，例如 team_42")
    parser.add_argument("--save", action="store_true", help="保存到 dumps/ 时间戳目录")
    parser.add_argument(
        "--consume-one", action="store_true", help="危险：LPOP 消费一个 task"
    )
    args = parser.parse_args()

    r = connect()

    output_dir = None
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("dumps") / ts

    if args.consume_one:
        consume_one_task(r)
        return

    if not any([args.task, args.result, args.processing, args.leaderboard, args.all]):
        args.all = True

    if args.watch:
        teams = {x for x in (args.team or []) if x}
        watch(r, args.interval, args.limit, teams, None if args.duration <= 0 else args.duration)
        return

    if args.all or args.task:
        dump_list(r, TASK_Q, args.limit, output_dir, args.compact)

    if args.all or args.result:
        dump_list(r, RESULT_Q, args.limit, output_dir, args.compact)

    if args.all or args.leaderboard:
        dump_list(r, LEADERBOARD, args.limit, output_dir, args.compact)

    if args.all or args.processing:
        dump_processing(r, output_dir, args.compact)


if __name__ == "__main__":
    main()
