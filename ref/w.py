#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watch_openoperator.py - 给人看的 OpenOperator 关注队伍榜单查看器

用途：把关注队伍写死在脚本顶部，然后直接查看这些队伍最近提交、各题名次、登顶题目等。
不依赖第三方库，只使用 Python 标准库。

先改这里：
  WATCH_TEAMS = [
      "BANGC08",
      "智算第八组",
      "@beiwenziyao/BANGC08",
  ]

常用命令：
  # 默认：看关注队伍最近 30 条提交，按提交时间倒序
  python watch_openoperator.py

  # 看最近 80 条，扫描全部题目
  python watch_openoperator.py recent --limit 80

  # 只看 1-40 题
  python watch_openoperator.py recent 1-40

  # 看关注队伍在每个题目的最好名次
  python watch_openoperator.py ranks all

  # 看关注队伍登顶了哪些题
  python watch_openoperator.py wins all

  # 看概览统计：登顶数、前3数、前10数、有提交题数等
  python watch_openoperator.py summary all

  # 看某题 top-k，并标记关注队伍
  python watch_openoperator.py top 085 --top 10

  # 临时追加关注队伍，不改文件
  python watch_openoperator.py recent --team BANGC09 --team 某队名

  # 输出 TSV，方便复制进表格
  python watch_openoperator.py ranks all --format tsv

字段说明：
  rank=原榜名次，problem=题号，name=题名，team=队伍/用户，repo=github/repo，
  score=分数，lat_us=延时微秒，time=提交时间，leader=该题当前第一名，gap=与第一名分数差。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# ========== 你主要改这里 ==========
WATCH_TEAMS = [
    "@geotle77/AICS-BangC_Game-17",
    "@AICS-2025-63H/openoperator",
    "@wbr258/openoperator-start-kit",
]

BASE_URL = "https://openoperator.cn/result/{:03d}.json"
DEFAULT_MAX_PROBLEM = 139
DEFAULT_TIMEOUT = 10.0
DEFAULT_JOBS = 24
DEFAULT_TZ = "Asia/Shanghai"

# 可选：如果脚本放在 AICS-openoperator/ref 旁边，能读到 get_tasks.py，就补充难度/分类。
try:
    from get_tasks import TASKS as _TASKS_TABLE  # type: ignore
except Exception:  # pragma: no cover
    _TASKS_TABLE = []

TASK_META: Dict[str, Dict[str, Any]] = {
    str(x.get("p") or "").zfill(3): x
    for x in _TASKS_TABLE
    if isinstance(x, dict) and x.get("p")
}


@dataclass
class Row:
    problem: str
    name: str
    rank: Optional[int]
    team: str
    repo: str
    score: Optional[float]
    latency: Optional[float]
    timestamp_raw: str
    timestamp_sort: float
    leader_team: str
    leader_score: Optional[float]
    leader_latency: Optional[float]
    difficulty: str = ""
    category: str = ""

    @property
    def is_win(self) -> bool:
        return self.rank == 1

    @property
    def gap(self) -> Optional[float]:
        if self.score is None or self.leader_score is None:
            return None
        return self.leader_score - self.score


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def parse_nums(args: Iterable[str], max_problem: int) -> List[int]:
    nums: List[int] = []
    if not list(args):
        args = ["all"]
    for arg in args:
        for part in str(arg).replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if part.lower() == "all":
                nums.extend(range(1, max_problem + 1))
            elif "-" in part:
                a, b = part.split("-", 1)
                start, end = int(a), int(b)
                if start > end:
                    start, end = end, start
                nums.extend(range(start, end + 1))
            else:
                nums.append(int(part))
    seen: Set[int] = set()
    out: List[int] = []
    for n in nums:
        if 1 <= n <= max_problem and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def team_key(x: Any) -> str:
    return str(x or "").strip().lower()


def aliases_for_item(item: Dict[str, Any]) -> Set[str]:
    vals: List[str] = []
    for k in ("user", "github", "repo", "repository"):
        v = str(item.get(k) or "").strip()
        if v:
            vals.append(v)
    out: Set[str] = set()
    for s in vals:
        out.add(team_key(s))
        if s.startswith("@"):
            out.add(team_key(s[1:]))
        if "/" in s:
            tail = s.rsplit("/", 1)[-1]
            out.add(team_key(tail))
            out.add(team_key("@" + s.lstrip("@")))
    return out


def is_watched(item: Dict[str, Any], teams: Set[str]) -> bool:
    return bool(aliases_for_item(item) & teams)


def to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def to_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def fmt_num(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return ""
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return (f"{x:.{digits}f}").rstrip("0").rstrip(".")


def parse_time_any(x: Any) -> Tuple[float, str]:
    """Return sortable epoch-ish seconds and original string."""
    if x is None:
        return (0.0, "")
    s = str(x).strip()
    if not s:
        return (0.0, "")

    # ISO-like: 2026-05-15T15:43:21Z / 2026-05-15 15:43:21
    ss = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            # OpenOperator 页面时间一般可当成本地展示时间；排序只需相对稳定。
            return (dt.timestamp(), s)
        return (dt.timestamp(), s)
    except Exception:
        pass

    digits = re.sub(r"\D", "", s)
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%y%m%d%H%M%S", "%y%m%d%H%M"):
        try:
            need = len(datetime.now().strftime(fmt))
            if len(digits) >= need:
                dt = datetime.strptime(digits[:need], fmt)
                return (dt.timestamp(), s)
        except Exception:
            continue
    return (0.0, s)


def format_time(x: str, tz_name: str) -> str:
    if not x:
        return ""
    ss = x.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is not None and ZoneInfo is not None:
            dt = dt.astimezone(ZoneInfo(tz_name))
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        pass
    digits = re.sub(r"\D", "", x)
    if len(digits) >= 12:
        if digits.startswith("20"):
            return f"{digits[4:6]}-{digits[6:8]} {digits[8:10]}:{digits[10:12]}"
        return f"{digits[2:4]}-{digits[4:6]} {digits[6:8]}:{digits[8:10]}"
    return x[:16]


def fetch_problem(num: int, timeout: float) -> Dict[str, Any]:
    url = BASE_URL.format(num)
    req = urllib.request.Request(
        url, headers={"User-Agent": "watch-openoperator-human/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        if not isinstance(data, dict):
            return {"problem_id": num, "error": "non-dict-json"}
        if not data.get("problem_id"):
            data["problem_id"] = num
        return data
    except urllib.error.HTTPError as e:
        return {"problem_id": num, "error": f"http-{e.code}"}
    except Exception as e:
        return {"problem_id": num, "error": str(e)[:160]}


def fetch_all(nums: Sequence[int], jobs: int, timeout: float) -> List[Dict[str, Any]]:
    jobs = max(1, min(jobs, 64))
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(fetch_problem, n, timeout) for n in nums]
        return [f.result() for f in futs]


def problem_meta(data: Dict[str, Any]) -> Tuple[str, str, str, str]:
    pid = f"{int(data.get('problem_id') or 0):03d}" if data.get("problem_id") else ""
    name = str(data.get("problem_name") or data.get("name") or "")
    meta = TASK_META.get(pid, {})
    difficulty = str(meta.get("d") or meta.get("difficulty") or "")
    category = str(meta.get("c") or meta.get("cat") or meta.get("category") or "")
    return pid, name, difficulty, category


def result_rows(data: Dict[str, Any], teams: Set[str]) -> List[Row]:
    if data.get("error"):
        return []
    results = data.get("results") or []
    if not isinstance(results, list) or not results:
        return []
    pid, name, difficulty, category = problem_meta(data)
    leader = results[0] if isinstance(results[0], dict) else {}
    leader_team = str(leader.get("user") or "")
    leader_score = to_float(leader.get("score"))
    leader_latency = to_float(leader.get("latency"))
    out: List[Row] = []
    for item in results:
        if not isinstance(item, dict) or not is_watched(item, teams):
            continue
        ts_sort, ts_raw = parse_time_any(item.get("timestamp"))
        out.append(
            Row(
                problem=pid,
                name=name,
                rank=to_int(item.get("rank")),
                team=str(item.get("user") or ""),
                repo=str(
                    item.get("github")
                    or item.get("repo")
                    or item.get("repository")
                    or ""
                ),
                score=to_float(item.get("score")),
                latency=to_float(item.get("latency")),
                timestamp_raw=ts_raw,
                timestamp_sort=ts_sort,
                leader_team=leader_team,
                leader_score=leader_score,
                leader_latency=leader_latency,
                difficulty=difficulty,
                category=category,
            )
        )
    return out


def best_rows(rows: Sequence[Row]) -> List[Row]:
    best: Dict[Tuple[str, str], Row] = {}
    for r in rows:
        # 一个队伍可能用 user/repo 多个别名；这里按 problem+team 聚合，保留最好名次。
        key = (r.problem, r.team or r.repo)
        old = best.get(key)
        if old is None:
            best[key] = r
            continue
        old_rank = old.rank if old.rank is not None else 10**9
        new_rank = r.rank if r.rank is not None else 10**9
        if (new_rank, -(r.timestamp_sort or 0)) < (
            old_rank,
            -(old.timestamp_sort or 0),
        ):
            best[key] = r
    return list(best.values())


def table(
    rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]], fmt: str
) -> None:
    if fmt == "json":
        print(json.dumps(list(rows), ensure_ascii=False, indent=2))
        return
    if fmt == "jsonl":
        for r in rows:
            print(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
        return
    if fmt == "tsv":
        print("\t".join(title for _, title in columns))
        for r in rows:
            print("\t".join(str(r.get(k, "")) for k, _ in columns))
        return

    # pretty
    widths: List[int] = []
    for k, title in columns:
        mx = len(title)
        for r in rows:
            mx = max(mx, len(str(r.get(k, ""))))
        widths.append(min(mx, 42))

    def cell(s: Any, w: int) -> str:
        text = str(s)
        if len(text) > w:
            text = text[: max(0, w - 1)] + "…"
        return text.ljust(w)

    print("  ".join(cell(title, w) for (_, title), w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(cell(r.get(k, ""), w) for (k, _), w in zip(columns, widths)))


def row_dict(r: Row, tz_name: str) -> Dict[str, Any]:
    return {
        "rank": r.rank or "",
        "problem": r.problem,
        "name": r.name,
        "team": r.team,
        "repo": r.repo,
        "score": fmt_num(r.score),
        "lat_us": fmt_num(r.latency),
        "time": format_time(r.timestamp_raw, tz_name),
        "win": "YES" if r.is_win else "",
        "leader": r.leader_team,
        "gap": fmt_num(r.gap),
        "diff": r.difficulty,
        "cat": r.category,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="给人看的 OpenOperator 关注队伍榜单查看器")
    sub = ap.add_subparsers(dest="cmd")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "ranges",
            nargs="*",
            default=["all"],
            help="题号范围：085、1-20、1,2,3、all；默认 all",
        )
        p.add_argument(
            "--team",
            action="append",
            default=[],
            help="临时追加关注队伍/用户/repo，可重复",
        )
        p.add_argument("--max-problem", type=int, default=DEFAULT_MAX_PROBLEM)
        p.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
        p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
        p.add_argument("--tz", default=DEFAULT_TZ, help="显示时区，默认 Asia/Shanghai")
        p.add_argument(
            "--format", choices=["pretty", "tsv", "json", "jsonl"], default="pretty"
        )
        p.add_argument(
            "--show-repo", action="store_true", help="pretty/tsv 中显示 repo 列"
        )

    p = sub.add_parser("recent", help="最近提交，默认命令")
    common(p)
    p.add_argument("--limit", type=int, default=30)

    p = sub.add_parser("ranks", help="关注队伍在每题的最好名次")
    common(p)
    p.add_argument("--limit", type=int, default=0, help="限制输出行数；0 表示不限制")

    p = sub.add_parser("wins", help="关注队伍登顶题目")
    common(p)

    p = sub.add_parser("summary", help="关注队伍概览统计")
    common(p)

    p = sub.add_parser("top", help="查看指定题目 top-k，并标记关注队伍")
    common(p)
    p.add_argument("--top", type=int, default=10)

    return ap


def watched_set(extra: Sequence[str]) -> Set[str]:
    vals = list(WATCH_TEAMS) + list(extra)
    out = {team_key(x) for x in vals if str(x).strip()}
    # 同时兼容有无 @ 的写法。
    out |= {team_key(x[1:]) for x in vals if str(x).strip().startswith("@")}
    return out


def load_rows(args: argparse.Namespace) -> Tuple[List[Row], List[Dict[str, Any]]]:
    nums = parse_nums(args.ranges, args.max_problem)
    if not nums:
        raise SystemExit("没有合法题号")
    teams = watched_set(args.team)
    if not teams:
        raise SystemExit("WATCH_TEAMS 为空：请先在脚本顶部写入关注队伍")
    data = fetch_all(nums, args.jobs, args.timeout)
    rows: List[Row] = []
    for d in data:
        rows.extend(result_rows(d, teams))
    return rows, data


def cmd_recent(args: argparse.Namespace) -> int:
    rows, _ = load_rows(args)
    rows = sorted(
        rows, key=lambda r: (r.timestamp_sort, -(r.rank or 999999)), reverse=True
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    dicts = [row_dict(r, args.tz) for r in rows]
    cols = [
        ("time", "time"),
        ("rank", "rank"),
        ("problem", "problem"),
        ("name", "name"),
        ("team", "team"),
    ]
    if args.show_repo:
        cols.append(("repo", "repo"))
    cols += [
        ("score", "score"),
        ("lat_us", "lat_us"),
        ("win", "win"),
        ("leader", "leader"),
        ("gap", "gap"),
    ]
    table(dicts, cols, args.format)
    return 0


def cmd_ranks(args: argparse.Namespace) -> int:
    rows, _ = load_rows(args)
    rows = best_rows(rows)
    rows = sorted(
        rows,
        key=lambda r: (r.rank if r.rank is not None else 999999, r.problem, r.team),
    )
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    dicts = [row_dict(r, args.tz) for r in rows]
    cols = [
        ("rank", "rank"),
        ("problem", "problem"),
        ("name", "name"),
        ("team", "team"),
    ]
    if args.show_repo:
        cols.append(("repo", "repo"))
    cols += [
        ("score", "score"),
        ("lat_us", "lat_us"),
        ("time", "time"),
        ("win", "win"),
        ("leader", "leader"),
        ("gap", "gap"),
    ]
    table(dicts, cols, args.format)
    return 0


def cmd_wins(args: argparse.Namespace) -> int:
    rows, _ = load_rows(args)
    rows = [r for r in best_rows(rows) if r.is_win]
    rows = sorted(rows, key=lambda r: (r.problem, r.team))
    dicts = [row_dict(r, args.tz) for r in rows]
    cols = [("problem", "problem"), ("name", "name"), ("team", "team")]
    if args.show_repo:
        cols.append(("repo", "repo"))
    cols += [("score", "score"), ("lat_us", "lat_us"), ("time", "time")]
    table(dicts, cols, args.format)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    rows, data = load_rows(args)
    best = best_rows(rows)
    by_team: Dict[str, Dict[str, Any]] = {}
    for r in best:
        key = r.team or r.repo or "<unknown>"
        s = by_team.setdefault(
            key,
            {
                "team": key,
                "submitted": 0,
                "wins": 0,
                "top3": 0,
                "top10": 0,
                "best_rank": "",
                "latest": 0.0,
            },
        )
        s["submitted"] += 1
        rank = r.rank if r.rank is not None else 999999
        if rank == 1:
            s["wins"] += 1
        if rank <= 3:
            s["top3"] += 1
        if rank <= 10:
            s["top10"] += 1
        old_best = s["best_rank"] if isinstance(s["best_rank"], int) else 999999
        s["best_rank"] = min(old_best, rank)
        s["latest"] = max(float(s["latest"]), r.timestamp_sort)
    out: List[Dict[str, Any]] = []
    for s in by_team.values():
        latest_raw = ""
        team_rows = [r for r in best if (r.team or r.repo or "<unknown>") == s["team"]]
        if team_rows:
            latest_raw = max(team_rows, key=lambda r: r.timestamp_sort).timestamp_raw
        out.append(
            {
                "team": s["team"],
                "submitted_problems": s["submitted"],
                "wins": s["wins"],
                "top3": s["top3"],
                "top10": s["top10"],
                "best_rank": s["best_rank"] if s["best_rank"] != 999999 else "",
                "latest": format_time(latest_raw, args.tz),
            }
        )
    out.sort(
        key=lambda x: (
            -int(x["wins"]),
            -int(x["top3"]),
            -int(x["top10"]),
            str(x["team"]),
        )
    )
    cols = [
        ("team", "team"),
        ("submitted_problems", "submitted"),
        ("wins", "wins"),
        ("top3", "top3"),
        ("top10", "top10"),
        ("best_rank", "best"),
        ("latest", "latest"),
    ]
    table(out, cols, args.format)

    errors = [d for d in data if d.get("error")]
    if errors:
        eprint(
            f"warning: {len(errors)} problems failed to fetch; use a smaller range or larger --timeout if needed"
        )
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    nums = parse_nums(args.ranges, args.max_problem)
    if not nums:
        raise SystemExit("没有合法题号")
    teams = watched_set(args.team)
    data = fetch_all(nums, args.jobs, args.timeout)
    out: List[Dict[str, Any]] = []
    for d in data:
        if d.get("error"):
            continue
        pid, name, _, _ = problem_meta(d)
        results = d.get("results") or []
        for item in results[: max(1, args.top)]:
            if not isinstance(item, dict):
                continue
            ts_sort, ts_raw = parse_time_any(item.get("timestamp"))
            _ = ts_sort
            out.append(
                {
                    "mark": "*" if is_watched(item, teams) else "",
                    "rank": to_int(item.get("rank")) or "",
                    "problem": pid,
                    "name": name,
                    "team": str(item.get("user") or ""),
                    "repo": str(item.get("github") or ""),
                    "score": fmt_num(to_float(item.get("score"))),
                    "lat_us": fmt_num(to_float(item.get("latency"))),
                    "time": format_time(ts_raw, args.tz),
                }
            )
    cols = [
        ("mark", "*"),
        ("rank", "rank"),
        ("problem", "problem"),
        ("name", "name"),
        ("team", "team"),
    ]
    if args.show_repo:
        cols.append(("repo", "repo"))
    cols += [("score", "score"), ("lat_us", "lat_us"), ("time", "time")]
    table(out, cols, args.format)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd is None:
        # 默认等价于 recent all
        args = ap.parse_args(["recent", "all"])

    if args.cmd == "recent":
        return cmd_recent(args)
    if args.cmd == "ranks":
        return cmd_ranks(args)
    if args.cmd == "wins":
        return cmd_wins(args)
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.cmd == "top":
        return cmd_top(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
