#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact / agent-friendly OpenOperator leaderboard fetcher.

新增用法（默认 JSONL，字段尽量短）：
  # 每题当前登顶者：1 行 1 题，最省 token
  python get_leaderboard.py all --mode leaders
  # 输出示例：
  # {"p":"001","u":"PRO-2684","s":"300.25","us":"2","t":"2605151543","d":"basic","cat":"pointwise"}

  # 指定几支队伍当前登顶了哪些题（rank1 属于这些队伍）
  python get_leaderboard.py all --team BANGC08 --team 智算第八组 --mode team-leaders
  # 输出示例：
  # {"p":"085","u":"智算第八组","s":"300.25","us":"2","t":"2605151543"}

  # 当前登顶者中排除指定队伍（rank1 不属于这些队伍）
  python get_leaderboard.py all --team BANGC08 --mode other-leaders

  # 每题“排除指定队伍后”的最高名次者（不是原榜 rank1 过滤，而是重新找第一个非指定队伍）
  python get_leaderboard.py all --team BANGC08 --mode best-excl-team
  # 输出示例：
  # {"p":"085","r":2,"u":"other","s":"299.1","us":"2.1","t":"2605151601"}

  # 每题指定队伍自己的最好成绩（不是必须登顶）
  python get_leaderboard.py all --team BANGC08 --mode team-best

  # 常规每题 top-k；默认不输出仓库/GitHub 字段，除非加 --gh；默认不输出题名，除非加 --name
  python get_leaderboard.py 1-10 --top 3 --mode top --gh --name

输出字段说明（短字段）：
  p=题号, n=题名(可选), r=原榜名次, u=队伍/用户, g=github/repo(仅 --gh),
  s=score, us=latency_us, t=压缩时间 YYMMDDHHMM, d=难度, cat=领域/category,
  lu=榜单更新时间, c=结果数(仅 --count), st=状态, h=HTTP 状态, e=错误信息。

依赖：pip install aiohttp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Set

import aiohttp

try:
    from get_tasks import TASKS as TASKS_TABLE
except Exception:  # pragma: no cover
    TASKS_TABLE = []

BASE_URL = "https://openoperator.cn/result/{:03d}.json"
DEFAULT_MAX_PROBLEM = 139
TASK_META: Dict[str, Dict[str, Any]] = {
    str(x.get("p") or "").zfill(3): x for x in TASKS_TABLE if x.get("p")
}


def jdumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def parse_nums(args: Iterable[str], max_problem: int) -> List[int]:
    nums: List[int] = []
    for arg in args:
        for part in str(arg).split(","):
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


def compact_time(x: Any) -> str:
    """Return YYMMDDHHMM when parseable, else a compact digits prefix."""
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    # Handles ISO-ish strings like 2026-05-15T15:43:21Z or 2026-05-15 15:43.
    ss = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.strftime("%y%m%d%H%M")
    except Exception:
        pass
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 12:
        return digits[2:12] if digits.startswith("20") else digits[:10]
    return s[:16].replace(" ", "T")


def slim_num(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    try:
        f = float(x)
    except Exception:
        return x
    if f.is_integer():
        return str(int(f))
    return ("%.6f" % f).rstrip("0").rstrip(".")


def team_key(x: Any) -> str:
    return str(x or "").strip().lower()


def is_team(item: Dict[str, Any], teams: Set[str]) -> bool:
    if not teams:
        return False
    vals = [item.get("user"), item.get("github")]
    # Also match the repo/name tail because GitHub strings may be user/repo or @user/repo.
    for v in list(vals):
        s = str(v or "").strip()
        if "/" in s:
            vals.append(s.rsplit("/", 1)[-1])
        if s.startswith("@"):
            vals.append(s[1:])
    return any(team_key(v) in teams for v in vals if v)


def norm_entry(
    item: Dict[str, Any],
    *,
    gh: bool,
    name: str = "",
    pid: str = "",
    include_rank: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if pid:
        out["p"] = pid
    if name:
        out["n"] = name
    if include_rank and item.get("rank") not in (None, 1, "1"):
        out["r"] = item.get("rank")
    elif include_rank and item.get("rank") in (0, "0"):
        out["r"] = item.get("rank")
    user = item.get("user")
    if user not in (None, ""):
        out["u"] = user
    if gh and item.get("github") not in (None, ""):
        out["g"] = item.get("github")
    score = slim_num(item.get("score"))
    if score is not None:
        out["s"] = score
    latency = slim_num(item.get("latency"))
    if latency is not None:
        out["us"] = latency
    t = compact_time(item.get("timestamp"))
    if t:
        out["t"] = t
    return out


async def fetch_problem(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore, num: int
) -> Dict[str, Any]:
    pid = f"{num:03d}"
    async with sem:
        try:
            async with session.get(BASE_URL.format(num)) as resp:
                if resp.status != 200:
                    return {"p": pid, "st": "empty", "h": resp.status}
                data = await resp.json(content_type=None)
                if not isinstance(data, dict):
                    return {"p": pid, "st": "bad", "e": "non-dict-json"}
                return data
        except Exception as e:
            return {"p": pid, "st": "err", "e": str(e)[:120]}


async def fetch_all(
    nums: Sequence[int], concurrency: int, timeout: float
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    headers = {"User-Agent": "get-openoperator-compact/1.0"}
    async with aiohttp.ClientSession(timeout=timeout_obj, headers=headers) as session:
        return await asyncio.gather(*(fetch_problem(session, sem, n) for n in nums))


def problem_out(data: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    # Error/empty objects from fetch_problem are already short.
    if "results" not in data:
        return [data] if args.keep_empty else []

    pid = f"{int(data.get('problem_id') or 0):03d}" if data.get("problem_id") else ""
    # Some endpoints may omit problem_id; caller sorts by requested order but we still prefer provided id.
    name = data.get("problem_name") or ""
    name_s = name if args.name else ""
    lu = compact_time(data.get("last_updated"))
    results = data.get("results") or []
    teams = {team_key(x) for x in args.team if str(x).strip()}

    def with_meta(row: Dict[str, Any]) -> Dict[str, Any]:
        if args.task_meta:
            meta = TASK_META.get(row.get("p") or pid)
            if meta:
                if meta.get("d"):
                    row["d"] = meta["d"]
                if meta.get("c"):
                    row["cat"] = meta["c"]
        if lu and args.lu:
            row["lu"] = lu
        if args.count:
            row["c"] = len(results)
        return row

    if not results:
        return [with_meta({"p": pid, "st": "empty"})] if args.keep_empty else []

    if args.mode == "top":
        rows = [
            norm_entry(x, gh=args.gh, name=name_s, pid=pid, include_rank=True)
            for x in results[: args.top]
        ]
        return [with_meta(r) for r in rows]

    if args.mode == "leaders":
        return [
            with_meta(
                norm_entry(
                    results[0], gh=args.gh, name=name_s, pid=pid, include_rank=False
                )
            )
        ]

    if args.mode == "team-leaders":
        if is_team(results[0], teams):
            return [
                with_meta(
                    norm_entry(
                        results[0], gh=args.gh, name=name_s, pid=pid, include_rank=False
                    )
                )
            ]
        return []

    if args.mode == "other-leaders":
        if not is_team(results[0], teams):
            return [
                with_meta(
                    norm_entry(
                        results[0], gh=args.gh, name=name_s, pid=pid, include_rank=False
                    )
                )
            ]
        return []

    if args.mode == "best-excl-team":
        for x in results:
            if not is_team(x, teams):
                return [
                    with_meta(
                        norm_entry(
                            x, gh=args.gh, name=name_s, pid=pid, include_rank=True
                        )
                    )
                ]
        return [with_meta({"p": pid, "st": "no-other"})] if args.keep_empty else []

    if args.mode == "team-best":
        for x in results:
            if is_team(x, teams):
                return [
                    with_meta(
                        norm_entry(
                            x, gh=args.gh, name=name_s, pid=pid, include_rank=True
                        )
                    )
                ]
        return [with_meta({"p": pid, "st": "no-team"})] if args.keep_empty else []

    if args.mode == "team-all":
        rows = [
            norm_entry(x, gh=args.gh, name=name_s, pid=pid, include_rank=True)
            for x in results
            if is_team(x, teams)
        ]
        if rows:
            return [with_meta(r) for r in rows]
        return [with_meta({"p": pid, "st": "no-team"})] if args.keep_empty else []

    raise ValueError(f"unknown mode: {args.mode}")


def print_rows(rows: List[Dict[str, Any]], fmt: str) -> None:
    if fmt == "json":
        print(jdumps(rows))
    elif fmt == "tsv":
        # Stable compact columns; missing fields become empty.
        cols = [
            "p",
            "n",
            "r",
            "u",
            "g",
            "s",
            "us",
            "t",
            "d",
            "cat",
            "lu",
            "c",
            "st",
            "h",
            "e",
        ]
        for r in rows:
            print("\t".join(str(r.get(k, "")) for k in cols).rstrip("\t"))
    else:
        for r in rows:
            print(jdumps(r))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Fetch compact OpenOperator leaderboard data; default output is short JSONL."
    )
    ap.add_argument("ranges", nargs="+", help="problem ids: 1 2 3, 1-10, 1,2,3, or all")
    ap.add_argument(
        "--mode",
        choices=[
            "leaders",
            "top",
            "team-leaders",
            "other-leaders",
            "best-excl-team",
            "team-best",
            "team-all",
        ],
        default="leaders",
        help="leaders=each problem rank1; top=top-k rows; team-leaders=rank1 in teams; other-leaders=rank1 not in teams; best-excl-team=best row excluding teams; team-best=best team row; team-all=all matched team rows",
    )
    ap.add_argument(
        "--team",
        action="append",
        default=[],
        help="team/user/github/repo matcher; repeatable",
    )
    ap.add_argument(
        "--top",
        "--top-k",
        dest="top",
        type=int,
        default=1,
        help="rows per problem for --mode top",
    )
    ap.add_argument(
        "--fmt", "--format", choices=["jsonl", "json", "tsv"], default="jsonl"
    )
    ap.add_argument(
        "--gh",
        action="store_true",
        help="include github/repo field g; omitted by default to save tokens",
    )
    ap.add_argument(
        "--name",
        action="store_true",
        help="include problem name field n; omitted by default",
    )
    ap.add_argument(
        "--task-meta",
        dest="task_meta",
        action="store_true",
        default=True,
        help="include task difficulty d and category cat from ref/get_tasks.py (default)",
    )
    ap.add_argument(
        "--no-task-meta",
        dest="task_meta",
        action="store_false",
        help="omit task difficulty/category fields for minimum output",
    )
    ap.add_argument(
        "--lu", action="store_true", help="include leaderboard last_updated as lu"
    )
    ap.add_argument("--count", action="store_true", help="include result count c")
    ap.add_argument(
        "--keep-empty",
        action="store_true",
        help="emit empty/error/no-match rows instead of dropping them",
    )
    ap.add_argument(
        "--max-problem",
        type=int,
        default=DEFAULT_MAX_PROBLEM,
        help="range used by all; default: 139",
    )
    ap.add_argument(
        "--concurrency", type=int, default=32, help="max concurrent requests"
    )
    ap.add_argument(
        "--timeout", type=float, default=10.0, help="total request timeout seconds"
    )
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    args.top = max(1, args.top)
    nums = parse_nums(args.ranges, args.max_problem)
    if not nums:
        print(jdumps({"st": "err", "e": "no-valid-problem-id"}))
        return 2
    if (
        args.mode
        in {"team-leaders", "other-leaders", "best-excl-team", "team-best", "team-all"}
        and not args.team
    ):
        print(jdumps({"st": "err", "e": "--team-required-for-mode"}))
        return 2

    fetched = asyncio.run(fetch_all(nums, max(1, args.concurrency), args.timeout))
    rows: List[Dict[str, Any]] = []
    for n, data in zip(nums, fetched):
        if "results" in data and not data.get("problem_id"):
            data = dict(data)
            data["problem_id"] = n
        rows.extend(problem_out(data, args))
    print_rows(rows, args.fmt)
    return 0 if all(r.get("st") not in {"err", "bad"} for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
