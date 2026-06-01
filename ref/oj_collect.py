#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import get_result


def jdumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def parse_hash_file(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        commit, op = parts[0], parts[1].zfill(3) if parts[1].isdigit() else parts[1]
        idx = parts[-1] if parts[-1].isdigit() else ""
        msg = " ".join(parts[2:-1] if idx else parts[2:])
        out.append({"commit": commit, "op": op, "msg": msg, "idx": idx})
    return out


def select_entries(entries: List[Dict[str, str]], args) -> List[Dict[str, str]]:
    if args.tail:
        entries = entries[-args.tail :]
    if args.ops:
        want = {x.zfill(3) if x.isdigit() else x for x in args.ops.replace(",", " ").split()}
        entries = [e for e in entries if e["op"] in want]
    if args.grep:
        entries = [e for e in entries if args.grep in e["msg"]]
    if args.commits:
        existing = {e["commit"] for e in entries}
        for c in args.commits:
            if c not in existing:
                entries.append({"commit": c, "op": "", "msg": "adhoc", "idx": ""})
    seen = set()
    out = []
    for e in entries:
        c = e["commit"]
        if c and c not in seen:
            seen.add(c)
            out.append(e)
    return out


def fetch_all(repo: str, entries: List[Dict[str, str]], jobs: int) -> Dict[str, Dict[str, Any]]:
    ns = SimpleNamespace(
        no_resolve=False,
        verbose=False,
        full=False,
        max_output_chars=0,
        token=get_result.gh_token(),
        gh_cli=False,
    )
    commits = [e["commit"] for e in entries]
    results = get_result.fetch_many(repo, commits, ns, max(1, min(jobs, len(entries) or 1)))
    return {r.get("commit", ""): r for r in results}


def median(xs: List[float]) -> float | None:
    return statistics.median(xs) if xs else None


def q(xs: List[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = p * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def summarize(entries: List[Dict[str, str]], results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], Dict[str, Any]] = {}
    for e in entries:
        key = (e.get("op", ""), e.get("msg", ""))
        g = groups.setdefault(
            key,
            {
                "op": key[0],
                "msg": key[1],
                "commits": 0,
                "ok_commits": 0,
                "empty": 0,
                "rows": 0,
                "best_us": None,
                "best_commit": "",
                "best_acc": "",
                "values": [],
            },
        )
        g["commits"] += 1
        r = results.get(e["commit"]) or {}
        if r.get("status") != "ok":
            g["empty"] += 1
            continue
        g["ok_commits"] += 1
        for row in r.get("rows", []):
            us = row.get("latency_us")
            if us is None:
                continue
            g["rows"] += 1
            g["values"].append(float(us))
            if g["best_us"] is None or float(us) < g["best_us"]:
                g["best_us"] = float(us)
                g["best_commit"] = e["commit"]
                g["best_acc"] = row.get("acc", "")

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        vals = g.pop("values")
        g["median_us"] = median(vals)
        g["p25_us"] = q(vals, 0.25)
        g["p75_us"] = q(vals, 0.75)
        out.append(g)
    return sorted(out, key=lambda x: (x.get("op", ""), x.get("msg", "")))


def print_tsv(rows: Iterable[Dict[str, Any]]) -> None:
    print("op\tmsg\tcommits\tok\tempty\trows\tbest_us\tmedian_us\tp25_us\tp75_us\tbest_commit\tbest_acc")
    for r in rows:
        def f(x):
            return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        print(
            "\t".join(
                [
                    f(r.get("op")),
                    f(r.get("msg")),
                    f(r.get("commits")),
                    f(r.get("ok_commits")),
                    f(r.get("empty")),
                    f(r.get("rows")),
                    f(r.get("best_us")),
                    f(r.get("median_us")),
                    f(r.get("p25_us")),
                    f(r.get("p75_us")),
                    f(r.get("best_commit")),
                    f(r.get("best_acc")),
                ]
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact OJ result aggregation from ref/.oj_repeat_hashes")
    ap.add_argument("commits", nargs="*")
    ap.add_argument("--hash-file", default="ref/.oj_repeat_hashes")
    ap.add_argument("--ops", default="")
    ap.add_argument("--grep", default="")
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--repo", default="")
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--format", choices=["tsv", "jsonl", "json"], default="tsv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    entries = parse_hash_file(root / args.hash_file)
    entries = select_entries(entries, args)
    if not entries:
        print("no entries", file=sys.stderr)
        return 2
    repo = get_result.repo_detect(args.repo or None)
    results = fetch_all(repo, entries, args.jobs)
    rows = summarize(entries, results)
    if args.format == "json":
        print(jdumps(rows))
    elif args.format == "jsonl":
        for r in rows:
            print(jdumps(r))
    else:
        print_tsv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
