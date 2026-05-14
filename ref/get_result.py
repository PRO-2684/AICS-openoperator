#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-friendly extractor for OpenOperator eval tables in GitHub commit comments.

Default output is compact JSONL, one object per commit:
  {"type":"eval","status":"ok","commit":"...","rows":[...]}

Use --format text for human-readable TSV, --verbose for comment metadata,
and --full for raw **输出:** code blocks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional


def jdumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "command failed").strip()
        raise RuntimeError(f"{' '.join(cmd)}: {msg}")
    return p.stdout


def repo_detect(repo: Optional[str]) -> str:
    if repo:
        return repo
    return run_cmd(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]
    ).strip()


def resolve_sha(ref: str, no_resolve: bool) -> str:
    if no_resolve:
        return ref
    try:
        return run_cmd(["git", "rev-parse", ref]).strip()
    except Exception:
        return ref


def gh_comments(repo: str, sha: str) -> List[Dict[str, Any]]:
    out = run_cmd(["gh", "api", f"repos/{repo}/commits/{sha}/comments", "--paginate"])
    data = json.loads(out)
    return data if isinstance(data, list) else []


def clean_cell(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        s = s[1:-1]
    # Markdown image/status icons sometimes carry extra spaces; keep the semantic text only.
    return re.sub(r"\s+", " ", s).strip()


def split_md_row(line: str) -> List[str]:
    # Minimal markdown table splitter. Handles escaped pipes well enough for common bot tables.
    line = line.strip().strip("|")
    cells: List[str] = []
    cur: List[str] = []
    esc = False
    for ch in line:
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
            cur.append(ch)
        elif ch == "|":
            cells.append(clean_cell("".join(cur)))
            cur = []
        else:
            cur.append(ch)
    cells.append(clean_cell("".join(cur)))
    return cells


def is_sep(cells: List[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells)


def find_overview_table(body: str) -> List[Dict[str, str]]:
    in_overview = False
    table: List[str] = []

    for line in body.splitlines():
        if re.match(r"^\s*#{2,4}\s+.*评测总览", line):
            in_overview = True
            table = []
            continue

        if not in_overview:
            continue

        if re.match(r"^\s*#{2,4}\s+", line):
            break

        if line.strip().startswith("|"):
            table.append(line)
        elif table:
            break

    if len(table) < 2:
        return []

    rows = [split_md_row(x) for x in table]
    header = rows[0]
    data = rows[1:]
    if data and is_sep(data[0]):
        data = data[1:]

    out: List[Dict[str, str]] = []
    for r in data:
        if len(r) == len(header):
            out.append(dict(zip(header, r)))
    return out


def parse_latency_us(s: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s or "")
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def norm_row(row: Dict[str, str]) -> Dict[str, Any]:
    acc = row.get("精度检查", "")
    state = row.get("状态", "")
    latency_raw = row.get("延迟", "")
    return {
        "problem": row.get("题目", ""),
        "score": row.get("得分", ""),
        "acc": acc,
        "latency_us": parse_latency_us(latency_raw),
        "latency_raw": latency_raw,
        "state": state,
    }


def extract_outputs(body: str, max_chars: int = 20000) -> List[str]:
    outs = re.findall(r"\*\*输出:\*\*\s*```(?:\w+)?\n(.*?)\n```", body, re.S)
    clipped: List[str] = []
    for text in outs:
        if len(text) > max_chars:
            clipped.append(
                text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>"
            )
        else:
            clipped.append(text)
    return clipped


def fetch_one(repo: str, commit: str, args: argparse.Namespace) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "type": "eval",
        "status": "empty",
        "commit": commit,
        "rows": [],
    }
    try:
        sha = resolve_sha(commit, args.no_resolve)
        if args.verbose or sha != commit:
            r["sha"] = sha

        comments = gh_comments(repo, sha)
        if args.verbose:
            r["n_comments"] = len(comments)

        for cmt in comments:
            body = cmt.get("body") or ""
            rows = find_overview_table(body)
            if rows:
                r["rows"].extend(norm_row(x) for x in rows)
                if args.verbose:
                    r.setdefault("comments", []).append(
                        {
                            "user": ((cmt.get("user") or {}).get("login")),
                            "at": cmt.get("created_at"),
                            "url": cmt.get("html_url"),
                        }
                    )

            if args.full:
                outs = extract_outputs(body, args.max_output_chars)
                if outs:
                    r.setdefault("outputs", []).extend(
                        {
                            "user": ((cmt.get("user") or {}).get("login")),
                            "at": cmt.get("created_at"),
                            "url": cmt.get("html_url"),
                            "text": out,
                        }
                        for out in outs
                    )

        if r["rows"]:
            r["status"] = "ok"
        else:
            r["reason"] = "no_table"
    except Exception as e:
        r.update({"status": "error", "kind": "fetch", "msg": str(e)})
    return r


def read_commits(args: argparse.Namespace) -> List[str]:
    xs = list(args.commits)
    if args.stdin:
        xs += re.split(r"[\s,，]+", sys.stdin.read().strip())
    # Preserve order while deduplicating.
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def print_text(results: Iterable[Dict[str, Any]]) -> None:
    for r in results:
        if r.get("status") != "ok":
            print(
                f"{r.get('commit')}\t{str(r.get('status')).upper()}\t{r.get('reason') or r.get('msg') or ''}"
            )
            continue
        for row in r.get("rows", []):
            us = row.get("latency_us")
            us_s = "" if us is None else f"{us:g}us"
            print(
                f"{r.get('commit')}\t{row.get('problem')}\t{row.get('score')}\t{row.get('acc')}\t{us_s}\t{row.get('state')}"
            )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract compact eval overview from GitHub commit comments."
    )
    ap.add_argument("commits", nargs="*", help="commit hashes/refs")
    ap.add_argument("--repo", help="OWNER/REPO; default: current gh repo")
    ap.add_argument(
        "--stdin",
        action="store_true",
        help="read commits from stdin, split by whitespace/comma",
    )
    ap.add_argument(
        "--no-resolve", action="store_true", help="do not git rev-parse refs"
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="include sha/comment metadata"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="include raw **输出:** blocks from eval comments",
    )
    ap.add_argument(
        "--max-output-chars",
        type=int,
        default=20000,
        help="clip each --full output block; default: 20000",
    )
    ap.add_argument(
        "--format",
        choices=["jsonl", "json", "text"],
        default="jsonl",
        help="default: jsonl",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    commits = read_commits(args)
    if not commits:
        print("STATUS: ERROR\nREASON: no commits provided", file=sys.stderr)
        print(
            "USAGE: get_result_optimized.py COMMIT [COMMIT...] [--repo OWNER/REPO]",
            file=sys.stderr,
        )
        return 2

    try:
        repo = repo_detect(args.repo)
    except Exception as e:
        err = {"type": "eval", "status": "error", "kind": "repo_detect", "msg": str(e)}
        print(
            jdumps(err)
            if args.format != "text"
            else f"STATUS: ERROR\nKIND: repo_detect\nMSG: {e}"
        )
        return 2

    results = [fetch_one(repo, c, args) for c in commits]

    if args.format == "jsonl":
        for r in results:
            print(jdumps(r))
    elif args.format == "json":
        print(jdumps(results))
    else:
        print_text(results)

    return 0 if all(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
