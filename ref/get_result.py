#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


def sh(cmd: List[str]) -> str:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or "command failed")
    return p.stdout


def repo_detect(repo: Optional[str]) -> str:
    if repo:
        return repo
    return sh(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]
    ).strip()


def resolve_sha(x: str, no_resolve: bool) -> str:
    if no_resolve:
        return x
    try:
        return sh(["git", "rev-parse", x]).strip()
    except Exception:
        return x


def gh_comments(repo: str, sha: str) -> List[Dict[str, Any]]:
    out = sh(["gh", "api", f"repos/{repo}/commits/{sha}/comments", "--paginate"])
    data = json.loads(out)
    return data if isinstance(data, list) else []


def clean_cell(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        s = s[1:-1]
    return s.strip()


def split_md_row(line: str) -> List[str]:
    return [clean_cell(x) for x in line.strip().strip("|").split("|")]


def is_sep(cells: List[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells)


def find_table(body: str) -> List[Dict[str, str]]:
    in_overview = False
    table: List[str] = []

    for line in body.splitlines():
        if re.match(r"^###\s+.*评测总览", line):
            in_overview = True
            continue

        if not in_overview:
            continue

        if re.match(r"^###\s+", line):
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

    out = []
    for r in data:
        if len(r) == len(header):
            out.append(dict(zip(header, r)))
    return out


def norm_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "p": row.get("题目", ""),
        "s": row.get("得分", ""),
        "acc": row.get("精度检查", ""),
        "us": row.get("延迟", "").replace(" us", ""),
        "ok": row.get("状态", ""),
    }


def extract_outputs(body: str) -> List[str]:
    return re.findall(r"\*\*输出:\*\*\s*```(?:\w+)?\n(.*?)\n```", body, re.S)


def fetch_one(
    repo: str, commit: str, no_resolve: bool, verbose: bool, full: bool
) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "c": commit,
        "ok": 0,
        "rows": [],
    }

    try:
        sha = resolve_sha(commit, no_resolve)
        if verbose:
            r["sha"] = sha

        comments = gh_comments(repo, sha)
        if verbose:
            r["n_comments"] = len(comments)

        for cmt in comments:
            body = cmt.get("body") or ""
            if full:
                outs = extract_outputs(body)
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

            rows = find_table(body)
            if not rows:
                continue

            r["rows"].extend(norm_row(x) for x in rows)

            if verbose:
                r.setdefault("comments", []).append(
                    {
                        "user": ((cmt.get("user") or {}).get("login")),
                        "at": cmt.get("created_at"),
                        "url": cmt.get("html_url"),
                    }
                )

        r["ok"] = 1 if r["rows"] else 0
        if not r["ok"]:
            r["err"] = "no_table"

    except Exception as e:
        r["err"] = str(e)

    return r


def read_commits(args: argparse.Namespace) -> List[str]:
    xs = list(args.commits)
    if args.stdin:
        xs += re.split(r"[\s,，]+", sys.stdin.read().strip())
    return [x for x in xs if x]


def print_text(results: List[Dict[str, Any]]) -> None:
    for r in results:
        if not r.get("rows"):
            print(f"{r.get('c')}\tERR\t{r.get('err', 'no_table')}")
            continue
        for row in r["rows"]:
            print(
                f"{r.get('c')}\t"
                f"{row.get('p')}\t"
                f"{row.get('s')}\t"
                f"{row.get('acc')}\t"
                f"{row.get('us')}us\t"
                f"{row.get('ok')}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract compact eval overview from GitHub commit comments."
    )
    ap.add_argument("commits", nargs="*", help="commit hashes")
    ap.add_argument("--repo", help="OWNER/REPO; default: current gh repo")
    ap.add_argument("--stdin", action="store_true", help="read commits from stdin")
    ap.add_argument("--no-resolve", action="store_true", help="do not git rev-parse")
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="include sha/comment metadata"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="include raw stdout/stderr output blocks from eval comments",
    )
    ap.add_argument(
        "--format",
        choices=["jsonl", "json", "text"],
        default="jsonl",
        help="default: jsonl",
    )
    args = ap.parse_args()

    commits = read_commits(args)
    if not commits:
        print(
            "usage: get_eval.py COMMIT [COMMIT...] [--repo OWNER/REPO]", file=sys.stderr
        )
        return 2

    try:
        repo = repo_detect(args.repo)
    except Exception as e:
        print(f"repo_error: {e}", file=sys.stderr)
        return 2

    results = [
        fetch_one(repo, c, args.no_resolve, args.verbose, args.full)
        for c in commits
    ]

    if args.format == "jsonl":
        for r in results:
            print(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
    elif args.format == "json":
        print(json.dumps(results, ensure_ascii=False, separators=(",", ":")))
    else:
        print_text(results)

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
