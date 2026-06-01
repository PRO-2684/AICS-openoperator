#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_URL = "http://43.143.241.66:8000/webhook"
DEFAULT_REPO = "PRO-2684/AICS-openoperator"
DEFAULT_BRANCH = "main"
SCRIPT_DIR = Path(__file__).resolve().parent
SECRET_FILE = SCRIPT_DIR / "webhook_secrets.json"


def run(cmd: List[str], *, check: bool = True, echo: bool = False) -> str:
    if echo:
        print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(
        cmd, cwd=repo_root(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if check and p.returncode != 0:
        msg = p.stderr.strip() or p.stdout.strip() or f"{cmd[0]} failed"
        raise RuntimeError(msg)
    return p.stdout.strip()


def repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        return Path(out)
    except Exception:
        return Path.cwd()


def git_config(key: str, default: str = "") -> str:
    return run(["git", "config", "--get", key], check=False) or default


def current_branch() -> str:
    return run(["git", "branch", "--show-current"], check=False) or DEFAULT_BRANCH


def normalize_repo(repo: Optional[str]) -> str:
    if repo:
        return repo
    url = run(["git", "remote", "get-url", "origin"], check=False)
    if not url:
        return DEFAULT_REPO
    url = url.removesuffix(".git")
    m = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/]+)$", url)
    return m.group("repo") if m else DEFAULT_REPO


def repo_name(full_name: str) -> str:
    return full_name.rsplit("/", 1)[-1]


def resolve_commit(x: str) -> str:
    return run(["git", "rev-parse", x])


def short(sha: str) -> str:
    return sha[:7]


def commit_meta(sha: str) -> Dict[str, Any]:
    fmt = "%H%x00%P%x00%an%x00%ae%x00%aI%x00%s"
    raw = run(["git", "show", "-s", f"--format={fmt}", sha])
    full, parents, author_name, author_email, timestamp, message = raw.split("\x00", 5)
    files = run(["git", "show", "--name-only", "--format=", full], check=False)
    changed = [x for x in files.splitlines() if x.strip()]
    parent_list = parents.split() if parents else []
    return {
        "sha": full,
        "before": parent_list[0] if parent_list else "0" * 40,
        "author_name": author_name,
        "author_email": author_email,
        "timestamp": timestamp,
        "message": message,
        "changed": changed,
    }


def commits_from_last(n: int) -> List[str]:
    if n <= 0:
        return []
    out = run(["git", "rev-list", f"--max-count={n}", "HEAD"])
    return list(reversed([x for x in out.splitlines() if x]))


def commits_from_range(spec: str) -> List[str]:
    out = run(["git", "rev-list", "--reverse", spec])
    return [x for x in out.splitlines() if x]


def read_stdin_commits() -> List[str]:
    data = sys.stdin.read().strip()
    return [x for x in re.split(r"[\s,，]+", data) if x]


def load_secret_file() -> Dict[str, str]:
    if not SECRET_FILE.exists():
        return {}
    with SECRET_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{SECRET_FILE} must contain a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def secret_keys_for_team(team: str) -> List[str]:
    t = team.strip().lower().replace("team_", "")
    return [
        f"team_{t}",
        f"team{t}",
        t,
        f"TEAM_{t}",
        f"TEAM{t}",
    ]


def resolve_secret(args: argparse.Namespace, team: Optional[str] = None) -> Tuple[str, str]:
    if args.secret and team in (None, args.team):
        return args.secret, "--secret"

    env_keys = []
    team = team or args.team
    if team:
        t = team.strip().replace("team_", "")
        env_keys += [
            f"TEAM{t}_WEBHOOK_SECRET",
            f"TEAM_{t}_WEBHOOK_SECRET",
            f"WEBHOOK_SECRET_TEAM{t}",
        ]
    env_keys.append("WEBHOOK_SECRET")

    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return value, f"${key}"

    secrets = load_secret_file()
    if team:
        for key in secret_keys_for_team(team):
            if secrets.get(key):
                return secrets[key], str(SECRET_FILE)
    for key in ("default", "team_42", "team42", "42"):
        if secrets.get(key):
            return secrets[key], str(SECRET_FILE)

    raise RuntimeError(
        "missing webhook secret; use --secret, WEBHOOK_SECRET, "
        "TEAM42_WEBHOOK_SECRET, or ref/webhook_secrets.json"
    )


def parse_teams(value: str) -> List[str]:
    if value.lower() in ("all", "both"):
        return ["42", "91"]
    xs = [x.strip().replace("team_", "") for x in re.split(r"[,，\s]+", value) if x.strip()]
    return xs or ["42"]


def make_payload(meta: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    repo = normalize_repo(args.repo)
    name = repo_name(repo)
    branch = args.branch or current_branch() or DEFAULT_BRANCH
    html_url = f"https://github.com/{repo}"
    clone_url = f"{html_url}.git"
    login = args.sender or git_config("github.user", "MosRat")
    pusher_name = args.pusher_name or git_config("user.name", login)
    pusher_email = args.pusher_email or git_config("user.email", "")

    commit = {
        "id": meta["sha"],
        "tree_id": "",
        "distinct": True,
        "message": meta["message"],
        "timestamp": meta["timestamp"],
        "url": f"{html_url}/commit/{meta['sha']}",
        "author": {
            "name": meta["author_name"],
            "email": meta["author_email"],
            "username": login,
        },
        "committer": {
            "name": pusher_name,
            "email": pusher_email,
            "username": login,
        },
        "added": [],
        "removed": [],
        "modified": meta["changed"],
    }

    return {
        "ref": f"refs/heads/{branch}",
        "before": args.before or meta["before"],
        "after": meta["sha"],
        "created": False,
        "deleted": False,
        "forced": False,
        "base_ref": None,
        "compare": f"{html_url}/compare/{args.before or meta['before']}...{meta['sha']}",
        "commits": [commit],
        "head_commit": commit,
        "repository": {
            "id": args.repository_id,
            "node_id": "",
            "name": name,
            "full_name": repo,
            "private": True,
            "owner": {
                "name": repo.split("/", 1)[0],
                "email": pusher_email,
                "login": repo.split("/", 1)[0],
                "id": 0,
            },
            "html_url": html_url,
            "clone_url": clone_url,
            "default_branch": branch,
        },
        "pusher": {"name": pusher_name, "email": pusher_email},
        "sender": {"login": login, "id": 0, "type": "User"},
    }


def encode_payload(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def headers(payload_bytes: bytes, secret: str, args: argparse.Namespace) -> Dict[str, str]:
    signature = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "User-Agent": args.user_agent,
        "X-GitHub-Event": "push",
        "X-GitHub-Delivery": args.delivery or str(uuid.uuid4()),
        "X-Hub-Signature-256": f"sha256={signature}",
    }


def send_one(sha: str, secret: str, args: argparse.Namespace) -> bool:
    meta = commit_meta(resolve_commit(sha))
    payload = make_payload(meta, args)
    body = encode_payload(payload)
    h = headers(body, secret, args)

    if args.print_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(f"dry c={short(meta['sha'])} repo={payload['repository']['full_name']} ref={payload['ref']}")
        return True

    r = requests.post(args.url, data=body, headers=h, timeout=args.timeout)
    text = r.text.strip()
    ok = 200 <= r.status_code < 300
    mark = "ok" if ok else "err"
    if args.verbose:
        print(f"{mark} c={short(meta['sha'])} status={r.status_code} url={args.url} resp={text[:1000]}")
    else:
        print(f"{mark} c={short(meta['sha'])} status={r.status_code} resp={text[:160]}")
    return ok


def git_staged_dirty() -> bool:
    return subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0


def commit_push_if_requested(args: argparse.Namespace) -> Optional[str]:
    if not args.commit_msg and not args.push:
        return None

    if args.dry_run:
        if args.commit_msg:
            stage = " ".join(args.stage or ["*.mlu", "config"])
            print(f"dry git-add={stage} commit={args.commit_msg!r} push=1")
        elif args.push:
            print("dry push=1")
        return resolve_commit("HEAD")

    if args.commit_msg:
        stage = args.stage or ["*.mlu", "config"]
        run(["git", "add", "--", *stage], echo=args.verbose)
        if git_staged_dirty() or args.allow_empty:
            run(["git", "commit", "-m", args.commit_msg, "--allow-empty"], echo=args.verbose)
        elif args.verbose:
            print("skip commit: no staged/default changes")

    sha = resolve_commit("HEAD")
    if args.push or args.commit_msg:
        run(["git", "push"], echo=args.verbose)
    return sha


def unique_keep_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_config_ids() -> List[str]:
    path = repo_root() / "config"
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def write_config_if_requested(args: argparse.Namespace) -> None:
    if not args.op:
        return
    ops = [x.strip().zfill(3) for x in re.split(r"[,，\s]+", args.op) if x.strip()]
    if not ops:
        return
    path = repo_root() / "config"
    text = "\n".join(ops) + "\n"
    if args.dry_run:
        print(f"dry config={','.join(ops)}")
        return
    path.write_text(text)


def build_commit_list(args: argparse.Namespace) -> List[str]:
    xs: List[str] = []
    xs.extend(args.commits)
    if args.stdin:
        xs.extend(read_stdin_commits())
    if args.last:
        xs.extend(commits_from_last(args.last))
    if args.range:
        xs.extend(commits_from_range(args.range))
    if not xs:
        xs.append("HEAD")
    return unique_keep_order(resolve_commit(x) for x in xs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Manually send GitHub push webhooks to the temporary BangC OJ server."
    )
    ap.add_argument("commits", nargs="*", help="commit/rev to submit; default: HEAD")
    ap.add_argument(
        "-m",
        "--commit-msg",
        help="stage default files, commit, push, then webhook HEAD",
    )
    ap.add_argument("--op", help="write config before -m, e.g. 004 or 109,111")
    ap.add_argument(
        "--stage",
        nargs="+",
        help="pathspecs for -m; default: *.mlu config",
    )
    ap.add_argument("--push", action="store_true", help="git push before webhook")
    ap.add_argument("--allow-empty", action="store_true", help="allow empty -m commit")
    ap.add_argument("--no-webhook", action="store_true", help="commit/push only")
    ap.add_argument("--url", default=os.environ.get("WEBHOOK_URL", DEFAULT_URL))
    ap.add_argument("--repo", help="OWNER/REPO; default: git remote origin")
    ap.add_argument("--branch", default=os.environ.get("WEBHOOK_BRANCH", DEFAULT_BRANCH))
    ap.add_argument("--team", default=os.environ.get("WEBHOOK_TEAM", "42"), help="42, 91, 42,91, or all")
    ap.add_argument("--secret", help="webhook secret; prefer env or secret file")
    ap.add_argument("--before", help="override payload before SHA")
    ap.add_argument("--sender", default=os.environ.get("WEBHOOK_SENDER", "MosRat"))
    ap.add_argument("--pusher-name", default=os.environ.get("GIT_AUTHOR_NAME"))
    ap.add_argument("--pusher-email", default=os.environ.get("GIT_AUTHOR_EMAIL"))
    ap.add_argument("--repository-id", type=int, default=123456789)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between commits")
    ap.add_argument("--last", type=int, help="submit the last N commits, oldest first")
    ap.add_argument("--range", help="git rev-list range, for example OLD..HEAD")
    ap.add_argument("--stdin", action="store_true", help="read commits from stdin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--delivery", help="fixed X-GitHub-Delivery id for one request")
    ap.add_argument("--user-agent", default="GitHub-Hookshot/manual-aics")
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    write_config_if_requested(args)
    submitted_head = commit_push_if_requested(args)
    explicit_commits = bool(args.commits or args.stdin or args.last or args.range)
    if submitted_head and not explicit_commits:
        commits = [submitted_head]
    else:
        commits = build_commit_list(args)

    if args.no_webhook:
        print(f"done c={short(resolve_commit('HEAD'))} webhook=skip")
        return 0

    teams = parse_teams(args.team)
    config_ids = ",".join(read_config_ids()) or "-"

    if args.verbose:
        print(
            f"repo={normalize_repo(args.repo)} branch={args.branch} team={','.join(teams)} "
            f"config={config_ids} commits={len(commits)}"
        )

    failures = 0
    for team in teams:
        args.team = team
        if args.dry_run:
            secret, source = "", "dry-run"
        else:
            secret, source = resolve_secret(args, team)
        if args.verbose:
            print(f"team={team} secret={source}")

        for i, sha in enumerate(commits):
            try:
                if args.verbose or len(teams) > 1:
                    print(f"team={team}", end=" ")
                if not send_one(sha, secret, args):
                    failures += 1
                    if not args.keep_going:
                        return 1
            except Exception as e:
                failures += 1
                print(f"err c={short(sha)} type={type(e).__name__} msg={e}", file=sys.stderr)
                if not args.keep_going:
                    return 1
            if args.sleep and (i + 1 < len(commits) or team != teams[-1]):
                time.sleep(args.sleep)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
