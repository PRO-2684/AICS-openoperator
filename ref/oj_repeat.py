#!/usr/bin/env python3
"""Repeat OJ submit helper.

Default mode pushes after every commit because the OJ trigger path may be
push-based rather than per-commit scanning.  Use --push-at-end only for
throughput experiments where a missed webhook can be tolerated or separately
verified.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, cwd, check=True):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if check and p.returncode:
        raise RuntimeError(f"{' '.join(cmd)}\n{p.stdout.strip()}")
    return p


def out(cmd, cwd):
    return run(cmd, cwd).stdout.strip()


def norm_ops(s):
    return [x.strip().zfill(3) if x.strip().isdigit() else x.strip() for x in s.split(",") if x.strip()]


def git_commit(repo, config_text, msg, add, nonce):
    suffix = "\n" * ((nonce % 3) + 1) if nonce else "\n"
    (repo / "config").write_text(config_text.rstrip() + suffix, encoding="utf-8")
    run(["git", "add", "--", *add], repo)
    run(["git", "commit", "--allow-empty", "-m", msg], repo)
    return out(["git", "rev-parse", "--short", "HEAD"], repo)


def submit_items(ops, joint):
    if joint:
        label = ",".join(ops)
        return [(label, "\n".join(ops))]
    return [(op, op) for op in ops]


def replay_webhook(repo, commit, args):
    if not args.webhook:
        return True
    cmd = [
        sys.executable,
        "ref/webhook_commit.py",
        commit,
        "--team",
        args.webhook_team,
        "--keep-going",
    ]
    if args.webhook_verbose:
        cmd.append("--verbose")
    proc = run(cmd, repo, check=False)
    ok = proc.returncode == 0
    if not args.quiet or not ok or args.webhook_verbose:
        print(proc.stdout.strip(), flush=True)
    return ok


def direct_submit(args, repo, ops):
    if args.pull:
        upstream = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
        if "/" in upstream:
            remote, branch = upstream.split("/", 1)
            run(["git", "pull", "--rebase", "--autostash", remote, branch], repo)
        else:
            run(["git", "pull", "--rebase", "--autostash"], repo)

    commits = []
    hash_path = repo / args.hash_file
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.count + 1):
        for label, config_text in submit_items(ops, args.joint_config):
            msg = f"{label} {args.msg} {i}"
            try:
                commit = git_commit(
                    repo, config_text, msg, args.add, i if args.config_nonce else 0
                )
            except Exception as e:
                print(f"repeat submit failed: {e}", file=sys.stderr, flush=True)
                continue
            commits.append(commit)
            if args.quiet:
                print(f"{commit} {label} {args.msg} {i}", flush=True)
            else:
                print(json.dumps({"ok": 1, "op": label.split(","), "c": commit, "cm": 1, "ps": 0 if args.push_at_end else 1}, separators=(",", ":")), flush=True)
            with hash_path.open("a", encoding="utf-8") as f:
                f.write(f"{commit} {label} {args.msg} {i}\n")
            if not args.push_at_end:
                run(["git", "push"], repo)
                replay_webhook(repo, commit, args)
                if args.sleep:
                    time.sleep(args.sleep)

    pushed = False
    if args.push_at_end and commits:
        proc = run(["git", "push"], repo, check=False)
        pushed = proc.returncode == 0
        if not args.quiet or proc.returncode != 0:
            print(proc.stdout.strip(), flush=True)
        if proc.returncode != 0:
            print(f"batch push failed rc={proc.returncode}", file=sys.stderr, flush=True)
        elif args.webhook:
            for commit in commits:
                replay_webhook(repo, commit, args)
                if args.sleep:
                    time.sleep(args.sleep)

    return commits, pushed, hash_path


def oj_git_submit(args, repo, ops):
    hash_path = repo / args.hash_file
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    commits = []
    pushed = False
    pulled = False
    for i in range(1, args.count + 1):
        for label, config_text in submit_items(ops, args.joint_config):
            cmd = [
                sys.executable,
                "ref/oj_git.py",
                "-o",
                label if args.joint_config else config_text,
                "-m",
                f"{label} {args.msg} {i}",
                "-a",
                *args.add,
                "--empty",
                "--no-lock",
            ]
            if args.push_at_end:
                cmd.append("--no-push")
            if args.pull and not pulled:
                cmd += ["--pull", "--retry", args.retry]
                pulled = True
            proc = subprocess.run(
                cmd,
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if not args.quiet:
                print(proc.stdout.strip(), flush=True)
            if proc.returncode != 0:
                print(f"repeat submit failed rc={proc.returncode}", file=sys.stderr, flush=True)
                continue
            for line in proc.stdout.splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    commit = obj.get("c")
                    if commit:
                        commits.append(commit)
                        if args.quiet:
                            print(f"{commit} {label} {args.msg} {i}", flush=True)
                        with hash_path.open("a", encoding="utf-8") as f:
                            f.write(f"{commit} {label} {args.msg} {i}\n")
    if args.push_at_end and commits:
        proc = run(["git", "push"], repo, check=False)
        pushed = proc.returncode == 0
        if not args.quiet or proc.returncode != 0:
            print(proc.stdout.strip(), flush=True)
        if proc.returncode != 0:
            print(f"batch push failed rc={proc.returncode}", file=sys.stderr, flush=True)
    return commits, pushed, hash_path


def main():
    ap = argparse.ArgumentParser(description="repeat OJ submissions and record hashes")
    ap.add_argument("-o", "--ops", required=True)
    ap.add_argument("-n", "--count", type=int, required=True)
    ap.add_argument("-m", "--msg", required=True)
    ap.add_argument("--hash-file", default="ref/.oj_repeat_hashes")
    ap.add_argument("--add", nargs="*", default=["ELU.mlu", "HardSigmoid.mlu", "HardTanh.mlu", "config"])
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--retry", default="3")
    ap.add_argument("--push-at-end", action="store_true", help="create all commits locally, then push once; faster but webhook semantics may miss commits")
    ap.add_argument("--quiet", action="store_true", help="print one compact line per commit instead of oj_git output")
    ap.add_argument("--via-oj-git", action="store_true", help="use ref/oj_git.py per commit instead of the direct fast path")
    ap.add_argument("--joint-config", action="store_true", help="write all -o ids into one config per repeat instead of one commit per op")
    ap.add_argument("--config-nonce", action="store_true", help="alternate harmless blank lines in config so repeat commits are non-empty")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep after each pushed/webhooked commit")
    ap.add_argument("--webhook", action="store_true", help="manually replay the OJ webhook after each successful push")
    ap.add_argument("--webhook-team", default="91", help="team id(s) for manual webhook, e.g. 91 or 42,91")
    ap.add_argument("--webhook-verbose", action="store_true", help="print webhook replay details")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ops = norm_ops(args.ops)
    if args.via_oj_git:
        commits, pushed, hash_path = oj_git_submit(args, repo, ops)
    else:
        commits, pushed, hash_path = direct_submit(args, repo, ops)

    print("COMMITS", " ".join(commits), flush=True)
    if args.push_at_end:
        print("PUSHED", int(pushed), flush=True)
    print("HASH_FILE", hash_path, flush=True)


if __name__ == "__main__":
    main()
