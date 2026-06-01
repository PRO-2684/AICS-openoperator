#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ref/oj_git.py — agent-friendly OJ git submit helper

Purpose
  Make "write config + git add + commit + push (+ optional result query)" as close
  as possible to one local atomic operation by using a repo-local flock lock.

Typical usage
  python ref/oj_git.py -o 041 -m "041 Opname: optimize kernel"
  python ref/oj_git.py -o 041,042 -m "batch submit" --result --sleep 8
  python ref/oj_git.py -o 041 --empty
  python ref/oj_git.py -o 041 --add config 041_Op.mlu
  python ref/oj_git.py -o 041 --pull --retry 3
  python ref/oj_git.py -o 041 --no-push
  python ref/oj_git.py --dry -o 041 -m test

Compact JSON output examples
  success:
    {"ok":1,"op":["041"],"c":"8ee50f8","cm":1,"ps":1}
  no staged changes:
    {"ok":1,"op":["041"],"c":"8ee50f8","cm":0,"ps":1}
  command failure:
    {"ok":0,"cmd":"git push","rc":1,"err":"..."}

Fields
  ok  : 1 success, 0 failure
  op  : written config ids, omitted if none
  c   : short HEAD commit hash
  cm  : committed? 1/0
  ps  : pushed? 1/0
  rrc : optional result command return code

Notes
  - Local atomicity only works between processes that use the same lock file.
  - Remote push can still race with other machines/users; use --pull --retry N.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def die(obj, code: int = 1) -> None:
    print(jdump(obj), file=sys.stderr)
    raise SystemExit(code)


def cmd_s(cmd: Sequence[str]) -> str:
    return " ".join(str(x) for x in cmd)


def run(
    cmd: Sequence[str], *, check: bool = True, cwd: Optional[Path] = None, env=None
) -> subprocess.CompletedProcess:
    p = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode:
        err = (p.stderr or p.stdout or "").strip()
        die({"ok": 0, "cmd": cmd_s(cmd), "rc": p.returncode, "err": err})
    return p


def out(cmd: Sequence[str], *, check: bool = True, cwd: Optional[Path] = None) -> str:
    return run(cmd, check=check, cwd=cwd).stdout.strip()


def repo_root() -> Path:
    p = run(["git", "rev-parse", "--show-toplevel"], check=False)
    if p.returncode == 0 and p.stdout.strip():
        return Path(p.stdout.strip()).resolve()
    return Path.cwd().resolve()


def git(
    args: Sequence[str], *, root: Path, check: bool = True
) -> subprocess.CompletedProcess:
    return run(["git", *args], check=check, cwd=root)


def git_out(args: Sequence[str], *, root: Path, check: bool = True) -> str:
    return git(args, root=root, check=check).stdout.strip()


def short(root: Path, rev: str = "HEAD") -> str:
    return git_out(["rev-parse", "--short", rev], root=root)


def norm_ops(s: Optional[str]) -> List[str]:
    if not s:
        return []
    ops: List[str] = []
    for x in s.replace(",", " ").split():
        x = x.strip()
        if not x:
            continue
        if x.isdigit():
            x = x.zfill(3)
        ops.append(x)
    return ops


def write_config(root: Path, ops: Sequence[str]) -> None:
    if ops:
        (root / "config").write_text("\n".join(ops) + "\n", encoding="utf-8")


def staged_dirty(root: Path) -> bool:
    return git(["diff", "--cached", "--quiet"], root=root, check=False).returncode != 0


def worktree_dirty(root: Path) -> bool:
    return bool(git_out(["status", "--porcelain"], root=root, check=False))


def lock_path(root: Path, user_lock: Optional[str]) -> Path:
    if user_lock:
        p = Path(user_lock)
        return p if p.is_absolute() else root / p
    gd = git_out(["rev-parse", "--git-dir"], root=root, check=False) or ".git"
    gp = Path(gd)
    if not gp.is_absolute():
        gp = root / gp
    return gp / "oj_git.lock"


class FileLock:
    def __init__(self, path: Path, wait: float):
        self.path = path
        self.wait = wait
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o666)
        start = time.time()
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(self.fd, 0)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except BlockingIOError:
                if self.wait == 0 or (
                    self.wait > 0 and time.time() - start >= self.wait
                ):
                    die({"ok": 0, "err": "lock_busy", "lock": str(self.path)})
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
        return False


def maybe_pull(root: Path, do_pull: bool) -> None:
    if do_pull:
        upstream = git_out(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            root=root,
            check=False,
        )
        if "/" in upstream:
            remote, branch = upstream.split("/", 1)
            git(["pull", "--rebase", "--autostash", remote, branch], root=root)
        else:
            git(["pull", "--rebase", "--autostash"], root=root)


def do_add(
    root: Path, add: Optional[List[str]], ops: Sequence[str], all_files: bool
) -> None:
    if all_files:
        git(["add", "-A"], root=root)
        return
    if add is None:
        git(["add", "-u"], root=root)
        if ops:
            git(["add", "config"], root=root)
        return
    if add:
        git(["add", "--", *add], root=root)


def commit_once(root: Path, msg: str, allow_empty: bool, no_verify: bool) -> int:
    if not staged_dirty(root) and not allow_empty:
        return 0
    cmd = ["commit", "-m", msg]
    if allow_empty:
        cmd.append("--allow-empty")
    if no_verify:
        cmd.append("--no-verify")
    git(cmd, root=root)
    return 1


def push_once(root: Path, no_push: bool) -> int:
    if no_push:
        return 0
    git(["push"], root=root)
    return 1


def result(root: Path, c: str, sleep_s: float, extra: Sequence[str]) -> int:
    if sleep_s > 0:
        time.sleep(sleep_s)
    script = root / "ref" / "get_result.py"
    if not script.exists():
        print(jdump({"ok": 0, "err": "missing ref/get_result.py"}), file=sys.stderr)
        return 127
    p = subprocess.run(
        [sys.executable, str(script), c, *extra], cwd=str(root), text=True
    )
    return p.returncode


def submit_attempt(root: Path, args, ops: Sequence[str]) -> Tuple[int, int, str]:
    maybe_pull(root, args.pull)
    write_config(root, ops)
    do_add(root, args.add, ops, args.all)
    cm = commit_once(root, args.msg, args.empty, args.no_verify)
    c = short(root)
    ps = push_once(root, args.no_push)
    return cm, ps, c


def main() -> None:
    ap = argparse.ArgumentParser(description="agent-friendly OJ git submit helper")
    ap.add_argument("-o", "--op", help="write config ids, e.g. 041 or 041,042")
    ap.add_argument("-m", "--msg", help="commit message")
    ap.add_argument(
        "-a",
        "--add",
        nargs="*",
        default=None,
        help="pathspecs; default: git add -u plus config when --op",
    )
    ap.add_argument("-A", "--all", action="store_true", help="git add -A")
    ap.add_argument("--empty", action="store_true", help="allow empty commit")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument(
        "--pull",
        action="store_true",
        help="git pull --rebase --autostash before changing/committing",
    )
    ap.add_argument(
        "--retry",
        type=int,
        default=1,
        help="attempt count for pull/commit/push; useful with --pull",
    )
    ap.add_argument("--lock", help="lock file path; default: .git/oj_git.lock")
    ap.add_argument(
        "--lock-wait",
        type=float,
        default=-1,
        help="seconds to wait for lock; -1 forever, 0 no wait",
    )
    ap.add_argument("--no-lock", action="store_true", help="disable flock")
    ap.add_argument(
        "--no-verify", action="store_true", help="pass --no-verify to git commit"
    )
    ap.add_argument(
        "--result", action="store_true", help="run ref/get_result.py after push"
    )
    ap.add_argument(
        "--result-arg",
        action="append",
        default=[],
        help="extra arg passed to ref/get_result.py; repeatable",
    )
    ap.add_argument("--sleep", type=float, default=0, help="sleep before --result")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument(
        "--status",
        action="store_true",
        help="print compact git status summary and exit",
    )
    args = ap.parse_args()

    root = repo_root()
    ops = norm_ops(args.op)
    args.msg = args.msg or ((ops[0] + " submit") if ops else "OJ submit")
    args.retry = max(1, args.retry)

    if args.status:
        print(
            jdump(
                {
                    "ok": 1,
                    "root": str(root),
                    "c": short(root),
                    "dirty": int(worktree_dirty(root)),
                }
            )
        )
        return

    if args.dry:
        print(
            jdump(
                {
                    "dry": 1,
                    "op": ops,
                    "m": args.msg,
                    "push": int(not args.no_push),
                    "root": str(root),
                }
            )
        )
        return

    def body():
        last = None
        for i in range(1, args.retry + 1):
            try:
                cm, ps, c = submit_attempt(root, args, ops)
                out_obj = {"ok": 1}
                if ops:
                    out_obj["op"] = list(ops)  # ty:ignore[invalid-assignment]
                out_obj.update({"c": c, "cm": cm, "ps": ps})  # ty:ignore[no-matching-overload]
                print(jdump(out_obj))
                if args.result:
                    rc = result(root, c, args.sleep, args.result_arg)
                    if rc:
                        print(jdump({"ok": 0, "rrc": rc}), file=sys.stderr)
                return
            except SystemExit as e:
                last = e
                if i >= args.retry:
                    raise
                time.sleep(min(1.0 * i, 5.0))
        if last:
            raise last

    if args.no_lock:
        body()
    else:
        lp = lock_path(root, args.lock)
        with FileLock(lp, args.lock_wait):
            body()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        die({"ok": 0, "err": "interrupted"}, 130)
