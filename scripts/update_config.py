#!/usr/bin/env python3
import subprocess
from typing import Optional

from problem_utils import task_index_by_base


TASK_INDICES = task_index_by_base()


def run(*args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_staged_mlu_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "--", "*.mlu"],
        text=True,
    )
    return [f for f in out.splitlines() if f]


def has_real_change(f: str) -> bool:
    r = run("git", "diff", "--cached", "-w", "--ignore-blank-lines", "--quiet", "--", f)
    return r.returncode == 1


def to_index(f: str) -> Optional[int]:
    task = f.removesuffix(".mlu")
    return TASK_INDICES.get(task)


def main():
    changed_files = [f for f in get_staged_mlu_files() if has_real_change(f)]
    indices = []

    for f in changed_files:
        index = to_index(f)
        if index is not None:
            indices.append(index)

    with open("config", "w") as fp:
        for index in indices:
            fp.write(f"{index:>03}\n")

    subprocess.check_call(["git", "add", "config"])


if __name__ == "__main__":
    main()
