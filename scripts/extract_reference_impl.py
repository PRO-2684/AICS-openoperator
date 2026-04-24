#!/usr/bin/env python3

import json
from pathlib import Path

from problem_utils import REPO_ROOT, load_problems


def main():
    _, tasks, _ = load_problems()
    out_dir = REPO_ROOT / "reference-impl"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for task in tasks:
        py_path = out_dir / f"{task['base_name']}.py"
        py_path.write_text(task["reference_implementation"].rstrip() + "\n")
        manifest.append(
            {
                "id": task["id"],
                "name": task["name"],
                "base_name": task["base_name"],
                "cpp_wrapper": task["description"]["cpp_wrapper"],
                "dtype": task["description"].get("dtype", []),
                "path": str(py_path.relative_to(REPO_ROOT)),
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"extracted {len(tasks)} reference implementations to {out_dir}")


if __name__ == "__main__":
    main()
