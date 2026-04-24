#!/usr/bin/env python3

from pathlib import Path

from problem_utils import REPO_ROOT, build_template, load_problems


def is_empty_file(path):
    return not path.exists() or not path.read_text(errors="ignore").strip()


def main():
    _, tasks, _ = load_problems()
    created = []
    skipped = []

    for task in tasks:
        path = REPO_ROOT / f"{task['base_name']}.mlu"
        if is_empty_file(path):
            path.write_text(build_template(task))
            created.append(path.name)
        else:
            skipped.append(path.name)

    print(f"generated {len(created)} template files")
    if created:
      print("updated:")
      for name in created:
          print(f"  {name}")
    print(f"skipped {len(skipped)} non-empty files")


if __name__ == "__main__":
    main()
