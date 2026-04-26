#!/usr/bin/env python3

import json

from problem_utils import REPO_ROOT, build_template, is_generated_template, load_problems


def main():
    _, tasks, _ = load_problems()
    reference_dir = REPO_ROOT / "reference-impl"
    reference_dir.mkdir(parents=True, exist_ok=True)

    reference_count = 0
    template_written = []
    template_skipped = []
    task_names = []

    for task in tasks:
        base_name = task["base_name"]
        task_names.append(base_name)

        ref_path = reference_dir / f"{base_name}.py"
        ref_path.write_text(task["reference_implementation"].rstrip() + "\n")
        reference_count += 1

        mlu_path = REPO_ROOT / f"{base_name}.mlu"
        if is_generated_template(mlu_path, task):
            mlu_path.write_text(build_template(task))
            template_written.append(mlu_path.name)
        else:
            template_skipped.append(mlu_path.name)

    tasks_path = REPO_ROOT / "scripts" / "tasks.json"
    tasks_path.write_text(json.dumps(task_names, ensure_ascii=False, indent=4) + "\n")

    print(f"fetched {len(tasks)} tasks")
    print(f"updated {reference_count} reference implementations")
    print(f"updated scripts/tasks.json with {len(task_names)} task names")
    print(f"wrote {len(template_written)} generated .mlu templates")
    print(f"skipped {len(template_skipped)} non-template .mlu files")


if __name__ == "__main__":
    main()
