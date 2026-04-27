import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_PATH = REPO_ROOT / "reference-impl" / "problems.json"


def load_problems():
    if not PROBLEMS_PATH.exists():
        raise FileNotFoundError(
            f"{PROBLEMS_PATH} is missing; download https://openoperator.cn/problems.json there"
        )
    data = json.loads(PROBLEMS_PATH.read_text())
    tasks = []
    by_base = {}
    for raw_task in data["tasks"]:
        task = dict(raw_task)
        task["description"] = dict(raw_task["description"])
        task = dict(task)
        task["base_name"] = strip_task_prefix(task["name"])
        tasks.append(task)
        by_base[task["base_name"]] = task
    return data, tasks, by_base


def strip_task_prefix(name):
    return re.sub(r"^\d+_", "", name)


def task_for_source(source_name):
    _, _, by_base = load_problems()
    key = Path(source_name).stem
    if key not in by_base:
        raise KeyError(f"No task metadata for {source_name}")
    return by_base[key]


def task_names():
    _, tasks, _ = load_problems()
    return [task["base_name"] for task in tasks]


def task_index_by_base():
    return {name: index + 1 for index, name in enumerate(task_names())}


def task_dtypes(task):
    return task.get("description", {}).get("dtype", []) or [None]


def build_template(task):
    desc = task["description"]
    dtype = ", ".join(desc.get("dtype", [])) or "unknown"
    wrapper = desc["cpp_wrapper"].strip()
    return f"""#include <bang.h>
#include <cnrt.h>
#include <torch/extension.h>

// Task: {task["id"]} {task["name"]}
// Category: {desc.get("category", "")}
// Difficulty: {desc.get("difficulty", "")}
// DType: {dtype}
// Description: {desc.get("desc", "")}
// Wrapper: {wrapper}

{wrapper[:-1]} {{
  TORCH_CHECK(false, "TODO: implement {task["base_name"]}");
}}
"""


def is_generated_template(path, task):
    if not path.exists():
        return True
    text = path.read_text(errors="ignore")
    if not text.strip():
        return True
    return f'TORCH_CHECK(false, "TODO: implement {task["base_name"]}");' in text
