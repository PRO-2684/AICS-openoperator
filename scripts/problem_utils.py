import json
import re
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_URL = "https://openoperator.cn/problems.json"


def load_problems():
    with urlopen(PROBLEMS_URL, timeout=30) as response:
        data = json.load(response)
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


def first_dtype(task):
    dtypes = task.get("description", {}).get("dtype", [])
    return dtypes[0] if dtypes else None


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
