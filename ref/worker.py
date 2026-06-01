#!/usr/bin/env python3
"""
worker.py — 评估节点（部署在计算服务器 B1..Bn）

职责：
  1. 阻塞监听 Redis 任务队列
  2. 通过 HTTP 从 Server A 下载选手代码（tar.gz）
  3. 读取选手仓库中的 config 文件，筛选本次需评估的题目
  4. 将选手 BANG C 代码插入外围 Python 模板，逐题调用 bangc_torch_tester.py
  5. 将结果推回 Redis 供排行榜更新与 commit comment 发布

安全设计：
  选手仓库中只提交 .mlu 文件（纯 BANG C 源码），Worker 将其插入
  本地 mlu/ 目录下的外围 Python 模板的 bang_func_source 变量中，
  避免直接执行选手提交的任意 Python 代码。

config.yaml 中 problems 示例：
  problems:
    - id: '001'
      name: 001_LeakyReLU
      torch_model: 001_LeakyReLU.py
      solution_path: 001_LeakyReLU_mlu.py
      mlu_path: LeakyReLU.mlu

改动：
  1. fetch_code 增加下载重试（默认 3 次）
  2. main 启动时增加文件服务连通性预检
"""

import os
import io
import json
import shutil
import subprocess
import tarfile
import logging
import re
import socket
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

import yaml
import redis
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("worker")

# ── 路径 & 配置 ─────────────────────────────────────
WORKER_DIR = Path(__file__).parent
REF_DIR = WORKER_DIR / "ref"
MLU_DIR = WORKER_DIR / "mlu"  # 外围 Python 模板目录
TESTER_SCRIPT = WORKER_DIR / "bangc_torch_tester.py"

with open(WORKER_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)

R = redis.Redis(
    host=CFG["redis"]["worker_host"],
    port=CFG["redis"]["port"],
    password=CFG["redis"].get("password") or None,
    db=CFG["redis"].get("db", 0),
    decode_responses=True,
)

TASK_Q = "bangc:task_queue"
RESULT_Q = "bangc:result_queue"
PROCESSING = (
    "bangc:processing"  # Hash: task_id -> JSON（完整任务快照 + processing 元信息）
)
WCFG = CFG["worker"]
PROBLEMS = CFG.get("problems", [])

WORKER_ID = os.environ.get("WORKER_ID", "unknown_worker")

# 选手仓库中的题目配置文件名
PROBLEM_CONFIG_FILE = WCFG.get("problem_config_file", "config")
# bangc_torch_tester 性能测试迭代次数
EVAL_RUNS = WCFG.get("iterations", 3)
# 单题评估超时（秒），包含编译时间
EVAL_TIMEOUT = WCFG.get("timeout", 300)
# stdout/stderr 最大保留字符数
MAX_OUTPUT_LEN = 24000

# HTTP 文件服务配置
FILE_SERVER_URL = WCFG.get("file_server_url", "http://152.136.18.42:13000")
FILE_SERVER_TOKEN = WCFG.get("file_server_token", "change-me-to-a-strong-secret")
# HTTP 下载超时（秒）
HTTP_TIMEOUT = WCFG.get("http_timeout", 180)
# HTTP 下载重试次数
HTTP_MAX_RETRIES = WCFG.get("http_max_retries", 3)

# bang_func_source 占位符（外围模板中的空变量）
BANG_SOURCE_PLACEHOLDER = 'bang_func_source = """\n"""'
# 安全加载片段：运行时从环境变量指定的文件读取选手 .mlu 源码
SAFE_BANG_SOURCE_SNIPPET = (
    "import os\n"
    "from pathlib import Path\n"
    'bang_func_source = Path(os.environ["BANG_SOURCE_FILE"]).read_text(encoding="utf-8")'
)

BANGC_AUDIT_RULES_FILE = WORKER_DIR / "security" / "bangc_audit_rules.json"

# requests Session 用于连接复用
_http_session = requests.Session()
_http_session.headers.update(
    {
        "Authorization": f"Bearer {FILE_SERVER_TOKEN}",
    }
)


def _load_bangc_audit_rules():
    if not BANGC_AUDIT_RULES_FILE.exists():
        raise FileNotFoundError(
            f"BangC 审计规则文件不存在: {BANGC_AUDIT_RULES_FILE}。"
            "请先运行 security/generate_bangc_audit_rules.py 生成静态规则。"
        )

    payload = json.loads(BANGC_AUDIT_RULES_FILE.read_text(encoding="utf-8"))
    general_rules = []
    for item in payload.get("general_rules", []):
        pattern_text = item.get("pattern")
        if not isinstance(pattern_text, str) or not pattern_text:
            raise ValueError(
                f"BangC 审计规则无效: general rule {item.get('id', '<unknown>')} "
                f"的 pattern 不是有效字符串: {pattern_text!r}"
            )
        general_rules.append(
            {
                "id": item["id"],
                "pattern": re.compile(pattern_text, item.get("flags", 0)),
                "message": item["message"],
            }
        )
    problem_rules = {
        name: {
            "id": item["id"],
            "pattern": re.compile(item["pattern"], item.get("flags", 0)),
            "message": item["message"],
        }
        for name, item in payload.get("problem_rules", {}).items()
    }
    return general_rules, problem_rules


BANGC_SOURCE_RULES, PROBLEM_OPERATOR_RULES = _load_bangc_audit_rules()


# ═══════════════════════════════════════════════════════
#  大小写不敏感的文件查找
# ═══════════════════════════════════════════════════════


def find_file_icase(directory: Path, filename: str) -> Path | None:
    """
    在 directory 中查找文件名与 filename 大小写不敏感匹配的文件。
    支持多级相对路径，如 "subdir/LeakyReLU.mlu"。
    优先返回完全匹配的路径，其次返回大小写不敏感匹配的第一个结果。
    未找到返回 None。
    """
    target = directory / filename
    # 优先精确匹配
    if target.exists():
        return target

    # 分离父目录和文件名
    rel = Path(filename)
    parent = directory / rel.parent
    name_lower = rel.name.lower()

    if not parent.is_dir():
        return None

    for entry in parent.iterdir():
        if entry.is_file() and entry.name.lower() == name_lower:
            return entry

    return None


# ═══════════════════════════════════════════════════════
#  Processing Hash 辅助函数
# ═══════════════════════════════════════════════════════


def mark_processing(task: dict):
    """将任务标记为正在处理，写入 Redis Hash。"""
    task_id = task["task_id"]
    info = {
        **task,
        "worker_id": WORKER_ID,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    R.hset(PROCESSING, task_id, json.dumps(info))
    log.info(f"  已标记 processing: {task_id}")


def unmark_processing(task_id: str):
    """任务处理完毕，从 Redis Hash 中移除。"""
    R.hdel(PROCESSING, task_id)
    log.info(f"  已移除 processing: {task_id}")


# ═══════════════════════════════════════════════════════
#  安全 tar 解压
# ═══════════════════════════════════════════════════════


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path):
    """
    安全解压 tar，防止路径穿越攻击（如 ../../etc/passwd）。
    """
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest) + os.sep) and member_path != dest:
            raise RuntimeError(f"Tar 路径穿越: {member.name}")
        # 跳过符号链接
        if member.issym() or member.islnk():
            log.warning(f"  跳过符号链接: {member.name}")
            continue
    tar.extractall(path=dest)


# ═══════════════════════════════════════════════════════
#  代码同步（HTTP 方式，带重试）
# ═══════════════════════════════════════════════════════


def fetch_code(team_id: str, remote_path: str) -> Path:
    unique = uuid.uuid4().hex[:12]
    local = Path(WCFG["workspace"]) / f"{team_id}_{unique}"
    local.mkdir(parents=True, exist_ok=True)

    url = f"{FILE_SERVER_URL}/archive"
    last_err = None

    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            log.info(
                f"  HTTP 下载 (尝试 {attempt}/{HTTP_MAX_RETRIES}): {url}?path={remote_path}"
            )

            resp = _http_session.get(
                url,
                params={"path": remote_path},
                timeout=HTTP_TIMEOUT,
                stream=True,
            )

            if resp.status_code == 503:
                raise RuntimeError(f"HTTP 503: {resp.text[:500]} (仓库可能正在 clone)")

            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP 下载失败: status={resp.status_code} body={resp.text[:500]}"
                )

            content = resp.content
            log.info(f"  下载完成: {len(content)} bytes")

            if local.exists():
                shutil.rmtree(local)
            local.mkdir(parents=True, exist_ok=True)

            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                _safe_extract_tar(tar, local)

            log.info(f"  解压完成: {local}")
            return local

        except Exception as e:
            last_err = e
            log.warning(f"  下载失败 (尝试 {attempt}/{HTTP_MAX_RETRIES}): {e}")
            if attempt < HTTP_MAX_RETRIES:
                wait = 3 * attempt
                log.info(f"  等待 {wait}s 后重试...")
                time.sleep(wait)

    # 全部失败，清理残留目录
    if local.exists():
        shutil.rmtree(local, ignore_errors=True)

    raise RuntimeError(f"代码下载重试 {HTTP_MAX_RETRIES} 次全部失败: {last_err}")


# ═══════════════════════════════════════════════════════
#  启动预检
# ═══════════════════════════════════════════════════════


def _preflight_check():
    """
    Worker 启动时验证关键依赖是否可用：
      1. 文件服务连通性
      2. Redis 连通性
      3. 本地必要文件/目录
    """
    ok = True

    # ── 检查文件服务 ──
    health_url = f"{FILE_SERVER_URL}/health"
    try:
        resp = _http_session.get(health_url, timeout=10)
        if resp.status_code == 200:
            log.info(f"[预检] 文件服务连通: {FILE_SERVER_URL}")
        else:
            log.warning(
                f"[预检] 文件服务返回非 200: {resp.status_code} {resp.text[:200]}"
            )
            ok = False
    except Exception as e:
        log.error(f"[预检] 无法连接文件服务 {FILE_SERVER_URL}: {e}")
        ok = False

    # ── 检查 Redis ──
    try:
        R.ping()
        log.info("[预检] Redis 连通")
    except Exception as e:
        log.error(f"[预检] Redis 不可用: {e}")
        ok = False

    # ── 检查本地关键路径 ──
    if not TESTER_SCRIPT.exists():
        log.error(f"[预检] 评测脚本不存在: {TESTER_SCRIPT}")
        ok = False
    else:
        log.info(f"[预检] 评测脚本存在: {TESTER_SCRIPT}")

    if not BANGC_AUDIT_RULES_FILE.exists():
        log.error(f"[预检] BangC 审计规则文件不存在: {BANGC_AUDIT_RULES_FILE}")
        ok = False
    else:
        log.info(f"[预检] BangC 审计规则文件存在: {BANGC_AUDIT_RULES_FILE}")

    if not MLU_DIR.is_dir():
        log.error(f"[预检] MLU 模板目录不存在: {MLU_DIR}")
        ok = False
    else:
        log.info(f"[预检] MLU 模板目录存在: {MLU_DIR}")

    if not REF_DIR.is_dir():
        log.warning(f"[预检] 参考模型目录不存在: {REF_DIR}")

    workspace = Path(WCFG["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    log.info(f"[预检] 工作目录: {workspace}")

    if not ok:
        log.warning("[预检] 部分检查未通过，Worker 仍将启动但可能无法正常工作")
    else:
        log.info("[预检] 所有检查通过")


# ═══════════════════════════════════════════════════════
#  选手 config 读取与题目筛选
# ═══════════════════════════════════════════════════════


def read_problem_config(code_dir: Path) -> list[str]:
    """
    读取选手仓库中的 config 文件，每行一个题目名前缀。
    空行和 # 开头的注释行被忽略。
    文件查找忽略大小写。
    """
    config_file = find_file_icase(code_dir, PROBLEM_CONFIG_FILE)
    if config_file is None:
        log.warning(f"选手 config 文件不存在: {code_dir / PROBLEM_CONFIG_FILE}")
        return []
    prefixes = []
    with open(config_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prefixes.append(line)
    log.info(f"  config 指定 {len(prefixes)} 个题目前缀: {prefixes}")
    return prefixes


def read_config_content(code_dir: Path) -> str:
    """读取选手 config 文件的原始文本内容（用于反馈展示）。"""
    config_file = find_file_icase(code_dir, PROBLEM_CONFIG_FILE)
    if config_file is None:
        return "no config found"
    try:
        return config_file.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def filter_problems(problems: list[dict], prefixes: list[str]) -> list[dict]:
    """根据前缀列表筛选题目。prefixes 为空时返回全部。比较时忽略大小写。"""
    if not prefixes:
        return list(problems)
    prefixes_lower = [p.lower() for p in prefixes]
    return [
        p
        for p in problems
        if any(p["name"].lower().startswith(pfx) for pfx in prefixes_lower)
    ]


# ═══════════════════════════════════════════════════════
#  仓库文件列表
# ═══════════════════════════════════════════════════════


def list_code_dir(code_dir: Path, max_files: int = 200) -> list[str]:
    """
    列出选手代码目录中的文件（相对路径），跳过 .git 等隐藏目录。
    最多返回 max_files 条，防止目录过大时消息爆炸。
    """
    files = []
    try:
        for p in sorted(code_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(code_dir)
                if any(part.startswith(".") for part in rel.parts):
                    continue
                files.append(str(rel))
                if len(files) >= max_files:
                    break
    except Exception:
        pass
    return files


def _strip_c_comments(text: str) -> str:
    """
    删除 C/C++ 风格注释，同时保留换行，便于后续定位行号。
    """
    text = re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S
    )
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def audit_bangc_source(student_path: Path, prob: dict):
    """
    编译前扫描选手 .mlu 源码，阻断直接调用 PyTorch / ATen / CNNL / Python 运行时的作弊实现。
    """
    text = student_path.read_text(encoding="utf-8")
    scan_text = _strip_c_comments(text)

    rules_to_check = list(BANGC_SOURCE_RULES)
    problem_rule = PROBLEM_OPERATOR_RULES.get(prob["name"])
    if problem_rule is not None:
        rules_to_check.append(problem_rule)

    for rule in rules_to_check:
        match = rule["pattern"].search(scan_text)
        if not match:
            continue

        line_no = scan_text.count("\n", 0, match.start()) + 1
        snippet = ""
        lines = scan_text.splitlines()
        if 1 <= line_no <= len(lines):
            snippet = lines[line_no - 1].strip()[:200]

        raise ValueError(
            f"源码安全审计未通过 [{rule['id']}]：第 {line_no} 行命中规则，"
            f"{rule['message']} 片段: {snippet}"
        )


# ═══════════════════════════════════════════════════════
#  代码组装：外围模板 + 选手 BANG C 源码
# ═══════════════════════════════════════════════════════


def assemble_solution(prob: dict, code_dir: Path, tmp_dir: Path) -> tuple[Path, Path]:
    """
    将选手的 BANG C 代码插入外围 Python 模板，生成可执行的评估文件。

    流程：
      1. 读取 mlu/{solution_path} 外围模板
      2. 查找选手仓库中 {mlu_path} 的 BANG C 源码文件（忽略大小写）
      3. 将固定的安全加载代码插入模板，由运行时读取源码文件
      4. 写入临时目录，返回组装后文件路径与选手源码路径
    """
    wrapper_path = MLU_DIR / prob["solution_path"]

    # 选手文件查找忽略大小写
    student_path = find_file_icase(code_dir, prob["mlu_path"])

    if not wrapper_path.exists():
        raise FileNotFoundError(f"外围模板不存在: {wrapper_path}")
    if student_path is None:
        raise FileNotFoundError(
            f"选手 BANG C 文件不存在: {prob['mlu_path']}（已忽略大小写）"
        )

    audit_bangc_source(student_path, prob)

    wrapper_code = wrapper_path.read_text(encoding="utf-8")
    if BANG_SOURCE_PLACEHOLDER not in wrapper_code:
        raise ValueError(
            f"外围模板中未找到 bang_func_source 占位符: {prob['solution_path']}"
        )

    assembled = wrapper_code.replace(
        BANG_SOURCE_PLACEHOLDER,
        SAFE_BANG_SOURCE_SNIPPET,
        1,
    )

    output_path = tmp_dir / prob["solution_path"]
    output_path.write_text(assembled, encoding="utf-8")
    return output_path, student_path


# ═══════════════════════════════════════════════════════
#  单题评估（调用 bangc_torch_tester.py）
# ═══════════════════════════════════════════════════════


def _empty_result(error: str | None = None) -> dict:
    return {
        "passed": False,
        "score": 0.0,
        "max_abs_diff": float("inf"),
        "torch_us": 0.0,
        "bangc_us": 0.0,
        "error": error,
        "stdout": "",
        "stderr": "",
    }


def _run_single_evaluation(
    torch_model: Path, bangc_model: Path, student_mlu: Path
) -> dict:
    cmd = [
        "python3",
        str(TESTER_SCRIPT),
        "-t",
        str(torch_model),
        "-b",
        str(bangc_model),
    ]

    try:
        env = os.environ.copy()
        env["BANG_SOURCE_FILE"] = str(student_mlu)
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=EVAL_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _empty_result(f"评估超时 ({EVAL_TIMEOUT}s)")

    stdout_text = (r.stdout or "")[:MAX_OUTPUT_LEN]
    stderr_text = (r.stderr or "")[:MAX_OUTPUT_LEN]

    if r.returncode != 0:
        if r.returncode == 2:
            err_label = "BangC 编译错误"
        elif r.returncode == 3:
            err_label = "BangC 运行时错误"
        else:
            err_label = "评估脚本异常"

        err_hint = (
            stderr_text.strip()[:3000]
            if stderr_text.strip()
            else f"exit code {r.returncode}"
        )
        res = _empty_result(f"{err_label}: {err_hint}")
        res["stdout"] = stdout_text
        res["stderr"] = stderr_text
        return res

    result = _empty_result()
    result["stdout"] = stdout_text
    result["stderr"] = stderr_text

    try:
        m = re.search(r"@@RESULT@@(.+)", stdout_text)
        if m:
            data = json.loads(m.group(1))
            result["passed"] = data["passed"]
            result["max_abs_diff"] = data["max_abs_diff"]
            result["torch_us"] = data["torch_us"]
            result["bangc_us"] = data["bangc_us"]
            result["score"] = data["score"]
            result["latency"] = data["bangc_us"]
        else:
            result["error"] = "未找到 @@RESULT@@ 输出行"
    except (json.JSONDecodeError, KeyError) as e:
        result["error"] = f"结果解析失败: {e}"

    return result


def _aggregate_evaluation_runs(results: list[dict]) -> dict:
    if not results:
        return _empty_result("没有获得任何评测结果")

    aggregated = _empty_result()
    aggregated["stdout"] = "\n\n".join(
        f"[run {idx}]\n{item['stdout']}".strip()
        for idx, item in enumerate(results, start=1)
        if item.get("stdout")
    )[:MAX_OUTPUT_LEN]
    aggregated["stderr"] = "\n\n".join(
        f"[run {idx}]\n{item['stderr']}".strip()
        for idx, item in enumerate(results, start=1)
        if item.get("stderr")
    )[:MAX_OUTPUT_LEN]

    all_passed = all(item.get("passed", False) for item in results)
    all_clean = all(not item.get("error") for item in results)

    aggregated["passed"] = all_passed and all_clean
    aggregated["max_abs_diff"] = max(
        item.get("max_abs_diff", float("inf")) for item in results
    )

    torch_values = [item.get("torch_us", 0.0) for item in results]
    bangc_values = [item.get("bangc_us", 0.0) for item in results]
    aggregated["torch_us"] = sum(torch_values) / len(torch_values)
    aggregated["bangc_us"] = sum(bangc_values) / len(bangc_values)
    aggregated["latency"] = aggregated["bangc_us"]

    if aggregated["passed"] and aggregated["bangc_us"] > 0:
        aggregated["score"] = aggregated["torch_us"] / aggregated["bangc_us"]
    else:
        aggregated["score"] = 0.0

    if not all_clean:
        errors = [
            f"run {idx}: {item['error']}"
            for idx, item in enumerate(results, start=1)
            if item.get("error")
        ]
        aggregated["error"] = " | ".join(errors)[:3000]

    return aggregated


def evaluate_one(prob: dict, code_dir: Path, tmp_dir: Path) -> dict:
    """
    对一道题目进行评估：组装代码 → 调用 bangc_torch_tester.py。

    返回:
      {
        "passed":       bool,
        "score":        float,
        "max_abs_diff": float,
        "torch_us":     float,
        "bangc_us":     float,
        "error":        str|None,
        "stdout":       str,
        "stderr":       str,
      }
    """
    torch_model = REF_DIR / prob["torch_model"]

    if not torch_model.exists():
        return _empty_result(f"参考模型不存在: {prob['torch_model']}")

    # ── 组装选手代码 ──
    try:
        bangc_model, student_mlu = assemble_solution(prob, code_dir, tmp_dir)
    except FileNotFoundError as e:
        return _empty_result(str(e))
    except ValueError as e:
        return _empty_result(str(e))
    except Exception as e:
        return _empty_result(f"代码组装失败: {e}")

    run_results = []
    for run_idx in range(1, EVAL_RUNS + 1):
        log.info(f"  │  启动子评测进程 {run_idx}/{EVAL_RUNS}")
        run_results.append(
            _run_single_evaluation(torch_model, bangc_model, student_mlu)
        )

    return _aggregate_evaluation_runs(run_results)


# ═══════════════════════════════════════════════════════
#  任务处理主流程
# ═══════════════════════════════════════════════════════


def process_task(task: dict):
    """处理一条评估任务：同步代码 → 读取 config → 筛选题目 → 组装并逐题评估 → 结果入队"""
    task_id = task["task_id"]
    team_id = task["team_id"]
    team_name = task["team_name"]
    log.info(f"┌ 开始任务 {task_id} [{team_id}]")

    # 标记正在处理
    mark_processing(task)

    try:
        _do_process(task, task_id, team_id, team_name)
    finally:
        # 无论成功失败，处理完毕后移除标记
        unmark_processing(task_id)


def _do_process(task: dict, task_id: str, team_id: str, team_name: str):
    server_code_path = task.get("code_path", "")  # Server A 上的唯一目录

    # 1. 同步代码（带重试）
    code_dir = None
    try:
        code_dir = fetch_code(team_id, server_code_path)
    except Exception as e:
        log.error(f"  代码同步失败: {e}")
        R.rpush(
            RESULT_Q,
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task_id": task_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "commit_sha": task.get("commit_sha", ""),
                    "repo_full_name": task.get("repo_full_name", ""),
                    "code_path": server_code_path,
                    "scores": {},
                    "error": str(e),
                    "config_content": "",
                    "file_listing": [],
                }
            ),
        )
        return

    try:
        # ── 收集仓库元信息 ──
        config_content = read_config_content(code_dir)
        file_listing = list_code_dir(code_dir)

        # 2. 读取选手 config，筛选题目
        prefixes = read_problem_config(code_dir)
        selected = filter_problems(PROBLEMS, prefixes)

        if not selected:
            msg = (
                f"config 中的题目前缀 {prefixes} 未匹配到任何已知题目"
                if prefixes
                else "config 文件为空或不存在，且全局题目列表为空"
            )
            log.warning(f"  {msg}")
            R.rpush(
                RESULT_Q,
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "task_id": task_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "commit_sha": task.get("commit_sha", ""),
                        "repo_full_name": task.get("repo_full_name", ""),
                        "code_path": server_code_path,
                        "scores": {},
                        "error": msg,
                        "config_content": config_content,
                        "file_listing": file_listing,
                    }
                ),
            )
            return

        log.info(f"  已选择 {len(selected)}/{len(PROBLEMS)} 题进行评估")

        # 3. 创建临时目录
        tmp_dir = Path(WCFG["workspace"]) / f"{team_id}_eval_{uuid.uuid4().hex[:12]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"  临时目录: {tmp_dir}")

        try:
            # 4. 逐题评估
            scores = {}
            for prob in selected:
                pname = prob["name"]
                log.info(f"  ├ 评估 {pname} ...")
                try:
                    info = evaluate_one(prob, code_dir, tmp_dir)
                except Exception as e:
                    log.exception(f"  │  评估 {pname} 时出现未预期异常")
                    info = _empty_result(f"未预期异常: {type(e).__name__}: {e}")
                scores[pname] = info
                if info.get("error"):
                    log.warning(
                        f"  │  passed={info['passed']}  score={info['score']:.3f}  "
                        f"error={info['error'][:80]}"
                    )
                else:
                    log.info(
                        f"  │  passed={info['passed']}  score={info['score']:.3f}  "
                        f"torch={info['torch_us']:.1f}us  bangc={info['bangc_us']:.1f}us"
                    )

            # 5. 结果推回 Redis
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "team_id": team_id,
                "team_name": team_name,
                "commit_sha": task.get("commit_sha", ""),
                "repo_full_name": task.get("repo_full_name", ""),
                "code_path": server_code_path,
                "scores": scores,
                "error": None,
                "config_content": config_content,
                "file_listing": file_listing,
            }
            R.rpush(RESULT_Q, json.dumps(result))
            log.info(f"└ 任务完成 {task_id} [{team_id}]，已推送结果")

        except KeyboardInterrupt:
            log.warning(f"  任务 {task_id} 被人工中断，回传已完成题目的部分结果")
            try:
                R.rpush(
                    RESULT_Q,
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "task_id": task_id,
                            "team_id": team_id,
                            "team_name": team_name,
                            "commit_sha": task.get("commit_sha", ""),
                            "repo_full_name": task.get("repo_full_name", ""),
                            "code_path": server_code_path,
                            "scores": scores if "scores" in locals() else {},
                            "error": "任务被人工中断（partial result returned）",
                            "config_content": config_content,
                            "file_listing": file_listing,
                        }
                    ),
                )
            except Exception:
                log.exception("  人工中断后推送部分结果到 Redis 失败")
            raise

        except Exception as e:
            log.exception(f"  任务 {task_id} 处理过程中出现未预期异常")
            try:
                R.rpush(
                    RESULT_Q,
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "task_id": task_id,
                            "team_id": team_id,
                            "team_name": team_name,
                            "commit_sha": task.get("commit_sha", ""),
                            "repo_full_name": task.get("repo_full_name", ""),
                            "code_path": server_code_path,
                            "scores": scores if "scores" in dir() else {},
                            "error": f"任务未完整完成: {type(e).__name__}: {e}",
                            "config_content": config_content,
                            "file_listing": file_listing,
                        }
                    ),
                )
            except Exception:
                log.exception("  推送部分结果到 Redis 也失败了")

        finally:
            # 清理临时目录
            try:
                shutil.rmtree(tmp_dir)
                log.info(f"  已清理临时目录: {tmp_dir}")
            except Exception as e:
                log.warning(f"  清理临时目录失败: {e}")

    finally:
        # ★ 清理 Worker 本地的代码目录
        if code_dir and code_dir.exists():
            try:
                shutil.rmtree(code_dir)
                log.info(f"  已清理本地代码目录: {code_dir}")
            except Exception as e:
                log.warning(f"  清理本地代码目录失败: {e}")


# ═══════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════


def main():
    # ── 启动预检 ──
    _preflight_check()

    log.info(f"Worker 启动 | id={WORKER_ID} | 题目数={len(PROBLEMS)} | 等待任务...")
    while True:
        try:
            item = R.blpop(TASK_Q, timeout=5)
            if item is None:
                continue
            _, raw = item
            process_task(json.loads(raw))
        except redis.ConnectionError:
            log.warning("Redis 连接中断，5s 后重试...")
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Worker 正在关闭...")
            break
        except Exception:
            log.exception("未预期异常，1s 后继续")
            time.sleep(1)


if __name__ == "__main__":
    main()
