#!/usr/bin/env python3
"""
批量审计历史提交中可能存在的 Python 注入 payload。

用法:
  source ~/.bashrc
  python audit_python_injection.py
  python audit_python_injection.py --date 2026-04-27
  python audit_python_injection.py --team 70
  python audit_python_injection.py --limit 20 --keep-repos

说明:
  - 从 result_archive 读取历史评测记录
  - 按 (repo_full_name, commit_sha) 去重
  - 使用环境变量 GITHUB_TOKEN / GH_TOKEN 精确拉取对应 commit
  - 默认重点扫描仓库中的 .mlu 文件
  - 输出控制台摘要，并将详细结果写入 JSON 报告
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_DIR = SCRIPT_DIR / "result_archive"
DEFAULT_WORKSPACE = SCRIPT_DIR / ".audit_repos"
DEFAULT_REPORT = SCRIPT_DIR / "python_injection_audit_report.json"

SEVERITY_ORDER = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


@dataclass
class Rule:
    rule_id: str
    severity: str
    pattern: re.Pattern[str]
    message: str


@dataclass
class Finding:
    rule_id: str
    severity: str
    file: str
    line: int
    snippet: str
    message: str


RULES = [
    Rule(
        "STRING_ESCAPE_THEN_PYTHON",
        "critical",
        re.compile(
            r'"""\s*(?:\r?\n|\r)\s*(?:#.*(?:\r?\n|\r)\s*)*(?:import|from|def|class)\b'
        ),
        "出现三引号后直接接 Python 语句，极可能用于逃逸 bang_func_source 三引号字符串。",
    ),
    Rule(
        "TRIPLE_DOUBLE_QUOTE",
        "high",
        re.compile(r'"""'),
        "出现三引号，可能用于闭合外围 Python 字符串。",
    ),
    Rule(
        "TRIPLE_SINGLE_QUOTE",
        "high",
        re.compile(r"'''"),
        "出现三单引号，可能用于构造 Python 多行字符串逃逸。",
    ),
    Rule(
        "PYTHON_IMPORT",
        "high",
        re.compile(r"(?m)^\s*(import\s+\w+|from\s+\w+\s+import\s+)"),
        "在提交源码中发现 Python import 语句。",
    ),
    Rule(
        "PYTHON_DEF_CLASS",
        "high",
        re.compile(r"(?m)^\s*(def|class)\s+\w+"),
        "在提交源码中发现 Python 函数或类定义。",
    ),
    Rule(
        "PYTHON_EXEC_PRIMITIVE",
        "high",
        re.compile(r"\b(exec|eval|compile|__import__)\s*\("),
        "发现 Python 动态执行原语。",
    ),
    Rule(
        "PYTHON_MONKEY_PATCH",
        "high",
        re.compile(
            r"(__init_subclass__|gc\.get_objects|sys\.modules|setattr\s*\(|torch\.nn\.Module|_real_nn\.Module)"
        ),
        "发现 monkey patch / 运行时反射特征。",
    ),
    Rule(
        "PYTHON_FILE_ENV_ACCESS",
        "medium",
        re.compile(r"\b(os\.environ|Path\s*\(|read_text\s*\(|write_text\s*\()"),
        "发现环境变量或文件访问模式，常见于注入 payload。",
    ),
]


def parse_args():
    p = argparse.ArgumentParser(description="审计历史提交中的 Python 注入风险")
    p.add_argument(
        "--archive-dir", default=str(DEFAULT_ARCHIVE_DIR), help="result_archive 根目录"
    )
    p.add_argument(
        "--workspace", default=str(DEFAULT_WORKSPACE), help="拉取仓库的临时目录"
    )
    p.add_argument("--report", default=str(DEFAULT_REPORT), help="输出 JSON 报告路径")
    p.add_argument(
        "--token", default=None, help="直接传入 GitHub token；不传则从环境变量读取"
    )
    p.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="优先读取的 token 环境变量名，默认 GITHUB_TOKEN",
    )
    p.add_argument("--date", default=None, help="只检查指定日期目录，如 2026-04-27")
    p.add_argument("--team", default=None, help="只检查指定 team_id")
    p.add_argument(
        "--limit", type=int, default=0, help="最多检查多少个唯一 commit，0 表示不限制"
    )
    p.add_argument(
        "--all-text", action="store_true", help="扫描所有小型文本文件，不仅限于 .mlu"
    )
    p.add_argument("--keep-repos", action="store_true", help="保留拉取下来的仓库目录")
    return p.parse_args()


def get_github_token(args) -> str:
    token = (
        args.token
        or os.environ.get(args.token_env, "")
        or os.environ.get("GITHUB_TOKEN", "")
        or os.environ.get("GH_TOKEN", "")
    )
    if not token:
        raise RuntimeError(
            f"未找到 GitHub token。请传 --token，或先导出 {args.token_env}（也支持 GITHUB_TOKEN / GH_TOKEN）"
        )
    return token


def load_archive_records(
    archive_dir: Path, date: str | None, team: str | None
) -> list[dict]:
    root = archive_dir / date if date else archive_dir
    if not root.exists():
        raise FileNotFoundError(f"archive 目录不存在: {root}")

    records = []
    for jf in sorted(root.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if team and str(data.get("team_id", "")) != str(team):
            continue
        commit_sha = str(data.get("commit_sha", "")).strip()
        repo_full_name = str(data.get("repo_full_name", "")).strip()
        if not commit_sha or not repo_full_name:
            continue
        data["_source"] = str(jf)
        records.append(data)
    return records


def dedup_records(records: list[dict]) -> list[dict]:
    unique = {}
    for record in records:
        key = (record["repo_full_name"], record["commit_sha"])
        existing = unique.get(key)
        if existing is None or record.get("timestamp", "") > existing.get(
            "timestamp", ""
        ):
            unique[key] = record
    return sorted(unique.values(), key=lambda x: (x["repo_full_name"], x["commit_sha"]))


def build_clone_url(repo_full_name: str, token: str) -> str:
    return f"https://x-access-token:{token}@github.com/{repo_full_name}.git"


def clone_commit(
    repo_full_name: str, commit_sha: str, workspace: Path, token: str
) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = (
        workspace
        / f"{repo_full_name.replace('/', '__')}__{commit_sha[:8]}__{uuid.uuid4().hex[:8]}"
    )
    repo_url = build_clone_url(repo_full_name, token)

    subprocess.run(
        ["git", "init", str(repo_dir)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "add", "origin", repo_url],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", commit_sha],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "FETCH_HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return repo_dir


def is_text_file(path: Path, max_bytes: int = 2 * 1024 * 1024) -> bool:
    try:
        if path.stat().st_size > max_bytes:
            return False
        path.read_text(encoding="utf-8")
        return True
    except Exception:
        return False


def iter_candidate_files(repo_dir: Path, all_text: bool) -> list[Path]:
    candidates = []
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(repo_dir).parts
        if ".git" in rel_parts:
            continue
        if all_text:
            if is_text_file(path):
                candidates.append(path)
        else:
            if path.suffix.lower() == ".mlu" and is_text_file(path):
                candidates.append(path)
    return candidates


def _line_of_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _snippet_at_line(text: str, line_no: int) -> str:
    lines = text.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()[:200]
    return ""


def scan_file(path: Path, repo_dir: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            line_no = _line_of_offset(text, match.start())
            findings.append(
                Finding(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    file=str(path.relative_to(repo_dir)),
                    line=line_no,
                    snippet=_snippet_at_line(text, line_no),
                    message=rule.message,
                )
            )
    return findings


def highest_severity(findings: list[Finding]) -> str | None:
    if not findings:
        return None
    return max(findings, key=lambda x: SEVERITY_ORDER[x.severity]).severity


def audit_one_record(
    record: dict, workspace: Path, token: str, all_text: bool, keep_repos: bool
) -> dict:
    repo_dir = None
    findings: list[Finding] = []
    error = None
    scanned_files = []
    try:
        repo_dir = clone_commit(
            record["repo_full_name"], record["commit_sha"], workspace, token
        )
        candidates = iter_candidate_files(repo_dir, all_text=all_text)
        scanned_files = [str(p.relative_to(repo_dir)) for p in candidates]
        for path in candidates:
            findings.extend(scan_file(path, repo_dir))
    except subprocess.CalledProcessError as e:
        error = e.stderr.strip() or e.stdout.strip() or str(e)
    except Exception as e:
        error = str(e)
    finally:
        if repo_dir and repo_dir.exists() and not keep_repos:
            shutil.rmtree(repo_dir, ignore_errors=True)

    findings.sort(key=lambda x: (-SEVERITY_ORDER[x.severity], x.file, x.line))
    severity = highest_severity(findings)
    return {
        "team_id": record.get("team_id", ""),
        "team_name": record.get("team_name", ""),
        "repo_full_name": record["repo_full_name"],
        "commit_sha": record["commit_sha"],
        "source_record": record.get("_source", ""),
        "scanned_files": scanned_files,
        "suspicious": bool(findings),
        "highest_severity": severity,
        "finding_count": len(findings),
        "findings": [asdict(f) for f in findings],
        "error": error,
    }


def print_summary(results: list[dict]):
    suspicious = [r for r in results if r["suspicious"]]
    clean = [r for r in results if not r["suspicious"] and not r["error"]]
    errors = [r for r in results if r["error"]]

    print(f"Total commits checked : {len(results)}")
    print(f"Suspicious commits    : {len(suspicious)}")
    print(f"Clean commits         : {len(clean)}")
    print(f"Clone/scan errors     : {len(errors)}")
    print("")

    for item in suspicious:
        print(
            f"[SUSPICIOUS] team={item['team_id']} repo={item['repo_full_name']} "
            f"commit={item['commit_sha'][:8]} severity={item['highest_severity']} findings={item['finding_count']}"
        )
        for finding in item["findings"][:10]:
            print(
                f"  - {finding['severity']:>8} {finding['file']}:{finding['line']} "
                f"{finding['rule_id']} | {finding['snippet']}"
            )
        if item["finding_count"] > 10:
            print(f"  - ... {item['finding_count'] - 10} more findings")
        print("")

    for item in errors:
        print(
            f"[ERROR] team={item['team_id']} repo={item['repo_full_name']} "
            f"commit={item['commit_sha'][:8]} error={item['error'][:200]}"
        )


def main():
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    workspace = Path(args.workspace)
    report_path = Path(args.report)

    token = get_github_token(args)
    records = load_archive_records(archive_dir, args.date, args.team)
    records = dedup_records(records)
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        print("No archive records matched.")
        return 0

    results = []
    for idx, record in enumerate(records, start=1):
        print(
            f"[{idx}/{len(records)}] checking {record['repo_full_name']} "
            f"{record['commit_sha'][:8]} ..."
        )
        results.append(
            audit_one_record(
                record,
                workspace=workspace,
                token=token,
                all_text=args.all_text,
                keep_repos=args.keep_repos,
            )
        )

    report = {
        "archive_dir": str(archive_dir),
        "date": args.date,
        "team": args.team,
        "all_text": args.all_text,
        "checked_commits": len(results),
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("")
    print_summary(results)
    print("")
    print(f"JSON report written to: {report_path}")
    return 0


if __name__ == "__main__":
    print(scan_file(Path("Linear.mlu"), Path(".")))
    # raise SystemExit(main())
