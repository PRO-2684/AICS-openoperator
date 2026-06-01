#!/usr/bin/env python3
"""
按 result_archive 中的历史提交记录回放拉取代码，并使用当前 BangC 审计规则检查违规提交。

行为：
  1. 按历史记录时间顺序遍历 result_archive/*.json
  2. 根据 repo_full_name + commit_sha 精确拉取对应代码
  3. 复用 worker.py 当前的题目筛选与 audit_bangc_source 规则
  4. 一旦某个 team_id 命中违规，立即写入 skip 文件
  5. 后续再遇到该 team_id，直接跳过
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import worker


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_DIR = SCRIPT_DIR / "result_archive"
DEFAULT_WORKSPACE = SCRIPT_DIR / ".bangc_archive_audit_repos"
DEFAULT_REPORT = SCRIPT_DIR / "bangc_archive_rule_audit_report.json"
DEFAULT_SKIP_FILE = SCRIPT_DIR / "bangc_archive_rule_bad_teams.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bangc_archive_audit")


def parse_args():
    p = argparse.ArgumentParser(
        description="回放 result_archive 并使用当前 BangC 审计规则检查历史提交"
    )
    p.add_argument(
        "--archive-dir", default=str(DEFAULT_ARCHIVE_DIR), help="result_archive 根目录"
    )
    p.add_argument(
        "--workspace", default=str(DEFAULT_WORKSPACE), help="临时拉取仓库目录"
    )
    p.add_argument("--report", default=str(DEFAULT_REPORT), help="输出 JSON 报告路径")
    p.add_argument(
        "--skip-file", default=str(DEFAULT_SKIP_FILE), help="违规 team_id 持久化文件"
    )
    p.add_argument("--token", default=None, help="直接传入 GitHub token")
    p.add_argument(
        "--token-env", default="GITHUB_TOKEN", help="优先读取的 token 环境变量名"
    )
    p.add_argument("--date", default=None, help="只检查指定日期目录，如 2026-04-27")
    p.add_argument("--team", default=None, help="只检查指定 team_id")
    p.add_argument(
        "--limit", type=int, default=0, help="最多处理多少条未跳过的记录，0 表示不限制"
    )
    p.add_argument("--keep-repos", action="store_true", help="保留拉取下来的临时仓库")
    return p.parse_args()


def get_github_token(args) -> str:
    token = (
        args.token
        or os.environ.get(args.token_env, "")
        or os.environ.get("GITHUB_TOKEN", "")
        or os.environ.get("GH_TOKEN", "")
        or worker.CFG.get("server", {}).get("github_token", "")
    )
    if not token:
        raise RuntimeError(
            f"未找到 GitHub token。请传 --token，或设置 {args.token_env} / GITHUB_TOKEN / GH_TOKEN，"
            "或在 config.yaml 的 server.github_token 中配置。"
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
        except Exception as e:
            log.warning(f"跳过无法解析的归档记录 {jf}: {e}")
            continue

        if team and str(data.get("team_id", "")) != str(team):
            continue

        commit_sha = str(data.get("commit_sha", "")).strip()
        repo_full_name = str(data.get("repo_full_name", "")).strip()
        team_id = str(data.get("team_id", "")).strip()
        if not commit_sha or not repo_full_name or not team_id:
            continue

        data["_source"] = str(jf)
        records.append(data)

    records.sort(key=lambda item: (item.get("timestamp", ""), item["_source"]))
    return records


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


def load_skip_state(skip_file: Path) -> dict:
    if not skip_file.exists():
        return {"team_ids": {}, "updated_at": None}

    try:
        data = json.loads(skip_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"无法解析 skip 文件 {skip_file}: {e}") from e

    if "team_ids" not in data or not isinstance(data["team_ids"], dict):
        raise RuntimeError(f"skip 文件格式不正确: {skip_file}")
    return data


def save_skip_state(skip_file: Path, state: dict):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    skip_file.parent.mkdir(parents=True, exist_ok=True)
    skip_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_repo(repo_dir: Path) -> dict:
    prefixes = worker.read_problem_config(repo_dir)
    selected = worker.filter_problems(worker.PROBLEMS, prefixes)

    violations = []
    checked_files = []
    missing_files = []

    for prob in selected:
        student_path = worker.find_file_icase(repo_dir, prob["mlu_path"])
        if student_path is None:
            missing_files.append(prob["mlu_path"])
            continue

        checked_files.append(str(student_path.relative_to(repo_dir)))
        try:
            worker.audit_bangc_source(student_path, prob)
        except ValueError as e:
            violations.append(
                {
                    "problem_id": str(prob["id"]),
                    "problem_name": prob["name"],
                    "mlu_path": prob["mlu_path"],
                    "file": str(student_path.relative_to(repo_dir)),
                    "message": str(e),
                }
            )

    return {
        "prefixes": prefixes,
        "selected_problem_count": len(selected),
        "selected_problem_names": [prob["name"] for prob in selected],
        "checked_files": checked_files,
        "missing_files": missing_files,
        "violations": violations,
    }


def mark_team_bad(skip_state: dict, record: dict, repo_audit: dict):
    team_id = str(record["team_id"])
    if team_id in skip_state["team_ids"]:
        return

    skip_state["team_ids"][team_id] = {
        "team_name": record.get("team_name", ""),
        "repo_full_name": record.get("repo_full_name", ""),
        "commit_sha": record.get("commit_sha", ""),
        "timestamp": record.get("timestamp", ""),
        "source_record": record.get("_source", ""),
        "violation_count": len(repo_audit["violations"]),
        "violations": repo_audit["violations"],
    }


def build_report_payload(
    archive_dir: Path,
    skip_file: Path,
    records: list[dict],
    findings: list[dict],
    clone_errors: list[dict],
    skipped_records: list[dict],
    processed_count: int,
    checked_count: int,
    bad_team_ids: set[str],
) -> dict:
    return {
        "archive_dir": str(archive_dir),
        "skip_file": str(skip_file),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "processed_record_count": processed_count,
        "checked_record_count": checked_count,
        "flagged_team_count": len(bad_team_ids),
        "flagged_team_ids": sorted(bad_team_ids, key=lambda x: (len(x), x)),
        "finding_count": len(findings),
        "clone_error_count": len(clone_errors),
        "skipped_record_count": len(skipped_records),
        "findings": findings,
        "clone_errors": clone_errors,
        "skipped_records": skipped_records,
    }


def main():
    args = parse_args()

    archive_dir = Path(args.archive_dir)
    workspace = Path(args.workspace)
    report_path = Path(args.report)
    skip_file = Path(args.skip_file)

    token = get_github_token(args)
    records = load_archive_records(archive_dir, args.date, args.team)
    skip_state = load_skip_state(skip_file)
    bad_team_ids = set(skip_state["team_ids"].keys())

    findings = []
    clone_errors = []
    skipped_records = []
    processed_count = 0
    checked_count = 0

    if not records:
        log.info("没有匹配到任何归档记录。")
        report_path.write_text(
            json.dumps(
                build_report_payload(
                    archive_dir=archive_dir,
                    skip_file=skip_file,
                    records=[],
                    findings=[],
                    clone_errors=[],
                    skipped_records=[],
                    processed_count=0,
                    checked_count=0,
                    bad_team_ids=bad_team_ids,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    log.info(f"待处理归档记录数: {len(records)}")
    if bad_team_ids:
        log.info(f"已加载历史跳过 team_id 数: {len(bad_team_ids)}")

    for record in records:
        team_id = str(record["team_id"])
        if team_id in bad_team_ids:
            skipped_records.append(
                {
                    "team_id": team_id,
                    "team_name": record.get("team_name", ""),
                    "repo_full_name": record.get("repo_full_name", ""),
                    "commit_sha": record.get("commit_sha", ""),
                    "timestamp": record.get("timestamp", ""),
                    "source_record": record.get("_source", ""),
                    "reason": "team already flagged",
                }
            )
            continue

        if args.limit and processed_count >= args.limit:
            break

        processed_count += 1
        repo_dir = None
        log.info(
            f"[{processed_count}] 检查 team={team_id} repo={record['repo_full_name']} "
            f"commit={record['commit_sha'][:8]}"
        )

        try:
            repo_dir = clone_commit(
                record["repo_full_name"], record["commit_sha"], workspace, token
            )
            checked_count += 1
            repo_audit = audit_repo(repo_dir)
        except subprocess.CalledProcessError as e:
            clone_errors.append(
                {
                    "team_id": team_id,
                    "team_name": record.get("team_name", ""),
                    "repo_full_name": record.get("repo_full_name", ""),
                    "commit_sha": record.get("commit_sha", ""),
                    "timestamp": record.get("timestamp", ""),
                    "source_record": record.get("_source", ""),
                    "error": (e.stderr or e.stdout or str(e))[:4000],
                }
            )
            continue
        except Exception as e:
            clone_errors.append(
                {
                    "team_id": team_id,
                    "team_name": record.get("team_name", ""),
                    "repo_full_name": record.get("repo_full_name", ""),
                    "commit_sha": record.get("commit_sha", ""),
                    "timestamp": record.get("timestamp", ""),
                    "source_record": record.get("_source", ""),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue
        finally:
            if repo_dir and repo_dir.exists() and not args.keep_repos:
                shutil.rmtree(repo_dir, ignore_errors=True)

        if repo_audit["violations"]:
            finding = {
                "team_id": team_id,
                "team_name": record.get("team_name", ""),
                "repo_full_name": record.get("repo_full_name", ""),
                "commit_sha": record.get("commit_sha", ""),
                "timestamp": record.get("timestamp", ""),
                "source_record": record.get("_source", ""),
                "prefixes": repo_audit["prefixes"],
                "selected_problem_count": repo_audit["selected_problem_count"],
                "selected_problem_names": repo_audit["selected_problem_names"],
                "checked_files": repo_audit["checked_files"],
                "missing_files": repo_audit["missing_files"],
                "violations": repo_audit["violations"],
            }
            findings.append(finding)
            bad_team_ids.add(team_id)
            mark_team_bad(skip_state, record, repo_audit)
            save_skip_state(skip_file, skip_state)
            log.warning(
                f"  命中违规: team={team_id} violations={len(repo_audit['violations'])}，"
                "后续记录将直接跳过"
            )

    report = build_report_payload(
        archive_dir=archive_dir,
        skip_file=skip_file,
        records=records,
        findings=findings,
        clone_errors=clone_errors,
        skipped_records=skipped_records,
        processed_count=processed_count,
        checked_count=checked_count,
        bad_team_ids=bad_team_ids,
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    log.info(
        f"完成: processed={processed_count} checked={checked_count} "
        f"flagged_teams={len(bad_team_ids)} findings={len(findings)} "
        f"skipped={len(skipped_records)} clone_errors={len(clone_errors)}"
    )
    log.info(f"报告已写入: {report_path}")
    log.info(f"违规 team_id 已写入: {skip_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
