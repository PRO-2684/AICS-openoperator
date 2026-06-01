#!/usr/bin/env python3
"""
leaderboard_scheduler.py — 排行榜定时更新进程（独立运行）

职责：
  1. 每隔 update_interval 秒从 LEADERBOARD_Q 消费结果
  2. 与 result/{prob_id}.json 比对，更新各题排名
  3. 推送至 GitHub Pages

消费的每题结果字段（统一接口）：
  passed, score, max_abs_diff, latency, stdout, stderr, error

排行榜 JSON 中每条记录字段：
  rank, user, github, score, latency, timestamp

启动：
  python leaderboard_scheduler.py
"""

import json
import time
import subprocess
import logging
from pathlib import Path
import datetime
import shutil

from common import CFG, R, LEADERBOARD_Q

log = logging.getLogger("leaderboard_scheduler")


def remove_submissions_beta():
    target_repo = "PRO-2684/AICS-openoperator"
    prob_ids = [
        "018",
        "037",
        "048",
        "110",
        "111",
        "120",
        "121",
        "129",
        "130",
        "122",
        "134",
        "135",
    ]

    src_data_dir = Path(CFG["server"]["leaderboard_data_dir"])

    beta_data_dir = src_data_dir.with_name(f"{src_data_dir.name}_beta_copy")

    if beta_data_dir.exists():
        shutil.rmtree(beta_data_dir)

    shutil.copytree(src_data_dir, beta_data_dir)

    changed_files = []

    for pid in prob_ids:
        file = beta_data_dir / f"{pid}.json"
        if not file.exists():
            continue

        with open(file, encoding="utf-8") as f:
            board = json.load(f)

        old_results = board.get("results", [])
        new_results = [e for e in old_results if e.get("github") != target_repo]

        if len(new_results) == len(old_results):
            continue

        board["results"] = new_results
        board["results"].sort(key=lambda x: x.get("score", 0), reverse=True)

        for i, r in enumerate(board["results"]):
            r["rank"] = i + 1

        board["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with open(file, "w", encoding="utf-8") as f:
            json.dump(board, f, ensure_ascii=False, indent=2)

        changed_files.append(str(file))

    print("测试版处理完成")
    print(f"原始目录未修改: {src_data_dir}")
    print(f"副本目录: {beta_data_dir}")
    print(f"修改文件数: {len(changed_files)}")

    for file in changed_files:
        print(f"- {file}")

    shutil.rmtree(beta_data_dir)
    return {
        "source_dir": str(src_data_dir),
        "beta_dir": str(beta_data_dir),
        "changed_files": changed_files,
    }

# 想要加入的1函数


def remove_submissions():
    target_repo = "team_best" # 我们的队名
    prob_ids = [
        "018",
        "037",
        "048",
        "110",
        "111",
        "120",
        "121",
        "129",
        "130",
        "122",
        "134",
        "135",
    ]

    data_dir = Path(CFG["server"]["leaderboard_data_dir"])

    for pid in prob_ids:
        file = data_dir / f"{pid}.json"
        if not file.exists():
            continue

        with open(file, encoding="utf-8") as f:
            board = json.load(f)

        board["results"] = [
            e for e in board.get("results", []) if e.get("github") != target_repo
        ]

        board["results"].sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, r in enumerate(board["results"]):
            r["rank"] = i + 1

        board["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with open(file, "w", encoding="utf-8") as f:
            json.dump(board, f, ensure_ascii=False, indent=2)

    # Git 推送
    lb_repo = Path(CFG["server"]["leaderboard_repo"])
    rel_dir = str(data_dir.relative_to(lb_repo))
    subprocess.run(["git", "-C", str(lb_repo), "add", rel_dir], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(lb_repo),
            "commit",
            "-m",
            f"Remove {target_repo} submissions",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(lb_repo), "push"], check=True, timeout=60)

# 想要加入的函数
def update_leaderboard():
    """从 LEADERBOARD_Q 取出待处理结果，按题目更新对应的 result JSON 文件。"""
    lb_repo = Path(CFG["server"]["leaderboard_repo"])
    data_dir = Path(CFG["server"]["leaderboard_data_dir"])

    if not data_dir.exists():
        log.error(f"排行榜数据目录不存在: {data_dir}")
        return

    prob_name_to_id = {p["name"]: p["id"] for p in CFG.get("problems", [])}

    results = []
    while True:
        raw = R.lpop(LEADERBOARD_Q)
        if raw is None:
            break
        results.append(json.loads(raw))

    if not results:
        log.debug("无新结果，跳过更新")
        return

    log.info(f"读取到 {len(results)} 条新结果，开始更新排行榜")

    loaded: dict[str, dict] = {}
    changed_ids: set[str] = set()

    for res in results:
        team_name = res["team_name"]
        repo_full_name = res.get("repo_full_name", "")  # ← 改动①

        for prob_name, info in res.get("scores", {}).items():
            new_score = info.get("score", 0.0)
            if new_score <= 0:
                continue

            prob_id = prob_name_to_id.get(prob_name)
            if prob_id is None:
                log.warning(f"题目 '{prob_name}' 不在配置中，跳过")
                continue

            if prob_id not in loaded:
                result_file = data_dir / f"{prob_id}.json"
                if result_file.exists():
                    with open(result_file, encoding="utf-8") as f:
                        loaded[prob_id] = json.load(f)
                else:
                    log.info(f"题目结果文件不存在，将创建: {result_file}")
                    loaded[prob_id] = {
                        "problem_id": prob_id,
                        "problem_name": prob_name,
                        "results": [],
                        "last_updated": None,
                    }
            board = loaded[prob_id]
            entries = board.setdefault("results", [])
            existing = next((e for e in entries if e.get("user") == team_name), None)

            new_latency = info.get("latency", 0.0)  # ← 改动②：保留 float (μs)

            if existing:
                if new_score > existing.get("score", 0):
                    existing["score"] = new_score
                    existing["latency"] = new_latency  # ← 改动②
                    existing["github"] = repo_full_name  # ← 改动①
                    existing["timestamp"] = res["timestamp"]  # ← 改动③：不截断
                    changed_ids.add(prob_id)
                    log.info(
                        f"  更新 {team_name} @ {prob_name}: "
                        f"{new_score:.6f}  latency={new_latency:.3f}us"
                    )
            else:
                entries.append(
                    {
                        "rank": 0,
                        "user": team_name,
                        "github": repo_full_name,  # ← 改动①
                        "score": new_score,
                        "latency": new_latency,  # ← 改动②
                        "timestamp": res["timestamp"],  # ← 改动③
                    }
                )
                changed_ids.add(prob_id)
                log.info(
                    f"  新增 {team_name} @ {prob_name}: "
                    f"{new_score:.6f}  latency={new_latency:.3f}us"
                )

    if not changed_ids:
        log.info("所有结果均未超越历史最优，排行榜无变化")
        return

    for prob_id in changed_ids:
        board = loaded[prob_id]
        rl = board.get("results", [])
        rl.sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, r in enumerate(rl):
            r["rank"] = i + 1
        board["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        result_file = data_dir / f"{prob_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(board, f, ensure_ascii=False, indent=2)

    log.info(f"已更新 {len(changed_ids)} 个题目文件: {sorted(changed_ids)}")

    try:
        rel_data_dir = str(data_dir.relative_to(lb_repo))
        subprocess.run(
            ["git", "-C", str(lb_repo), "add", rel_data_dir],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(lb_repo),
                "commit",
                "-m",
                f"leaderboard update {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M}",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(lb_repo), "push"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        log.info("排行榜已推送至 GitHub Pages ✓")
    except subprocess.CalledProcessError as e:
        log.error(f"Git 推送失败: {e.stderr}")


def run():
    interval = CFG["server"].get("update_interval", 300)
    log.info(f"排行榜调度器已启动，每 {interval}s 更新一次")
    while True:
        time.sleep(interval)
        try:
            update_leaderboard()
        except Exception:
            log.exception("排行榜更新异常")


if __name__ == "__main__":
    run()
