#!/usr/bin/env python3
"""
IB-Robot INTRO Skill - 仓库状态检测与 Skill 推荐脚本

根据当前仓库状态（git status、编译产物、PR 评论等）
智能推荐用户最应该使用的 skill。
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录（通过 git rev-parse）"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return Path.cwd()


def check_uncommitted_changes() -> dict:
    """检测是否有未提交的代码改动"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        staged = [l for l in lines if l[0] in ("A", "M", "D", "R", "C")]
        unstaged = [l for l in lines if l[1] in ("M", "D")]

        return {
            "has_changes": len(lines) > 0,
            "total_files": len(lines),
            "staged_count": len(staged),
            "unstaged_count": len(unstaged),
            "files": lines[:10],  # 最多显示 10 个文件
        }
    except subprocess.CalledProcessError:
        return {"has_changes": False, "total_files": 0, "error": "git status failed"}


def check_build_artifacts(project_root: Path) -> dict:
    """检测编译产物是否存在"""
    install_dir = project_root / "install"
    build_dir = project_root / "build"

    install_exists = install_dir.exists() and any(install_dir.iterdir())
    build_exists = build_dir.exists() and any(build_dir.iterdir())

    return {
        "install_exists": install_exists,
        "build_exists": build_exists,
        "needs_build": not install_exists,
    }


def check_open_prs_with_comments() -> dict:
    """检测是否有 open 的 PR 且有未回复评论（需要 AtomGit SDK）"""
    result = {
        "has_open_prs": False,
        "prs_with_comments": [],
        "available": False,
    }

    try:
        from atomgit_sdk.config import AtomGitConfig
        from atomgit_sdk.client import AtomGitClient
    except ImportError:
        result["error"] = "atomgit_sdk not in PYTHONPATH (请先 source .shrc_local)"
        return result

    try:
        config = AtomGitConfig.from_json("config.json")
        client = AtomGitClient(config)
        result["available"] = True

        prs = client.get_pull_requests(state="open")
        if not prs:
            return result

        result["has_open_prs"] = True

        for pr in prs[:5]:  # 最多检查 5 个 PR
            pr_number = pr.get("number")
            title = pr.get("title", "")
            try:
                comments = client.get_pr_comments(pr_number)
                if comments:
                    result["prs_with_comments"].append({
                        "number": pr_number,
                        "title": title,
                        "comment_count": len(comments),
                    })
            except Exception:
                continue

    except Exception as e:
        result["error"] = f"AtomGit API 调用失败: {str(e)}"

    return result


def check_current_branch() -> dict:
    """获取当前分支信息"""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True
        )
        branch = result.stdout.strip()

        # 检查是否有未推送的 commit
        try:
            ahead_result = subprocess.run(
                ["git", "rev-list", "--count", f"origin/{branch}..HEAD"],
                capture_output=True, text=True, check=True
            )
            unpushed = int(ahead_result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            unpushed = 0

        return {
            "branch": branch,
            "unpushed_commits": unpushed,
        }
    except subprocess.CalledProcessError:
        return {"branch": "unknown", "unpushed_commits": 0}


def generate_recommendations(
    changes: dict,
    build: dict,
    prs: dict,
    branch: dict,
) -> list:
    """根据仓库状态生成推荐列表"""
    recommendations = []

    # 优先级 1: 编译产物缺失
    if build.get("needs_build"):
        recommendations.append({
            "skill": "ibrobot-build",
            "reason": "编译产物缺失（install/ 目录为空或不存在），建议先编译项目",
            "priority": "🔴 高",
            "trigger": "编译一下项目",
        })

    # 优先级 2: 有未回复评论的 PR（最多显示一个）
    if prs.get("prs_with_comments"):
        for pr_info in prs["prs_with_comments"][:1]:
            recommendations.append({
                "skill": "atomgit-review-resolution",
                "reason": f"PR #{pr_info['number']}「{pr_info['title']}」有 {pr_info['comment_count']} 条评论待处理",
                "priority": "🟡 中",
                "trigger": f"修复 #{pr_info['number']} 号 PR 的评审意见",
            })

    # 优先级 3: 有未提交的改动
    if changes.get("has_changes"):
        recommendations.append({
            "skill": "ibrobot-git-flow",
            "reason": f"检测到 {changes['total_files']} 个文件有未提交的改动",
            "priority": "🟡 中",
            "trigger": "提交代码",
        })

    # 优先级 4: 有未推送的 commit
    if branch.get("unpushed_commits", 0) > 0:
        recommendations.append({
            "skill": "atomgit-pr",
            "reason": f"当前分支 `{branch['branch']}` 有 {branch['unpushed_commits']} 个未推送的 commit",
            "priority": "🟢 低",
            "trigger": "帮我提交一个 PR",
        })

    # 默认推荐
    if not recommendations:
        recommendations.append({
            "skill": "ibrobot-launch",
            "reason": "当前工作区状态良好！可以试试启动机器人仿真环境进行开发测试",
            "priority": "💡 推荐",
            "trigger": "启动机器人仿真",
        })

    return recommendations


def print_report(
    changes: dict,
    build: dict,
    prs: dict,
    branch: dict,
    recommendations: list,
):
    """输出检测报告"""
    print("=" * 60)
    print("🔍 IB-Robot 仓库状态检测报告")
    print("=" * 60)
    print()

    # 分支状态
    print(f"📌 当前分支: {branch.get('branch', 'unknown')}")
    if branch.get("unpushed_commits", 0) > 0:
        print(f"   ⚠️  有 {branch['unpushed_commits']} 个未推送的 commit")
    print()

    # Git 状态
    print("📂 工作区状态:")
    if changes.get("has_changes"):
        print(f"   ⚠️  {changes['total_files']} 个文件有改动")
        for f in changes.get("files", [])[:5]:
            print(f"       {f}")
        if changes["total_files"] > 5:
            print(f"       ... 还有 {changes['total_files'] - 5} 个文件")
    else:
        print("   ✅ 工作区干净，无未提交改动")
    print()

    # 编译状态
    print("🔨 编译状态:")
    if build.get("needs_build"):
        print("   ⚠️  install/ 目录缺失或为空，需要编译")
    else:
        print("   ✅ 编译产物存在")
    print()

    # PR 状态
    print("🌐 AtomGit PR 状态:")
    if not prs.get("available"):
        error = prs.get("error", "未知错误")
        print(f"   ⏭️  跳过（{error}）")
    elif prs.get("prs_with_comments"):
        for pr_info in prs["prs_with_comments"]:
            print(f"   📝 PR #{pr_info['number']}「{pr_info['title']}」- {pr_info['comment_count']} 条评论")
    elif prs.get("has_open_prs"):
        print("   ✅ 有 open PR，但无待处理评论")
    else:
        print("   ℹ️  无 open 状态的 PR")
    print()

    # 推荐
    print("=" * 60)
    print("🎯 推荐操作")
    print("=" * 60)
    print()

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. [{rec['priority']}] {rec['skill']}")
        print(f"     原因: {rec['reason']}")
        print(f"     触发: \"{rec['trigger']}\"")
        print()

    print("=" * 60)
    print("💡 提示: 直接用自然语言告诉 Agent 你想做什么即可！")
    print("=" * 60)


def main():
    project_root = get_project_root()
    os.chdir(project_root)

    print("正在检测仓库状态...\n")

    # 收集仓库状态
    changes = check_uncommitted_changes()
    build = check_build_artifacts(project_root)
    branch = check_current_branch()
    prs = check_open_prs_with_comments()

    # 生成推荐
    recommendations = generate_recommendations(changes, build, prs, branch)

    # 输出报告
    print_report(changes, build, prs, branch, recommendations)

    # 同时输出 JSON 格式到 tmp 目录（供 Agent 解析）
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    report = {
        "branch": branch,
        "changes": changes,
        "build": build,
        "prs": prs,
        "recommendations": recommendations,
    }
    report_path = tmp_dir / "intro_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()
