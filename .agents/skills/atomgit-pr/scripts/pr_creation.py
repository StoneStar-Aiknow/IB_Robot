#!/usr/bin/env python3
"""
AtomGit Pull Request Creation Tool

自动创建 PR，从当前分支生成 PR 标题和描述

用法：
    python3 pr_creation.py --branch <branch> --fork-owner <owner> --description-file <file> [--title "标题"] [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys

from ai_compliance import add_ai_disclosure, validate_agent_tool, validate_commit_ai_model
from atomgit_sdk import AtomGitClient, resolve_atomgit_context
from verification_gate import (
    file_triggers_dual_docker_gate,
    resolve_pr_stage,
    validate_verified_tree,
)


def run_git(args: list[str], cwd: str = None) -> str:
    """运行 git 命令并返回输出"""
    result = subprocess.run(["git"] + args, cwd=cwd or os.getcwd(), capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Git 命令失败: {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


def get_current_branch() -> str:
    """获取当前分支名"""
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"])


def is_valid_branch_name(name: str) -> bool:
    """验证分支名是否有效"""
    if not name or name in ("master", "main", "HEAD"):
        return False
    try:
        run_git(["rev-parse", "--verify", name])
        return True
    except Exception:
        return False


def get_best_base_ref(base_branch: str) -> str:
    """智能解析最佳基准引用，优先使用远程分支"""
    remotes = run_git(["remote"]).split()

    # 优先序: upstream -> origin -> local
    if "upstream" in remotes:
        return f"upstream/{base_branch}"
    if "origin" in remotes:
        return f"origin/{base_branch}"
    return base_branch


def get_commit_messages(branch: str, base_branch: str = "master") -> list[dict]:
    """获取分支相对于基线的提交信息"""
    base_ref = get_best_base_ref(base_branch)
    log_format = "%H%n%s%n%b%n---COMMIT_END---"

    try:
        output = run_git(["log", f"{base_ref}..{branch}", f"--format={log_format}"])
    except Exception:
        # 如果远程引用不存在，回退到本地
        output = run_git(["log", f"{base_branch}..{branch}", f"--format={log_format}"])

    commits = []
    for block in output.split("---COMMIT_END---"):
        if not block.strip():
            continue
        lines = block.strip().split("\n", 2)
        if len(lines) >= 2:
            commits.append(
                {
                    "hash": lines[0],
                    "subject": lines[1],
                    "body": lines[2] if len(lines) > 2 else "",
                }
            )

    return commits


def get_changed_files(branch: str, base_branch: str = "master") -> list[str]:
    """获取变更文件列表"""
    base_ref = get_best_base_ref(base_branch)
    try:
        output = run_git(["diff", "--name-only", f"{base_ref}...{branch}"])
    except Exception:
        output = run_git(["diff", "--name-only", f"{base_branch}...{branch}"])
    return [f for f in output.split("\n") if f.strip()]


def dual_docker_gate_triggered(branch: str, base_branch: str, files: list[str]) -> bool:
    """Return whether the PR diff requires tree-bound dual Docker verification."""
    base_ref = get_best_base_ref(base_branch)
    for filename in files:
        patch = ""
        if filename.endswith("/package.xml"):
            try:
                patch = run_git(["diff", "--unified=0", f"{base_ref}...{branch}", "--", filename])
            except Exception:
                patch = run_git(["diff", "--unified=0", f"{base_branch}...{branch}", "--", filename])
        if file_triggers_dual_docker_gate(filename, patch):
            return True
    return False


def get_remote_branch_head(branch: str) -> str:
    output = run_git(["ls-remote", "--heads", "origin", f"refs/heads/{branch}"])
    if not output:
        raise ValueError(f"origin/{branch} does not exist; push the verified commit before creating the PR")
    return output.split()[0].lower()


def get_local_tree(ref: str) -> str:
    return run_git(["rev-parse", f"{ref}^{{tree}}"])


def get_diff_stats(branch: str, base_branch: str = "master") -> dict:
    """获取 diff 统计信息"""
    base_ref = get_best_base_ref(base_branch)
    try:
        output = run_git(["diff", "--stat", f"{base_ref}...{branch}"])
    except Exception:
        output = run_git(["diff", "--stat", f"{base_branch}...{branch}"])

    stats = {"files_changed": 0, "insertions": 0, "deletions": 0}

    for line in output.split("\n"):
        if "file" in line and "changed" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if "file" in part:
                    stats["files_changed"] = int(part.split()[0])
                elif "insertion" in part:
                    stats["insertions"] = int(part.split()[0])
                elif "deletion" in part:
                    stats["deletions"] = int(part.split()[0])

    return stats


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if not config.get("atomgit") or not config["atomgit"].get("token"):
        raise Exception("配置文件中缺少 atomgit.token")

    return config


def main():
    parser = argparse.ArgumentParser(description="AtomGit PR 创建工具")
    parser.add_argument("--branch", type=str, help="源分支名")
    parser.add_argument("--base", type=str, default="master", help="目标分支名")
    parser.add_argument("--title", type=str, help="PR 标题")
    parser.add_argument("--description-file", type=str, required=True, help="从文件读取 PR 描述 (Markdown 格式)")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--owner", type=str, help="目标仓库 owner，覆盖 config.json")
    parser.add_argument("--repo", type=str, help="目标仓库 repo，覆盖 config.json")
    parser.add_argument(
        "--url",
        type=str,
        help="AtomGit/GitCode 仓库或 PR 链接，用于自动解析 owner/repo",
    )
    parser.add_argument(
        "--fork-owner",
        type=str,
        required=True,
        help="Fork 仓库的 owner（必需，通过 git remote -v 获取）",
    )
    parser.add_argument("--draft", action="store_true", help="创建为草稿 PR")
    parser.add_argument(
        "--pr-stage",
        choices=("wip", "review"),
        help="PR 阶段；命中双 Docker 门禁时必须在询问用户后指定",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不实际创建")
    parser.add_argument("-y", "--yes", action="store_true", help="自动确认创建 PR")
    parser.add_argument(
        "--agent-tool",
        required=True,
        help="Coding agent 执行 <tool> --version 后传入实际工具名和版本",
    )
    parser.add_argument(
        "--ai-model",
        required=True,
        help="PR 使用的 AI 模型名称及版本；多个模型用逗号分隔，不含 provider 前缀",
    )
    parser.add_argument("--prompt-summary", required=True, help="核心提示词或核心意图摘要")
    parser.add_argument("--third-party-materials", required=True, help="第三方材料、来源及许可证；没有时明确写无")
    parser.add_argument("--human-reviewed", action="store_true", help="确认开发者已人工审查 AI 辅助内容")
    args = parser.parse_args()

    try:
        sdk_config, parsed_url = resolve_atomgit_context(args.config, owner=args.owner, repo=args.repo, url=args.url)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        sys.exit(1)

    branch = args.branch or get_current_branch()

    if not is_valid_branch_name(branch):
        print(f"❌ 无效的分支名: {branch}")
        sys.exit(1)

    if branch in ("master", "main"):
        print("❌ 不能从 master/main 分支创建 PR")
        sys.exit(1)

    print("=" * 60)
    print("🚀 AtomGit PR Creator")
    print("=" * 60)
    print(f"源分支: {branch}")
    print(f"目标分支: {args.base}")
    print(f"目标仓库: {sdk_config.owner}/{sdk_config.repo}")
    print(f"Fork owner: {args.fork_owner}")
    if args.url:
        print(f"解析链接: {args.url}")
        if parsed_url.get("pr_number") is not None:
            print("ℹ️  已忽略链接中的 PR 编号，仅使用其中的仓库信息创建 PR")
        if parsed_url.get("issue_number") is not None:
            print("ℹ️  已忽略链接中的 Issue 编号，仅使用其中的仓库信息创建 PR")
    print()

    print(">>> 获取提交信息...")
    commits = get_commit_messages(branch, args.base)
    print(f"✓ 找到 {len(commits)} 个提交")

    if not commits:
        print(f"❌ 在 {args.base} 和 {branch} 之间未找到提交")
        print("   请确保你的分支与 upstream/master 同步")
        sys.exit(1)

    try:
        validate_commit_ai_model(commits, args.ai_model)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(">>> 获取变更文件...")
    files = get_changed_files(branch, args.base)
    print(f"✓ 找到 {len(files)} 个变更文件")

    print(">>> 计算变更统计...")
    stats = get_diff_stats(branch, args.base)
    print(f"✓ +{stats['insertions']}/-{stats['deletions']} 行, {stats['files_changed']} 文件")

    print(">>> 生成 PR 描述...")

    try:
        with open(args.description_file, encoding="utf-8") as f:
            description = f.read()
        print(f"✓ 从文件读取 Markdown 描述: {args.description_file}")
    except Exception as e:
        print(f"❌ 读取描述文件失败: {e}")
        sys.exit(1)

    try:
        agent_tool = validate_agent_tool(args.agent_tool)
        description = add_ai_disclosure(
            description,
            agent_tool=agent_tool,
            ai_model=args.ai_model,
            prompt_summary=args.prompt_summary,
            third_party_materials=args.third_party_materials,
        )
    except ValueError as e:
        print(f"❌ AI 披露信息无效: {e}")
        sys.exit(1)

    title = args.title or commits[0]["subject"]
    gate_required = dual_docker_gate_triggered(branch, args.base, files)
    try:
        title, gate_triggered = resolve_pr_stage(title, args.pr_stage, gate_required)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    if gate_required and not gate_triggered:
        print("ℹ️  [WIP] PR 跳过双平台 Docker 验证；移除 [WIP] 转为正式检视时门禁会恢复。")
    remote_head = None
    verified_tree = None
    if gate_triggered:
        try:
            local_head = run_git(["rev-parse", branch]).lower()
            remote_head = get_remote_branch_head(branch)
            if local_head != remote_head:
                raise ValueError(
                    f"local branch head {local_head} does not match origin/{branch} {remote_head}; push the latest commit first"
                )
            verified_tree = get_local_tree(branch)
            validate_verified_tree(description, verified_tree)
        except ValueError as e:
            print(f"❌ 双 Docker 验证 tree 校验失败: {e}")
            sys.exit(1)

    print(f"✓ 标题: {title}")

    print()
    print("=" * 60)
    print("生成的 PR 描述:")
    print("=" * 60)
    print(description)
    print("=" * 60)
    print()

    if args.dry_run:
        print("⚠ Dry run 模式，未创建 PR")
        sys.exit(0)

    if not args.human_reviewed:
        print("❌ 创建 PR 前必须由开发者人工审查，并显式传入 --human-reviewed")
        sys.exit(1)

    if args.yes:
        answer = "y"
    else:
        try:
            answer = input("\n是否创建 PR？(y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

    if answer in ("y", "yes"):
        print(">>> 创建 PR...")
        api = AtomGitClient(sdk_config)

        pr_head = f"{args.fork_owner}:{branch}"

        try:
            pr = api.create_pull_request(
                title=title,
                body=description,
                head=pr_head,
                base=args.base,
                draft=args.draft,
            )
            pr_number = pr.get("number")
            if gate_triggered:
                created_pr = api.get_pull_request(pr_number)
                created_head = (created_pr.get("head", {}).get("sha") or "").lower()
                if created_head != remote_head:
                    raise ValueError(f"PR head changed during creation: expected {remote_head}, got {created_head}")
                validate_verified_tree(description, verified_tree)
            pr_url = api.get_pr_url(pr_number)

            print()
            print("✅ PR 创建成功!")
            print(f"PR 编号: #{pr_number}")
            print(f"PR 链接: {pr_url}")

        except Exception as e:
            print(f"❌ 创建 PR 失败: {e}")
            sys.exit(1)
    else:
        print(">>> 已取消创建 PR")
        sys.exit(0)


if __name__ == "__main__":
    main()
