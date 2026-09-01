#!/usr/bin/env python3
"""
AtomGit PR Management Tool (Agent 驱动版)

专门为 AI Agent 设计，提供两个核心能力：
1. --fetch-info: 为 Agent 提取 PR 的完整上下文 (提交、文件、Diff)
2. --update-pr: 接收 Agent 生成的精美描述并同步到服务器
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ai_compliance import (
    add_ai_disclosure,
    reject_handwritten_disclosure,
    validate_agent_tool,
    validate_commit_ai_model,
)
from atomgit_sdk import AtomGitClient, resolve_atomgit_context
from reuse_gate import count_changed_lines, reuse_gate_required, reuse_self_check_status, validate_reuse_self_check
from verification_gate import (
    compute_verification_inputs,
    extract_verification_metadata,
    is_wip_title,
    prepare_update_verification,
    resolve_pr_head_tree,
    resolve_pr_stage,
    validate_verification_metadata,
)


def load_config(config_path: str = "config.json") -> dict:
    """加载配置文件"""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if not config.get("atomgit") or not config["atomgit"].get("token"):
        raise Exception("配置文件中缺少 atomgit.token")

    return config


def mode_fetch_info(args, api: AtomGitClient):
    """
    模式1: 获取 PR 信息 (Agent 学习模式)

    此模式会将 PR 的所有上下文导出为 JSON，Agent 读取后可根据代码 Diff
    生成高质量的 PR 描述。
    """
    print("\n" + "=" * 60)
    print("📥 模式: 提取 PR 上下文 (Agent 学习中)")
    print("=" * 60)

    print(f"\n📝 正在从 #{args.pr} 抓取元数据...")

    try:
        pr = api.get_pull_request(args.pr)
        commits = api.get_pr_commits(args.pr)
        files = api.get_pr_files(args.pr)
        comments = [] if args.no_comments else api.get_all_pr_comments(args.pr)
        verification_inputs = compute_verification_inputs(files)
        head_tree = None
        verification_metadata = None
        if verification_inputs and not is_wip_title(pr.get("title") or ""):
            try:
                verification_metadata = extract_verification_metadata(pr.get("body") or "")
            except ValueError:
                verification_metadata = None
            if not verification_metadata or verification_metadata.get("mode") != "reused-environment":
                try:
                    head_tree = resolve_pr_head_tree(pr)
                except ValueError:
                    head_tree = None

        # 统计信息
        additions = sum(f.get("additions", 0) for f in files)
        deletions = sum(f.get("deletions", 0) for f in files)

        pr_context = {
            "pr_number": args.pr,
            "fetch_time": datetime.now().isoformat(),
            "metadata": {
                "title": pr.get("title", ""),
                "author": pr.get("user", {}).get("login", ""),
                "state": pr.get("state", ""),
                "branch": f"{pr.get('head', {}).get('ref', '')} -> {pr.get('base', {}).get('ref', '')}",
                "head_sha": pr.get("head", {}).get("sha", ""),
                "head_tree": head_tree,
                "wip": is_wip_title(pr.get("title") or ""),
                "verification_inputs": verification_inputs,
                "verification": verification_metadata,
                "reuse_self_check": {
                    "changed_lines": count_changed_lines(files),
                    "required": reuse_gate_required(files),
                    "status": reuse_self_check_status(pr.get("body") or ""),
                },
                "stats": {
                    "files_changed": len(files),
                    "additions": additions,
                    "deletions": deletions,
                    "comments": len(comments),
                },
            },
            "commits": [
                {
                    "sha": c.get("sha", ""),
                    "author": c.get("commit", {}).get("author", {}).get("name", ""),
                    "message": c.get("commit", {}).get("message", ""),
                }
                for c in commits
            ],
            "changed_files": [
                {
                    "filename": f.get("filename", ""),
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch", ""),  # 关键：包含代码 Diff
                }
                for f in files
            ],
            "comments": comments,
        }

        # 保存为 Agent 可读的 JSON
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        repo_name = api.config.repo.lower().replace("-", "_")
        output_file = output_dir / f"{repo_name}_pr_{args.pr}_context.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pr_context, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 上下文已准备就绪: {output_file}")
        print(f"📊 统计: {len(commits)} 提交, {len(files)} 文件修改, +{additions}/-{deletions}, {len(comments)} 评论")
        print("\n💡 Agent 指令:")
        print("  请分析该 JSON 中的 patch 字段，总结变更背后的逻辑，然后调用:")
        print("  --update-pr <description_json>")

    except Exception as e:
        print(f"\n❌ 获取数据失败: {e}")
        sys.exit(1)


def mode_update_pr(args, api: AtomGitClient):
    """
    模式2: 同步 PR 描述 (Agent 回传模式)

    接收 Agent 生成的高质量内容并同步到服务器。
    """
    print("\n" + "=" * 60)
    print("📝 模式: 同步 Agent 描述到 AtomGit")
    print("=" * 60)

    try:
        # 读取 description.json
        with open(args.update_pr, encoding="utf-8") as f:
            description_data = json.load(f)

        title = description_data.get("title", "")
        description = description_data.get("description", "")

        if not description:
            print("❌ 错误: JSON 中未找到 'description' 字段")
            return

        commits = api.get_pr_commits(args.pr)
        pr = api.get_pull_request(args.pr)
        files = api.get_pr_files(args.pr)
        title = title or pr.get("title") or ""
        validate_commit_ai_model(commits, args.ai_model)
        agent_tool = validate_agent_tool(args.agent_tool)
        reject_handwritten_disclosure(description)
        description = add_ai_disclosure(
            description,
            agent_tool=agent_tool,
            ai_model=args.ai_model,
            prompt_summary=args.prompt_summary,
            third_party_materials=args.third_party_materials,
        )

        verification_inputs = compute_verification_inputs(files)
        gate_required = verification_inputs is not None
        title, gate_triggered = resolve_pr_stage(title, args.pr_stage, gate_required)
        if gate_required and not gate_triggered:
            print("ℹ️  [WIP] PR 跳过双平台 Docker 验证；移除 [WIP] 转为正式检视时门禁会恢复。")
        if gate_triggered:
            current_tree = resolve_pr_head_tree(pr)
            description, verification = prepare_update_verification(
                description,
                pr.get("body") or "",
                verification_inputs,
                current_tree,
            )

        try:
            validate_reuse_self_check(description, files)
        except ValueError as e:
            raise ValueError(f"large-PR reuse self-check failed: {e}") from e

        if not args.human_reviewed and not args.dry_run:
            raise ValueError("更新 PR 前必须由开发者人工审查，并显式传入 --human-reviewed")

        if args.dry_run:
            print("\n⚠️  Dry Run 模式 (预览内容):")
            print(f"标题: {title}")
            print(f"描述正文:\n{description}")
            return

        # 更新 PR
        api.update_pull_request(args.pr, title=title, body=description)
        if gate_triggered:
            updated_pr = api.get_pull_request(args.pr)
            validate_verification_metadata(
                description,
                verification_inputs,
                resolve_pr_head_tree(updated_pr),
                allow_reuse=True,
            )

        print(f"\n✅ PR #{args.pr} 描述已更新")
        print(f"🔗 链接: {api.get_pr_url(args.pr)}")

    except Exception as e:
        print(f"\n❌ 更新失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AtomGit PR 管理工具 (Agent 驱动)")

    parser.add_argument("--pr", type=int, help="PR 编号，可由 --url 自动解析")

    # 模式选择 (互斥)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--fetch-info", action="store_true", help="提取 PR 上下文供 Agent 学习")
    mode_group.add_argument("--update-pr", type=str, metavar="JSON_FILE", help="同步 Agent 生成的描述 (从 JSON 读取)")

    # 可选参数
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="配置文件路径 (默认: config.json)",
    )
    parser.add_argument("--owner", type=str, help="目标仓库 owner，覆盖 config.json")
    parser.add_argument("--repo", type=str, help="目标仓库 repo，覆盖 config.json")
    parser.add_argument(
        "--url",
        type=str,
        help="PR 链接，用于自动解析 owner/repo/PR 编号",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tmp",
        help="上下文输出目录 (默认: ./tmp)",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="在 --fetch-info 模式下跳过抓取 PR 评论",
    )
    parser.add_argument("--agent-tool", help="Coding agent 执行 <tool> --version 后传入实际工具名和版本")
    parser.add_argument("--ai-model", help="PR 使用的 AI 模型名称及版本；多个模型用逗号分隔，不含 provider 前缀")
    parser.add_argument("--prompt-summary", help="核心提示词或核心意图摘要（更新 PR 时必填）")
    parser.add_argument("--third-party-materials", help="第三方材料、来源及许可证（更新 PR 时必填）")
    parser.add_argument("--human-reviewed", action="store_true", help="确认开发者已人工审查 AI 辅助内容")
    parser.add_argument(
        "--pr-stage",
        choices=("wip", "review"),
        help="PR 阶段；命中双 Docker 门禁时必须在询问用户后指定",
    )
    parser.add_argument("--dry-run", action="store_true", help="预览但不提交")

    args = parser.parse_args()

    if args.update_pr:
        required_metadata = {
            "--agent-tool": args.agent_tool,
            "--ai-model": args.ai_model,
            "--prompt-summary": args.prompt_summary,
            "--third-party-materials": args.third_party_materials,
        }
        missing = [name for name, value in required_metadata.items() if not value]
        if missing:
            parser.error(f"更新 PR 时缺少 AI 合规参数: {', '.join(missing)}")

    try:
        sdk_config, parsed_url = resolve_atomgit_context(args.config, owner=args.owner, repo=args.repo, url=args.url)
    except Exception as e:
        print(f"\n❌ 配置错误: {e}")
        sys.exit(1)

    if args.pr is None:
        args.pr = parsed_url.get("pr_number")
    if args.pr is None:
        print("\n❌ 缺少 PR 编号。请通过 --pr 指定，或传入包含 PR 编号的 --url。")
        sys.exit(1)

    api = AtomGitClient(sdk_config)
    print(f"\n🏠 仓库: {sdk_config.owner}/{sdk_config.repo}")
    if args.url:
        print(f"🔗 链接: {args.url}")

    # 执行模式
    if args.fetch_info:
        mode_fetch_info(args, api)
    elif args.update_pr:
        mode_update_pr(args, api)


if __name__ == "__main__":
    main()
