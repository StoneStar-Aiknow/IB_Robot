#!/usr/bin/env python3
"""
AtomGit Architecture Review Workflow
支持三种模式：
1. --extract-info: 提取 PR 信息（输出JSON）- AI Agent 使用
2. --submit-review: 提交审查结果（从JSON读取）- AI Agent 使用
3. --auto: 自动审查（调用LLM）- CI 使用，需要配置 LLM

Platform access goes through the official AtomGit CLI via the shared boundary
in ``.agents/skills/_shared/atomgit_cli.py`` (OpenSpec change
``migrate-atomgit-skills-to-cli``); review-specific argv building lives in
``.agents/skills/_shared/review_helpers.py``. Authentication is owned by the
CLI: ``gitcode auth login --hostname atomgit.com --with-token``.
"""

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

import atomgit_cli
import review_helpers
from comment_formatter import ArchitectureIssue, CommentFormatter
from llm_architecture_reviewer import LLMArchitectureReviewer


class ArchitectureReviewer:
    """架构审查器"""

    def __init__(self, cli: str, target: atomgit_cli.RepoTarget, formatter: CommentFormatter):
        self.cli = cli
        self.target = target
        self.formatter = formatter

    def _candidate_readme_paths(self, file_path: str) -> list[str]:
        """Return nearest README candidates for a repository path."""
        path = PurePosixPath(file_path)
        candidates = []
        for parent in path.parents:
            parent_text = str(parent)
            if parent_text in (".", ""):
                continue
            candidate = f"{parent_text}/README.md"
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def get_readme_context(
        self,
        file_path: str,
        head_source: review_helpers.HeadSource | None,
        head_incomplete_reason: str | None = None,
    ) -> dict:
        """Fetch nearest package README context for a changed file.

        Failures are recorded structurally (``content`` empty + ``errors``
        list); they never become fake README text that could be reviewed as
        documentation. Unprovable head sources make no platform call at all.
        """
        candidates = self._candidate_readme_paths(file_path)
        if head_source is None:
            return {
                "path": None,
                "content": "",
                "candidates": candidates,
                "errors": [head_incomplete_reason or "PR head content source is unprovable"],
            }
        errors = []
        for candidate in candidates:
            try:
                content = review_helpers.fetch_file_content(self.cli, head_source.target, candidate, head_source.sha)
                return {
                    "path": candidate,
                    "content": review_helpers.add_line_numbers(content),
                    "candidates": candidates,
                }
            except atomgit_cli.CLIError as e:
                errors.append(f"{candidate}: {e}")

        return {
            "path": None,
            "content": "",
            "candidates": candidates,
            "errors": errors,
        }

    def extract_pr_info(self, pr_number: int) -> dict:
        """提取 PR 信息"""
        pr = review_helpers.fetch_pr_record(self.cli, self.target, pr_number)
        files = review_helpers.fetch_pr_files(self.cli, self.target, pr_number)
        # PR head content lives in the head repo at the complete head SHA
        # (cross-fork gap fix); unprovable metadata marks content unavailable
        # instead of silently reading the base repo at an ambiguous ref.
        head_source, head_incomplete_reason = review_helpers.resolve_head_source(pr)
        head_sha = pr.get("head", {}).get("sha") if isinstance(pr.get("head"), dict) else None

        changed_files = []
        for f in files:
            filename = f.get("filename") or ""
            is_arch_relevant = filename.endswith(".py") or filename.endswith("README.md")
            if f.get("status") != "removed" and is_arch_relevant:
                file_data = {
                    "filename": filename,
                    "status": f.get("status"),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch"),
                }

                # Content unavailability is recorded structurally (content:
                # None + content_error); errors are never stringified into a
                # fake file body that could be reviewed as code.
                if head_source is None:
                    file_data["content"] = None
                    file_data["content_error"] = head_incomplete_reason
                else:
                    try:
                        content = review_helpers.fetch_file_content(
                            self.cli, head_source.target, filename, head_source.sha
                        )
                        file_data["content"] = review_helpers.add_line_numbers(content)
                    except atomgit_cli.CLIError as e:
                        file_data["content"] = None
                        file_data["content_error"] = str(e)

                file_data["readme_context"] = self.get_readme_context(filename, head_source, head_incomplete_reason)
                changed_files.append(file_data)

        return {
            # Explicit target identity so a submit can prove the context was
            # extracted for exactly this owner/repo/PR (same-named forks at
            # the same head SHA are not interchangeable).
            "target": {
                "owner": self.target.owner,
                "repo": self.target.repo,
                "pr_number": pr_number,
            },
            "pr": {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "body": pr.get("body") or "",
                "author": pr.get("user", {}).get("login"),
                "branch": f"{pr.get('head', {}).get('ref')} → {pr.get('base', {}).get('ref')}",
                "head_sha": head_sha,
                "head_source": {
                    "repo": head_source.repo_slug if head_source else None,
                    "sha": head_source.sha if head_source else None,
                    "incomplete_reason": head_incomplete_reason,
                },
                "changed_files": changed_files,
            },
        }

    def load_issues_from_json(self, json_path: str) -> list[ArchitectureIssue]:
        """从 JSON 文件加载问题（整批先验校验，review 回补 4）。

        任何一个 finding 无效都在返回前拒绝整批并列出所有坏条目；
        调用方拿不到部分构造的列表，因此任何平台写入之前整批已校验。
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"arch-issues file must contain a JSON array, got {type(data).__name__}")

        issues = []
        errors = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"item {index}: not a JSON object")
                continue
            try:
                issue = ArchitectureIssue(
                    file=item.get("file", ""),
                    line=item.get("line", 0),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    severity=item.get("severity", "warning"),
                    pillar=item.get("pillar", "python"),
                    fix=item.get("fix"),
                    context_code=item.get("context_code"),
                )
            except (ValueError, TypeError) as error:
                errors.append(f"item {index}: {error}")
                continue
            issues.append(issue)

        if errors:
            raise ValueError(
                f"arch-issues file rejected before any platform write ({len(errors)} invalid of {len(data)}): "
                + "; ".join(errors)
            )

        return issues

    def submit_review(self, pr_number: int, issues: list[ArchitectureIssue]) -> None:
        """提交审查结果"""
        if not issues:
            summary = self.formatter.format_summary(issues)
            review_helpers.submit_pr_comment(self.cli, self.target, pr_number, summary)
            print(f"✅ 提交架构审查通过评论到 PR #{pr_number}")
            return

        pr_files = review_helpers.fetch_pr_files(self.cli, self.target, pr_number)
        diffs = {str(f.get("filename") or ""): f for f in pr_files}
        changed_filenames = set(diffs)

        inline_issues = []
        pr_level_issues = []
        for issue in issues:
            if issue.file in changed_filenames:
                inline_issues.append(issue)
            else:
                pr_level_issues.append(issue)

        if inline_issues:
            # 与 pr_review 相同的 CLI 定位语义：``pr comment --position`` 接受
            # new 侧文件行号，仅 diff new 侧可见的行可携带行内评论；不可见行
            # 由 submit_inline_comments 逐条显式报告失败（无平台调用）。
            positions: dict[str, dict[int, int]] = {}
            for issue in inline_issues:
                patch = diffs.get(issue.file, {}).get("patch") or ""
                if isinstance(patch, dict):
                    patch = patch.get("diff") or ""
                if issue.line in review_helpers.diff_new_side_lines(patch):
                    positions.setdefault(issue.file, {})[issue.line] = issue.line
            comments = self.formatter.format_issues(inline_issues, positions)
            results = review_helpers.submit_inline_comments(self.cli, self.target, pr_number, comments)

            success_count = sum(1 for r in results if r["success"])
            print(f"✅ 提交 {success_count}/{len(results)} 条行内评论到 PR #{pr_number}")

            for result in results:
                if not result["success"]:
                    print(f"  ❌ 失败: {result['comment']['path']} - {result['error']}")

        if pr_level_issues:
            body = self.formatter.format_pr_level_issues(pr_level_issues)
            try:
                review_helpers.submit_pr_comment(self.cli, self.target, pr_number, body)
                print(f"✅ 提交 {len(pr_level_issues)} 条 PR 评论到 PR #{pr_number}（非变更文件，降级为 PR 级评论）")
            except Exception as e:
                print(f"  ❌ PR 评论提交失败: {e}")


def arch_context_path(output_dir: str | Path, repo: str, pr_number: int) -> Path:
    """Path of the extract-info architecture context for one PR (contracts.md §5.6)."""
    return Path(output_dir) / f"{repo.lower().replace('-', '_')}_pr_{pr_number}_arch_info.json"


def validate_arch_context_for_submit(reviewer: "ArchitectureReviewer", pr_number: int, output_dir: str | Path) -> dict:
    """Bind a manual submit to a complete extract-info context and revision.

    Delegates to the shared :func:`review_helpers.validate_extract_context_file`:
    target identity (owner/repo/pr_number), a structurally complete head source
    object with a full 40/64-hex SHA consistent with the recorded head SHA,
    complete per-file head content, and a head SHA matching the live PR head.
    Any mismatch refuses the submission before a single platform write.
    """
    return review_helpers.validate_extract_context_file(
        arch_context_path(output_dir, reviewer.target.repo, pr_number),
        cli=reviewer.cli,
        target=reviewer.target,
        pr_number=pr_number,
        what="architecture review",
    )


def mode_extract_info(args, reviewer: ArchitectureReviewer):
    """模式1: 提取 PR 信息（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📥 模式: 提取 PR 信息")
    print("=" * 60)

    pr_info = reviewer.extract_pr_info(args.pr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从配置中获取仓库名称
    repo_name = reviewer.target.repo.lower().replace("-", "_")
    output_file = output_dir / f"{repo_name}_pr_{args.pr}_arch_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pr_info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 已保存到: {output_file}")
    print("\n📊 变更摘要:")
    print(f"   标题: {pr_info['pr']['title']}")
    print(f"   作者: {pr_info['pr']['author']}")
    print(f"   分支: {pr_info['pr']['branch']}")
    print(f"   架构相关文件: {len(pr_info['pr']['changed_files'])} 个")

    print("\n💡 下一步:")
    print("  AI Agent 应该:")
    print("  1. 读取此文件并进行架构审查")
    print("  2. 检查是否符合 IB_Robot 架构四大支柱")
    print("  3. 对比每个 changed_files[].readme_context，检查 README 是否与代码同步")
    print("  4. 生成 arch-issues.json（包含所有架构问题和建议）")
    print("  5. ⚠️ 将审查结果以用户可读的格式展示给用户确认")
    print("  6. 用户确认后，运行提交命令")
    print(
        f"\n     python3 architecture_review.py --pr {args.pr} --submit-review arch-issues.json --ai-model <your-model-name>"
    )


def mode_submit_review(args, reviewer: ArchitectureReviewer):
    """模式2: 提交审查结果（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📤 模式: 提交审查结果")
    print("=" * 60)

    print(f"\n📂 从 JSON 加载问题: {args.submit_review}\n")

    issues = reviewer.load_issues_from_json(args.submit_review)
    print(f"📝 加载了 {len(issues)} 个架构问题\n")

    if args.dry_run:
        print("ℹ️  Dry run 模式：将显示提交计划但不执行\n")
        for issue in issues:
            print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        print("")
        return

    # The submit must be bound to a complete reviewed context and the current
    # revision: findings JSON alone never reaches the platform (task 4 gap fix).
    try:
        validate_arch_context_for_submit(reviewer, args.pr, args.output_dir)
    except (ValueError, atomgit_cli.CLIError) as error:
        print(f"\n❌ 拒绝提交审查结果: {error}")
        sys.exit(1)

    reviewer.submit_review(args.pr, issues)

    print("\n" + "=" * 60)
    print("✅ 审查完成")
    print("=" * 60 + "\n")
    print("📊 统计:")
    print(f"   总问题数: {len(issues)}")
    print(f"\n🔗 PR 链接: {review_helpers.pr_url(reviewer.target, args.pr)}\n")


def mode_auto(args, cli: str, target: atomgit_cli.RepoTarget, reviewer: ArchitectureReviewer, config: dict) -> bool:
    """模式3: 自动审查（CI 使用，需要 LLM 配置）.

    Returns ``True`` only when the review ran over a complete context and its
    result was published (or previewed, for dry-run). Any head/file read
    failure is accumulated and blocks every automatic publication — including
    the architecture pass summary — because a partial review must never be
    presented as review coverage.
    """
    print("\n" + "=" * 60)
    print("🤖 模式: 自动审查（LLM驱动）")
    print("=" * 60)

    # 检查 LLM 配置
    if not config.get("anthropic", {}).get("apiKey"):
        print("\n❌ 自动模式需要配置 Anthropic API Key")
        print("   请在 config.json 中添加:")
        print("   {")
        print('     "anthropic": {')
        print('       "apiKey": "sk-ant-..."')
        print("     }")
        print("   }")
        print("\n或者使用手动模式（AI Agent 调用）:")
        print(f"   python3 architecture_review.py --pr {args.pr} --extract-info")
        return False

    # 创建 LLM 审查器
    llm_reviewer = LLMArchitectureReviewer(
        api_key=config["anthropic"]["apiKey"],
        base_url=config["anthropic"].get("baseUrl", ""),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    # 获取 PR 信息以证明 head 内容来源（跨 fork 修复：head.repo + 完整 head SHA）
    pr_info = review_helpers.fetch_pr_record(cli, target, args.pr)
    head_source, head_incomplete_reason = review_helpers.resolve_head_source(pr_info)

    print("\n📝 获取 PR 文件变更...")
    files = review_helpers.fetch_pr_files(cli, target, args.pr)

    all_issues = []
    incomplete_reasons: list[str] = []

    for i, file_info in enumerate(files, 1):
        file_path = file_info["filename"]

        is_readme = file_path.endswith("README.md")
        is_python = file_path.endswith(".py")

        # 只审查 Python 文件和 README
        if not (is_python or is_readme):
            continue

        # 跳过测试文件
        if is_python and "test" in file_path.lower():
            continue

        # 跳过 __init__.py
        if is_python and file_path.endswith("__init__.py"):
            continue

        print(f"\n[{i}/{len(files)}] 审查 {file_path}")

        if head_source is None:
            # 没有可证明的 head 内容来源时该文件无法审查（不用 base 仓库的
            # 猜测 ref 冒充 PR head 内容，跨 fork 修复）；累计不完整原因，
            # 任何自动发布前都会被阻止。
            reason = f"{file_path}: PR head 内容来源无法证明 ({head_incomplete_reason})"
            incomplete_reasons.append(reason)
            print(f"  ⚠️  跳过：{reason}")
            continue

        try:
            # 获取文件内容（head.repo @ 完整 head SHA）
            content = review_helpers.fetch_file_content(cli, head_source.target, file_path, head_source.sha)
            diff = file_info.get("patch", "")
            readme_context = reviewer.get_readme_context(file_path, head_source, head_incomplete_reason)

            # 调用 LLM 审查
            print("  ⏳ 调用 LLM 进行架构审查...")
            issues = llm_reviewer.review_file(
                file_path,
                content,
                diff,
                readme_context=readme_context,
            )

            if issues:
                print(f"  ✓ 发现 {len(issues)} 个架构问题")
                all_issues.extend(issues)
            else:
                print("  ✓ 未发现架构问题")

        except Exception as e:
            reason = f"{file_path}: 架构审查失败: {e}"
            incomplete_reasons.append(reason)
            print(f"  ✗ {reason}")

    if incomplete_reasons:
        print("\n" + "=" * 60)
        print(f"❌ 自动架构审查上下文不完整（{len(incomplete_reasons)} 项），已阻止任何发布（含通过摘要）:")
        for reason in incomplete_reasons:
            print(f"   - {reason}")
        print("=" * 60)
        if args.dry_run:
            print(f"\n（Dry Run）已发现的 {len(all_issues)} 个架构问题不发布:")
            for issue in all_issues:
                print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        return False

    if args.dry_run:
        print("\n" + "=" * 60)
        print("⚠️  Dry run 模式，未提交评论")
        print("=" * 60)
        print(f"\n发现 {len(all_issues)} 个架构问题：")
        for issue in all_issues:
            print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        return True

    # 提交审查结果
    if all_issues:
        print(f"\n📦 提交 {len(all_issues)} 个架构审查结果...")
        reviewer.submit_review(args.pr, all_issues)

        print("\n" + "=" * 60)
        print("✅ 审查完成")
        print("=" * 60)
        print("\n📊 统计:")
        print(
            f"   审查文件: {len([f for f in files if f['filename'].endswith('.py') and 'test' not in f['filename'].lower()])} 个"
        )
        print(f"   发现问题: {len(all_issues)} 个")
    else:
        # 提交通过评论
        print("\n📦 提交架构审查通过评论...")
        reviewer.submit_review(args.pr, [])

        print("\n" + "=" * 60)
        print("✅ 审查完成 - 未发现架构问题")
        print("=" * 60)

    print(f"\n🔗 PR 链接: {review_helpers.pr_url(target, args.pr)}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AtomGit 架构审查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必需参数
    parser.add_argument("--pr", type=int, required=True, help="PR 编号")

    # 模式选择（互斥）
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--extract-info",
        action="store_true",
        help="模式1: 提取 PR 信息（AI Agent 使用）",
    )
    mode_group.add_argument(
        "--submit-review",
        type=str,
        metavar="JSON_FILE",
        help="模式2: 提交审查结果（AI Agent 使用）",
    )
    mode_group.add_argument("--auto", action="store_true", help="模式3: 自动审查（CI 使用，需要 LLM 配置）")

    # 通用参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: config.json；仅作 owner/repo 默认与 --auto 模式 LLM 配置，不再用于认证)",
    )
    parser.add_argument("--output-dir", type=str, default="./tmp", help="输出目录 (默认: ./tmp)")
    parser.add_argument(
        "--ai-model",
        type=str,
        default="ai",
        help="AI模型名称，用于签名 (默认: ai)",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不提交")

    # 自动模式专用参数（LLM 配置）
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="anthropic",
        help="LLM 提供商（仅 --auto 模式，默认: anthropic）",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="LLM 模型名称（仅 --auto 模式，默认: claude-sonnet-4-20250514）",
    )

    args = parser.parse_args()

    # 警告：如果使用默认的 ai-model，提醒指定真实模型名称
    if args.ai_model == "ai":
        print("\n⚠️  警告: 未指定 --ai-model 参数，将使用默认值 'ai'")
        print("   建议指定真实模型名称，例如：")
        print("   --ai-model glm-5.2")
        print("   --ai-model gpt-5.6-sol")
        print("   --ai-model claude-fable-5")
        print()

    print("\n" + "=" * 60)
    print("🔍 AtomGit 架构审查工具")
    print("=" * 60)

    # 初始化 CLI 边界与目标仓库
    try:
        cli = atomgit_cli.resolve_cli()
        atomgit_cli.check_cli_identity(cli)
        target = atomgit_cli.resolve_target(
            config_path=args.config or "config.json",
            explicit_config=args.config is not None,
            remotes=atomgit_cli.read_git_remotes(),
        )
    except (atomgit_cli.CLIError, atomgit_cli.TargetResolutionError) as e:
        print(f"\n❌ 初始化失败: {e}")
        sys.exit(1)

    # config.json 只为 --auto 模式提供 LLM 配置；owner/repo 默认已由
    # resolve_target 读取，认证由 CLI 自己管理（design.md D2）。
    config = {}
    if args.auto:
        try:
            with open(args.config or "config.json", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            pass  # mode_auto 会给出缺少 Anthropic API Key 的指引
        except Exception as e:
            print(f"\n❌ 加载配置失败: {e}")
            sys.exit(1)

    formatter = CommentFormatter(ai_model=args.ai_model)
    reviewer = ArchitectureReviewer(cli, target, formatter)

    print(f"\n📋 PR: #{args.pr}")
    print(f"🏠 仓库: {target.slug}")
    print(f"🤖 AI模型: {args.ai_model}")

    # 检查模式
    if args.auto:
        print(f"🧠 LLM模型: {args.llm_model} (provider: {args.llm_provider})")
        print("📦 模式: 自动（CI 模式，Skill 内部调用 LLM）")
    elif args.extract_info:
        print("📥 模式: 提取信息（AI Agent 模式）")
    elif args.submit_review:
        print("📤 模式: 提交审查（AI Agent 模式）")
    else:
        # 默认：提取信息模式（AI Agent 友好）
        args.extract_info = True
        print("📥 模式: 提取信息（默认，AI Agent 模式）")

    if args.dry_run:
        print("⚠️  Dry Run 模式（仅显示计划）")

    # 根据模式执行
    if args.extract_info:
        mode_extract_info(args, reviewer)
    elif args.submit_review:
        mode_submit_review(args, reviewer)
    elif args.auto and not mode_auto(args, cli, target, reviewer, config):
        sys.exit(1)


if __name__ == "__main__":
    main()
