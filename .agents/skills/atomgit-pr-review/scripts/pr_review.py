#!/usr/bin/env python3
"""
AtomGit PR Review Workflow
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
import re
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

import atomgit_cli
import review_helpers
from comment_formatter import CodeIssue, CommentFormatter
from llm_reviewer import LLMCodeReviewer
from reuse_gate import (
    REUSE_FIELD_LABELS,
    REUSE_SELF_CHECK_THRESHOLD,
    count_changed_lines,
    extract_reuse_self_check,
    missing_reuse_fields,
    reuse_gate_required,
    reuse_self_check_status,
)
from verification_gate import (
    VERIFICATION_MODE_REUSED,
    compute_verification_inputs,
    extract_verification_metadata,
    is_wip_title,
    resolve_pr_head_tree,
    validate_verification_metadata,
)

_AGENT_TOOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._/@:+~\- ]*\s+v?\d+(?:\.\d+){1,5}(?:[-+][0-9A-Za-z.-]+)?$")


class CodeReviewer:
    """代码审查器"""

    def __init__(self, cli: str, target: atomgit_cli.RepoTarget, formatter: CommentFormatter):
        self.cli = cli
        self.target = target
        self.formatter = formatter

    @staticmethod
    def _build_mandatory_review_checks(files: list[dict]) -> list[dict]:
        """Flag repository-specific changes that require explicit review."""
        checks = []
        for file_info in files:
            filename = file_info.get("filename") or file_info.get("new_path") or ""
            paths = {
                filename,
                file_info.get("old_path") or "",
                file_info.get("new_path") or "",
            }
            patch = file_info.get("patch") or ""
            if isinstance(patch, dict):
                patch = patch.get("diff") or ""

            modes = {
                str(file_info.get("mode") or ""),
                str(file_info.get("old_mode") or file_info.get("a_mode") or ""),
                str(file_info.get("new_mode") or file_info.get("b_mode") or ""),
            }
            is_lerobot_path = "libs/lerobot" in paths
            is_lerobot_gitlink = is_lerobot_path and (
                "160000" in modes or "Subproject commit " in patch or filename == "libs/lerobot"
            )
            if not is_lerobot_gitlink:
                continue

            commits = re.findall(r"Subproject commit ([0-9a-fA-F]{7,40})", patch)
            checks.append(
                {
                    "id": "lerobot_gitlink_changed",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "libs/lerobot",
                    "detected_commits": commits,
                    "message": (
                        "PR changes the libs/lerobot submodule pointer. Verify that this is an explicit "
                        "upstream base upgrade, that the new commit is fetchable from the .gitmodules "
                        "remote, and that INDEX.yaml, manifest.yaml, series files, and patch-stack tests "
                        "were migrated. Ordinary LeRobot source changes must be exported under "
                        "third_party/patches/lerobot instead of committing a gitlink change."
                    ),
                }
            )
            break
        return checks

    @staticmethod
    def _build_ai_metadata_checks(pr: dict, commits: list[dict]) -> list[dict]:
        """Flag incomplete or inconsistent openEuler AI contribution metadata."""
        body = pr.get("body") or ""
        commit_models = set()
        for commit in commits:
            message = commit.get("commit", {}).get("message", "")
            for line in message.splitlines():
                if not line.strip().startswith("Co-Authored-By:"):
                    continue
                value = line.partition(":")[2].strip()
                if value and not re.search(r"<[^<>]+>\s*$", value):
                    commit_models.add(value)

        ai_declared = "[x] 是" in body.lower() or "[x] yes" in body.lower()
        if not ai_declared and not commit_models:
            return []

        required_patterns = {
            "Agent platform/version": r"Agent平台信息|Agent\s*(?:platform|tool)",
            "model name/version": r"模型信息|\bModel\b",
            "Prompt summary": r"Prompt摘要|Prompt\s*Summary",
            "human review": r"人工审查|human\s*review",
            "third-party materials/licenses": r"第三方材料|third[- ]party\s*materials?",
        }
        missing_fields = [
            name for name, pattern in required_patterns.items() if re.search(pattern, body, re.IGNORECASE) is None
        ]
        checks = []
        if missing_fields:
            checks.append(
                {
                    "id": "ai_disclosure_incomplete",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": "AI participation is indicated, but the PR disclosure is missing: "
                    + ", ".join(missing_fields),
                }
            )

        tool_match = re.search(
            r"(?:Agent平台信息\s*（?Tool）?|Agent\s*(?:platform|tool))\s*[:：]\s*([^\n]+)",
            body,
            re.IGNORECASE,
        )
        if tool_match and _AGENT_TOOL_RE.fullmatch(tool_match.group(1).strip()) is None:
            checks.append(
                {
                    "id": "ai_tool_version_invalid",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": "Agent platform must contain the tool name and version reported by the coding agent.",
                }
            )

        model_match = re.search(
            r"(?:模型信息\s*(?:\(Model\))?|\bModel\b)\s*[:：]\s*([^\n]+)",
            body,
            re.IGNORECASE,
        )
        pr_model_text = model_match.group(1).strip() if model_match else ""
        pr_models = {model.strip() for model in re.split(r"[,，]", pr_model_text) if model.strip()}
        if any("/" in model for model in pr_models | commit_models):
            checks.append(
                {
                    "id": "ai_model_provider_prefix",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description and commit messages",
                    "message": "AI model metadata must contain only the model name and version, without provider prefixes.",
                }
            )
        if not commit_models:
            checks.append(
                {
                    "id": "ai_commit_metadata_missing",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "commit messages",
                    "message": (
                        "The PR discloses AI participation, but no commit contains "
                        "Co-Authored-By: <model name and version>."
                    ),
                }
            )
        elif not pr_models or not commit_models.issubset(pr_models):
            missing_models = commit_models - pr_models
            checks.append(
                {
                    "id": "ai_model_metadata_mismatch",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description and commit messages",
                    "message": (
                        f"PR model disclosure ({pr_model_text or 'missing'}) must include every AI model used by "
                        f"commits; missing: {', '.join(sorted(missing_models))}."
                    ),
                }
            )
        return checks

    @staticmethod
    def _build_reuse_self_check_checks(pr: dict, files: list[dict]) -> list[dict]:
        """Require a complete Reuse Self-Check section on PRs above the line threshold."""
        if not reuse_gate_required(files):
            return []

        body = pr.get("body") or ""
        try:
            fields = extract_reuse_self_check(body)
        except ValueError as exc:
            return [
                {
                    "id": "large_pr_reuse_self_check_invalid",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": str(exc),
                }
            ]
        if fields is None:
            return [
                {
                    "id": "large_pr_reuse_self_check_missing",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": (
                        f"This PR changes more than {REUSE_SELF_CHECK_THRESHOLD} lines and must document a "
                        "'## Reuse Self-Check' section stating whether existing workflows were reinvented, "
                        "what was reused from this repository and libs/lerobot, whether any reinvention is "
                        "justified, and how the change follows the architecture of similar features."
                    ),
                }
            ]
        missing = missing_reuse_fields(fields)
        if missing:
            return [
                {
                    "id": "large_pr_reuse_self_check_incomplete",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": (
                        "The Reuse Self-Check section is incomplete; every field ("
                        + ", ".join(f"'**{label}:**'" for label in REUSE_FIELD_LABELS)
                        + ") must carry a concrete answer. Missing: "
                        + ", ".join(missing)
                        + "."
                    ),
                }
            ]
        return []

    @staticmethod
    def _build_verification_tree_checks(pr: dict, files: list[dict], head_tree: str | None) -> list[dict]:
        """Validate Docker verification evidence against the current PR inputs and tree."""
        if is_wip_title(pr.get("title") or ""):
            return []

        verification_inputs = compute_verification_inputs(files)
        if verification_inputs is None:
            return []

        try:
            metadata = extract_verification_metadata(pr.get("body") or "")
        except ValueError:
            metadata = None

        if metadata is None:
            return [
                {
                    "id": "docker_verification_missing",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": (
                        "This PR requires dual Docker verification, but its description does not contain a valid "
                        "'## Docker Verification' block with mode, Verified inputs, Tested source tree, and "
                        "Docker environment fields. Record the evidence tested by both platforms."
                    ),
                }
            ]

        is_reused = metadata.get("mode") == VERIFICATION_MODE_REUSED
        if not is_reused and head_tree is None:
            return [
                {
                    "id": "docker_verification_missing",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": "PR head tree is unavailable; cannot verify full Docker evidence.",
                }
            ]

        try:
            if is_reused:
                validate_verification_metadata(
                    pr.get("body") or "",
                    verification_inputs,
                    "",
                    allow_reuse=True,
                )
            else:
                validate_verification_metadata(
                    pr.get("body") or "",
                    verification_inputs,
                    head_tree or "",
                )
        except ValueError as exc:
            return [
                {
                    "id": "docker_verification_mismatch",
                    "severity": "error",
                    "blocking_until_reviewed": True,
                    "file": "PR description",
                    "message": str(exc),
                }
            ]
        return []

    def extract_pr_info(self, pr_number: int, include_comments: bool = True) -> dict:
        """提取适合 review 场景的完整 PR 上下文"""
        pr = review_helpers.fetch_pr_record(self.cli, self.target, pr_number)
        files = review_helpers.fetch_pr_files(self.cli, self.target, pr_number)
        commits = review_helpers.fetch_pr_commits(self.cli, self.target, pr_number)
        comments = [] if not include_comments else review_helpers.fetch_pr_comments(self.cli, self.target, pr_number)
        # PR head content lives in the head repo at the complete head SHA
        # (cross-fork gap fix); unprovable metadata marks content unavailable
        # instead of silently reading the base repo at an ambiguous ref.
        head_source, head_incomplete_reason = review_helpers.resolve_head_source(pr)
        head_sha = pr.get("head", {}).get("sha") if isinstance(pr.get("head"), dict) else None
        additions = sum(f.get("additions", 0) for f in files)
        deletions = sum(f.get("deletions", 0) for f in files)
        mandatory_review_checks = self._build_mandatory_review_checks(files)
        verification_inputs = compute_verification_inputs(files)
        head_tree = None
        if verification_inputs and not is_wip_title(pr.get("title") or ""):
            with suppress(ValueError):
                head_tree = resolve_pr_head_tree(pr)
        mandatory_review_checks.extend(self._build_verification_tree_checks(pr, files, head_tree))
        mandatory_review_checks.extend(self._build_ai_metadata_checks(pr, commits))
        mandatory_review_checks.extend(self._build_reuse_self_check_checks(pr, files))

        changed_files = []
        for f in files:
            if f.get("status") != "removed":
                file_data = {
                    "filename": f.get("filename"),
                    "status": f.get("status"),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": f.get("patch"),
                }

                # Content unavailability is recorded structurally (content:
                # None + content_error). Errors are never stringified into a
                # fake file body that could be reviewed as code.
                if head_source is None:
                    file_data["content"] = None
                    file_data["content_error"] = head_incomplete_reason
                else:
                    try:
                        content = review_helpers.fetch_file_content(
                            self.cli, head_source.target, f.get("filename"), head_source.sha
                        )
                        file_data["content"] = review_helpers.add_line_numbers(content)
                    except atomgit_cli.CLIError as error:
                        file_data["content"] = None
                        file_data["content_error"] = str(error)

                changed_files.append(file_data)

        inline_comment_count = sum(1 for comment in comments if comment.get("path") or comment.get("diff_file"))
        # ``resolved_at`` unreadable (key absent) is not evidence of an
        # unresolved thread: only explicit null/negative states count, and the
        # aggregate is marked unknown so consumers can withhold conclusions
        # (matrix B2: resolved-state readability is pending verification).
        unresolved_comment_count = 0
        unresolved_comments_unknown = False
        for comment in comments:
            if "resolved_at" not in comment:
                unresolved_comments_unknown = True
            elif not comment.get("resolved_at"):
                unresolved_comment_count += 1

        return {
            "fetch_time": datetime.now().isoformat(),
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
                "state": pr.get("state"),
                "branch": f"{pr.get('head', {}).get('ref')} → {pr.get('base', {}).get('ref')}",
                "head_sha": head_sha,
                "head_source": {
                    "repo": head_source.repo_slug if head_source else None,
                    "sha": head_source.sha if head_source else None,
                    "incomplete_reason": head_incomplete_reason,
                },
                "head_tree": head_tree,
                "wip": is_wip_title(pr.get("title") or ""),
                "mandatory_review_checks": mandatory_review_checks,
                "reuse_self_check": {
                    "changed_lines": count_changed_lines(files),
                    "required": reuse_gate_required(files),
                    "status": reuse_self_check_status(pr.get("body") or ""),
                },
                "stats": {
                    "files_changed": len(changed_files),
                    "commits": len(commits),
                    "comments": len(comments),
                    "inline_comments": inline_comment_count,
                    "unresolved_comments": unresolved_comment_count,
                    "unresolved_comments_state": "unknown"
                    if unresolved_comments_unknown
                    else "resolved_state_readable",
                    "additions": additions,
                    "deletions": deletions,
                },
                "changed_files": changed_files,
            },
            "commits": [
                {
                    "sha": commit.get("sha", ""),
                    "author": commit.get("commit", {}).get("author", {}).get("name", ""),
                    "message": commit.get("commit", {}).get("message", ""),
                }
                for commit in commits
            ],
            "comments": comments,
        }

    def load_issues_from_json(self, json_path: str) -> list[CodeIssue]:
        """从 JSON 文件加载问题（整批先验校验，review 回补 4）。

        任何一个 finding 无效都在返回前拒绝整批并列出所有坏条目；
        调用方拿不到部分构造的列表，因此任何平台写入之前整批已校验。
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"issues file must contain a JSON array, got {type(data).__name__}")

        issues = []
        errors = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"item {index}: not a JSON object")
                continue
            try:
                issue = CodeIssue(
                    file=item.get("file", ""),
                    line=item.get("line", 0),
                    type=item.get("type", "bug"),
                    severity=item.get("severity", "warning"),
                    confidence=item.get("confidence", 80),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    context_code=item.get("contextCode") or item.get("context_code"),
                    fix_code=item.get("fix", {}).get("code")
                    if isinstance(item.get("fix"), dict)
                    else item.get("fix_code"),
                    fix_explanation=item.get("fix", {}).get("explanation")
                    if isinstance(item.get("fix"), dict)
                    else item.get("fix_explanation"),
                )
            except (ValueError, TypeError) as error:
                errors.append(f"item {index}: {error}")
                continue
            issues.append(issue)

        if errors:
            raise ValueError(
                f"issues file rejected before any platform write ({len(errors)} invalid of {len(data)}): "
                + "; ".join(errors)
            )

        return issues

    def submit_issues(self, pr_number: int, issues: list[CodeIssue]) -> dict:
        """提交问题到 PR"""
        pr = review_helpers.fetch_pr_record(self.cli, self.target, pr_number)
        files = review_helpers.fetch_pr_files(self.cli, self.target, pr_number)
        diffs = {str(f.get("filename") or ""): f for f in files}

        issues = self.formatter.deduplicate(issues)

        positions = {}
        for issue in issues:
            if issue.file not in positions:
                positions[issue.file] = {}

            diff_info = diffs.get(issue.file, {})
            patch = diff_info.get("patch") or ""
            if isinstance(patch, dict):
                patch = patch.get("diff") or ""
            # CLI ``pr comment --position`` takes the new-side file line number
            # directly; only lines visible on the new side of the diff can
            # carry inline comments (review_helpers.diff_new_side_lines).
            if issue.line in review_helpers.diff_new_side_lines(patch):
                positions[issue.file][issue.line] = issue.line

        comments = self.formatter.format_issues(issues, positions)

        summary = self.formatter.format_summary(issues, pr_number, pr.get("title", ""))
        review_helpers.submit_pr_comment(self.cli, self.target, pr_number, summary)
        print("✅ 已提交摘要评论\n")

        if comments:
            results = review_helpers.submit_inline_comments(self.cli, self.target, pr_number, comments)
            success_count = sum(1 for r in results if r["success"])

            print(f"✅ 提交 {success_count}/{len(results)} 条评论\n")

            for result in results:
                if result["success"]:
                    print(f"  ✅ {result['comment']['path']} → {result['comment_url']}")
                else:
                    print(f"  ❌ {result['comment']['path']} - {result['error']}")
        else:
            print("⚠️  没有符合条件的问题需要提交\n")

        return {
            "total_issues": len(issues),
            "submitted_comments": len(comments),
            "summary_submitted": True,
        }


def review_context_path(output_dir: str | Path, repo: str, pr_number: int) -> Path:
    """Path of the extract-info review context for one PR (contracts.md §5.5)."""
    return Path(output_dir) / f"{repo.lower().replace('-', '_')}_pr_{pr_number}_info.json"


def validate_review_context_for_submit(reviewer: "CodeReviewer", pr_number: int, output_dir: str | Path) -> dict:
    """Bind a manual submit to a complete extract-info context and revision.

    Delegates to the shared :func:`review_helpers.validate_extract_context_file`:
    target identity (owner/repo/pr_number), a structurally complete head source
    object with a full 40/64-hex SHA consistent with the recorded head SHA,
    complete per-file head content, and a head SHA matching the live PR head.
    Any mismatch refuses the submission before a single platform write.
    """
    return review_helpers.validate_extract_context_file(
        review_context_path(output_dir, reviewer.target.repo, pr_number),
        cli=reviewer.cli,
        target=reviewer.target,
        pr_number=pr_number,
        what="review",
    )


def mode_extract_info(args, reviewer: CodeReviewer):
    """模式1: 提取 PR 信息（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📥 模式: 提取 PR 信息")
    print("=" * 60)

    pr_info = reviewer.extract_pr_info(args.pr, include_comments=not args.no_comments)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_name = reviewer.target.repo.lower().replace("-", "_")
    output_file = output_dir / f"{repo_name}_pr_{args.pr}_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pr_info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 已保存到: {output_file}")
    print("\n📊 变更摘要:")
    print(f"   标题: {pr_info['pr']['title']}")
    print(f"   作者: {pr_info['pr']['author']}")
    print(f"   分支: {pr_info['pr']['branch']}")
    print(f"   文件: {pr_info['pr']['stats']['files_changed']} 个")
    print(f"   提交: {pr_info['pr']['stats']['commits']} 个")
    print(f"   评论: {pr_info['pr']['stats']['comments']} 条")
    if not args.no_comments:
        print(f"   未解决评论: {pr_info['pr']['stats']['unresolved_comments']} 条")
        if pr_info["pr"]["stats"]["unresolved_comments_state"] == "unknown":
            print("   （resolved_at 字段不可读，未解决数按 unknown 处理，不含状态未知评论）")

    mandatory_checks = pr_info["pr"].get("mandatory_review_checks", [])
    if mandatory_checks:
        print(f"\n强制专项检查: {len(mandatory_checks)} 项，完成前两步后在第 3 步读取详情")

    print("\n评审三步法（默认强制执行，无需用户提醒）:")
    print("  禁止直接读取完整 info.json；前两步不得读取 changed_files、comments 或 PR head 源码")
    print("  1. 仅读取 title、body、关联 Issue 和 commit message，展示问题与目标")
    print("  2. 不看实现，展示自己的独立方案（职责边界、复用点、契约、验证场景）")
    print("  3. 前两步展示后，处理 mandatory_review_checks，再读实现和评论，逐项对比方案")
    print("  完成对比后生成 issues.json；展示具体结果并获得用户确认后才能提交")
    print(f"\n     python3 pr_review.py --pr {args.pr} --submit-review issues.json --ai-model <your-model-name>")


def mode_submit_review(args, reviewer: CodeReviewer):
    """模式2: 提交审查结果（AI Agent 使用）"""
    print("\n" + "=" * 60)
    print("📤 模式: 提交审查结果")
    print("=" * 60)

    print(f"\n📂 从 JSON 加载问题: {args.submit_review}\n")

    issues = reviewer.load_issues_from_json(args.submit_review)
    print(f"📝 加载了 {len(issues)} 个问题\n")

    if args.dry_run:
        print("ℹ️  Dry run 模式：将显示提交计划但不执行\n")
        for issue in issues:
            if issue.confidence >= args.threshold:
                print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        print("")
        return

    # The submit must be bound to a complete reviewed context and the current
    # revision: findings JSON alone never reaches the platform (task 4 gap fix).
    try:
        validate_review_context_for_submit(reviewer, args.pr, args.output_dir)
    except (ValueError, atomgit_cli.CLIError) as error:
        print(f"\n❌ 拒绝提交审查结果: {error}")
        sys.exit(1)

    result = reviewer.submit_issues(args.pr, issues)

    print("\n" + "=" * 60)
    print("✅ 审查完成")
    print("=" * 60 + "\n")
    print("📊 统计:")
    print(f"   总问题数: {result['total_issues']}")
    print(f"   提交评论数: {result['submitted_comments']}")
    print(f"\n🔗 PR 链接: {review_helpers.pr_url(reviewer.target, args.pr)}\n")


def mode_auto(args, cli: str, target: atomgit_cli.RepoTarget, reviewer: CodeReviewer, config: dict) -> bool:
    """模式3: 自动审查（CI 使用，需要 LLM 配置）.

    Returns ``True`` only when the review ran over a complete context and its
    result was published (or previewed, for dry-run). Any head/file read
    failure is accumulated and blocks every automatic publication — including
    the "no issues found" pass summary — because a partial review must never
    be presented as review coverage.
    """
    print("\n" + "=" * 60)
    print("🤖 模式: 自动审查（LLM驱动）")
    print("=" * 60)

    if not config.get("anthropic", {}).get("apiKey"):
        print("\n❌ 自动模式需要配置 Anthropic API Key")
        print("   请在 config.json 中添加:")
        print("   {")
        print('     "anthropic": {')
        print('       "apiKey": "sk-ant-..."')
        print("     }")
        print("   }")
        print("\n或者使用手动模式（AI Agent 调用）:")
        print(f"   python3 pr_review.py --pr {args.pr} --extract-info")
        return False

    pr_info = review_helpers.fetch_pr_record(cli, target, args.pr)
    head_source, head_incomplete_reason = review_helpers.resolve_head_source(pr_info)

    llm_reviewer = LLMCodeReviewer(
        api_key=config["anthropic"]["apiKey"],
        base_url=config["anthropic"].get("baseUrl", ""),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    print("\n📝 获取 PR 文件变更...")
    files = review_helpers.fetch_pr_files(cli, target, args.pr)

    mandatory_checks = reviewer._build_mandatory_review_checks(files)
    all_issues = [
        CodeIssue(
            file="libs/lerobot",
            line=1,
            type="maintainability",
            severity="error",
            confidence=100,
            title="禁止直接提交 LeRobot submodule 指针",
            description=check["message"],
            context_code="libs/lerobot (gitlink mode 160000)",
            fix_code=(
                "Restore libs/lerobot to the base branch gitlink. Export the LeRobot source change "
                "as a mailbox patch under third_party/patches/lerobot, then update the target "
                "series file, manifest.yaml, and scripts/setup/tests/test_lerobot_filter.sh."
            ),
            fix_explanation=(
                "只有明确、可从权威 submodule remote 获取并完成整个 patch stack 迁移的上游基线升级，"
                "才允许修改 gitlink。"
            ),
        )
        for check in mandatory_checks
        if check["id"] == "lerobot_gitlink_changed"
    ]

    incomplete_reasons: list[str] = []
    for i, file_info in enumerate(files, 1):
        file_path = file_info["filename"]

        if not file_path.endswith(".py"):
            continue

        if "test" in file_path.lower():
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
            content = review_helpers.fetch_file_content(cli, head_source.target, file_path, head_source.sha)
            diff = file_info.get("patch", "")

            print("  ⏳ 调用 LLM 进行审查...")
            issues = llm_reviewer.review_file(file_path, content, diff)

            if issues:
                print(f"  ✓ 发现 {len(issues)} 个问题")
                all_issues.extend(issues)
            else:
                print("  ✓ 未发现问题")

        except Exception as e:
            reason = f"{file_path}: 审查失败: {e}"
            incomplete_reasons.append(reason)
            print(f"  ✗ {reason}")

    if incomplete_reasons:
        print("\n" + "=" * 60)
        print(f"❌ 自动审查上下文不完整（{len(incomplete_reasons)} 项），已阻止任何发布（含通过摘要）:")
        for reason in incomplete_reasons:
            print(f"   - {reason}")
        print("=" * 60)
        if args.dry_run:
            print(f"\n（Dry Run）已发现的 {len(all_issues)} 个问题不发布:")
            for issue in all_issues:
                print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        return False

    if args.dry_run:
        print("\n" + "=" * 60)
        print("⚠️  Dry run 模式，未提交评论")
        print("=" * 60)
        print(f"\n发现 {len(all_issues)} 个问题：")
        for issue in all_issues:
            print(f"  - {issue.file}:{issue.line} [{issue.severity}] {issue.title}")
        return True

    if all_issues:
        print(f"\n📦 提交 {len(all_issues)} 个审查结果...")
        result = reviewer.submit_issues(args.pr, all_issues)

        print("\n" + "=" * 60)
        print("✅ 审查完成")
        print("=" * 60)
        print("\n📊 统计:")
        print(f"   审查文件: {len([f for f in files if f['filename'].endswith('.py')])} 个")
        print(f"   发现问题: {result['total_issues']} 个")
        print(f"   提交评论: {result['submitted_comments']} 条")
    else:
        pr = review_helpers.fetch_pr_record(cli, target, args.pr)
        summary = reviewer.formatter.format_summary([], args.pr, pr.get("title", ""))
        review_helpers.submit_pr_comment(cli, target, args.pr, summary)

        print("\n" + "=" * 60)
        print("✅ 审查完成 - 未发现问题")
        print("=" * 60)

    print(f"\n🔗 PR 链接: {review_helpers.pr_url(target, args.pr)}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AtomGit 代码审查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--pr", type=int, help="PR 编号，可由 --url 自动解析")

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

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: config.json；仅作 owner/repo 默认与 --auto 模式 LLM 配置，不再用于认证)",
    )
    parser.add_argument("--owner", type=str, help="目标仓库 owner，覆盖 config.json")
    parser.add_argument("--repo", type=str, help="目标仓库 repo，覆盖 config.json")
    parser.add_argument("--url", type=str, help="PR 链接，用于自动解析 owner/repo/PR 编号")
    parser.add_argument("--output-dir", type=str, default="./tmp", help="输出目录 (默认: ./tmp)")
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="在 --extract-info 模式下跳过抓取现有 PR 评论",
    )
    parser.add_argument("--threshold", type=int, default=80, help="置信度阈值")
    parser.add_argument(
        "--ai-model",
        type=str,
        default="ai",
        help="AI模型名称，用于签名 (默认: ai)",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不提交")

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

    if args.ai_model == "ai":
        print("\n⚠️  警告: 未指定 --ai-model 参数，将使用默认值 'ai'")
        print("   建议指定真实模型名称，例如：")
        print("   --ai-model glm-5.2")
        print("   --ai-model gpt-5.6-sol")
        print("   --ai-model claude-fable-5")
        print()

    print("\n" + "=" * 60)
    print("🔍 AtomGit 代码审查工具")
    print("=" * 60)

    try:
        cli = atomgit_cli.resolve_cli()
        atomgit_cli.check_cli_identity(cli)
        target = atomgit_cli.resolve_target(
            owner=args.owner,
            repo=args.repo,
            url=args.url,
            config_path=args.config or "config.json",
            explicit_config=args.config is not None,
            remotes=atomgit_cli.read_git_remotes(),
        )
    except (atomgit_cli.CLIError, atomgit_cli.TargetResolutionError) as e:
        print(f"\n❌ 初始化失败: {e}")
        sys.exit(1)

    if args.pr is None:
        args.pr = (target.url_info or {}).get("pr_number")
    if args.pr is None:
        print("\n❌ 缺少 PR 编号。请通过 --pr 指定，或传入包含 PR 编号的 --url。")
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

    formatter = CommentFormatter(confidence_threshold=args.threshold, ai_model=args.ai_model)
    reviewer = CodeReviewer(cli, target, formatter)

    print(f"\n📋 PR: #{args.pr}")
    print(f"🏠 仓库: {target.slug}")
    if args.url:
        print(f"🔗 链接: {args.url}")
    print(f"🤖 AI模型: {args.ai_model}")

    if args.auto:
        print(f"🧠 LLM模型: {args.llm_model} (provider: {args.llm_provider})")
        print("📦 模式: 自动（CI 模式，Skill 内部调用 LLM）")
    elif args.extract_info:
        print("📥 模式: 提取信息（AI Agent 模式）")
    elif args.submit_review:
        print("📤 模式: 提交审查（AI Agent 模式）")
    else:
        args.extract_info = True
        print("📥 模式: 提取信息（默认，AI Agent 模式）")

    if args.dry_run:
        print("⚠️  Dry Run 模式（仅显示计划）")

    if args.extract_info:
        mode_extract_info(args, reviewer)
    elif args.submit_review:
        mode_submit_review(args, reviewer)
    elif args.auto and not mode_auto(args, cli, target, reviewer, config):
        sys.exit(1)


if __name__ == "__main__":
    main()
