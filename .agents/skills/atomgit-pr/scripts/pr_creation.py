#!/usr/bin/env python3
"""
AtomGit Pull Request Creation Tool

自动创建 PR，从当前分支生成 PR 标题和描述

用法：
    python3 pr_creation.py --branch <branch> --fork-owner <owner> --description-file <file> [--title "标题"] [--dry-run]

Platform access goes through the official AtomGit CLI via the shared boundary
in ``.agents/skills/_shared/atomgit_cli.py`` (OpenSpec change
``migrate-atomgit-skills-to-cli``). Authentication is owned by the CLI:
``gitcode auth login --hostname atomgit.com --with-token``; config.json only
supplies owner/repo defaults.

Cross-fork creation maps the SDK head ``fork_owner:branch`` to the CLI form
``--fork fork_owner/repo --head branch`` (gc_pr_create.go resolveHead).

Fingerprint policy (reviews/round-3.md R3-01 option (b), design.md D3): fresh
verification evidence uses the v2 fingerprint; a complete legacy v1 record is
recognized by recomputation (``legacy_inputs``) and may be reused as v1
evidence without rewriting its hash. Inputs whose patches are missing never
support fingerprint-based publication.

Uncertain write outcomes (design.md D5): when the create times out, open PRs
are reconciled by full source/target/head/base identity plus body equality —
the same branch alone never proves association — and any doubt is reported as
uncertain for manual reconciliation instead of re-creating. Once a PR number
is known, later failures are reported as "created but post-write validation
pending", never as "not created".
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))

import atomgit_cli
from ai_compliance import (
    add_ai_disclosure,
    reject_handwritten_disclosure,
    validate_agent_tool,
    validate_commit_ai_model,
)
from pr_management import fetch_pr_record, pr_url
from reuse_gate import REUSE_SELF_CHECK_THRESHOLD, count_changed_lines, reuse_gate_required, validate_reuse_self_check
from verification_gate import (
    compute_legacy_verification_inputs,
    compute_verification_inputs,
    resolve_pr_stage,
    validate_verification_metadata,
    verification_inputs_placeholder_tainted,
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


def get_changed_file_infos(branch: str, base_branch: str = "master") -> list[dict]:
    """Return changed paths with zero-context patches for input fingerprinting."""
    base_ref = get_best_base_ref(base_branch)
    files = get_changed_files(branch, base_branch)
    infos = []
    for filename in files:
        try:
            patch = run_git(["diff", "--unified=0", f"{base_ref}...{branch}", "--", filename])
        except Exception:
            patch = run_git(["diff", "--unified=0", f"{base_branch}...{branch}", "--", filename])
        infos.append({"filename": filename, "patch": patch})
    return infos


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


def _head_repo_slug(head: dict) -> str | None:
    """Full "owner/repo" identity of the head repository, or None.

    Task 3.2 review: matching only the owner is not enough — a different
    fork repository under the same owner with the same branch and body
    would be mistaken for this write. Any field shape that cannot prove
    the complete identity returns None, so the caller keeps the outcome
    uncertain instead of guessing.
    """
    repo = head.get("repo")
    if not isinstance(repo, dict):
        repo = {}
    full_name = repo.get("full_name") or ""
    if "/" in full_name:
        return full_name
    owner = (repo.get("owner") or {}).get("login") or ""
    name = repo.get("name") or ""
    if owner and name:
        return f"{owner}/{name}"
    return None


def _print_reconcile_context(
    target: atomgit_cli.RepoTarget, source: atomgit_cli.RepoTarget, branch: str, base: str
) -> None:
    print(
        "   保留以下上下文供人工核对（禁止盲目重试或重建 PR）：\n"
        f"   - 目标仓库: {target.slug}\n"
        f"   - 源 fork: {source.slug}（head 分支 {branch} → base {base}）\n"
        f"   - 手工对账: gitcode pr list -R {target.slug} --state open --json"
    )


def reconcile_created_pr(
    cli: str,
    target: atomgit_cli.RepoTarget,
    source: atomgit_cli.RepoTarget,
    branch: str,
    base: str,
    description: str,
) -> int:
    """D5: prove or refute an accepted create whose outcome is unknown.

    Lists every open PR (``pr list --json --paginate``, server-side paging)
    and keeps only entries whose head branch/ref and base match, whose head
    repository is provably the exact source fork (full owner/repo identity:
    same-owner different-repo entries are excluded, and entries whose
    identity cannot be read keep the outcome uncertain instead of being
    silently skipped), and whose stored body equals the description this
    run sent. Branch identity alone never proves association — an older PR
    from the same branch must not be mistaken for this write. Any doubt
    (failed query, empty result with delayed visibility, unverifiable
    candidates, multiple matches, unreadable fields) is reported as
    ``uncertain`` for manual reconciliation; the caller never re-creates.
    """
    print(">>> 对账：确认平台是否已接受本次创建（只读查询，不会重建）...")
    try:
        result = atomgit_cli.run_json(
            ["pr", "list", "-R", target.slug, "--state", "open", "--json", "--paginate", "--per-page", "100"],
            cli=cli,
        )
        prs = result.require_data()
        if not isinstance(prs, list):
            raise atomgit_cli.CLIJSONError(
                f"pr list returned {type(prs).__name__} instead of a JSON array",
                argv=result.argv,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
    except atomgit_cli.CLIError as e:
        print(f"❌ 对账结果 uncertain：查询失败，无法确认创建是否已被平台接受。\n{e}")
        _print_reconcile_context(target, source, branch, base)
        sys.exit(1)

    matches = []
    unverifiable = None
    for pr in prs:
        if not isinstance(pr, dict):
            unverifiable = "列表含非对象条目"
            continue
        head = pr.get("head")
        base_entry = pr.get("base")
        if not isinstance(head, dict) or not isinstance(base_entry, dict):
            unverifiable = f"PR #{pr.get('number') or '?'} 缺 head/base 结构"
            continue
        if head.get("ref") != branch:
            continue
        if base_entry.get("ref") != base:
            continue
        # head/base 同名后必须能证明候选不是本写入：完整 owner/repo 身份
        # 缺失时无法排除，保持 uncertain 而不是静默跳过（task 3.2 review）。
        head_slug = _head_repo_slug(head)
        if head_slug is None:
            unverifiable = f"PR #{pr.get('number') or '?'} 的 head 仓库缺少完整 owner/repo 身份"
            continue
        if head_slug != source.slug:
            if head_slug.lower() == source.slug.lower():
                # design.md D2：大小写差异无法证明同一仓库，也不猜测折叠。
                unverifiable = (
                    f"PR #{pr.get('number') or '?'} 的 head 仓库身份与源仓库仅大小写不一致"
                    f"（{head_slug} != {source.slug}），无法证明是否同一仓库"
                )
            continue
        if (pr.get("body") or "").strip() != description.strip():
            continue
        matches.append(pr)

    if unverifiable is not None:
        print(f"❌ 对账结果 uncertain：open PR 列表存在无法核实身份的候选（{unverifiable}），不能排除其为本次写入。")
        _print_reconcile_context(target, source, branch, base)
        sys.exit(1)
    if not matches:
        print("❌ 对账结果 uncertain：未找到可证明为本次写入的 open PR（可能延迟可见或未被接受）。")
        _print_reconcile_context(target, source, branch, base)
        sys.exit(1)
    if len(matches) > 1:
        print(f"❌ 对账结果 uncertain：{len(matches)} 个 open PR 与本次写入同源同正文，无法唯一确认。")
        _print_reconcile_context(target, source, branch, base)
        sys.exit(1)
    number = matches[0].get("number")
    if number is None:
        print("❌ 对账结果 uncertain：匹配的 PR 记录缺少编号。")
        _print_reconcile_context(target, source, branch, base)
        sys.exit(1)
    print(f"✅ 对账确认：open PR #{number} 的 head/base 与正文均与本次写入一致，视为本次创建结果（未重复创建）。")
    return int(number)


def create_pr_or_reconcile(
    cli: str,
    target: atomgit_cli.RepoTarget,
    source: atomgit_cli.RepoTarget,
    branch: str,
    base: str,
    description: str,
    create_argv: list[str],
) -> int:
    """Create the PR; on an uncertain outcome reconcile instead of re-creating."""
    try:
        result = atomgit_cli.run_json(create_argv, cli=cli, stdin_text=description)
        pr = result.require_data()
        if not isinstance(pr, dict) or pr.get("number") is None:
            # rc=0 means the mutation was very likely applied; never auto-retry.
            print("❌ pr create 未返回 PR 编号，PR 可能已创建；请先用 pr list 对账确认后再决定，不要盲目重试。")
            sys.exit(1)
        return int(pr["number"])
    except atomgit_cli.CLITimeoutError as e:
        print(f"\n⚠️  创建 PR 请求超时，平台写入结果未知: {e}")
        return reconcile_created_pr(cli, target, source, branch, base, description)
    except atomgit_cli.CLIConflictError as e:
        print(f"❌ 创建 PR 失败（平台状态冲突；同 head/base 的 open PR 可能已存在）: {e}")
        print("   请先只读对账 open PR 列表，不要重复创建。")
        sys.exit(1)
    except atomgit_cli.CLIError as e:
        print(f"❌ 创建 PR 失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AtomGit PR 创建工具")
    parser.add_argument("--branch", type=str, help="源分支名")
    parser.add_argument("--base", type=str, default="master", help="目标分支名")
    parser.add_argument("--title", type=str, help="PR 标题")
    parser.add_argument("--description-file", type=str, required=True, help="从文件读取 PR 描述 (Markdown 格式)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: config.json；仅作 owner/repo 默认，不再用于认证)",
    )
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
        help="Coding agent 执行 <tool> --version 确认版本后传入产品全称和版本（不用 CLI 命令名）",
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
        source = atomgit_cli.resolve_source(args.fork_owner, target)
    except (atomgit_cli.CLIError, atomgit_cli.TargetResolutionError) as e:
        print(f"\n❌ 初始化失败: {e}")
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
    print(f"目标仓库: {target.slug}")
    print(f"Fork owner: {source.owner}")
    if args.url:
        print(f"解析链接: {args.url}")
        url_info = target.url_info or {}
        if url_info.get("pr_number") is not None:
            print("ℹ️  已忽略链接中的 PR 编号，仅使用其中的仓库信息创建 PR")
        if url_info.get("issue_number") is not None:
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
        reject_handwritten_disclosure(description)
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
    file_infos = get_changed_file_infos(branch, args.base)
    verification_inputs = compute_verification_inputs(file_infos)
    legacy_inputs = compute_legacy_verification_inputs(file_infos)
    gate_required = verification_inputs is not None
    try:
        title, gate_triggered = resolve_pr_stage(title, args.pr_stage, gate_required)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    if gate_required and not gate_triggered:
        print("ℹ️  [WIP] PR 跳过双平台 Docker 验证；移除 [WIP] 转为正式检视时门禁会恢复。")
    try:
        validate_reuse_self_check(description, file_infos)
    except ValueError as e:
        print(f"❌ 大型 PR 复用自查未通过: {e}")
        print(
            "   描述中需补充如下章节（标签固定，正文默认中文，逐项给出具体答案）：\n"
            "   ## Reuse Self-Check\n"
            "   **Reinvented workflows:** ...\n"
            "   **Reused components:** ...\n"
            "   **Reinvention justification:** ...\n"
            "   **Architecture conformance:** ..."
        )
        sys.exit(1)
    if reuse_gate_required(file_infos):
        print(
            f"ℹ️  变更 {count_changed_lines(file_infos)} 行（> {REUSE_SELF_CHECK_THRESHOLD}）："
            "已包含 ## Reuse Self-Check 复用自查章节。"
        )
    remote_head = None
    verified_tree = None
    if gate_triggered:
        # R3-01 option (b) / design.md D3: fingerprints over incomplete
        # (placeholder) inputs never support publication; stop before any write.
        if verification_inputs_placeholder_tainted(file_infos):
            print("❌ 门禁相关文件的本地 diff patch 为空，验证指纹不完整；拒绝基于不完整输入发布。")
            sys.exit(1)
        try:
            local_head = run_git(["rev-parse", branch]).lower()
            remote_head = get_remote_branch_head(branch)
            if local_head != remote_head:
                raise ValueError(
                    f"local branch head {local_head} does not match origin/{branch} {remote_head}; push the latest commit first"
                )
            verified_tree = get_local_tree(branch)
            validate_verification_metadata(description, verification_inputs, verified_tree, legacy_inputs=legacy_inputs)
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
        # SDK 语义 head=f"{fork_owner}:{branch}" 映射为 CLI 跨仓形式：
        # --fork fork_owner/repo --head branch（服务端解析为 head "fork_owner/repo:branch"）。
        create_argv = [
            "pr",
            "create",
            "-R",
            target.slug,
            "--fork",
            source.slug,
            "--head",
            branch,
            "--base",
            args.base,
            "--title",
            title,
            "--body-file",
            "-",
        ]
        if args.draft:
            create_argv.append("--draft")
        create_argv.append("--json")
        pr_number = create_pr_or_reconcile(cli, target, source, branch, args.base, description, create_argv)

        # PR 编号已确认（创建成功或对账确认）。此后的任何失败都不得报告为“未创建”：
        # 保留 PR 编号/链接与 mutation_applied_verification_pending 状态，
        # 只读对账后再决定 (design.md D5)。
        if gate_triggered:
            try:
                created_pr = fetch_pr_record(cli, target, pr_number)
                created_head = (created_pr.get("head", {}).get("sha") or "").lower()
                if created_head != remote_head:
                    raise ValueError(f"PR head changed during creation: expected {remote_head}, got {created_head}")
                validate_verification_metadata(
                    description, verification_inputs, verified_tree, legacy_inputs=legacy_inputs
                )
            except Exception as e:
                print(f"\n⚠️  PR #{pr_number} 已创建，但写后校验未完成: {e}")
                print(f"🔗 链接: {pr_url(target, pr_number)}")
                print("   请先只读对账 PR 当前状态（head SHA 与验证元数据），不要重复创建或盲目更新。")
                sys.exit(1)

        print()
        print("✅ PR 创建成功!")
        print(f"PR 编号: #{pr_number}")
        print(f"PR 链接: {pr_url(target, pr_number)}")

    else:
        print(">>> 已取消创建 PR")
        sys.exit(0)


if __name__ == "__main__":
    main()
