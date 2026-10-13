#!/usr/bin/env python3
"""Offline IB-Robot PR description gate (single interface, local Git as SSOT).

Generates and validates the final PR description body for ``gitcode pr
create`` / ``gitcode pr edit``. It performs NO platform access and no
network Git: commits, files and patches are read from local Git (branch vs
base), and the finished body is written to ``--output`` for the agent to
publish directly with the official CLI:

    gitcode pr create -R <owner/repo> [--fork <fork/repo>] --head <branch> \
        --base <base> --title "<printed title>" --body-file <output> --json
    gitcode pr edit <N> -R <owner/repo> --title "<printed title>" \
        --body-file <output> --json

Project rules enforced here (rule core: ``_shared/pr_gates.py``):

- openEuler AI disclosure: generated only from --agent-tool/--ai-model/
  --prompt-summary/--third-party-materials into the managed marker block;
  hand-written disclosure is rejected; --ai-model must cover every AI
  ``Co-Authored-By`` model of ALL commits in the PR's commit range (commits
  pushed from other sessions count — fetch the branch first so local Git is
  current). --publication-authorized records publication authorization; human review
  stays pending unless completion has been explicitly confirmed.
- WIP staging: --pr-stage wip|review normalizes the ``[WIP]`` title prefix;
  ``wip`` defers the dual-platform Docker evidence, ``review`` enforces it.
- Dual-platform Docker verification (create): with the gate triggered at
  review stage, the draft must already carry a complete ``## Docker
  Verification`` block whose Verified inputs fingerprint matches the PR
  inputs (the error message prints the expected fingerprint) and whose
  Tested source tree equals the branch head tree. The tree comes from local
  Git only: the branch head must match the local ``origin/<branch>``
  tracking ref (run ``git fetch origin`` first; the skill's pre-publish
  protocol confirms the remote head).
- Dual-platform Docker verification (update): pass --previous-body-file with
  the current remote PR body; reusable evidence is carried forward as
  ``reused-environment`` automatically, otherwise a fresh full block is
  required.
- Large-PR Reuse Self-Check (>2000 changed lines): one complete
  ``## Reuse Self-Check`` section is mandatory; [WIP] does not waive it.

Base resolution is deterministic and fixed for the whole run: an explicit
``--base upstream/master`` (any ref containing ``/``) must resolve as-is;
a plain ``--base master`` resolves to the first existing of
``upstream/master``, ``origin/master`` or the local branch. The resolved
ref is printed and used for every command — there is no per-command
fallback, so a missing base is an error, never a silently different
baseline.

Publishing safety rules that used to live in the deleted wrappers belong to
the agent protocol documented in ``atomgit-pr/SKILL.md``: on an uncertain
create outcome (timeout, or rc=0 without a PR number) reconcile via
``gitcode pr list --paginate`` before any retry — never re-create blindly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))

from pr_gates import (
    REUSE_SELF_CHECK_THRESHOLD,
    TESTED_SOURCE_TREE_LABEL,
    VERIFIED_INPUTS_LABEL,
    add_ai_disclosure,
    collect_diff_file_infos,
    compute_verification_inputs,
    count_changed_lines,
    extract_verification_metadata,
    prepare_update_verification,
    reject_handwritten_disclosure,
    resolve_pr_stage,
    reuse_gate_required,
    validate_agent_tool,
    validate_commit_ai_model,
    validate_reuse_self_check,
    validate_review_status_transition,
    validate_verification_metadata,
    verification_inputs_placeholder_tainted,
)


def make_git(root: str | None):
    """Local git runner bound to --repo-root; stderr text stays in the error.

    Returns **verbatim stdout** (``-z`` listings may begin with a path whose
    first character is whitespace); call sites that want a single value
    strip explicitly. Pathspec-literal behaviour is requested per command by
    ``collect_diff_file_infos`` via the global ``--literal-pathspecs`` flag.
    """

    def git(args: list[str]) -> str:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    return git


def resolve_base_ref(git, base: str) -> str:
    """Deterministically resolve --base to one fixed ref for the whole run.

    An explicit ref (containing ``/``, e.g. ``upstream/master``) must resolve
    as-is or the run stops. A plain branch name resolves to the first existing
    of ``upstream/<base>``, ``origin/<base>`` or the local ref; if none
    resolves the run stops with fetch guidance instead of guessing.
    """
    if "/" in base:
        git(["rev-parse", "--verify", base])
        return base
    remotes = git(["remote"]).strip().split()
    candidates = [f"{remote}/{base}" for remote in ("upstream", "origin") if remote in remotes]
    for ref in candidates:
        try:
            git(["rev-parse", "--verify", ref])
        except ValueError:
            continue
        return ref
    git(["rev-parse", "--verify", base])
    return base


def resolve_branch(git, branch: str | None) -> str:
    branch = branch or git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch in ("master", "main", "HEAD"):
        raise ValueError(f"cannot gate a PR description for {branch}")
    git(["rev-parse", "--verify", branch])
    return branch


def require_pushed_tracking_head(git, branch: str) -> str:
    """Branch head must equal the local origin/<branch> tracking ref.

    ``git fetch origin`` is a skill-protocol prerequisite, not something this
    offline gate performs: the comparison uses the local tracking ref only.
    A missing tracking ref or a mismatch is an error — the tree that Docker
    verified must be exactly the pushed head.
    """
    local_head = git(["rev-parse", branch]).strip().lower()
    try:
        tracking_head = git(["rev-parse", "--verify", f"origin/{branch}"]).strip().lower()
    except ValueError:
        raise ValueError(
            f"local tracking ref origin/{branch} does not exist; run 'git fetch origin' first so the "
            "gate can bind the verified tree to the pushed head"
        ) from None
    if local_head != tracking_head:
        raise ValueError(
            f"local branch head {local_head} does not match origin/{branch} {tracking_head}; "
            "push the latest commit (or reset to the pushed head) before gating"
        )
    return local_head


class LocalPrInputs:
    """PR commit/file inputs read from local Git against one fixed base ref."""

    def __init__(self, git, branch: str, base_ref: str):
        self.git = git
        self.branch = branch
        self.base_ref = base_ref

    def commits(self) -> list[dict]:
        log_format = "%H%n%s%n%b%n---COMMIT-END---"
        output = self.git(["log", f"{self.base_ref}..{self.branch}", f"--format={log_format}"])
        commits = []
        for block in output.split("---COMMIT-END---"):
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

    def file_infos(self) -> list[dict]:
        """Deterministic diff inputs (shared reviewer source, fixed flags).

        Same contract as the reviewer side: refs peeled to OIDs, diffed from
        the explicit merge base with rename pairing disabled, so platform
        rename detection or local git configuration cannot fork the
        fingerprint inputs.
        """
        return collect_diff_file_infos(self.git, self.base_ref, self.branch)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline PR description gate: generate/validate the final PR body for direct gitcode CLI publishing",
    )
    parser.add_argument("--description-file", required=True, help="Draft PR description (Markdown)")
    parser.add_argument("--output", required=True, help="Path for the final gated PR body (Markdown)")
    parser.add_argument("--title", help="PR title (default: first commit subject)")
    parser.add_argument("--branch", help="Source branch (default: current branch)")
    parser.add_argument(
        "--base", default="master", help="Target branch or explicit ref (e.g. upstream/master; default: master)"
    )
    parser.add_argument("--repo-root", default=None, help="Git worktree root (default: current directory)")
    parser.add_argument(
        "--pr-stage",
        choices=("wip", "review"),
        help="PR stage; required when the dual Docker gate triggers, after asking the user (WIP or review)",
    )
    parser.add_argument("--previous-body-file", help="Current remote PR body (update scenario): enables evidence reuse")
    parser.add_argument(
        "--agent-tool",
        required=True,
        help="Product name and version confirmed via '<tool> --version' (use the tool's own name, not the CLI command name)",
    )
    parser.add_argument(
        "--ai-model",
        required=True,
        help="AI model name(s) and version, comma-separated, no provider prefix; must cover every PR commit's Co-Authored-By model",
    )
    parser.add_argument("--prompt-summary", required=True, help="Core prompt / intent summary")
    parser.add_argument("--third-party-materials", required=True, help="Third-party materials and licenses, or '无'")
    parser.add_argument("--publication-authorized", action="store_true", help="Confirm the user authorized publication")
    parser.add_argument(
        "--human-review-status",
        choices=("pending", "completed"),
        default="pending",
        help="Actual human review status; completed requires explicit confirmation",
    )
    parser.add_argument(
        "--review-status-reset",
        action="store_true",
        help="Explicitly downgrade a previously disclosed completed human review after re-asking the developer",
    )
    args = parser.parse_args()

    if not args.publication_authorized:
        print("❌ 发布必须已有用户授权，并显式传入 --publication-authorized")
        return 1

    git = make_git(args.repo_root)
    try:
        branch = resolve_branch(git, args.branch)
        base_ref = resolve_base_ref(git, args.base)
        print(f"✓ 基线 ref: {base_ref}（--base {args.base}）")
        inputs = LocalPrInputs(git, branch, base_ref)

        print(f">>> 读取本地提交（{base_ref}..{branch}）...")
        commits = inputs.commits()
        if not commits:
            raise ValueError(f"no commits found between {base_ref} and {branch}; sync the branch with the base first")
        print(f"✓ {len(commits)} 个提交")

        validate_commit_ai_model(commits, args.ai_model)

        print(">>> 读取变更文件与 patch...")
        file_infos = inputs.file_infos()
        print(f"✓ {len(file_infos)} 个文件")

        with open(args.description_file, encoding="utf-8") as f:
            description = f.read()

        previous_body = None
        if args.previous_body_file:
            with open(args.previous_body_file, encoding="utf-8") as f:
                previous_body = f.read()
            validate_review_status_transition(previous_body, args.human_review_status, reset=args.review_status_reset)

        agent_tool = validate_agent_tool(args.agent_tool)
        reject_handwritten_disclosure(description)
        description = add_ai_disclosure(
            description,
            agent_tool=agent_tool,
            ai_model=args.ai_model,
            prompt_summary=args.prompt_summary,
            third_party_materials=args.third_party_materials,
            human_review_status=args.human_review_status,
            review_status_reset=args.review_status_reset,
        )

        title = args.title or commits[0]["subject"]
        verification_inputs = compute_verification_inputs(file_infos)
        gate_required = verification_inputs is not None
        title, gate_triggered = resolve_pr_stage(title, args.pr_stage, gate_required)
        if gate_required and not gate_triggered:
            print("ℹ️  [WIP] PR 跳过双平台 Docker 验证；移除 [WIP] 转为正式检视时门禁会恢复。")

        validate_reuse_self_check(description, file_infos)
        if reuse_gate_required(file_infos):
            print(
                f"ℹ️  变更 {count_changed_lines(file_infos)} 行（> {REUSE_SELF_CHECK_THRESHOLD}）："
                "已包含 ## Reuse Self-Check 复用自查章节。"
            )

        if gate_triggered:
            if verification_inputs_placeholder_tainted(file_infos):
                raise ValueError(
                    "gate-relevant file patches are incomplete locally; verification fingerprints must be "
                    "computed from complete inputs"
                )
            require_pushed_tracking_head(git, branch)
            head_tree = git(["rev-parse", f"{branch}^{{tree}}"]).strip().lower()
            if previous_body is not None:
                description, _metadata = prepare_update_verification(
                    description,
                    previous_body,
                    verification_inputs,
                    head_tree,
                )
            else:
                try:
                    current_metadata = extract_verification_metadata(description)
                except ValueError:
                    current_metadata = None
                if current_metadata is None:
                    raise ValueError(
                        "a full Docker verification is required; the description has no complete "
                        "'## Docker Verification' block. Run both platform verifications (Ubuntu + "
                        f"openEuler) against the head tree {head_tree}, then set "
                        f"**{VERIFIED_INPUTS_LABEL}:** to {verification_inputs.lower()} — the "
                        "fingerprint of the gate-relevant file list and patches, NOT the commit or "
                        f"tree SHA — and **{TESTED_SOURCE_TREE_LABEL}:** to {head_tree}."
                    )
                validate_verification_metadata(description, verification_inputs, head_tree)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(description, encoding="utf-8")

        print(f"✅ PR 描述已通过门禁: {output_path}")
        print(f"✓ 标题: {title}")
        print(f"✓ 源分支: {branch}（基线 {base_ref}）")
        if gate_triggered:
            mode = "reused-environment（复用证据）" if args.previous_body_file else "full（完整证据）"
            print(f"✓ Docker 验证块: {mode}")
            print(f"✓ 已绑定本地 head tree: {git(['rev-parse', f'{branch}^{{tree}}']).strip().lower()}")
        print("下一步: 按 atomgit-pr/SKILL.md 的 gitcode pr create/edit 模板发布该文件（勿改动正文），")
        print("发布前确认远端 head 与本地一致；创建结果未知时先 gitcode pr list --paginate 对账，禁止盲目重试。")
        return 0
    except ValueError as e:
        print(f"❌ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
