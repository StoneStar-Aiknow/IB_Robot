#!/usr/bin/env python3
"""Offline reviewer-side audit gate for IB_Robot PR review (deterministic).

Reads a PR context that the agent fetched with the official CLI and reports
the mandatory review checks as a JSON array. No platform access and no
network Git: the only subprocess use is local git — ``rev-parse`` to resolve
the PR head tree, and the shared deterministic diff between the locally
fetched PR base (upstream master) and head commits that feeds the
verification fingerprint. Fetch both commits first; when either is missing
the audit reports the inputs as unavailable instead of falling back to the
platform files list or guessing a tree.

Inputs (plain CLI output saved to files; multi-page ``api`` results must be
concatenated into one JSON array first, e.g. ``jq -s 'add'``):

- ``--pr-json``: ``gitcode pr view <N> -R <owner/repo> --json``
- ``--files-json``: ``gitcode api repos/<owner>/<repo>/pulls/<N>/files``
- ``--commits-json``: ``gitcode api repos/<owner>/<repo>/pulls/<N>/commits``

Checks (ported from the retired pr_review.py wrapper; rule core in
``_shared/pr_gates.py``):

- ``lerobot_gitlink_changed``: the PR moves the libs/lerobot submodule
  pointer (blocking).
- ``ai_*``: openEuler AI disclosure completeness and model-set coverage of
  every commit's AI ``Co-Authored-By`` (blocking).
- ``large_pr_reuse_self_check_*``: the >2000-line Reuse Self-Check section
  (blocking; [WIP] does not waive it).
- ``docker_verification_missing`` / ``docker_verification_mismatch`` /
  ``docker_verification_inputs_incomplete`` /
  ``docker_verification_inputs_unavailable``: the structured
  ``## Docker Verification`` block of non-WIP gated PRs, including the
  Verified inputs fingerprint and Tested source tree binding (full mode
  only; reused-environment checks the fingerprint and environment), the
  refusal of fingerprints computed over incomplete patches, and the
  requirement that both the PR base and head commits exist locally so the
  fingerprint comes from the same deterministic local diff the author gate
  used (platform files-list rename pairing is not a substitute input).

Exit codes: 0 = audit completed (the checks array may be non-empty; every
entry with ``blocking_until_reviewed: true`` is mandatory to process before
publishing review findings), 1 = input error (missing/unreadable files).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))

from pr_gates import (
    AGENT_TOOL_RE,
    HUMAN_COAUTHOR_RE,
    REUSE_FIELD_LABELS,
    REUSE_SELF_CHECK_THRESHOLD,
    VERIFICATION_MODE_REUSED,
    collect_diff_file_infos,
    commit_message,
    compute_verification_inputs,
    count_changed_lines,
    extract_reuse_self_check,
    extract_verification_metadata,
    is_wip_title,
    missing_reuse_fields,
    reuse_gate_required,
    validate_verification_metadata,
    verification_inputs_placeholder_tainted,
)


def _git(args: list[str], cwd: str | None) -> str:
    """Verbatim-stdout local git runner for the deterministic diff helper.

    Pathspec-literal behaviour is requested per command by
    ``collect_diff_file_infos`` via the global ``--literal-pathspecs`` flag.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env, timeout=60)
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args[:3])}... failed: {result.stderr.strip()}")
    return result.stdout


def load_json_array(path: Path, what: str) -> list:
    data = load_json_object_or_array(path, what)
    if not isinstance(data, list):
        raise ValueError(f"{what} file {path} must contain a JSON array, got {type(data).__name__}")
    return data


def validate_collection(items: list, *, what: str, key: str, declared_total) -> None:
    """Reject collections that cannot be the complete PR files/commits list.

    A fetch or page-concatenation failure must never masquerade as "no
    files": non-dict entries, entries without their identity field, duplicate
    identities (repeated pages pasted twice) and a count that disagrees with
    the PR record's own total are all input errors. Only shape and totals —
    no new schema.
    """
    bad = sum(1 for item in items if not isinstance(item, dict))
    if bad:
        raise ValueError(f"{what} list contains {bad} non-object entries; refetch the complete collection")
    identities = [item.get(key) for item in items]
    missing = sum(1 for identity in identities if not identity)
    if missing:
        raise ValueError(
            f"{what} list contains {missing} entries without a {key!r} field; refetch the complete collection"
        )
    if len(set(identities)) != len(identities):
        raise ValueError(
            f"{what} list contains duplicate {key!r} entries (repeated pages?); refetch and merge pages once"
        )
    total_is_int = isinstance(declared_total, int) and not isinstance(declared_total, bool)
    if total_is_int and len(items) != declared_total:
        raise ValueError(
            f"{what} list has {len(items)} entries but the PR record declares {declared_total}; "
            "refetch every page and merge with jq -s 'add'"
        )


def load_json_object_or_array(path: Path, what: str):
    if not path.is_file():
        raise ValueError(f"{what} file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{what} file {path} is not valid JSON: {error}") from error


def _check(check_id: str, message: str, *, file: str, severity: str = "error") -> dict:
    return {
        "id": check_id,
        "severity": severity,
        "blocking_until_reviewed": True,
        "file": file,
        "message": message,
    }


def build_mandatory_review_checks(files: list[dict]) -> list[dict]:
    """lerobot submodule gitlink change is always a blocking finding."""
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
        return [
            _check(
                "lerobot_gitlink_changed",
                "PR changes the libs/lerobot submodule pointer. Verify that this is an explicit "
                "upstream base upgrade, that the new commit is fetchable from the .gitmodules "
                "remote, and that INDEX.yaml, manifest.yaml, series files, and patch-stack tests "
                "were migrated. Ordinary LeRobot source changes must be exported under "
                "third_party/patches/lerobot instead of committing a gitlink change.",
                file="libs/lerobot",
            )
            | {"detected_commits": commits}
        ]
    return []


def build_ai_metadata_checks(pr: dict, commits: list[dict]) -> list[dict]:
    """openEuler AI contribution metadata completeness and coverage."""
    body = pr.get("body") or ""
    commit_models = set()
    for commit in commits:
        for line in commit_message(commit).splitlines():
            if not line.strip().startswith("Co-Authored-By:"):
                continue
            value = line.partition(":")[2].strip()
            if value and not HUMAN_COAUTHOR_RE.search(value):
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
            _check(
                "ai_disclosure_incomplete",
                "AI participation is indicated, but the PR disclosure is missing: " + ", ".join(missing_fields),
                file="PR description",
            )
        )

    tool_match = re.search(
        r"(?:Agent平台信息\s*（?Tool）?|Agent\s*(?:platform|tool))\s*[:：]\s*([^\n]+)",
        body,
        re.IGNORECASE,
    )
    if tool_match and AGENT_TOOL_RE.fullmatch(tool_match.group(1).strip()) is None:
        checks.append(
            _check(
                "ai_tool_version_invalid",
                "Agent platform must contain the tool name and version reported by the coding agent.",
                file="PR description",
            )
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
            _check(
                "ai_model_provider_prefix",
                "AI model metadata must contain only the model name and version, without provider prefixes.",
                file="PR description and commit messages",
            )
        )
    if not commit_models:
        checks.append(
            _check(
                "ai_commit_metadata_missing",
                "The PR discloses AI participation, but no commit contains Co-Authored-By: <model name and version>.",
                file="commit messages",
            )
        )
    elif not pr_models or not commit_models.issubset(pr_models):
        missing_models = commit_models - pr_models
        checks.append(
            _check(
                "ai_model_metadata_mismatch",
                f"PR model disclosure ({pr_model_text or 'missing'}) must include every AI model used by "
                f"commits; missing: {', '.join(sorted(missing_models))}.",
                file="PR description and commit messages",
            )
        )
    return checks


def build_reuse_self_check_checks(pr: dict, files: list[dict]) -> list[dict]:
    """Require a complete Reuse Self-Check section above the line threshold."""
    if not reuse_gate_required(files):
        return []

    body = pr.get("body") or ""
    try:
        fields = extract_reuse_self_check(body)
    except ValueError as exc:
        return [
            _check(
                "large_pr_reuse_self_check_invalid",
                str(exc),
                file="PR description",
            )
        ]
    if fields is None:
        return [
            _check(
                "large_pr_reuse_self_check_missing",
                f"This PR changes more than {REUSE_SELF_CHECK_THRESHOLD} lines and must document a "
                "'## Reuse Self-Check' section stating whether existing workflows were reinvented, "
                "what was reused from this repository and libs/lerobot, whether any reinvention is "
                "justified, and how the change follows the architecture of similar features.",
                file="PR description",
            )
        ]
    missing = missing_reuse_fields(fields)
    if missing:
        return [
            _check(
                "large_pr_reuse_self_check_incomplete",
                "The Reuse Self-Check section is incomplete; every field ("
                + ", ".join(f"'**{label}:**'" for label in REUSE_FIELD_LABELS)
                + ") must carry a concrete answer. Missing: "
                + ", ".join(missing)
                + ".",
                file="PR description",
            )
        ]
    return []


def local_head_tree(pr: dict, repo_root: str | None) -> str | None:
    """Resolve the PR head tree from the locally fetched head commit.

    Uses ``git rev-parse <head.sha>^{tree}`` only — no fetch, no platform
    access. An object that is not present locally yields None (reported as
    an unavailable tree), never a guessed tree.
    """
    head_sha = ((pr.get("head") or {}).get("sha") or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", head_sha):
        return None
    try:
        tree = _git(["rev-parse", "--verify", f"{head_sha}^{{tree}}"], repo_root).strip().lower()
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return None
    return tree or None


def local_diff_inputs(pr: dict, repo_root: str | None) -> tuple[list[dict] | None, str | None]:
    """Deterministic fingerprint inputs from local Git (shared author source).

    Resolves the PR base and head commits recorded in ``--pr-json`` and
    derives the same fixed-flag diff the author gate used. Any failure
    returns ``(None, reason)``; the caller must report it and never fall
    back to the platform files list, whose rename pairing diverges from
    local git rename detection.
    """
    head_sha = ((pr.get("head") or {}).get("sha") or "").strip()
    base_sha = ((pr.get("base") or {}).get("sha") or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", head_sha) or not re.fullmatch(r"[0-9a-fA-F]{40}", base_sha):
        return None, "the PR record lacks full base/head commit SHAs for the deterministic diff"
    try:
        return collect_diff_file_infos(lambda args: _git(args, repo_root), base_sha, head_sha), None
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        return None, (
            f"resolving the local diff between PR base {base_sha[:12]} and head {head_sha[:12]} "
            f"failed ({exc}); fetch the PR base (upstream master) and head commits locally"
        )


def build_verification_tree_checks(
    pr: dict,
    head_tree: str | None,
    local_inputs: list[dict] | None,
    local_inputs_error: str | None,
) -> list[dict]:
    """Validate Docker verification evidence against the current PR inputs and tree.

    Fingerprint inputs come from the deterministic local diff shared with
    the author gate; the platform files list is not a substitute (its
    rename pairing diverges from local git). Unresolvable base/head is a
    blocking finding, never a silent fallback to the platform list or the
    local HEAD.
    """
    if is_wip_title(pr.get("title") or ""):
        return []

    if local_inputs is None:
        return [
            _check(
                "docker_verification_inputs_unavailable",
                f"Docker verification needs the deterministic local diff between the PR base and "
                f"head commits, which is unavailable: {local_inputs_error}. Fetch both commits "
                "locally and re-run.",
                file="PR files",
            )
        ]

    verification_inputs = compute_verification_inputs(local_inputs)
    if verification_inputs is None:
        return []

    if verification_inputs_placeholder_tainted(local_inputs):
        # A fingerprint over incomplete patches contains placeholder records;
        # even a hand-written block quoting that exact hash proves nothing.
        # Formal acceptance needs complete inputs and renewed verification.
        return [
            _check(
                "docker_verification_inputs_incomplete",
                "Gate-relevant file patches are incomplete (missing or truncated diffs), so the "
                "Verified inputs fingerprint contains placeholder records and cannot support "
                "acceptance. Refetch complete diffs and obtain renewed verification evidence.",
                file="PR files",
            )
        ]

    try:
        metadata = extract_verification_metadata(pr.get("body") or "")
    except ValueError:
        metadata = None

    if metadata is None:
        return [
            _check(
                "docker_verification_missing",
                "This PR requires dual Docker verification, but its description does not contain a valid "
                "'## Docker Verification' block with mode, Verified inputs, Tested source tree, and "
                "Docker environment fields. Record the evidence tested by both platforms.",
                file="PR description",
            )
        ]

    is_reused = metadata.get("mode") == VERIFICATION_MODE_REUSED
    if not is_reused and head_tree is None:
        return [
            _check(
                "docker_verification_missing",
                "PR head tree is unavailable (fetch the PR head commit locally and re-run); "
                "cannot verify full Docker evidence.",
                file="PR description",
            )
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
            _check(
                "docker_verification_mismatch",
                str(exc),
                file="PR description",
            )
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline reviewer-side audit gate over a CLI-fetched PR context",
    )
    parser.add_argument("--pr-json", required=True, type=Path, help="gitcode pr view <N> --json output file")
    parser.add_argument("--files-json", required=True, type=Path, help="gitcode api .../pulls/<N>/files output file")
    parser.add_argument(
        "--commits-json", required=True, type=Path, help="gitcode api .../pulls/<N>/commits output file"
    )
    parser.add_argument("--repo-root", default=None, help="Local repo root for head tree resolution (default: cwd)")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write the checks JSON to this file (default: stdout)"
    )
    args = parser.parse_args()

    try:
        pr = load_json_object_or_array(args.pr_json, "PR context")
        if not isinstance(pr, dict):
            raise ValueError(f"PR context file {args.pr_json} must contain a JSON object")
        files = load_json_array(args.files_json, "PR files")
        commits = load_json_array(args.commits_json, "PR commits")
        validate_collection(files, what="PR files", key="filename", declared_total=pr.get("changed_files"))
        validate_collection(commits, what="PR commits", key="sha", declared_total=pr.get("commits"))
    except ValueError as error:
        print(f"❌ {error}")
        return 1

    checks: list[dict] = []
    wip = is_wip_title(pr.get("title") or "")
    if wip:
        # WIP defers the whole Docker verification branch; do not even probe
        # the local base/head inputs (they are not needed and may be absent).
        local_inputs, local_inputs_error = None, None
    else:
        local_inputs, local_inputs_error = local_diff_inputs(pr, args.repo_root)
    checks.extend(build_mandatory_review_checks(files))
    checks.extend(
        build_verification_tree_checks(
            pr, local_head_tree(pr, args.repo_root) if not wip else None, local_inputs, local_inputs_error
        )
    )
    checks.extend(build_ai_metadata_checks(pr, commits))
    checks.extend(build_reuse_self_check_checks(pr, files))

    summary = {
        "pr_number": pr.get("number"),
        "title": pr.get("title"),
        "wip": wip,
        "reuse_self_check": {
            "changed_lines": count_changed_lines(files),
            "required": reuse_gate_required(files),
        },
        "checks": checks,
    }

    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"✅ 审计完成: {len(checks)} 项检查 → {args.output}")
    else:
        print(rendered)
    if checks:
        blocking = [c for c in checks if c.get("blocking_until_reviewed")]
        print(f"ℹ️  {len(checks)} 项检查（{len(blocking)} 阻塞）；阻塞项必须在发布检视意见前逐项处理", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
