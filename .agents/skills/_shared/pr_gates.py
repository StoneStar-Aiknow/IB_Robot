"""Offline IB-Robot PR rule gates (project-specific, deterministic).

Single source for the author- and reviewer-side PR description gates:

- openEuler AI contribution disclosure (``ai_compliance`` section): the
  machine-marked disclosure block is generated only by :func:`add_ai_disclosure`
  from CLI metadata, hand-written copies are rejected, and the disclosed model
  set must cover every AI ``Co-Authored-By`` model of the PR commits.
- Large-PR Reuse Self-Check (``reuse_gate`` section): PRs above the changed-line
  threshold must carry one complete '## Reuse Self-Check' section.
- Dual-platform Docker verification (``verification_gate`` section): the
  structured '## Docker Verification' block, input fingerprints (policy v2),
  WIP title staging, and evidence reuse rules.

The author-side modules and reviewer reuse-gate copy previously lived under
``atomgit-pr/scripts`` and ``atomgit-pr-review/lib`` (verification was already
shared via a symlink). They are consolidated here as the shared rule
core consumed by the offline entry points of the direct-CLI workflows:

- ``atomgit-pr/scripts/pr_body_gate.py`` (author side: generate/validate the
  PR description body before ``gitcode pr create/edit``)
- ``atomgit-pr-review/scripts/review_gates.py`` (reviewer side: mandatory
  review checks over a PR context fetched with the official CLI)

Everything here is offline with respect to the AtomGit platform: no platform
API reads or writes, no token, no CLI invocation. PR/issue context is
supplied by the caller as plain JSON produced with the official CLI
(``gitcode pr view --json`` / ``gitcode api ...``); the author-side entry
point derives its inputs from local Git, which is the SSOT for unpushed work.
The only subprocess use is the deterministic local-git diff helper
(:func:`collect_diff_file_infos`), shared by the author and reviewer entry
points so both compute verification fingerprints from one fixed-flag source.
"""

from __future__ import annotations

import hashlib
import json
import re

# --------------------------------------------------------------------------- #
# openEuler AI contribution metadata (from ai_compliance.py)
# --------------------------------------------------------------------------- #

DISCLOSURE_START = "<!-- openEuler-ai-disclosure:start -->"
DISCLOSURE_END = "<!-- openEuler-ai-disclosure:end -->"
PLACEHOLDERS = {"ai", "agent", "model", "unknown", "n/a", "none"}
HUMAN_COAUTHOR_RE = re.compile(r"<[^<>]+>\s*$")
AGENT_TOOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._/@:+~\- ]*\s+v?\d+(?:\.\d+){1,5}(?:[-+][0-9A-Za-z.-]+)?$")
REVIEW_STATUS_PENDING_TEXT = "待开发者完成逐项实质审查；发布授权不等同于审查完成"
REVIEW_STATUS_COMPLETED_TEXT = "开发者已明确确认完成逐项实质审查"


def _require_value(name: str, value: str) -> str:
    value = value.strip()
    if not value or value.lower() in PLACEHOLDERS:
        raise ValueError(f"{name} must contain the actual value, not a placeholder")
    return value


def validate_agent_tool(value: str) -> str:
    """Validate the tool/version reported by the coding agent.

    The caller must run the tool's real version command before invoking the PR
    workflow. The repository intentionally does not maintain a tool allowlist.
    """
    value = _require_value("Agent platform", value)
    if AGENT_TOOL_RE.fullmatch(value) is None:
        raise ValueError(
            "Agent platform must contain the product name and version (for example, 'OpenCode 1.17.20'); "
            "take the version from '<tool> --version' and the product name from the tool's own name, not "
            "the CLI command name (for example 'DeepSeek Harness 0.1.7-rc.2', not 'dsh 0.1.7-rc.2')"
        )
    return value


def _require_model(value: str) -> str:
    value = _require_value("AI model", value)
    if "/" in value:
        raise ValueError(
            "AI model must not include a provider prefix; use only the model name and version "
            f"(for example, {value.rsplit('/', 1)[-1]!r})"
        )
    return value


def _require_models(value: str) -> list[str]:
    models = [_require_model(item) for item in re.split(r"[,，;；\n]", value) if item.strip()]
    if not models:
        raise ValueError("AI model must contain at least one model name and version")
    return list(dict.fromkeys(models))


_DISCLOSURE_BLOCK_RE = re.compile(
    rf"{re.escape(DISCLOSURE_START)}.*?{re.escape(DISCLOSURE_END)}",
    re.DOTALL,
)
_HANDWRITTEN_DISCLOSURE_RE = re.compile(r"当前PR是否有AI参与|希望检视人员了解|Agent平台信息（Tool）")


def extract_disclosure_review_status(body: str) -> str | None:
    """Read the disclosed human-review status from a managed disclosure block."""
    match = _DISCLOSURE_BLOCK_RE.search(body or "")
    if match is None:
        return None
    block = match.group(0)
    if REVIEW_STATUS_COMPLETED_TEXT in block:
        return "completed"
    if REVIEW_STATUS_PENDING_TEXT in block:
        return "pending"
    return None


def validate_review_status_transition(previous_body: str, human_review_status: str, *, reset: bool = False) -> None:
    """Reject a silent downgrade of an already-disclosed completed human review.

    The disclosure block is regenerated from CLI metadata on every gate run,
    so an update that re-runs with the default ``--human-review-status pending``
    would quietly rewrite a previously published "completed" review record.
    Both update doors are guarded here: the previous PR body passed via
    ``--previous-body-file`` and a draft that still carries the old managed
    block. A real downgrade must re-ask the developer and pass an explicit
    reset flag.
    """
    if reset:
        return
    if extract_disclosure_review_status(previous_body) == "completed" and human_review_status != "completed":
        raise ValueError(
            "the existing PR disclosure already records completed human review; regenerating with a lower "
            "--human-review-status would silently downgrade it — keep completed, or re-ask the developer and "
            "pass --review-status-reset to downgrade explicitly"
        )


def _ensure_disclosure_separator(description: str) -> str:
    """Keep exactly one horizontal rule between the disclosure block and the body."""
    head, marker, tail = description.partition(DISCLOSURE_END)
    if not marker:
        return description
    tail = tail.rstrip()
    if not tail:
        return head + DISCLOSURE_END
    # Strip blank lines and stray horizontal rules after the managed block, then
    # re-insert exactly one separator.
    tail = re.sub(r"\A(?:\s|-{3,}[ \t]*\n\s*)*", "", tail)
    return f"{head}{DISCLOSURE_END}\n\n---\n\n{tail}"


def reject_handwritten_disclosure(description: str) -> None:
    """Reject AI disclosure content written outside the managed marker block.

    The disclosure is generated exclusively by ``add_ai_disclosure`` from CLI
    metadata. A hand-written section would duplicate the managed block.
    """
    outside_block = _DISCLOSURE_BLOCK_RE.sub("", description)
    if _HANDWRITTEN_DISCLOSURE_RE.search(outside_block):
        raise ValueError(
            "hand-written AI disclosure content detected in the PR description; remove it — the openEuler "
            "disclosure block is generated automatically from --agent-tool/--ai-model/--prompt-summary/"
            "--third-party-materials and must stay inside the managed marker block"
        )


def add_ai_disclosure(
    description: str,
    *,
    agent_tool: str,
    ai_model: str,
    prompt_summary: str,
    third_party_materials: str,
    human_review_status: str = "pending",
    review_status_reset: bool = False,
) -> str:
    """Add or replace the machine-identifiable openEuler AI disclosure block."""
    agent_tool = _require_value("Agent platform", agent_tool)
    agent_tool = validate_agent_tool(agent_tool)
    ai_models = _require_models(ai_model)
    prompt_summary = _require_value("Prompt summary", prompt_summary)
    third_party_materials = _require_value("Third-party materials", third_party_materials)

    review_states = {
        "pending": REVIEW_STATUS_PENDING_TEXT,
        "completed": REVIEW_STATUS_COMPLETED_TEXT,
    }
    if human_review_status not in review_states:
        raise ValueError("human review status must be pending or completed")
    # A draft pasted from the current PR body still carries the old managed
    # block; replacing it with a lower status must not downgrade the record.
    validate_review_status_transition(description, human_review_status, reset=review_status_reset)

    disclosure = f"""{DISCLOSURE_START}
### 当前PR是否有AI参与:
- [ ] 否
- [x] 是

1. **Agent平台信息（Tool）**: {agent_tool}
2. **模型信息 (Model)**: {", ".join(ai_models)}
3. **Prompt摘要 (Prompt Summary)**: {prompt_summary}
4. **人工审查情况**: {review_states[human_review_status]}
5. **第三方材料及许可证**: {third_party_materials}

### 希望检视人员了解:
代码或文档由 AI 辅助开发者完成，最终正确性、安全性、合规性和维护责任由人类贡献者承担。
{DISCLOSURE_END}"""

    body = description.strip()
    if not body:
        return disclosure
    if _DISCLOSURE_BLOCK_RE.search(description) is None:
        if DISCLOSURE_START in description or DISCLOSURE_END in description:
            raise ValueError("PR description contains an incomplete openEuler AI disclosure block")
        return _ensure_disclosure_separator(f"{disclosure}\n\n{body}")
    return _ensure_disclosure_separator(_DISCLOSURE_BLOCK_RE.sub(lambda _m: disclosure, description, count=1))


def commit_message(commit: dict) -> str:
    api_message = (commit.get("commit") or {}).get("message")
    if isinstance(api_message, str):
        return api_message
    message = commit.get("message")
    if isinstance(message, str):
        return message
    return "\n".join(part for part in (commit.get("subject", ""), commit.get("body", "")) if part)


def validate_commit_ai_model(commits: list[dict], ai_model: str) -> None:
    """Require the PR disclosure to cover every AI model used by its commits."""
    disclosed_models = set(_require_models(ai_model))
    commit_models = set()

    for commit in commits:
        message = commit_message(commit)
        trailers = [line.strip() for line in message.splitlines() if line.strip().startswith("Co-Authored-By:")]
        identifier = commit.get("hash") or commit.get("sha") or "unknown"
        for trailer in trailers:
            model = trailer.partition(":")[2].strip()
            if not HUMAN_COAUTHOR_RE.search(model):
                try:
                    commit_models.add(_require_model(model))
                except ValueError as exc:
                    raise ValueError(f"invalid AI model metadata in {identifier[:12]}: {exc}") from exc

    problems = []
    if not commit_models:
        problems.append("no AI-assisted commit contains Co-Authored-By model metadata")
    missing = commit_models - disclosed_models
    if missing:
        problems.append(
            f"PR disclosure does not include commit models: {', '.join(sorted(missing))}; re-fetch the "
            "PR commits and pass the union of every commit's Co-Authored-By model as --ai-model "
            "(commits pushed from other sessions count too)"
        )
    if problems:
        raise ValueError("AI commit metadata check failed; " + "; ".join(problems))


# --------------------------------------------------------------------------- #
# Large-PR Reuse Self-Check (from reuse_gate.py)
# --------------------------------------------------------------------------- #

REUSE_SELF_CHECK_HEADING = "Reuse Self-Check"
REUSE_SELF_CHECK_THRESHOLD = 2000
REINVENTED_WORKFLOWS_LABEL = "Reinvented workflows"
REUSED_COMPONENTS_LABEL = "Reused components"
REINVENTION_JUSTIFICATION_LABEL = "Reinvention justification"
ARCHITECTURE_CONFORMANCE_LABEL = "Architecture conformance"
REUSE_FIELD_LABELS = (
    REINVENTED_WORKFLOWS_LABEL,
    REUSED_COMPONENTS_LABEL,
    REINVENTION_JUSTIFICATION_LABEL,
    ARCHITECTURE_CONFORMANCE_LABEL,
)
_REUSE_HEADING_RE = re.compile(r"^##\s*Reuse Self-Check\s*$", re.IGNORECASE | re.MULTILINE)
_NEXT_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)

# _normalise_patch is defined once in the verification section below; the
# reuse section shares it (the merged duplicates were byte-identical).


def _count_patch_lines(patch: str) -> int:
    """Count +/- content lines, ignoring hunk headers, file headers, and context."""
    count = 0
    for line in patch.splitlines():
        if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
            continue
        count += 1
    return count


def count_changed_lines(files: list[dict]) -> int:
    """Total additions plus deletions; falls back to patch line counting when stats are absent."""
    total = 0
    for file_info in files:
        additions = file_info.get("additions")
        deletions = file_info.get("deletions")
        if additions is None and deletions is None:
            total += _count_patch_lines(_normalise_patch(file_info.get("patch")))
            continue
        total += max(int(additions or 0), 0) + max(int(deletions or 0), 0)
    return total


def reuse_gate_required(files: list[dict]) -> bool:
    return count_changed_lines(files) > REUSE_SELF_CHECK_THRESHOLD


def _reuse_section(description: str) -> str | None:
    """Return the section body under the single Reuse Self-Check heading."""
    text = description or ""
    heading_matches = list(_REUSE_HEADING_RE.finditer(text))
    if not heading_matches:
        return None
    if len(heading_matches) > 1:
        raise ValueError("description must contain exactly one '## Reuse Self-Check' heading")
    rest = text[heading_matches[0].end() :]
    next_heading = _NEXT_HEADING_RE.search(rest)
    return rest[: next_heading.start()] if next_heading else rest


def extract_reuse_self_check(description: str) -> dict[str, str | None] | None:
    """Parse the four Reuse Self-Check fields; values are None when a field is absent or empty."""
    section = _reuse_section(description)
    if section is None:
        return None
    fields: dict[str, str | None] = {}
    for label in REUSE_FIELD_LABELS:
        # Horizontal whitespace only after the label, so an empty answer cannot
        # swallow the newline and capture the next field's line.
        pattern = re.compile(rf"^\*\*{re.escape(label)}:\*\*[^\S\n]*(\S.*)$", re.IGNORECASE | re.MULTILINE)
        matches = pattern.findall(section)
        if len(matches) > 1:
            raise ValueError(f"Reuse Self-Check contains multiple '{label}' fields")
        fields[label] = matches[0].strip() if matches else None
    return fields


def missing_reuse_fields(fields: dict[str, str | None]) -> list[str]:
    return [label for label in REUSE_FIELD_LABELS if not fields.get(label)]


def validate_reuse_self_check(description: str, files: list[dict]) -> dict[str, str]:
    """Require a complete Reuse Self-Check section when the PR exceeds the line threshold."""
    if not reuse_gate_required(files):
        return {}
    fields = extract_reuse_self_check(description)
    if fields is None:
        raise ValueError(
            "this PR changes more than "
            f"{REUSE_SELF_CHECK_THRESHOLD} lines and requires exactly one '## {REUSE_SELF_CHECK_HEADING}' "
            "section in the description stating whether existing workflows were reinvented, what was reused "
            "from this repository and libs/lerobot, whether any reinvention is justified, and how the change "
            "follows the architecture of similar features"
        )
    missing = missing_reuse_fields(fields)
    if missing:
        raise ValueError(f"Reuse Self-Check is missing concrete answers for: {', '.join(missing)}")
    return fields


# --------------------------------------------------------------------------- #
# Dual-platform Docker verification gate (from verification_gate.py)
# --------------------------------------------------------------------------- #

VERIFIED_TREE_LABEL = "Verified tree"
VERIFICATION_MODE_LABEL = "Docker verification mode"
VERIFIED_INPUTS_LABEL = "Verified inputs"
TESTED_SOURCE_TREE_LABEL = "Tested source tree"
VERIFICATION_ENVIRONMENT_LABEL = "Docker environment"
VERIFICATION_MODE_FULL = "full"
VERIFICATION_MODE_REUSED = "reused-environment"
# Fingerprint policy: a single current policy v2. The fingerprint hashes the
# dependency-relevant records — non-dependency files never enter it, so
# unrelated changes cannot invalidate recorded evidence — and the payload's
# policy field marks that the hash was computed over provably complete
# inputs (see verification_inputs_placeholder_tainted). The v2 calculator is
# frozen: do not change records, serialization or hashing, or every recorded
# evidence hash stops matching. Evidence recorded under the retired v1
# policy no longer verifies; the gate reports a mismatch and requires
# renewed verification instead of relabelling it.
VERIFICATION_POLICY_VERSION = "2"
PATCH_UNAVAILABLE_PLACEHOLDER = "<patch-unavailable>"
# Task 3.2 review: known truncation markers on a changed-file entry, or diff
# hunks whose declared line counts do not add up, mark the patch incomplete
# until the CLI-side capability to re-fetch complete diffs is verified
# (task 1.4). Entry-level flags are the authoritative truncation indicator;
# a non-empty patch without hunk headers is treated as structurally valid.
_PATCH_TRUNCATION_KEYS = ("patch_truncated", "truncated", "too_large", "is_truncated")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
WIP_PREFIX = "[WIP]"
_VERIFICATION_HEADING_RE = re.compile(r"^##[ \t]+Docker Verification[ \t]*\r?$", re.MULTILINE | re.IGNORECASE)
_WIP_PREFIX_RE = re.compile(r"^(?:\s*\[WIP\]\s*)+", re.IGNORECASE)
# Field lines accept an optional Markdown list prefix ("- " or "* ") so Agents
# can write them inside bullet lists without triggering duplicate-block errors.
# upsert_verification_metadata strips both forms before appending the canonical
# block, so formatting never yields two copies of the same field.
_FIELD_LINE_PREFIX = r"(?:-\s+)?"
_VERIFICATION_MODE_RE = re.compile(
    rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(VERIFICATION_MODE_LABEL)}:\*\*\s*`?(full|reused-environment)`?",
    re.IGNORECASE,
)
_VERIFICATION_SHA_FIELDS = {
    VERIFIED_INPUTS_LABEL: re.compile(
        rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(VERIFIED_INPUTS_LABEL)}:\*\*\s*`?([0-9a-f]{{40}})(?![0-9a-f])`?",
        re.IGNORECASE,
    ),
    TESTED_SOURCE_TREE_LABEL: re.compile(
        rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(TESTED_SOURCE_TREE_LABEL)}:\*\*\s*`?([0-9a-f]{{40}})(?![0-9a-f])`?",
        re.IGNORECASE,
    ),
}
_VERIFICATION_ENVIRONMENT_RE = re.compile(
    rf"{_FIELD_LINE_PREFIX}\*\*{re.escape(VERIFICATION_ENVIRONMENT_LABEL)}:\*\*\s*`([^`\n]+)`",
    re.IGNORECASE,
)
_PACKAGE_DEPENDENCY_RE = re.compile(
    r"^[+-](?![+-])\s*</?(?:depend|build_depend|build_export_depend|buildtool_depend|buildtool_export_depend|exec_depend|test_depend|doc_depend|group_depend)\b",
    re.MULTILINE,
)
_GLOBAL_GATE_FILES = {
    "CMakeLists.txt",
    "pyproject.toml",
    "scripts/build.sh",
    "scripts/install_ros.sh",
    "scripts/setup.sh",
    "scripts/setup/python_venv.sh",
    "scripts/setup/verify_env.sh",
    "scripts/setup/lerobot_patches.sh",
    ".gitmodules",
}
_LEROBOT_PATCH_PREFIX = "third_party/patches/lerobot/"


def file_triggers_dual_docker_gate(filename: str, patch: str = "") -> bool:
    if filename in _GLOBAL_GATE_FILES:
        return True
    if filename.startswith("scripts/setup/platforms/") and filename.endswith(".sh"):
        return True
    if filename.startswith("requirements/") and filename.endswith(".txt"):
        return True
    if filename == "libs/lerobot" or filename.startswith(_LEROBOT_PATCH_PREFIX):
        return True
    if filename != "package.xml" and not filename.endswith("/package.xml"):
        return False
    # Missing patches are treated as gated so API truncation cannot bypass the check.
    return not patch or bool(_PACKAGE_DEPENDENCY_RE.search(patch))


def _normalise_patch(patch: str | dict | None) -> str:
    if isinstance(patch, dict):
        patch = patch.get("diff") or ""
    return patch or ""


def _changed_lines(patch: str) -> list[str]:
    """Extract only +/- content lines, ignoring hunk headers and context."""
    lines = []
    for line in patch.splitlines():
        if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
            continue
        lines.append(line)
    return lines


def verification_input_record(filename: str, patch: str | dict | None = "") -> tuple[str, str] | None:
    """Return the setup/ dependency input represented by one changed file."""
    patch = _normalise_patch(patch)
    if filename == "package.xml" or filename.endswith("/package.xml"):
        if not patch:
            return filename, PATCH_UNAVAILABLE_PLACEHOLDER
        dependency_lines = [line for line in _changed_lines(patch) if _PACKAGE_DEPENDENCY_RE.search(line)]
        return (filename, "\n".join(sorted(dependency_lines))) if dependency_lines else None
    if file_triggers_dual_docker_gate(filename, patch):
        if not patch:
            return filename, PATCH_UNAVAILABLE_PLACEHOLDER
        return filename, "\n".join(_changed_lines(patch))
    return None


def _hunk_line_counts_add_up(patch: str) -> bool:
    """True when every hunk's declared line counts match its body lines.

    A truncated patch (platform-side diff size limits) usually loses body
    lines, so hunks whose declared counts do not add up mark the patch
    incomplete. A non-empty patch without any hunk header is treated as
    structurally valid here: entry-level truncation flags are the only
    authoritative indicator for such patches, and refusing every non-hunk
    patch would reject dependency lines extracted from them. Empty patches
    are handled by the placeholder logic.
    """
    lines = patch.splitlines()
    index = 0
    while index < len(lines):
        match = _HUNK_HEADER_RE.match(lines[index])
        index += 1
        if match is None:
            continue  # diff header lines (diff --git / index / --- / +++)
        old_expected = int(match.group(2) if match.group(2) is not None else 1)
        new_expected = int(match.group(4) if match.group(4) is not None else 1)
        old_actual = new_actual = 0
        while index < len(lines) and not _HUNK_HEADER_RE.match(lines[index]):
            line = lines[index]
            if not line.startswith("\\"):  # "\ No newline at end of file"
                if line.startswith("-") and not line.startswith("---"):
                    old_actual += 1
                elif line.startswith("+") and not line.startswith("+++"):
                    new_actual += 1
                else:
                    old_actual += 1
                    new_actual += 1
            index += 1
        if old_actual != old_expected or new_actual != new_expected:
            return False
    return True


def _file_patch_and_incompleteness(file_info: dict) -> tuple[str, str | None]:
    """Return (patch, incompleteness reason or None) for one changed-file entry.

    A known truncation flag on the entry, or diff hunks whose declared line
    counts do not add up, marks the patch as not provably complete; the
    fingerprint layer then treats it as a placeholder for gate-relevant
    files instead of trusting its visible content (task 3.2 review: a
    truncated package.xml patch with no visible depend lines must not
    bypass the gate, and truncated requirements patches must not enter the
    fingerprint as if they were complete).
    """
    for key in _PATCH_TRUNCATION_KEYS:
        if file_info.get(key):
            return _normalise_patch(file_info.get("patch")), f"entry declares {key}"
    patch = _normalise_patch(file_info.get("patch"))
    if patch and not _hunk_line_counts_add_up(patch):
        return patch, "diff hunk line counts do not add up (truncated patch)"
    return patch, None


def _may_contribute_gate_inputs(filename: str, patch: str) -> bool:
    """Could this file trigger the dual Docker gate or produce a record?"""
    if filename == "package.xml" or filename.endswith("/package.xml"):
        # The package.xml gate decision depends on patch content; an
        # incomplete patch can neither confirm nor exclude dependency changes.
        return True
    return file_triggers_dual_docker_gate(filename, patch)


def _collect_verification_inputs(files: list[dict]) -> list[tuple[str, str]]:
    """Return the dependency records for one input set.

    Files whose patch is missing or not provably complete contribute a
    placeholder record when they could affect the gate, so truncation can
    never silently shrink the fingerprint or bypass the package.xml check.
    """
    records: list[tuple[str, str]] = []
    for file_info in files:
        filename = file_info.get("filename") or file_info.get("new_path") or ""
        patch, incomplete = _file_patch_and_incompleteness(file_info)
        if incomplete and _may_contribute_gate_inputs(filename, patch):
            records.append((filename, PATCH_UNAVAILABLE_PLACEHOLDER))
            continue
        record = verification_input_record(filename, patch)
        if record is not None:
            records.append(record)
    return records


def _fingerprint(policy: str, records: list[tuple[str, str]]) -> str:
    payload = {"policy": policy, "records": sorted(set(records))}
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()


def compute_verification_inputs(files: list[dict]) -> str | None:
    """Fingerprint of the PR's dependency-relevant changes under policy v2.

    Non-dependency files never enter the fingerprint, so unrelated changes
    cannot invalidate recorded evidence. Returns None only when no
    gate-relevant record exists. Missing or truncated patches stay
    gate-relevant through the ``<patch-unavailable>`` placeholder record,
    but callers must refuse to reuse fingerprint-based evidence while any
    record is a placeholder (:func:`verification_inputs_placeholder_tainted`).
    """
    records = _collect_verification_inputs(files)
    if not records:
        return None
    return _fingerprint(VERIFICATION_POLICY_VERSION, records)


def verification_inputs_placeholder_tainted(files: list[dict]) -> bool:
    """True when a gate-relevant input is missing or not provably complete.

    Missing patches and non-empty-but-truncated patches (entry truncation
    flags such as ``patch_truncated``, or hunk line counts that do not add
    up) both count. Such inputs keep the gate triggered (conservative) but
    must not be used for evidence reuse: renewed verification from complete
    inputs is required instead of automatic reuse.
    """
    for file_info in files:
        filename = file_info.get("filename") or file_info.get("new_path") or ""
        patch, incomplete = _file_patch_and_incompleteness(file_info)
        if incomplete and _may_contribute_gate_inputs(filename, patch):
            return True
        record = verification_input_record(filename, patch)
        if record is not None and record[1] == PATCH_UNAVAILABLE_PLACEHOLDER:
            return True
    return False


# --------------------------------------------------------------------------- #
# Deterministic local-git diff inputs (shared author/reviewer fingerprint
# source). Platform files-API rename pairing diverges from any local git
# rename detection (PR460 review follow-up), so both sides derive the
# fingerprint inputs from the same fixed-flag local diff instead.
# --------------------------------------------------------------------------- #

_FIXED_DIFF_FLAGS = (
    "--no-renames",
    "--no-ext-diff",
    "--no-textconv",
    "--no-indent-heuristic",
    "--diff-algorithm=myers",
)
_OID_RE = re.compile(r"[0-9a-f]{40}")


def _resolve_commit_oid(git, ref: str) -> str:
    """Peel a ref to its full commit OID; only SHA output is stripped."""
    oid = git(["rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
    if not _OID_RE.fullmatch(oid):
        raise ValueError(f"cannot resolve {ref!r} to a full commit OID locally")
    return oid


def collect_diff_file_infos(git, base_ref: str, head_ref: str) -> list[dict]:
    """Deterministic complete diff inputs from local Git for both gates.

    ``git`` is a runner taking an argv list (without the leading ``git``)
    and returning **verbatim stdout** — ``-z`` listings may start with a
    path whose first character is whitespace, so a stripping runner would
    corrupt paths; only SHA output is stripped here. Pathspec lookup uses
    git's global ``--literal-pathspecs`` so metacharacters in filenames
    never glob, without any environment contract on the runner.

    Both refs are peeled to fixed commit OIDs first, then diffed from
    ``merge-base(base, head)`` to head with a frozen flag set that disables
    rename pairing, external diff drivers, textconv filters, the indent
    heuristic and alternative diff algorithms. The result therefore does
    not depend on user or platform git configuration: a rename always
    appears as a full delete of the old path plus a full add of the new
    path. Any failure (missing object, no merge base) raises ValueError —
    callers must report it, never fall back to another input source.
    """
    base_oid = _resolve_commit_oid(git, base_ref)
    head_oid = _resolve_commit_oid(git, head_ref)
    merge_base = git(["merge-base", base_oid, head_oid]).strip()
    if not _OID_RE.fullmatch(merge_base):
        raise ValueError(f"cannot prove a common base for {base_ref} and {head_ref}")
    listing = git(["diff", *_FIXED_DIFF_FLAGS, "-z", "--name-only", merge_base, head_oid])
    infos = []
    for name in listing.split("\0"):
        if not name:
            continue
        patch = git(
            ["--literal-pathspecs", "diff", *_FIXED_DIFF_FLAGS, "--unified=0", merge_base, head_oid, "--", name]
        )
        infos.append({"filename": name, "patch": patch})
    return infos


def is_wip_title(title: str) -> bool:
    return bool(_WIP_PREFIX_RE.match(title or ""))


def normalize_pr_title(title: str, stage: str) -> str:
    base_title = _WIP_PREFIX_RE.sub("", title or "").strip()
    if not base_title:
        raise ValueError("PR title must not be empty")
    if stage == "wip":
        return f"{WIP_PREFIX} {base_title}"
    if stage == "review":
        return base_title
    raise ValueError("PR stage must be 'wip' or 'review'")


def resolve_pr_stage(title: str, stage: str | None, gate_required: bool) -> tuple[str, bool]:
    if gate_required and stage is None:
        raise ValueError(
            "dual Docker verification gate requires an explicit PR stage after asking the user: use 'wip' or 'review'"
        )
    if stage is not None:
        title = normalize_pr_title(title, stage)
    return title, gate_required and not is_wip_title(title)


def extract_verification_metadata(description: str) -> dict | None:
    if len(_VERIFICATION_HEADING_RE.findall(description or "")) > 1:
        raise ValueError("description must contain exactly one '## Docker Verification' heading")
    mode_matches = _VERIFICATION_MODE_RE.findall(description or "")
    input_matches = _VERIFICATION_SHA_FIELDS[VERIFIED_INPUTS_LABEL].findall(description or "")
    tree_matches = _VERIFICATION_SHA_FIELDS[TESTED_SOURCE_TREE_LABEL].findall(description or "")
    environment_matches = _VERIFICATION_ENVIRONMENT_RE.findall(description or "")
    if not mode_matches and not input_matches and not tree_matches:
        return None
    if len(mode_matches) != 1 or len(input_matches) != 1 or len(tree_matches) != 1 or len(environment_matches) != 1:
        counts = {
            VERIFICATION_MODE_LABEL: len(mode_matches),
            VERIFIED_INPUTS_LABEL: len(input_matches),
            TESTED_SOURCE_TREE_LABEL: len(tree_matches),
            VERIFICATION_ENVIRONMENT_LABEL: len(environment_matches),
        }
        bad_fields = ", ".join(f"{label}={count}" for label, count in counts.items() if count != 1)
        raise ValueError(
            "Docker verification metadata requires exactly one of each field; "
            f"found: {bad_fields}. Each field must appear exactly once as "
            "'**<label>:** <value>' (a leading '- ' list prefix is also accepted)."
        )
    return {
        "mode": mode_matches[0].lower(),
        "verified_inputs": input_matches[0].lower(),
        "tested_tree": tree_matches[0].lower(),
        "environment": environment_matches[0],
    }


def format_verification_metadata(
    mode: str,
    verified_inputs: str,
    tested_tree: str,
    environment: str,
) -> str:
    if mode not in {VERIFICATION_MODE_FULL, VERIFICATION_MODE_REUSED}:
        raise ValueError(f"unsupported Docker verification mode: {mode}")
    return (
        "## Docker Verification\n\n"
        f"**{VERIFICATION_MODE_LABEL}:** `{mode}`\n"
        f"**{VERIFIED_INPUTS_LABEL}:** `{verified_inputs}`\n"
        f"**{TESTED_SOURCE_TREE_LABEL}:** `{tested_tree}`\n"
        f"**{VERIFICATION_ENVIRONMENT_LABEL}:** `{environment}`"
    )


def upsert_verification_metadata(
    description: str,
    mode: str,
    verified_inputs: str,
    tested_tree: str,
    environment: str,
) -> str:
    if len(_VERIFICATION_HEADING_RE.findall(description)) > 1:
        raise ValueError("description must contain exactly one '## Docker Verification' heading")
    line_patterns = [
        rf"^(?:-\s+)?\*\*{re.escape(VERIFICATION_MODE_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(VERIFIED_INPUTS_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(TESTED_SOURCE_TREE_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(VERIFICATION_ENVIRONMENT_LABEL)}:\*\*.*(?:\n|$)",
        rf"^(?:-\s+)?\*\*{re.escape(VERIFIED_TREE_LABEL)}:\*\*.*(?:\n|$)",
    ]
    cleaned = description
    for pattern in line_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)
    block = format_verification_metadata(mode, verified_inputs, tested_tree, environment)
    heading = _VERIFICATION_HEADING_RE.search(cleaned)
    if heading:
        before = cleaned[: heading.start()]
        after = cleaned[heading.end() :].lstrip("\r\n")
        return f"{before}{block}\n\n{after}".rstrip() + "\n"
    return f"{cleaned.rstrip()}\n\n{block}\n"


def validate_verification_metadata(
    description: str,
    expected_inputs: str,
    expected_tree: str,
    *,
    allow_reuse: bool = False,
    expected_environment: str | None = None,
) -> dict:
    """Validate Docker verification metadata.

    The recorded inputs must equal the current policy-v2 fingerprint of the
    PR. Evidence recorded under any other hash — including the retired v1
    policy — does not verify and requires renewed verification; it is never
    silently relabelled as current.

    For reused-environment mode, prior-evidence existence is verified by the
    author-side ``prepare_update_verification``; the reviewer only checks that
    inputs fingerprint and environment match.
    """
    metadata = extract_verification_metadata(description)
    if metadata is None:
        raise ValueError("Docker verification metadata is missing")
    if metadata["verified_inputs"] != (expected_inputs or "").lower():
        raise ValueError(
            f"Docker verification inputs {metadata['verified_inputs']} do not match the current inputs "
            f"(v{VERIFICATION_POLICY_VERSION} {(expected_inputs or '').lower()}). The expected value is a "
            "fingerprint computed from the PR file list and patches, not the commit or tree SHA. "
            "Evidence that matches no supported policy requires renewed verification: rerun the "
            "update so the gate reports the expected fingerprint, or reuse the fingerprint "
            "recorded by the last gate run."
        )
    if not metadata.get("environment"):
        raise ValueError("Docker verification environment is missing")
    if expected_environment is not None and metadata.get("environment") != expected_environment:
        raise ValueError(
            f"Docker environment {metadata.get('environment')} does not match current environment {expected_environment}"
        )
    if metadata["mode"] == VERIFICATION_MODE_FULL:
        if expected_tree and metadata["tested_tree"] != expected_tree.lower():
            raise ValueError(
                f"full Docker verification tested tree {metadata['tested_tree']} does not match current tree "
                f"{expected_tree.lower()}. The tested tree must equal the PR head tree; re-run both "
                "platform verifications against that tree."
            )
    elif metadata["mode"] == VERIFICATION_MODE_REUSED:
        if not allow_reuse:
            raise ValueError("reused-environment verification is only valid when prior evidence is available")
    else:
        raise ValueError(f"unsupported Docker verification mode: {metadata['mode']}")
    return metadata


def prepare_update_verification(
    description: str,
    previous_description: str,
    expected_inputs: str,
    current_tree: str,
) -> tuple[str, dict]:
    """Carry forward reusable evidence or require a fresh full verification.

    The description supplied by the Agent is treated as a *draft*: when it lacks a
    current Docker verification block but the previous PR description carries reusable
    evidence with the same inputs, the workflow automatically inserts a
    ``reused-environment`` block so reviewers see explicit provenance instead of a
    silent gap. A new ``full`` block is only required when the inputs changed or no
    reusable evidence exists.

    A recorded hash counts as reusable only when it equals the current
    policy-v2 fingerprint; v1-era evidence does not verify and triggers the
    renewed-verification path below.
    """
    current = extract_verification_metadata(description)
    try:
        previous = extract_verification_metadata(previous_description)
    except ValueError:
        previous = None

    def inputs_current(metadata: dict | None) -> bool:
        return bool(metadata) and metadata.get("verified_inputs") == (expected_inputs or "").lower()

    if current and current["mode"] == VERIFICATION_MODE_FULL:
        if not inputs_current(current):
            raise ValueError(
                f"full Docker verification inputs {current.get('verified_inputs')} do not match the "
                f"current PR inputs (v{VERIFICATION_POLICY_VERSION} {(expected_inputs or '').lower()}). "
                "The expected inputs fingerprint is computed from the PR file list and patches; take it "
                "from the previous gate error or ask the maintainer workflow to print it, then rerun "
                "both platform verifications."
            )
        if current.get("tested_tree") != current_tree.lower():
            raise ValueError(
                f"full Docker verification tested tree {current.get('tested_tree')} does not match the "
                f"current PR tree {current_tree.lower()}. Re-run both platform verifications against "
                "the PR head tree."
            )
        if not current.get("environment"):
            raise ValueError("full Docker verification environment is missing")
        validate_verification_metadata(description, expected_inputs, current_tree)
        return upsert_verification_metadata(
            description,
            VERIFICATION_MODE_FULL,
            current["verified_inputs"],
            current_tree,
            current["environment"],
        ), dict(current)

    previous_reusable = bool(
        previous
        and previous.get("mode") in {VERIFICATION_MODE_FULL, VERIFICATION_MODE_REUSED}
        and inputs_current(previous)
        and previous.get("tested_tree")
        and previous.get("environment")
    )

    if current and current["mode"] == VERIFICATION_MODE_REUSED:
        if not previous_reusable:
            raise ValueError("reused-environment verification requires previously recorded Docker evidence")
        if current.get("tested_tree") != previous["tested_tree"]:
            raise ValueError("reused-environment tested tree must match the previously verified tree")
        if not current.get("environment"):
            raise ValueError("reused-environment verification environment is missing")
        if not inputs_current(current):
            raise ValueError(
                "reused-environment verification inputs do not match the current PR inputs; "
                "renewed verification is required"
            )
        validate_verification_metadata(
            description,
            expected_inputs,
            current_tree,
            allow_reuse=True,
            expected_environment=previous["environment"],
        )
        return upsert_verification_metadata(
            description,
            VERIFICATION_MODE_REUSED,
            current["verified_inputs"],
            current["tested_tree"],
            current["environment"],
        ), dict(current)

    if previous_reusable:
        mode = VERIFICATION_MODE_FULL if previous["tested_tree"] == current_tree else VERIFICATION_MODE_REUSED
        tested_tree = current_tree if mode == VERIFICATION_MODE_FULL else previous["tested_tree"]
        metadata = {
            "mode": mode,
            "verified_inputs": previous["verified_inputs"],
            "tested_tree": tested_tree,
            "environment": previous["environment"],
        }
        return upsert_verification_metadata(
            description,
            mode,
            previous["verified_inputs"],
            tested_tree,
            previous["environment"],
        ), metadata

    raise ValueError(
        "a full Docker verification is required; no reusable evidence matches the current inputs. "
        "Run both platform verifications (Ubuntu + openEuler) against the PR head tree, then set "
        f"**{VERIFIED_INPUTS_LABEL}:** to {expected_inputs.lower()} — the fingerprint of the PR file "
        "list and patches, NOT the commit or tree SHA — and **Tested source tree:** to the head tree."
    )
