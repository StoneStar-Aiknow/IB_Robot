import re

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


def _normalise_patch(patch) -> str:
    if isinstance(patch, dict):
        patch = patch.get("diff") or ""
    return patch or ""


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


def format_reuse_self_check(
    reinvented_workflows: str,
    reused_components: str,
    reinvention_justification: str,
    architecture_conformance: str,
) -> str:
    return (
        f"## {REUSE_SELF_CHECK_HEADING}\n\n"
        f"**{REINVENTED_WORKFLOWS_LABEL}:** {reinvented_workflows}\n"
        f"**{REUSED_COMPONENTS_LABEL}:** {reused_components}\n"
        f"**{REINVENTION_JUSTIFICATION_LABEL}:** {reinvention_justification}\n"
        f"**{ARCHITECTURE_CONFORMANCE_LABEL}:** {architecture_conformance}\n"
    )


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


def reuse_self_check_status(description: str) -> str:
    """Classify the Reuse Self-Check section as absent/invalid/incomplete/complete."""
    try:
        fields = extract_reuse_self_check(description)
    except ValueError:
        return "invalid"
    if fields is None:
        return "absent"
    if missing_reuse_fields(fields):
        return "incomplete"
    return "complete"
