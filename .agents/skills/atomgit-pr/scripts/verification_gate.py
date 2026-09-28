import hashlib
import json
import os
import re
import subprocess
import tempfile
from urllib.parse import urlparse

VERIFIED_TREE_LABEL = "Verified tree"
VERIFICATION_MODE_LABEL = "Docker verification mode"
VERIFIED_INPUTS_LABEL = "Verified inputs"
TESTED_SOURCE_TREE_LABEL = "Tested source tree"
VERIFICATION_ENVIRONMENT_LABEL = "Docker environment"
VERIFICATION_MODE_FULL = "full"
VERIFICATION_MODE_REUSED = "reused-environment"
# Fingerprint policies (spec: verification fingerprints remain transport
# compatible; reviews/round-3.md R3-01 option (b)). Both policies hash the
# same dependency-relevant records — non-dependency files never enter the
# fingerprint, so unrelated changes cannot invalidate recorded evidence.
# The payload's policy field is the only difference: v1 is the fixed legacy
# calculator kept byte-identical for migration judgment, and new evidence
# uses v2 to mark that it was computed over provably complete inputs (see
# verification_inputs_placeholder_tainted). Complete-input recomputation
# under both policies is how a legacy record is recognized and reused
# without rewriting its hash.
VERIFICATION_POLICY_VERSION = "2"
LEGACY_VERIFICATION_POLICY_VERSION = "1"
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
_VERIFIED_TREE_RE = re.compile(
    rf"\*\*{VERIFIED_TREE_LABEL}:\*\*\s*`?([0-9a-f]{{40}})(?![0-9a-f])`?",
    re.IGNORECASE,
)
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
    patch would break byte-compatibility with the legacy v1 calculator,
    which extracted dependency lines from them. Empty patches are handled
    by the placeholder logic.
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

    v2 hashes the same dependency records as the legacy v1 calculator —
    non-dependency files never enter it (task 3.2 review reverted the
    file-list extension); the payload's policy field is the only
    difference, marking that this hash was computed over provably complete
    inputs. Returns None only when no gate-relevant record exists (the
    gate-trigger decision is unchanged from v1). Missing or truncated
    patches stay gate-relevant through the ``<patch-unavailable>``
    placeholder record, but callers must refuse to reuse fingerprint-based
    evidence while any record is a placeholder
    (:func:`verification_inputs_placeholder_tainted`).
    """
    records = _collect_verification_inputs(files)
    if not records:
        return None
    return _fingerprint(VERIFICATION_POLICY_VERSION, records)


def compute_legacy_verification_inputs(files: list[dict]) -> str | None:
    """Fingerprint under the fixed legacy policy v1 (R3-01 option (b)).

    Byte-identical to the pre-migration calculator so a complete legacy
    four-field record can be recognized by recomputation and reused as v1
    evidence — without rewriting its hash as v2 or claiming a new
    verification happened.
    """
    records = _collect_verification_inputs(files)
    if not records:
        return None
    return _fingerprint(LEGACY_VERIFICATION_POLICY_VERSION, records)


def verification_inputs_placeholder_tainted(files: list[dict]) -> bool:
    """True when a gate-relevant input is missing or not provably complete.

    Missing patches and non-empty-but-truncated patches (entry truncation
    flags such as ``patch_truncated``, or hunk line counts that do not add
    up) both count (task 3.2 review). Such inputs keep the gate triggered
    (conservative) but must not be used for evidence reuse: the spec
    requires renewed verification from complete inputs instead of automatic
    reuse.
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


def detect_verification_policy(
    recorded_inputs: str | None,
    expected_inputs: str | None,
    legacy_inputs: str | None,
) -> str | None:
    """Unique policy whose recomputed fingerprint equals the recorded hash.

    R3-01 option (b): for a legacy four-field record with no policy field,
    complete inputs are recomputed under both supported policies; only a
    unique match may enter the original tree/mode/environment checks. None
    (no match or ambiguous) means the evidence cannot be reused and renewed
    verification is required.
    """
    if not recorded_inputs:
        return None
    recorded = recorded_inputs.lower()
    matches: list[str] = []
    if expected_inputs is not None and recorded == expected_inputs.lower():
        matches.append(VERIFICATION_POLICY_VERSION)
    if legacy_inputs is not None and recorded == legacy_inputs.lower():
        matches.append(LEGACY_VERIFICATION_POLICY_VERSION)
    return matches[0] if len(matches) == 1 else None


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


def extract_verified_tree(description: str) -> str | None:
    matches = [match.lower() for match in _VERIFIED_TREE_RE.findall(description or "")]
    if len(matches) != 1:
        return None
    return matches[0]


def extract_verification_metadata(description: str) -> dict | None:
    if len(_VERIFICATION_HEADING_RE.findall(description or "")) > 1:
        raise ValueError("description must contain exactly one '## Docker Verification' heading")
    mode_matches = _VERIFICATION_MODE_RE.findall(description or "")
    input_matches = _VERIFICATION_SHA_FIELDS[VERIFIED_INPUTS_LABEL].findall(description or "")
    tree_matches = _VERIFICATION_SHA_FIELDS[TESTED_SOURCE_TREE_LABEL].findall(description or "")
    environment_matches = _VERIFICATION_ENVIRONMENT_RE.findall(description or "")
    if not mode_matches and not input_matches and not tree_matches:
        legacy_tree = extract_verified_tree(description)
        return {"mode": "legacy", "tested_tree": legacy_tree} if legacy_tree else None
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
    legacy_inputs: str | None = None,
) -> dict:
    """Validate Docker verification metadata.

    Fingerprints are policy-versioned (R3-01 option (b)). The recorded inputs
    must uniquely match one of the recomputed supported fingerprints — the
    current v2, plus the fixed legacy v1 when the caller supplies it — and the
    detected policy is reported as ``detected_policy`` without rewriting the
    recorded hash. A matched legacy v1 record still passes through every
    original tree/mode/environment check; no match at all means the evidence
    requires renewed verification.

    For reused-environment mode, prior-evidence existence is verified by the
    author-side ``prepare_update_verification``; the reviewer only checks that
    inputs fingerprint and environment match.
    """
    metadata = extract_verification_metadata(description)
    if metadata is None:
        raise ValueError("Docker verification metadata is missing")
    if metadata["mode"] == "legacy":
        if metadata.get("tested_tree", "").lower() != expected_tree.lower():
            raise ValueError("legacy Docker verification only matches the current source tree")
        raise ValueError("legacy Docker evidence must be migrated to full verification metadata")
    detected_policy = detect_verification_policy(metadata["verified_inputs"], expected_inputs, legacy_inputs)
    if detected_policy is None:
        expected_values = f"v{VERIFICATION_POLICY_VERSION} {expected_inputs.lower()}"
        if legacy_inputs is not None:
            expected_values += (
                f" or legacy v{LEGACY_VERIFICATION_POLICY_VERSION} {legacy_inputs.lower()}" if legacy_inputs else ""
            )
        raise ValueError(
            f"Docker verification inputs {metadata['verified_inputs']} do not match the current inputs "
            f"({expected_values}). The expected value is a fingerprint computed from the PR "
            "file list and patches, not the commit or tree SHA. Evidence that matches no supported "
            "policy requires renewed verification: rerun the update so the gate reports the expected "
            "fingerprint, or reuse the fingerprint recorded by the last gate run."
        )
    metadata["detected_policy"] = detected_policy
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
    *,
    legacy_inputs: str | None = None,
) -> tuple[str, dict]:
    """Carry forward reusable evidence or require a fresh full verification.

    The description supplied by the Agent is treated as a *draft*: when it lacks a
    current Docker verification block but the previous PR description carries reusable
    evidence with the same inputs, the workflow automatically inserts a
    ``reused-environment`` block so reviewers see explicit provenance instead of a
    silent gap. A new ``full`` block is only required when the inputs changed or no
    reusable evidence exists.

    Fingerprint comparisons use R3-01 option (b): a recorded hash uniquely
    matching either the current v2 or the fixed legacy v1 recomputation counts
    as reusable, and the carried-forward block keeps the recorded hash — a
    matched v1 record stays v1 evidence and is never rewritten as v2.
    """
    current = extract_verification_metadata(description)
    try:
        previous = extract_verification_metadata(previous_description)
    except ValueError:
        previous = None

    def recorded_policy(metadata: dict | None) -> str | None:
        if not metadata:
            return None
        return detect_verification_policy(metadata.get("verified_inputs"), expected_inputs, legacy_inputs)

    if current and current["mode"] == VERIFICATION_MODE_FULL:
        current_policy = recorded_policy(current)
        if current_policy is None:
            expected_values = f"v{VERIFICATION_POLICY_VERSION} {expected_inputs.lower()}"
            if legacy_inputs:
                expected_values += f" or legacy v{LEGACY_VERIFICATION_POLICY_VERSION} {legacy_inputs.lower()}"
            raise ValueError(
                f"full Docker verification inputs {current.get('verified_inputs')} do not match the "
                f"current PR inputs ({expected_values}). The expected inputs fingerprint is "
                "computed from the PR file list and patches; take it from the previous gate error or "
                "ask the maintainer workflow to print it, then rerun both platform verifications."
            )
        if current.get("tested_tree") != current_tree.lower():
            raise ValueError(
                f"full Docker verification tested tree {current.get('tested_tree')} does not match the "
                f"current PR tree {current_tree.lower()}. Re-run both platform verifications against "
                "the PR head tree."
            )
        if not current.get("environment"):
            raise ValueError("full Docker verification environment is missing")
        validate_verification_metadata(description, expected_inputs, current_tree, legacy_inputs=legacy_inputs)
        # Keep the recorded hash: a v1 match must stay v1 evidence; a v2 match is
        # already the current fingerprint. Upsert only normalizes formatting.
        return upsert_verification_metadata(
            description,
            VERIFICATION_MODE_FULL,
            current["verified_inputs"],
            current_tree,
            current["environment"],
        ), {**current, "detected_policy": current_policy}

    previous_policy = recorded_policy(previous)
    previous_reusable = bool(
        previous
        and previous.get("mode") in {VERIFICATION_MODE_FULL, VERIFICATION_MODE_REUSED}
        and previous_policy is not None
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
        current_policy = recorded_policy(current)
        if current_policy is None:
            raise ValueError(
                "reused-environment verification inputs do not match the current PR inputs under any "
                "supported fingerprint policy; renewed verification is required"
            )
        validate_verification_metadata(
            description,
            expected_inputs,
            current_tree,
            allow_reuse=True,
            legacy_inputs=legacy_inputs,
        )
        return upsert_verification_metadata(
            description,
            VERIFICATION_MODE_REUSED,
            current["verified_inputs"],
            current["tested_tree"],
            current["environment"],
        ), {**current, "detected_policy": current_policy}

    if previous_reusable:
        mode = VERIFICATION_MODE_FULL if previous["tested_tree"] == current_tree else VERIFICATION_MODE_REUSED
        tested_tree = current_tree if mode == VERIFICATION_MODE_FULL else previous["tested_tree"]
        metadata = {
            "mode": mode,
            "verified_inputs": previous["verified_inputs"],
            "tested_tree": tested_tree,
            "environment": previous["environment"],
            "detected_policy": previous_policy,
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


def _run_git(args: list[str], cwd: str) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_pr_head_tree(pr: dict) -> str:
    head = pr.get("head") or {}
    head_sha = (head.get("sha") or "").lower()
    head_ref = head.get("ref") or ""
    repo_url = (head.get("repo") or {}).get("html_url") or ""
    parsed_url = urlparse(repo_url)
    if len(head_sha) != 40 or not head_ref or parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("PR head commit, branch, or HTTPS repository URL is unavailable")

    with tempfile.TemporaryDirectory(prefix="ibrobot-pr-head-") as git_dir:
        _run_git(["init", "--bare"], git_dir)
        _run_git(["fetch", "--no-tags", "--depth=1", repo_url, f"refs/heads/{head_ref}"], git_dir)
        fetched_sha = _run_git(["rev-parse", "FETCH_HEAD"], git_dir).lower()
        if fetched_sha != head_sha:
            raise ValueError(f"PR head changed during tree resolution: API={head_sha}, fetched={fetched_sha}")
        return _run_git(["rev-parse", "FETCH_HEAD^{tree}"], git_dir).lower()
