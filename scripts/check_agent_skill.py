#!/usr/bin/env python3
"""Validate the repository contract for the ibrobot-control Agent Skill.

This script currently only validates the ``ibrobot-control`` skill: the
frontmatter name, description prefix, ordered ``robot-skill`` workflow,
explicit user motion confirmation placement, cancel ``--task-id`` command,
SIGINT/SIGTERM guidance, and the hardcoded REQUIRED_RULES prohibitions.
When additional Agent skills are added, extend this script to accept a
``--skill-name`` parameter and load per-skill rule sets instead of
duplicating the hardcoded checks below.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

COMMANDS = ("status", "list-skills", "describe", "validate", "execute")
REQUIRED_RULES = {
    "explicit user motion confirmation": "missing explicit user motion confirmation requirement",
    "must not launch or restart the pipeline": "missing prohibition: launch or restart pipeline",
    "must not enable motion authorization": "missing prohibition: enable motion authorization",
    "must not modify ros parameters": "missing prohibition: modify ROS parameters",
    "must not call primitive, moveit, controller, or raw ros2 motion commands": (
        "missing prohibition: primitive, MoveIt, controller, or raw ros2 motion commands"
    ),
    "cancellation requested is not robot stopped": "missing cancellation truthfulness requirement",
    "must not automatically retry after failure, timeout, or unknown result": (
        "missing prohibition: automatic retry after failure, timeout, or unknown result"
    ),
}


def _parse_frontmatter(content: str) -> tuple[dict[str, str], list[str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ["missing YAML frontmatter"]
    try:
        closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return {}, ["unterminated YAML frontmatter"]

    frontmatter_text = "\n".join(lines[: closing_index + 1])
    errors = []
    if len(frontmatter_text) > 1024:
        errors.append("YAML frontmatter exceeds 1024 characters")
    values = {}
    for line in lines[1:closing_index]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            errors.append(f"invalid YAML frontmatter line: {line}")
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values, errors


def _validate_command_order(content: str) -> bool:
    positions = []
    for command in COMMANDS:
        match = re.search(rf"`robot-skill\b[^`\n]*\b{re.escape(command)}\b[^`\n]*`", content)
        if match is None:
            return False
        positions.append(match.start())
    return positions == sorted(positions)


def _required_workflow(content: str) -> str | None:
    match = re.search(r"^## Required Workflow\s*$\n(.*?)(?=^## |\Z)", content, flags=re.MULTILINE | re.DOTALL)
    return None if match is None else match.group(1)


def validate_skill(skill_path: Path) -> list[str]:
    if not skill_path.is_file():
        return [f"skill file does not exist: {skill_path}"]
    content = skill_path.read_text(encoding="utf-8")
    frontmatter, errors = _parse_frontmatter(content)

    name = frontmatter.get("name", "")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        errors.append("frontmatter name must contain only lowercase letters, numbers, and hyphens")
    if name != "ibrobot-control" or skill_path.parent.name != name:
        errors.append("frontmatter name and skill directory must both be ibrobot-control")
    description = frontmatter.get("description", "")
    if not description.startswith("Use when "):
        errors.append("frontmatter description must start with 'Use when '")

    normalized = " ".join(content.lower().split())
    workflow = _required_workflow(content)
    if workflow is None or any(
        re.search(rf"`robot-skill\b[^`\n]*\b{re.escape(command)}\b[^`\n]*`", workflow) is None for command in COMMANDS
    ):
        errors.append("required workflow section is missing ordered robot-skill commands")
    elif not _validate_command_order(workflow):
        errors.append("robot-skill workflow commands are missing or out of order")
    validate_match = None if workflow is None else re.search(r"`robot-skill\b[^`\n]*\bvalidate\b[^`\n]*`", workflow)
    execute_match = None if workflow is None else re.search(r"`robot-skill\b[^`\n]*\bexecute\b[^`\n]*`", workflow)
    if (
        validate_match is not None
        and execute_match is not None
        and workflow.lower().find(
            "explicit user motion confirmation",
            validate_match.end(),
            execute_match.start(),
        )
        == -1
    ):
        errors.append("explicit user motion confirmation must appear between validate and execute")
    if re.search(r"`robot-skill\b[^`\n]*\bcancel\b[^`\n]*--task-id\b[^`\n]*`", content) is None:
        errors.append("missing task-ID cancel command")
    if "sigint/sigterm" not in normalized:
        errors.append("missing current execute signal cancellation guidance")
    for requirement, message in REQUIRED_RULES.items():
        if requirement not in normalized:
            errors.append(message)
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-path", required=True, type=Path)
    args = parser.parse_args(argv)
    errors = validate_skill(args.skill_path)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: ibrobot-control skill contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
