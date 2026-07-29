import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("check_agent_skill.py")


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_agent_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def valid_skill(tmp_path):
    skill_path = tmp_path / "ibrobot-control" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text(
        """---
name: ibrobot-control
description: Use when a user asks Hermes to inspect, validate, execute, stop, or cancel IB-Robot motion.
---

# IB-Robot Control

## Required Workflow

Run these commands in order:

1. `robot-skill --config-name NAME status`
2. `robot-skill --config-name NAME list-skills`
3. `robot-skill --config-name NAME describe SKILL`
4. `robot-skill --config-name NAME validate SKILL`
5. Obtain explicit user motion confirmation.
6. `robot-skill --config-name NAME execute SKILL --task-id ID`

Use SIGINT/SIGTERM for the current execute or `robot-skill --config-name NAME cancel --task-id ID` to stop.

MUST NOT launch or restart the pipeline.
MUST NOT enable motion authorization.
MUST NOT modify ROS parameters.
MUST NOT call primitive, MoveIt, controller, or raw ros2 motion commands.
Cancellation requested is not robot stopped.
MUST NOT automatically retry after failure, timeout, or unknown result.
""",
        encoding="utf-8",
    )
    return skill_path


def test_valid_skill_contract_passes(valid_skill):
    checker = _load_checker()

    assert checker.validate_skill(valid_skill) == []


def test_checker_requires_confirmation_and_cancel_truthfulness(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("Obtain explicit user motion confirmation.\n", "")
    content = content.replace("Cancellation requested is not robot stopped.\n", "")
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing explicit user motion confirmation requirement" in errors
    assert "missing cancellation truthfulness requirement" in errors


def test_checker_requires_command_order_and_all_prohibitions(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace(
        "1. `robot-skill --config-name NAME status`\n2. `robot-skill --config-name NAME list-skills`",
        "1. `robot-skill --config-name NAME list-skills`\n2. `robot-skill --config-name NAME status`",
    )
    content = content.replace("MUST NOT enable motion authorization.\n", "")
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "robot-skill workflow commands are missing or out of order" in errors
    assert "missing prohibition: enable motion authorization" in errors


def test_checker_requires_confirmation_between_validate_and_execute(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace(
        "5. Obtain explicit user motion confirmation.\n6. `robot-skill --config-name NAME execute SKILL --task-id ID`",
        "5. `robot-skill --config-name NAME execute SKILL --task-id ID`\n6. Obtain explicit user motion confirmation.",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "explicit user motion confirmation must appear between validate and execute" in errors


def test_checker_requires_commands_inside_required_workflow(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace(
        "## Required Workflow\n\nRun these commands in order:",
        "## Required Workflow\n\nFollow the safety contract.\n\n## Command Appendix\n\nRun these commands in order:",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "required workflow section is missing ordered robot-skill commands" in errors
