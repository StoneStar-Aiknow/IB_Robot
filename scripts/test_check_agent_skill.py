import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("check_agent_skill.py")
CANONICAL_SKILL = SCRIPT_PATH.parents[1] / ".agents" / "skills" / "ibrobot-control" / "SKILL.md"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_agent_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def valid_skill(tmp_path):
    skill_path = tmp_path / "ibrobot-control" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_bytes(CANONICAL_SKILL.read_bytes())
    return skill_path


def test_canonical_skill_contract_passes():
    assert _load_checker().validate_skill(CANONICAL_SKILL) == []


def test_checker_requires_confirmation_and_cancel_truthfulness(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("explicit user\n   motion confirmation", "operator approval")
    content = content.replace("explicit user motion confirmation", "operator approval")
    content = content.replace("Cancellation requested is not robot stopped", "Cancellation is complete")
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing explicit user motion confirmation requirement" in errors
    assert "missing cancellation truthfulness requirement" in errors


def test_checker_requires_plan_command_order_and_prohibitions(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    status = "1. Query the Gateway: `robot-skill status`."
    listing = "2. Discover capabilities: `robot-skill list-skills`."
    content = content.replace(status, "TEMP").replace(listing, status).replace("TEMP", listing)
    content = content.replace("must not enable motion authorization", "may enable motion authorization")
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "robot-skill workflow commands are missing or out of order" in errors
    assert "missing prohibition: enable motion authorization" in errors


def test_checker_requires_exact_display_before_confirmation(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("exact ordered steps", "summary")
    content = content.replace("explicit user\n   motion confirmation", "operator approval", 1)
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "workflow must display the exact plan, digest, registry identity, and fresh task ID" in errors
    assert "explicit user motion confirmation must appear between validate-plan and confirm-plan" in errors


def test_checker_requires_plan_cancel_and_routing(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("cancel-plan", "cancel")
    content = content.replace(
        "Natural-language single-Skill and Workflow requests both use the plan workflow above.",
        "Natural language is routed by the model.",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing Agent plan cancel command" in errors
    assert "missing natural-language single-Skill/Workflow routing rule" in errors


def test_checker_requires_flat_workflow_steps_and_terminal_nonzero_exit(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("Skill arguments are top-level fields", "Skill arguments may be nested")
    content = content.replace(
        "issue another `robot-skill` command in the same user request",
        "continue diagnosing with robot-skill commands",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing flat WorkflowStep argument rule" in errors
    assert "missing prohibition: follow-up robot-skill command after nonzero exit" in errors


def test_checker_requires_gateway_task_budget_default(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace("By default, omit `--timeout-sec` from both commands", "Choose a timeout")
    content = content.replace(
        "Do not derive a plan\nbudget from `default_skill_timeout_sec`",
        "Use the default skill timeout as the plan budget",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing default task-budget omission rule" in errors
    assert "missing skill-timeout/budget distinction rule" in errors


def test_checker_requires_direct_ids_and_distinguishes_command_approval(valid_skill):
    checker = _load_checker()
    content = valid_skill.read_text(encoding="utf-8")
    content = content.replace(
        "Construct request IDs and task IDs directly in the conversation and `robot-skill` arguments.",
        "Generate identifiers with a helper command.",
    )
    content = content.replace(
        "must not call Python, `uuidgen`, `date`, a shell, or another helper tool to generate request/task IDs",
        "may call a helper tool to generate request/task IDs",
    )
    content = content.replace(
        "A command approval, including session-wide\napproval, authorizes only that command and is not user motion "
        "confirmation.",
        "A command approval also confirms motion.",
    )
    valid_skill.write_text(content, encoding="utf-8")

    errors = checker.validate_skill(valid_skill)

    assert "missing direct request/task ID construction rule" in errors
    assert "missing prohibition: helper command ID generation" in errors
    assert "missing command-approval distinction rule" in errors
    assert "missing prohibition: command approval as motion confirmation" in errors
