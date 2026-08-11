import pytest

from embodied_common.workflow_contracts import (
    CanonicalWorkflowStep,
    compute_workflow_digest,
    normalize_workflow_steps,
    workflow_digest_preimage,
)


def _budget():
    return {
        "schema_version": 1,
        "started_at": {"sec": 10, "nanosec": 5},
        "deadline": {"sec": 20, "nanosec": 5},
    }


def test_workflow_digest_is_ordered_and_deterministic():
    steps = [
        CanonicalWorkflowStep(1, "open_gripper_skill"),
        CanonicalWorkflowStep(1, "recover_safe_pose"),
    ]
    kwargs = {
        "root_task_id": "task-1",
        "task_budget": _budget(),
        "expected_registry_epoch": "epoch-1",
        "expected_registry_generation": 1,
        "expected_registry_digest": "digest-1",
    }
    digest = compute_workflow_digest(workflow_steps=steps, **kwargs)
    assert digest == compute_workflow_digest(workflow_steps=steps, **kwargs)
    assert digest != compute_workflow_digest(workflow_steps=list(reversed(steps)), **kwargs)


def test_workflow_preimage_matches_frozen_contract():
    preimage = workflow_digest_preimage(
        root_task_id="task-1",
        task_budget=_budget(),
        expected_registry_epoch="epoch-1",
        expected_registry_generation=1,
        expected_registry_digest="digest-1",
        workflow_steps=[CanonicalWorkflowStep(1, "open_gripper_skill")],
    )
    assert set(preimage["task_budget"]) == {"started_at", "deadline"}
    assert set(preimage["workflow_steps"][0]) == {
        "schema_version",
        "skill_name",
        "target_name",
        "place_name",
        "motion_direction",
        "motion_distance",
        "timeout_sec",
    }


def test_workflow_step_normalization_is_typed():
    steps = normalize_workflow_steps(
        [
            {
                "schema_version": 1,
                "skill_name": " move_relative_ee ",
                "motion_direction": "LEFT",
                "motion_distance": 0.05,
            }
        ]
    )
    assert steps[0].to_dict()["skill_name"] == "move_relative_ee"
    assert steps[0].to_dict()["motion_direction"] == "left"


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -0.1])
def test_workflow_step_rejects_invalid_distance(distance):
    with pytest.raises(ValueError):
        CanonicalWorkflowStep(1, "move_relative_ee", motion_distance=distance)


def test_workflow_rejects_more_than_sixteen_steps():
    with pytest.raises(ValueError, match="maximum of 16"):
        normalize_workflow_steps([CanonicalWorkflowStep(1, f"skill-{index}") for index in range(17)])
