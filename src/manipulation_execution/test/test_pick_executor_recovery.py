from types import SimpleNamespace
from typing import Any, cast

import pytest

from manipulation_execution.pick_executor_models import FlowState
from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


def _prepared_candidate():
    return SimpleNamespace(
        plan=SimpleNamespace(
            grasp=(0.12, -0.20, 0.05),
            quaternion=(0.0, 0.0, 0.0, 1.0),
        )
    )


def _harness(*, observe_pose: str = "observe_table"):
    calls: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(
        _config={
            "observe_pose": observe_pose,
            "lift_velocity_scaling": 0.08,
            "observe_velocity_scaling": 0.40,
        },
        _gripper_open=1.0,
        _pose=PickExecutorNode._pose,
    )

    def run_primitive(_goal, _deadline, _task_id, primitive_name, **kwargs):
        calls.append((primitive_name, kwargs))

    harness._run_primitive = run_primitive
    return harness, calls


def test_close_failure_recovery_retracts_before_opening_and_observing():
    harness, calls = _harness()

    PickExecutorNode._recover_after_close_failure(
        cast(Any, harness),
        None,
        10.0,
        "pick-test",
        _prepared_candidate(),
        (0.11, -0.18, 0.10),
    )

    assert [name for name, _kwargs in calls] == ["move_to_pose", "open_gripper", "move_to_named_pose"]
    retreat_pose = calls[0][1]["pose"]
    assert retreat_pose.position.x == pytest.approx(0.12)
    assert retreat_pose.position.y == pytest.approx(-0.20)
    assert retreat_pose.position.z == pytest.approx(0.10)
    assert calls[0][1]["velocity_scaling"] == pytest.approx(0.08)
    assert calls[1][1] == {"gripper_position": pytest.approx(1.0)}
    assert calls[2][1] == {"pose_name": "observe_table", "velocity_scaling": pytest.approx(0.40)}


def test_close_failure_recovery_stops_if_closed_gripper_retreat_fails():
    harness, calls = _harness()

    def fail_retreat(_goal, _deadline, _task_id, primitive_name, **_kwargs):
        calls.append((primitive_name, {}))
        raise PickFlowError("PRIMITIVE_FAILED", "retreat failed")

    harness._run_primitive = fail_retreat

    with pytest.raises(PickFlowError, match="retreat failed"):
        PickExecutorNode._recover_after_close_failure(
            cast(Any, harness),
            None,
            10.0,
            "pick-test",
            _prepared_candidate(),
            (0.11, -0.18, 0.10),
        )

    assert calls == [("move_to_pose", {})]


def test_retention_failure_recovery_opens_then_returns_to_observe():
    harness, calls = _harness()

    PickExecutorNode._recover_after_retention_failure(cast(Any, harness), None, 10.0, "pick-test")

    assert [name for name, _kwargs in calls] == ["open_gripper", "move_to_named_pose"]


def test_recovery_skips_observe_when_no_observe_pose_is_configured():
    harness, calls = _harness(observe_pose="")

    PickExecutorNode._recover_after_retention_failure(cast(Any, harness), None, 10.0, "pick-test")

    assert [name for name, _kwargs in calls] == ["open_gripper"]


@pytest.mark.parametrize("failure_stage", ["motion", "verification"])
def test_retention_motion_or_verification_failure_runs_recovery(failure_stage: str):
    recovery_calls = []
    harness = SimpleNamespace(
        _config={"recover_after_retention_failure": True},
        _publish_feedback=lambda *_args: None,
        _recover_after_retention_failure=lambda *args: recovery_calls.append(args),
    )
    payload = SimpleNamespace(joint_state="lifted")

    def move(*_args, **_kwargs):
        if failure_stage == "motion":
            raise PickFlowError("RPC_TIMEOUT", "IK timed out")
        return payload

    def verify(*_args, **_kwargs):
        if failure_stage == "verification":
            raise PickFlowError("GRASP_VERIFICATION_FAILED", "marker slipped")

    harness._move_branch_locked_pose = move
    harness._verify = verify
    prepared = SimpleNamespace(ranked=SimpleNamespace(index=7, candidate="candidate"))
    state = FlowState(completed_phases=[])

    with pytest.raises(PickFlowError):
        PickExecutorNode._move_and_verify_retention(
            cast(Any, harness),
            "goal",
            10.0,
            state,
            "pick-test",
            "marker",
            prepared,
            phase="lift",
            feedback_phase="lift",
            feedback_detail="lifting verified target",
            xyz=(0.12, -0.20, 0.10),
            quaternion=(0.0, 0.0, 0.0, 1.0),
            velocity_scaling=0.08,
            seed="seed",
            validate_orientation=False,
        )

    assert len(recovery_calls) == 1
    assert recovery_calls[0][2] == "pick-test"
    assert state.pipeline_timings["subphase_recovery"] >= 0.0
