from types import SimpleNamespace
from typing import Any, cast

import pytest

from manipulation_execution.pick_executor_models import FlowState
from manipulation_execution.pick_executor_node import PickCancelled, PickExecutorNode, PickFlowError


def _plan():
    return SimpleNamespace(
        grasp=(0.12, -0.20, 0.05),
        lift=(0.12, -0.20, 0.10),
        quaternion=(0.0, 0.0, 0.0, 1.0),
        target_contact_ee=(0.0, 0.0, -0.08),
    )


def _harness(*, release_after_success: bool, release_drop_height_m: float):
    moves: list[tuple[tuple[float, float, float], float, bool]] = []
    primitives: list[tuple[str, dict[str, object]]] = []
    sleeps: list[float] = []
    harness = SimpleNamespace(
        _config={
            "release_velocity_scaling": 0.05,
            "release_settle_sec": 0.5,
        },
        _gripper_open=1.0,
        _pose=PickExecutorNode._pose,
        _publish_feedback=lambda *_args, **_kwargs: None,
        get_logger=lambda: SimpleNamespace(info=lambda *_a, **_k: None),
    )
    harness._move_branch_locked_pose = (
        lambda _goal, _deadline, _task, xyz, _quat, velocity, seed, validate_orientation=True: (
            moves.append((xyz, velocity, validate_orientation)),
            SimpleNamespace(joint_state=seed),
        )[1]
    )
    harness._run_primitive = lambda _goal, _deadline, _task, name, **kwargs: primitives.append((name, kwargs))
    harness._sleep_with_cancel = lambda _goal, _deadline, duration: sleeps.append(duration)
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            release_after_success=release_after_success,
            release_drop_height_m=release_drop_height_m,
        )
    )
    return harness, goal_handle, moves, primitives, sleeps


def test_release_lowers_target_then_opens_gripper():
    harness, goal_handle, moves, primitives, sleeps = _harness(
        release_after_success=True,
        release_drop_height_m=0.015,
    )
    state = FlowState(completed_phases=[])

    PickExecutorNode._release_after_success(
        cast(Any, harness),
        goal_handle,
        0.0,
        state,
        "task-1",
        _plan(),
        None,
    )

    # grasp z 0.05 + 0.015 drop height stays below the 0.10 lift, so descend first.
    assert [xyz for xyz, _, _ in moves] == [(0.12, -0.20, 0.065)]
    assert moves[0][1] == 0.05
    assert moves[0][2] is False
    assert [name for name, _ in primitives] == ["open_gripper"]
    assert primitives[0][1] == {"gripper_position": 1.0}
    assert sleeps == [0.5]
    assert state.released_after_success is True


def test_release_skips_descent_when_drop_height_is_at_or_above_lift():
    harness, goal_handle, moves, primitives, _ = _harness(
        release_after_success=True,
        release_drop_height_m=0.20,
    )
    state = FlowState(completed_phases=[])

    PickExecutorNode._release_after_success(
        cast(Any, harness),
        goal_handle,
        0.0,
        state,
        "task-1",
        _plan(),
        None,
    )

    assert moves == []
    assert [name for name, _ in primitives] == ["open_gripper"]
    assert state.released_after_success is True


def test_release_is_skipped_when_goal_does_not_request_it():
    harness, goal_handle, moves, primitives, sleeps = _harness(
        release_after_success=False,
        release_drop_height_m=0.015,
    )
    state = FlowState(completed_phases=[])

    PickExecutorNode._release_after_success(
        cast(Any, harness),
        goal_handle,
        0.0,
        state,
        "task-1",
        _plan(),
        None,
    )

    assert moves == []
    assert primitives == []
    assert sleeps == []
    assert state.released_after_success is False


def test_negative_drop_height_releases_at_the_lift_pose():
    harness, goal_handle, moves, primitives, _ = _harness(
        release_after_success=True,
        release_drop_height_m=-1.0,
    )
    state = FlowState(completed_phases=[])

    PickExecutorNode._release_after_success(
        cast(Any, harness),
        goal_handle,
        0.0,
        state,
        "task-1",
        _plan(),
        None,
    )

    assert moves == []
    assert [name for name, _ in primitives] == ["open_gripper"]
    assert state.released_after_success is True


@pytest.mark.parametrize("failure_stage", ["descent", "open"])
def test_release_failure_reports_explicit_error_and_runs_cleanup(failure_stage: str):
    logger = SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None)
    cleanup_calls = []
    harness = SimpleNamespace(get_logger=lambda: logger)

    def release(*_args):
        raise PickFlowError("RPC_TIMEOUT" if failure_stage == "descent" else "PRIMITIVE_FAILED", failure_stage)

    harness._release_after_success = release
    harness._recover_after_release_failure = lambda *args: cleanup_calls.append(args)
    state = FlowState(completed_phases=[])

    with pytest.raises(PickFlowError) as exc_info:
        PickExecutorNode._release_after_success_with_recovery(
            cast(Any, harness),
            SimpleNamespace(),
            10.0,
            state,
            "task-1",
            _plan(),
            None,
        )

    assert exc_info.value.code == "RELEASE_FAILED"
    assert str(exc_info.value) == failure_stage
    assert len(cleanup_calls) == 1
    assert state.pipeline_timings["subphase_recovery"] >= 0.0


def test_release_cleanup_failure_does_not_mask_original_failure():
    logger = SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None)
    harness = SimpleNamespace(get_logger=lambda: logger)
    harness._release_after_success = lambda *_args: (_ for _ in ()).throw(
        PickFlowError("PRIMITIVE_FAILED", "open failed")
    )
    harness._recover_after_release_failure = lambda *_args: (_ for _ in ()).throw(RuntimeError("cleanup crashed"))

    with pytest.raises(PickFlowError) as exc_info:
        PickExecutorNode._release_after_success_with_recovery(
            cast(Any, harness),
            SimpleNamespace(),
            10.0,
            FlowState(completed_phases=[]),
            "task-1",
            _plan(),
            None,
        )

    assert exc_info.value.code == "RELEASE_FAILED"
    assert str(exc_info.value) == "open failed"


def test_release_cancellation_is_not_rewritten_as_release_failure():
    logger = SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None)
    harness = SimpleNamespace(get_logger=lambda: logger)
    harness._release_after_success = lambda *_args: (_ for _ in ()).throw(PickCancelled())
    harness._recover_after_release_failure = lambda *_args: pytest.fail("cancellation must use the normal cancel path")

    with pytest.raises(PickCancelled):
        PickExecutorNode._release_after_success_with_recovery(
            cast(Any, harness),
            SimpleNamespace(),
            10.0,
            FlowState(completed_phases=[]),
            "task-1",
            _plan(),
            None,
        )
