from types import SimpleNamespace

import pytest
from test_banana_handeye_pick import BananaHandeyePickClient, PickExecutionError, TaskStep


class RecoveryHarness:
    def __init__(self, results: list[bool], *, skip_observe: bool = False):
        self.args = SimpleNamespace(
            lift_speed=0.05,
            open_settle_s=0.3,
            skip_observe=skip_observe,
            observe_x=0.1,
            observe_y=-0.16,
            observe_z=0.22,
            observe_speed=0.04,
            observe_settle_s=0.6,
        )
        self.current_execution_candidate_index = 31
        self.results = list(results)
        self.calls: list[tuple[str, list, float | None]] = []

    def run_task(self, task_id: str, description: str, steps: list, timeout_s: float | None = None) -> bool:
        del description
        self.calls.append((task_id, steps, timeout_s))
        return self.results.pop(0)


def run_recovery(harness: RecoveryHarness) -> None:
    BananaHandeyePickClient.recover_after_close_failure(
        harness,
        task_id="marker_pick",
        task_desc="pick marker",
        grasp=(0.12, -0.20, 0.05),
        pregrasp=(0.11, -0.18, 0.10),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def test_close_failure_recovery_retracts_vertically_before_opening() -> None:
    harness = RecoveryHarness([True, True])

    run_recovery(harness)

    assert len(harness.calls) == 2
    _, retreat_steps, retreat_timeout = harness.calls[0]
    assert retreat_timeout is None
    assert len(retreat_steps) == 1
    retreat = retreat_steps[0]
    assert retreat.type == TaskStep.MOVE_TO_POSE
    assert retreat.label == "retreat_closed_gripper_to_pregrasp"
    assert retreat.target_pose.position.x == pytest.approx(0.12)
    assert retreat.target_pose.position.y == pytest.approx(-0.20)
    assert retreat.target_pose.position.z == pytest.approx(0.10)

    _, reset_steps, reset_timeout = harness.calls[1]
    assert reset_timeout == pytest.approx(90.0)
    assert [step.type for step in reset_steps] == [
        TaskStep.GRIPPER,
        TaskStep.WAIT,
        TaskStep.MOVE_TO_POSE,
        TaskStep.WAIT,
    ]
    assert reset_steps[0].gripper_position == pytest.approx(1.0)
    assert reset_steps[2].target_pose.position.x == pytest.approx(0.1)
    assert reset_steps[2].target_pose.position.y == pytest.approx(-0.16)
    assert reset_steps[2].target_pose.position.z == pytest.approx(0.22)


def test_close_failure_recovery_stops_when_retreat_fails() -> None:
    harness = RecoveryHarness([False])

    with pytest.raises(PickExecutionError, match="could not retract") as exc_info:
        run_recovery(harness)

    assert exc_info.value.phase == "recover_close_retreat"
    assert exc_info.value.retryable is False
    assert len(harness.calls) == 1


def test_close_failure_recovery_respects_skip_observe() -> None:
    harness = RecoveryHarness([True, True], skip_observe=True)

    run_recovery(harness)

    _, reset_steps, _ = harness.calls[1]
    assert [step.type for step in reset_steps] == [TaskStep.GRIPPER, TaskStep.WAIT]
