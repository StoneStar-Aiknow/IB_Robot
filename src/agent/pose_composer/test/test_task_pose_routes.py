"""Bounded convergence and routed execution using real candidate construction, no transport."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.fixed_pose_backend import JOINTS, MAX_SPEED, FixedBackend

from pose_composer.errors import ComposeError
from pose_composer.task_poses import TaskPoseRunner
from robot_runtime.task_poses import normalize_task_poses


def backend():
    b = FixedBackend.__new__(FixedBackend)
    b.hard_limits = dict.fromkeys(JOINTS, (-3.0, 3.0))
    b.model = b.safety = b
    b.guard = lambda: None
    b.check_pose = lambda _: SimpleNamespace(ok=True)
    return b


def snapshot(*, route=True):
    zero = dict.fromkeys(JOINTS, 0.0)
    pose = {
        "joint_positions": dict.fromkeys(JOINTS, 0.3),
        "position_tolerance_rad": 0.1,
        "hardware_approved": True,
        "timeout_sec": 20.0,
        "arrival_wait_sec": 0.2,
    }
    if route:
        pose["transition_route"] = {
            "waypoints": [dict.fromkeys(JOINTS, -0.2), dict.fromkeys(JOINTS, 0.1)],
            "supported_starts": [{"joint_positions": zero, "tolerance_rad": 0.05}],
            "hardware_approved": False,
            "source": {"path": "recording.json", "sha256": "a" * 64, "sample_indices": [10, 20]},
        }
    return normalize_task_poses(
        {"schema_version": 1, "robot_type": "x2", "joint_names": list(JOINTS), "poses": {"grasp_ready": pose}}
    )


def execution(*, route=True, simulation=True, start=0.0, follow=True, read_hook=None, cancel=None, wait=0.2):
    b, now, moves = backend(), [10.0], []
    q = dict.fromkeys(JOINTS, start)
    cfg = snapshot(route=route)
    cfg.pop("digest")
    cfg["poses"]["grasp_ready"]["arrival_wait_sec"] = wait
    cfg = normalize_task_poses(cfg)
    runner = TaskPoseRunner(
        cfg, backend=b, simulation=simulation, clock=lambda: now[0], pause=lambda t: now.__setitem__(0, now[0] + t)
    )

    def read():
        if read_hook:
            read_hook(q, now[0], bool(moves))
        return dict(q), dict.fromkeys(JOINTS, now[0])

    def move(plan, deadline):
        assert deadline > now[0]
        moves.append(plan)
        if follow:
            q.update(zip(JOINTS, plan.traj.positions[-1], strict=True))
        return True, "", ""

    def run():
        return runner.execute(
            "grasp_ready",
            cfg["digest"],
            read_state=read,
            move=move,
            cancelled=lambda: cancel(now[0]) if cancel else False,
        )

    return runner, moves, run


def test_supported_route_builds_one_combined_minimum_jerk_candidate():
    runner, moves, run = execution()
    assert run() is False
    assert len(moves) == 1
    times, rows = np.asarray(moves[0].traj.times), np.asarray(moves[0].traj.positions)
    assert np.all(np.diff(times) > 0)
    assert np.max(abs(np.diff(rows, axis=0) / np.diff(times)[:, None])) <= MAX_SPEED + 1e-6
    assert any(np.allclose(row, -0.2) for row in rows)
    assert any(np.allclose(row, 0.1) for row in rows)
    assert np.allclose(rows[-1], 0.3)
    assert runner.report["arrival_check"]["max_error_rad"] == 0


@pytest.mark.parametrize(
    "options,code",
    [
        ({"simulation": False}, "TASK_POSE_ROUTE_NOT_APPROVED"),
        ({"start": -0.1}, "TASK_POSE_ROUTE_START_UNSUPPORTED"),
    ],
)
def test_route_never_falls_back_to_old_direct_path(options, code):
    _, moves, run = execution(**options)
    with pytest.raises(ComposeError) as error:
        run()
    assert error.value.code == code
    assert not moves


def test_already_at_target_skips_unapproved_route():
    _, moves, run = execution(simulation=False, start=0.3)
    assert run() is True
    assert not moves


def test_delayed_feedback_converges_without_widening_tolerance():
    def delayed(q, now, moving):
        if moving:
            q.update(dict.fromkeys(JOINTS, 0.3 if now >= 10.1 else 0.15))

    runner, _, run = execution(follow=False, read_hook=delayed)
    run()
    assert 0.1 <= runner.report["arrival_check"]["waited_sec"] <= 0.2
    assert runner.report["arrival_check"]["tolerance_rad"] == 0.1


def test_persistent_error_fails_after_bounded_wait_with_worst_joint():
    runner, _, run = execution(follow=False)
    with pytest.raises(ComposeError) as error:
        run()
    assert error.value.code == "TASK_POSE_NOT_REACHED"
    assert runner.report["arrival_check"]["waited_sec"] == pytest.approx(0.2)
    assert runner.report["arrival_check"]["worst_joint"] in JOINTS
    assert runner.report["arrival_check"]["max_error_rad"] == 0.3


def test_cancel_during_convergence_stops_waiting():
    _, _, run = execution(follow=False, cancel=lambda now: now >= 10.05)
    with pytest.raises(ComposeError) as error:
        run()
    assert error.value.code == "POSE_CANCELLED"


def test_entire_multisegment_path_reaches_existing_geometry_gate():
    b = backend()
    called = []

    def body(_joints, rows, **_):
        called.append(len(rows))
        if np.min(rows[:, 0]) < -0.1:
            raise ValueError("middle route collision")

    b.body_geometry = SimpleNamespace(check_path=body)
    b.geometry = SimpleNamespace(check_arm_path=lambda *_args, **_kwargs: None)
    plan = b.plan_safe(
        dict.fromkeys(JOINTS, 0.0), [dict.fromkeys(JOINTS, -0.2), dict.fromkeys(JOINTS, 0.3)], [0.0, 0.0]
    )
    assert not plan.ok
    assert called[0] > 2
    assert "middle route collision" in plan.report.notes
