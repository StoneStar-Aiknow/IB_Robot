"""The held object described while the arm lifts is the one the handover would compute."""

from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp import held_object
from aimdk_grasp.joints import arm_joints

from manipulation_execution.grasp_pipeline.contracts import ArmSegment, Move, SegmentKind

SIDE = "left"
JOINTS = arm_joints(SIDE)


def _move(name, start, end):
    positions = np.stack([np.full(7, start), np.full(7, end)])
    return Move(ArmSegment(name, SegmentKind.TRAJECTORY, JOINTS, positions, np.array([0.0, 1.0])))


def _plan():
    steps = (_move("approach", 0.0, 0.1), _move("grasp", 0.1, 0.2), _move("lift_carry", 0.2, 0.25))
    return SimpleNamespace(report={"arm_side": SIDE, "carry_lift_m": 0.05}, steps=steps)


def _apple(seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(4000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction[direction[:, 2] < 0.3]  # one-sided, as the camera sees it
    cloud = np.array([0.45, 0.1, 0.05]) + 0.04 * direction
    fringe = np.array([0.45, 0.1, 0.05]) + [0.0, 0.09, 0.0] + rng.normal(scale=0.003, size=(20, 3))
    noise = cloud[:15] + rng.normal(scale=0.03, size=(15, 3))
    return np.vstack([cloud, fringe, noise])


def _parts():
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=7)))
    executor = SimpleNamespace(
        io=SimpleNamespace(feedback=lambda: SimpleNamespace(positions=defaultdict(float))),
        measured=lambda names: np.zeros(len(names)),
    )
    effector = SimpleNamespace(_measured=lambda: np.zeros(10), snapshot=lambda: "{}")
    observation = SimpleNamespace(
        base_from_camera=np.eye(4), robot_state=SimpleNamespace(capture_id="c1"), stamp_ns=1, base_frame="torso_link"
    )
    return node, executor, effector, observation, SimpleNamespace(target_points=_apple())


def _same(a, b):
    assert a.radius_m == b.radius_m and a.points_in_palm == b.points_in_palm
    assert (a.arm_side, a.joint_names, a.joint_positions, a.home_positions) == (
        b.arm_side,
        b.joint_names,
        b.joint_positions,
        b.home_positions,
    )


def test_a_held_object_prepared_at_the_close_equals_the_one_built_at_the_handover(capfd):
    node, executor, effector, observation, proposals = _parts()
    plan = _plan()
    inline = held_object.held_builder(node, executor, effector, background=False)
    expected = inline("an apple", observation, proposals, plan)
    early = held_object.held_builder(node, executor, effector)  # a real worker process
    try:
        early.prepare(observation, proposals, plan)
        capfd.readouterr()
        _same(early("an apple", observation, proposals, plan), expected)
        assert '"prepared": true' in capfd.readouterr().err  # the worker's result was used
        # The fringe and noise were filtered: the radius is the apple's, not the fringe's.
        assert expected.radius_m < 0.07
    finally:
        early.close()


@pytest.mark.parametrize("case", ["worker failed", "another plan"])
def test_without_a_usable_early_result_the_handover_computes_it_inline(monkeypatch, case):
    class BrokenPool:
        def submit(self, *args):
            future = Future()
            future.set_exception(RuntimeError("worker died"))
            return future

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(held_object, "_filter_pool", BrokenPool)
    node, executor, effector, observation, proposals = _parts()
    plan = _plan()
    expected = held_object.held_builder(node, executor, effector, background=False)(
        "an apple", observation, proposals, plan
    )
    build = held_object.held_builder(node, executor, effector)
    build.prepare(observation, proposals, plan if case == "worker failed" else _plan())
    _same(build("an apple", observation, proposals, plan), expected)
