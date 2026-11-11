"""Motion arithmetic and per-command behaviour carried over from the first RoboDojo body (MR !551)."""

from __future__ import annotations

import math

import pytest

from robot_runtime_sim.direct_control.engine import (
    OUTCOME_EXECUTION_FAILED,
    OUTCOME_REJECTED,
    OUTCOME_SUCCEEDED,
    STATE_STOPPING,
    Command,
    MotionConfig,
    SimulationCore,
    motion_ticks,
)
from robot_runtime_sim.direct_control.kinematics import interpolate, nearest_equivalent, shift_pose
from robot_runtime_sim.direct_control.ports import GroupSpec, PlatformResult
from robot_runtime_sim.testing.direct_control_fakes import (
    FAST,
    DualArmCodec,
    DualArmEnv,
    PullPlatform,
    dual_arm_descriptor,
    submit_async,
    wait_for,
)


def rig(descriptor=None, env_cls=DualArmEnv):
    core = SimulationCore(descriptor or dual_arm_descriptor(), FAST)
    env = env_cls(core.descriptor)
    platform = PullPlatform(core, DualArmCodec(core.descriptor), env).start()
    return core, env, platform


def test_interpolate_ends_exactly_on_target():
    path = interpolate([0.0, 0.0], [1.0, -1.0], 4)
    assert len(path) == 4 and path[-1] == [1.0, -1.0] and path[0] == [0.25, -0.25]


def test_motion_ticks_scale_with_distance_velocity_and_dt():
    cfg = MotionConfig(max_joint_speed=1.0, min_motion_ticks=1, max_motion_ticks=1000)
    assert motion_ticks([0.0], [0.4], cfg, 1.0, 0.04) == 10
    assert motion_ticks([0.0], [0.4], cfg, 0.5, 0.04) == 20
    assert motion_ticks([0.0], [0.4], cfg, 1.0, 0.02) == 20  # a shorter tick needs more of them
    assert motion_ticks([0.0], [0.0], MotionConfig(min_motion_ticks=8), 1.0, 0.04) == 8


def test_nearest_equivalent_avoids_a_full_turn():
    assert nearest_equivalent([3.0], [-3.0], [10.0]) == pytest.approx([-3.0 + 2 * math.pi])
    assert nearest_equivalent([3.0], [-3.0], [3.14]) == [-3.0]


def test_motion_config_rejects_unknown_keys():
    assert MotionConfig.from_mapping({"settle_ticks": 2}).settle_ticks == 2
    with pytest.raises(ValueError):
        MotionConfig.from_mapping({"tick_s": 0.04})


def test_pose_ending_far_from_target_is_execution_failed():
    class Blocked(DualArmEnv):
        def step(self, native_action):  # contact: the arm never gets past z = 0.3
            out = super().step(native_action)
            for g in self.q:
                self.q[g][2] = max(self.q[g][2], 0.3)
            return {**out, **self.native()}

    core, _env, platform = rig(env_cls=Blocked)
    result = core.submit(Command(kind="pose", group="a", pose=[0.1, 0.1, 0.1, 1, 0, 0, 0]), 5.0)
    assert result["outcome"] == OUTCOME_EXECUTION_FAILED and "from the target" in result["message"]
    platform.join()


def test_ik_only_request_returns_joints_without_ticks():
    core, env, platform = rig()
    result = core.submit(Command(kind="ik", group="a", pose=[0.2, 0.2, 0.2, 1, 0, 0, 0]), 5.0)
    assert result["outcome"] == OUTCOME_SUCCEEDED and result["joints"][:3] == [0.2, 0.2, 0.2]
    assert env.tick == 0
    platform.join()


def test_gripper_moves_only_its_group_and_one_at_target_costs_no_ticks():
    core, env, platform = rig()
    result = core.submit(Command(kind="gripper", group="b", opening=0.0), 5.0)
    assert result["outcome"] == OUTCOME_SUCCEEDED and env.grip == {"a": 1.0, "b": 0.0}
    ticks = env.tick
    again = core.submit(Command(kind="gripper", group="b", opening=0.0), 5.0)
    assert again["outcome"] == OUTCOME_SUCCEEDED and env.tick == ticks
    platform.join()


def test_tcp_offset_shifts_ik_requests_and_reported_poses_along_the_tool():
    groups = tuple(
        GroupSpec(name=g, joints=tuple(f"{g}{i}" for i in range(6)), gripper=f"{g}g", tcp_offset=(0.0, 0.0, 0.1))
        for g in ("a", "b")
    )
    core, env, platform = rig(dual_arm_descriptor(groups=groups))
    result = core.submit(Command(kind="pose", group="a", pose=[0.2, 0.3, 0.5, 1, 0, 0, 0]), 5.0)
    assert result["outcome"] == OUTCOME_SUCCEEDED
    assert env.q["a"][:3] == pytest.approx([0.2, 0.3, 0.4])  # flange sits 0.1 behind the TCP
    frame, _tick = core.snapshot()
    assert core.tcp_pose(frame, "a")[:3] == pytest.approx([0.2, 0.3, 0.5])
    assert shift_pose([0, 0, 0, 1, 0, 0, 0], (0.0, 0.0, 0.1)) == pytest.approx([0, 0, 0.1, 1, 0, 0, 0])
    platform.join()


def test_session_end_fails_the_motion_in_flight_and_finish_holds():
    core, env, platform = rig()
    platform.join()  # drive ticks by hand
    thread, box = submit_async(core, Command(kind="joint", group="a", joints=[0.9] * 6))
    assert wait_for(lambda: core._queue)
    core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)
    core.session_end(
        {"session_id": platform.session_id, "generation": platform.generation},
        PlatformResult(native_success=False, native_reason="timeout"),
    )
    thread.join(timeout=2.0)
    assert box["result"]["outcome"] == OUTCOME_EXECUTION_FAILED


def test_finish_holds_and_refuses_new_motion():
    core, env, platform = rig()
    assert core.finish() and core.state == STATE_STOPPING
    assert wait_for(lambda: env.tick >= 3)  # hold ticks until the platform ends the session
    assert all(step["a"]["joints"] == [0.1] * 6 for step in env.steps)
    refused = core.submit(Command(kind="joint", group="a", joints=[0.2] * 6), 1.0)
    assert refused["outcome"] == OUTCOME_REJECTED and "finishing" in refused["message"]
    platform.join()
