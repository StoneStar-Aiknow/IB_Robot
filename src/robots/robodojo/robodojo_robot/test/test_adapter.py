"""RoboDojo codec and adapter: native keys in, RoboDojo's verdict only as the session's platform result."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from robodojo_robot.adapter import PUBLIC_EPISODE_FIELDS, RoboDojoAdapter

from robot_runtime_sim.direct_control.engine import MotionConfig, SessionError, SimulationCore

PROFILE = yaml.safe_load((Path(__file__).resolve().parents[1] / "profiles" / "arx_x5_dual.yaml").read_text())


def native_obs(tick, episode, left=0.0):
    state = {}
    for side, q in (("left", left), ("right", 0.0)):
        state[f"{side}_arm_joint_state"] = [q] * 6
        state[f"{side}_ee_joint_state"] = [1.0]
        state[f"{side}_ee_pose"] = [0.0, -0.3, 0.9, 1.0, 0.0, 0.0, 0.0]
    return {"state": state, "_body": {**episode, "tick": tick}, "reward": 1.0, "info": {"success": True}}


def rig():
    events, native = [], []
    adapter = RoboDojoAdapter(PROFILE, {"robodojo_root": "/unused"})
    core = SimulationCore(
        adapter.descriptor,
        MotionConfig(idle_wait_s=0.01),
        on_event=lambda k, i: events.append((k, i)),
        on_platform_result=native.append,
    )
    model = adapter.bind(core)
    return adapter, core, model, events, native


def test_codec_maps_sides_to_groups_and_drops_everything_else():
    adapter, core, model, _events, _native = rig()
    ident = model.reset()
    model.begin_episode({"task": "t", "instruction": "i", "step_lim": 10, "seed": 4, "layout": {"x": 1}})
    episode = {"episode_id": ident["episode_id"], "generation": ident["generation"]}
    model.update_obs(native_obs(0, episode, left=0.3))
    frame, tick = core.snapshot()
    assert tick == 0 and frame.joints["left_arm"] == [0.3] * 6 and frame.grippers["right_arm"] == 1.0
    assert set(frame.__dict__) == {
        "session_id",
        "generation",
        "step_id",
        "joints",
        "grippers",
        "ee_poses",
        "images",
        "wall_time",
    }
    assert core.tcp_pose(frame, "left_arm")[:3] == pytest.approx([0.13, -0.3, 0.9])
    action = adapter.codec.encode_action(
        {"left_arm": {"joints": [0.1] * 6, "gripper": 0.5}, "right_arm": {"joints": [0.2] * 6, "gripper": 1.0}}
    )
    assert action == {
        "left_arm_joint_state": [0.1] * 6,
        "left_ee_joint_state": [0.5],
        "right_arm_joint_state": [0.2] * 6,
        "right_ee_joint_state": [1.0],
    }


def test_public_episode_event_carries_no_seed_layout_or_calibration():
    _adapter, core, model, events, native = rig()
    ident = model.reset()
    model.begin_episode(
        {
            "run_id": "r",
            "task": "general_pickup",
            "instruction": "pick",
            "step_lim": 200,
            "seed": 7,
            "layout": {"scene_layout": "9002/1"},
            "camera_calibration": {"cam_head": {}},
        }
    )
    start = next(info for kind, info in events if kind == "session_start")
    assert set(start) <= {*PUBLIC_EPISODE_FIELDS, "session_id", "generation", "dt_s"}
    episode = {"episode_id": ident["episode_id"], "generation": ident["generation"]}
    model.update_obs(native_obs(0, episode))
    model.episode_end({**episode, "success": True, "failure_reason": None, "take_action_cnt": 3, "step_lim": 200})
    assert native[0]["native_success"] is True and native[0]["seed"] == 7
    assert native[0]["layout"] == {"scene_layout": "9002/1"}


def test_stale_generation_calls_are_refused_and_old_protocol_session_id_is_accepted():
    _adapter, core, model, _events, _native = rig()
    old = model.reset()
    model.begin_episode({"task": "t"})
    new = model.reset()
    assert new["generation"] == old["generation"] + 1
    with pytest.raises(SessionError):
        model.update_obs(native_obs(0, {"episode_id": old["episode_id"], "generation": old["generation"]}))
    with pytest.raises(SessionError):
        model.get_action({"episode_id": old["episode_id"], "generation": old["generation"], "replies": []})
    model.begin_episode({"task": "t"})
    model.update_obs(native_obs(0, {"session_id": new["session_id"]}))
    assert model.get_action({"session_id": new["session_id"]})["action"] is None  # idle: do not step


def test_the_adapter_refuses_a_clock_mode_it_cannot_drive():
    from robot_runtime_sim.direct_control.ports import DescriptorError

    robodojo = {**PROFILE["simulation"]["robodojo"], "clock_mode": "host_step"}
    profile = {**PROFILE, "simulation": {**PROFILE["simulation"], "robodojo": robodojo}}
    with pytest.raises(DescriptorError, match="platform_pull"):
        RoboDojoAdapter(profile, {"robodojo_root": "/unused"})


def test_the_shipped_profile_and_adapter_agree():
    from robot_runtime_sim.direct_control.ports import profile_mismatches

    adapter = RoboDojoAdapter(PROFILE, {"robodojo_root": "/unused"})
    assert profile_mismatches(PROFILE, adapter.descriptor) == []
