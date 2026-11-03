"""Contract v2 in the Gateway: arm_side selects the runtime motion group and its ee pose; gripper orientation."""

from __future__ import annotations

import contextlib
import time

from geometry_msgs.msg import PoseStamped

from skill_library.resolver import resolve_skill_primitives
from skill_library.skill_executor_node import SkillExecutorNode


def _gateway(arm_groups=None, ee_pose_topics=None):
    node = object.__new__(SkillExecutorNode)
    node._arm_groups = dict(arm_groups or {})
    node._ee_pose_topics = dict(ee_pose_topics or {})
    node._latest_group_ee_pose = {}
    node._latest_ee_pose = None
    node._latest_ee_pose_monotonic = None
    node._robot_state_freshness_sec = 0.5
    node._state_guard = contextlib.nullcontext
    return node


def _pose(x):
    msg = PoseStamped()
    msg.pose.position.x = x
    return msg


def test_single_arm_gateway_keeps_the_default_group():
    gateway = _gateway()
    assert gateway._target_group_for_arm_side("") == ("", "")
    assert gateway._target_group_for_arm_side("auto") == ("", "")
    group, error = gateway._target_group_for_arm_side("left")
    assert group == "" and "not an arm of this robot" in error


def test_dual_arm_gateway_maps_arm_side_to_the_runtime_group():
    gateway = _gateway({"left": "left_arm", "right": "right_arm"})
    assert gateway._target_group_for_arm_side("Right") == ("right_arm", "")
    assert gateway._target_group_for_arm_side("left") == ("left_arm", "")


def test_ee_pose_snapshot_is_taken_from_the_selected_arm():
    topics = {"left_arm": "/l", "right_arm": "/r"}
    gateway = _gateway({"left": "left_arm", "right": "right_arm"}, topics)
    gateway._handle_group_ee_pose("right_arm", _pose(0.3))
    gateway._latest_ee_pose, gateway._latest_ee_pose_monotonic = _pose(-0.3), time.monotonic()
    assert gateway._fresh_ee_pose_snapshot("right_arm").pose.pose.position.x == 0.3
    assert gateway._fresh_ee_pose_snapshot("").pose.pose.position.x == -0.3  # default topic
    assert gateway._fresh_ee_pose_snapshot("left_arm") is None  # no sample from the left arm yet


def _quat_rotate(q, v):
    x, y, z, w = q
    # v' = q v q*, written out for a unit quaternion
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    )


def _horizontal_x5_pose():
    # Approach (tool x) along world +y, fingers (tool y) along world -x.
    msg = PoseStamped()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = 0.3, -0.1, 0.9
    msg.pose.orientation.z = msg.pose.orientation.w = 2**-0.5
    return msg


def _axis(pose, local):
    o = pose.orientation
    return tuple(round(v, 6) + 0.0 for v in _quat_rotate((o.x, o.y, o.z, o.w), local))


def test_rotation_turns_about_the_declared_approach_axis():
    pose = SkillExecutorNode._pose_from_gripper_rotation(_horizontal_x5_pose(), "rotate_gripper_ccw", 90.0, axis="x")
    assert (pose.position.x, pose.position.y, pose.position.z) == (0.3, -0.1, 0.9)
    assert _axis(pose, (1, 0, 0)) == (0.0, 1.0, 0.0)  # still approaching forward
    assert _axis(pose, (0, 1, 0)) == (0.0, 0.0, 1.0)  # the fingers now close vertically


def test_rotation_axis_defaults_to_tool_z():
    default = SkillExecutorNode._pose_from_gripper_rotation(_horizontal_x5_pose(), "rotate_gripper_cw", 30.0)
    explicit = SkillExecutorNode._pose_from_gripper_rotation(_horizontal_x5_pose(), "rotate_gripper_cw", 30.0, axis="z")
    assert default == explicit


def test_gripper_approach_keeps_the_tcp_and_takes_the_preset_orientation():
    gateway = _gateway()
    gateway._gripper_approach = {
        "axis": "x",
        "presets": {"top_down": {"orientation": {"x": -0.5, "y": 0.5, "z": 0.5, "w": 0.5}}},
    }
    pose = gateway._pose_from_gripper_approach(_horizontal_x5_pose(), "top_down")
    assert (pose.position.x, pose.position.y, pose.position.z) == (0.3, -0.1, 0.9)
    assert _axis(pose, (1, 0, 0)) == (0.0, 0.0, -1.0)  # approach straight down
    assert _axis(pose, (0, 1, 0)) == (-1.0, 0.0, 0.0)  # fingers still close along x
    assert gateway._pose_from_gripper_approach(_horizontal_x5_pose(), "sideways") is None


def test_resolver_passes_the_requested_approach_as_the_primitive_pose_name():
    templates = {
        "set_gripper_approach": {
            "initial_gripper_state": "none",
            "primitive_sequence": [{"primitive_name": "set_gripper_approach", "approach_from_request": True}],
        }
    }
    primitives = resolve_skill_primitives(
        "set_gripper_approach", "", "", "", 0.0, {}, 1.0, 0.0, templates, approach="top_down"
    )
    assert [(p.primitive_name, p.pose_name) for p in primitives] == [("set_gripper_approach", "top_down")]
