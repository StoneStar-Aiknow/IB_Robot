"""Transactional joint-space ArmReturnHome tests for the SO-101 Placo node."""

import importlib.util
import os
import sys
import threading
import types

import numpy as np


def _install_stubs():
    def _mod(name):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    rclpy = _mod("rclpy")
    rclpy.ok = lambda: True
    for submodule in ("action", "node", "callback_groups", "duration", "executors", "time"):
        _mod(f"rclpy.{submodule}")
    _mod("rclpy.node").Node = type("Node", (), {})
    action = _mod("rclpy.action")
    action.ActionClient = type("ActionClient", (), {})
    action.ActionServer = type("ActionServer", (), {})
    action.CancelResponse = types.SimpleNamespace(ACCEPT="accept", REJECT="reject")
    action.GoalResponse = types.SimpleNamespace(ACCEPT="accept", REJECT="reject")
    callback_groups = _mod("rclpy.callback_groups")
    callback_groups.MutuallyExclusiveCallbackGroup = type("MECG", (), {})
    callback_groups.ReentrantCallbackGroup = type("RCG", (), {})
    _mod("rclpy.duration").Duration = type("Duration", (), {})
    executors = _mod("rclpy.executors")
    executors.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    executors.MultiThreadedExecutor = type("MultiThreadedExecutor", (), {})
    _mod("rclpy.time").Time = type("Time", (), {})

    for package, names in (
        ("geometry_msgs.msg", ("PoseStamped", "Vector3Stamped")),
        ("sensor_msgs.msg", ("JointState",)),
        ("std_msgs.msg", ("Bool", "Empty", "Float64MultiArray")),
    ):
        _mod(package.split(".")[0])
        message_module = _mod(package)
        for name in names:
            setattr(message_module, name, type(name, (), {}))

    _mod("std_srvs")
    services = _mod("std_srvs.srv")

    class _Trigger:
        class Request:
            pass

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""

    services.Trigger = _Trigger

    _mod("ibrobot_msgs")
    actions = _mod("ibrobot_msgs.action")

    class _ArmReturnHome:
        class Goal:
            def __init__(self):
                self.target_name = ""

        class Result:
            def __init__(self):
                self.success = False
                self.error_code = ""
                self.message = ""

        class Feedback:
            def __init__(self):
                self.state = ""
                self.max_joint_error_rad = 0.0

    actions.ArmReturnHome = _ArmReturnHome
    tf2_ros = _mod("tf2_ros")
    tf2_ros.Buffer = type("Buffer", (), {})
    tf2_ros.TransformListener = type("TransformListener", (), {})
    kinematics = _mod("so101_placo_kinematics")
    kinematics.SO101PlacoDiffIK = type("SO101PlacoDiffIK", (), {})


_install_stubs()

_NODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "so101_placo_servo_node.py",
)
_spec = importlib.util.spec_from_file_location("so101_placo_servo_node", _NODE_PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules["so101_placo_servo_node"] = mod
_spec.loader.exec_module(mod)

from std_srvs.srv import Trigger  # noqa: E402


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _FakeGoalHandle:
    def __init__(self):
        self.is_cancel_requested = False
        self.feedback = []
        self.terminal_state = None

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)

    def succeed(self):
        self.terminal_state = "succeeded"

    def canceled(self):
        self.terminal_state = "canceled"

    def abort(self):
        self.terminal_state = "aborted"


def _make_node():
    node = mod.SO101PlacoServoNode.__new__(mod.SO101PlacoServoNode)
    node.arm_joint_names = ["1", "2", "3", "4", "5"]
    node.joint_lo = np.full(5, -2.0)
    node.joint_hi = np.full(5, 2.0)
    node._home_q = np.array([0.2, -0.1, 0.3, -0.2, 0.1], dtype=np.float64)
    node._home_enabled = True
    node.home_joint_tolerance_rad = 0.05
    node.home_max_joint_speed = 1.0
    node.home_joint_state_stale_s = 0.2
    node.home_stable_duration_s = 0.2
    node.home_timeout_s = 10.0
    node.control_period = 0.02
    node.command_lease_timeout_s = 0.0
    node.target_reset_timeout = 2.0
    node.input_mode = "pose"
    node._enabled = True
    node._estop_active = False
    node._accept_velocity_commands = True
    node._accept_pose_commands = True
    node._latest_linear = object()
    node._latest_angular = object()
    node._latest_pose = object()
    node._latest_pose_stamp = 0.0
    node._latest_linear_stamp = 0.0
    node._latest_angular_stamp = 0.0
    node._last_input_time = 0.0
    node._last_lease_time = 123.0
    node._p_ref = np.ones(3)
    node._r_ref = np.eye(3)
    node._ee0_p = np.ones(3)
    node._ee0_R = np.eye(3)
    node._last_cmd = np.zeros(5)
    node._latest_js = object()
    node._latest_js_received_at = 123.0
    node._joint_state_generation = 1
    node._home_active = False
    node._home_started_at = 0.0
    node._home_stable_since = None
    node._home_last_joint_state_generation = 0
    node._home_request_lock = threading.Lock()
    node._home_goal_reserved = False
    node._home_preemption = None
    node._pending_home_request = None
    node._active_home_request = None
    node._dropped_frame_count = 0
    node._solve_count = 0
    node._measured_q = np.zeros(5)
    node._measured_arm_joints = lambda: node._measured_q.copy()
    node._now = lambda: 123.0
    node.get_logger = lambda: _FakeLogger()
    node.published = []
    node.cmd_pub = types.SimpleNamespace(publish=node.published.append)
    return node


def _activate_home(node):
    goal_handle = _FakeGoalHandle()
    request = mod._HomeActionRequest(goal_handle=goal_handle, done=threading.Event())
    node._home_goal_reserved = True
    node._pending_home_request = request
    node._process_home_action(123.0)
    return request


def _fresh_pose():
    return object()


def test_home_starts_joint_motion_and_closes_command_gates():
    node = _make_node()
    request = _activate_home(node)

    assert node._home_active is True
    assert node._active_home_request is request
    assert node._latest_pose is None
    assert node._latest_linear is None
    assert node._latest_angular is None
    assert node._accept_pose_commands is False
    assert node._accept_velocity_commands is False


def test_home_publishes_bounded_joint_space_step():
    node = _make_node()
    _activate_home(node)

    node._on_control_tick()

    expected = np.clip(node._home_q, -0.02, 0.02)
    np.testing.assert_allclose(node.published[-1].data, expected)


def test_pose_and_velocity_commands_are_rejected_until_next_start():
    node = _make_node()
    _activate_home(node)
    node._on_pose(_fresh_pose())
    node._on_linear(object())
    node._on_angular(object())
    assert node._latest_pose is None
    assert node._latest_linear is None
    assert node._latest_angular is None


def test_start_is_rejected_while_home_is_reserved():
    node = _make_node()
    node._home_goal_reserved = True

    response = node._on_start_srv(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert "ArmReturnHome" in response.message


def test_estop_aborts_home_and_blocks_new_home_and_start():
    node = _make_node()
    request = _activate_home(node)

    node._on_estop(types.SimpleNamespace(data=True))

    assert request.error_code == "EMERGENCY_STOP"
    assert node._enabled is False
    goal = types.SimpleNamespace(target_name="home")
    assert node._home_goal_callback(goal) == mod.GoalResponse.REJECT
    response = node._on_start_srv(Trigger.Request(), Trigger.Response())
    assert response.success is False

    node._on_estop(types.SimpleNamespace(data=False))
    assert node._estop_active is False
    assert node._enabled is False


def test_home_completes_only_after_fresh_stable_joint_samples():
    node = _make_node()
    node._measured_q = node._home_q.copy()
    request = _activate_home(node)

    node._joint_state_generation += 1
    node._update_home_progress(node._home_q, 123.0)
    node._update_home_progress(node._home_q, 124.0)
    assert request.done.is_set() is False

    node._joint_state_generation += 1
    node._update_home_progress(node._home_q, 123.21)
    assert request.done.is_set() is True
    assert request.outcome == "succeeded"
    assert node._enabled is False


def test_home_timeout_aborts_transaction():
    node = _make_node()
    request = _activate_home(node)

    assert node._home_timed_out(133.01) is True
    assert request.done.is_set() is True
    assert request.error_code == "TIMEOUT"
    assert request.max_joint_error_rad == 0.3
    assert "joint '3' error=0.3000rad" in request.message
    assert "target=0.3000" in request.message
    assert node._enabled is False


def test_stop_aborts_active_home():
    node = _make_node()
    request = _activate_home(node)

    node._on_stop_srv(Trigger.Request(), Trigger.Response())

    assert request.done.is_set() is True
    assert request.error_code == "STOP_REQUESTED"
    assert node._home_active is False


def test_prepare_shutdown_releases_active_action_waiter():
    node = _make_node()
    request = _activate_home(node)

    node.prepare_shutdown()

    assert request.done.is_set() is True
    assert request.error_code == "ROS_SHUTDOWN"


def test_stop_preempts_reserved_goal_before_execute_callback():
    node = _make_node()
    node._home_goal_reserved = True
    node._on_stop_srv(Trigger.Request(), Trigger.Response())
    goal_handle = _FakeGoalHandle()

    result = node._execute_home_action(goal_handle)

    assert result.success is False
    assert result.error_code == "STOP_REQUESTED"
    assert goal_handle.terminal_state == "aborted"
    assert node._home_goal_reserved is False


def test_cancel_request_aborts_without_starting_motion():
    node = _make_node()
    request = _activate_home(node)
    request.cancel_requested = True

    node._process_home_action(123.01)

    assert request.outcome == "canceled"
    assert node._enabled is False


def test_cancel_preempts_reserved_goal_before_execute_callback():
    node = _make_node()
    node._home_goal_reserved = True
    goal_handle = _FakeGoalHandle()

    assert node._home_cancel_callback(goal_handle) == mod.CancelResponse.ACCEPT
    result = node._execute_home_action(goal_handle)

    assert result.error_code == "CANCELED"
    assert goal_handle.terminal_state == "canceled"


def test_home_requires_fresh_joint_state():
    node = _make_node()
    node._latest_js_received_at = 122.0
    request = _activate_home(node)

    assert request.outcome == "aborted"
    assert request.error_code == "JOINT_STATE_UNAVAILABLE"


def test_home_aborts_on_non_finite_joint_feedback():
    node = _make_node()
    request = _activate_home(node)
    node._measured_q[2] = float("nan")

    node._on_control_tick()

    assert request.error_code == "JOINT_STATE_INVALID"
    assert node._enabled is False


def test_start_refreshes_command_lease():
    node = _make_node()
    node._home_goal_reserved = False
    node._last_lease_time = 0.0
    node.diffik = types.SimpleNamespace(
        ee_position=lambda _q: np.zeros(3),
        ee_rotation=lambda _q: np.eye(3),
    )

    response = node._on_start_srv(Trigger.Request(), Trigger.Response())

    assert response.success is True
    assert node._last_lease_time == 123.0
