"""Deadman / stop-latch state-machine tests for the SO-101 VR pose path.

These drive the *production* methods of ``VRTeleopNode`` (``_stop_so101_servo``,
``_on_so101_stop_response``, ``_handle_so101_stale``, ``_publish_all_zero``, and
the start/recalib/home callbacks) directly, so a regression in the real state
machine fails here. They cover the four scenarios the review asked for:

  1. Home no longer targets home+offset — asserted on the placo node instead
     (see the placo package); here we cover the VR-side deadman latch:
  2. stop service not ready → stop stays pending and is retried every watchdog
     tick until it confirms;
  3. stop racing an inflight recalib/home → the async success callback must NOT
     re-enable the arm while a stop is pending;
  4. recovered stream while the trigger is still held must NOT auto-resume — the
     stall clears only on a trigger release, and the next press re-baselines.

The node is built with ``__new__`` (skipping the ROS-heavy ``__init__``): only
the attributes the deadman logic touches are set, and the ROS service surface is
faked. The methods under test are the real ones — no logic is re-implemented.

Importing ``vr_teleop`` pulls ROS message/rclpy symbols and the cartesian
backend (which transitively needs pymoveit2, not installed in CI). We stub only
that *infrastructure* — never the code under test — before import.
"""

import os
import sys
import types

import pytest

# Package source dir resolved relative to THIS file, so the test works both from
# the source tree (bare pytest at workspace root) and under ``colcon test``
# (CWD is the build tree). ``<pkg>/test/..``/robot_teleop is the package dir.
_PKG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_teleop")


# --------------------------------------------------------------------------- #
# Infrastructure stubs (imported by vr_teleop but not exercised by these tests).
# --------------------------------------------------------------------------- #
def _install_stubs():
    def _mod(name):
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            sys.modules[name] = m
        return m

    # rclpy + rclpy.node.Node (only used as a base class; we bypass __init__).
    # Idempotent per-symbol: another test file may have installed a partial stub.
    _mod("rclpy")
    node_mod = _mod("rclpy.node")
    if not hasattr(node_mod, "Node"):
        node_mod.Node = type("_Node", (), {})
    sys.modules["rclpy"].node = node_mod
    action_mod = _mod("rclpy.action")
    action_mod.ActionClient = type("ActionClient", (), {})

    _mod("ibrobot_msgs")
    action_msgs = _mod("ibrobot_msgs.action")

    class _ArmReturnHome:
        class Goal:
            def __init__(self):
                self.target_name = ""

    action_msgs.ArmReturnHome = _ArmReturnHome

    # Message packages: only the names imported at module top-level are needed.
    if "geometry_msgs.msg" not in sys.modules:
        _mod("geometry_msgs")
        gm = _mod("geometry_msgs.msg")
        gm.PoseStamped = type("PoseStamped", (), {})
        gm.Vector3Stamped = type("Vector3Stamped", (), {})
    if "std_msgs.msg" not in sys.modules:
        _mod("std_msgs")
        sm = _mod("std_msgs.msg")
        sm.Bool = type("Bool", (), {})
        sm.Float64MultiArray = type("Float64MultiArray", (), {})
    if "std_srvs.srv" not in sys.modules:
        _mod("std_srvs")
        ss = _mod("std_srvs.srv")

        class _Trigger:
            class Request:  # noqa: D106
                pass

            class Response:  # noqa: D106
                def __init__(self):
                    self.success = False
                    self.message = ""

        ss.Trigger = _Trigger
    if "tf2_ros" not in sys.modules:
        tf = _mod("tf2_ros")
        tf.Buffer = type("Buffer", (), {})
        tf.TransformListener = type("TransformListener", (), {})

    # cartesian_backend.frame_adapter — vr_teleop imports ToolAngularAdapter from
    # it, and its package __init__ transitively needs pymoveit2. Provide the one
    # symbol so the import line resolves without dragging in the servo backends.
    pkg = types.ModuleType("robot_teleop")
    pkg.__path__ = [_PKG_DIR]
    sys.modules["robot_teleop"] = pkg
    cb = _mod("robot_teleop.cartesian_backend")
    cb.__path__ = [os.path.join(_PKG_DIR, "cartesian_backend")]
    fa = _mod("robot_teleop.cartesian_backend.frame_adapter")
    fa.ToolAngularAdapter = type("ToolAngularAdapter", (), {})


_install_stubs()

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("robot_teleop.vr_teleop", os.path.join(_PKG_DIR, "vr_teleop.py"))
vr_teleop = importlib.util.module_from_spec(_spec)
sys.modules["robot_teleop.vr_teleop"] = vr_teleop
_spec.loader.exec_module(vr_teleop)

from std_srvs.srv import Trigger  # noqa: E402  (our stub)


# --------------------------------------------------------------------------- #
# Fakes for the ROS service surface.
# --------------------------------------------------------------------------- #
class _FakeFuture:
    def __init__(self):
        self._cb = None
        self._result = None
        self._exc = None

    def add_done_callback(self, cb):
        self._cb = cb

    def resolve_success(self, message="ok"):
        resp = Trigger.Response()
        resp.success = True
        resp.message = message
        self._result = resp
        if self._cb:
            self._cb(self)

    def resolve_rejected(self, message="rejected"):
        resp = Trigger.Response()
        resp.success = False
        resp.message = message
        self._result = resp
        if self._cb:
            self._cb(self)

    def resolve_exception(self, exc=RuntimeError("boom")):
        self._exc = exc
        if self._cb:
            self._cb(self)

    def resolve_value(self, value):
        self._result = value
        if self._cb:
            self._cb(self)

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeClient:
    """A service client whose readiness is toggleable; records dispatched calls."""

    def __init__(self, ready=True):
        self._ready = ready
        self.futures = []

    def service_is_ready(self):
        return self._ready

    def set_ready(self, ready):
        self._ready = ready

    def call_async(self, _req):
        fut = _FakeFuture()
        self.futures.append(fut)
        return fut


class _FakeGoalHandle:
    def __init__(self):
        self.accepted = True
        self.cancel_calls = 0
        self.result_future = _FakeFuture()

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return _FakeFuture()

    def get_result_async(self):
        return self.result_future

    def finish(self, success=True, error_code="", message="done"):
        result = types.SimpleNamespace(success=success, error_code=error_code, message=message)
        self.result_future.resolve_value(types.SimpleNamespace(result=result))


class _FakeActionClient:
    def __init__(self, ready=True):
        self._ready = ready
        self.futures = []
        self.goal_handles = []

    def server_is_ready(self):
        return self._ready

    def send_goal_async(self, _goal):
        future = _FakeFuture()
        self.futures.append(future)
        return future

    def accept(self, index=-1):
        goal_handle = _FakeGoalHandle()
        self.goal_handles.append(goal_handle)
        self.futures[index].resolve_value(goal_handle)
        return goal_handle


class _FakeLogger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _make_node(stop_ready=True, start_ready=True, home_ready=True):
    """Build a VRTeleopNode with __init__ skipped and only deadman state set."""
    node = vr_teleop.VRTeleopNode.__new__(vr_teleop.VRTeleopNode)
    node._output_profile = "so101"
    node._so101_input_mode = "pose"
    node._so101_command_stale_s = 0.2
    node._controller_side = "right"
    # Deadman / start-path flags (mirror the constructor defaults).
    node._so101_started = False
    node._so101_start_inflight = False
    node._so101_stop_inflight = False
    node._so101_recalib_inflight = False
    node._so101_home_inflight = False
    node._so101_home_goal_handle = None
    node._so101_home_cancel_pending = False
    node._so101_stop_pending = False
    node._so101_reengage_required = False
    node._estop_active = False
    node._so101_stalled = False
    node._homing = False
    node._home_terminal = False
    node._home_deadman_held = False
    node._secondary_prev = False
    node._pose_calib_pos = None
    node._pose_calib_rot = None
    # Service clients.
    node._so101_start_cli = _FakeClient(start_ready)
    node._so101_stop_cli = _FakeClient(stop_ready)
    node._so101_home_cli = _FakeActionClient(home_ready)
    node._arm_state = {
        "left": vr_teleop._ArmState(),
        "right": vr_teleop._ArmState(),
    }
    # Logger.
    node._logger = _FakeLogger()
    node.get_logger = lambda: node._logger
    return node


# --------------------------------------------------------------------------- #
# Scenario 2: stop service not ready → pending, retried until it confirms.
# --------------------------------------------------------------------------- #
def test_stop_not_ready_latches_pending_and_retries():
    node = _make_node(stop_ready=False)
    node._so101_started = True

    # First stall: stop service not ready → nothing dispatched, but pending set.
    node._stop_so101_servo("stall")
    assert node._so101_stop_pending is True
    assert node._so101_started is False
    assert node._so101_stop_cli.futures == []  # nothing dispatched

    # Watchdog keeps firing while pending; still not ready → still no dispatch.
    node._stop_so101_servo("retry while not ready")
    assert node._so101_stop_pending is True
    assert node._so101_stop_cli.futures == []

    # Service becomes discoverable; next retry actually dispatches.
    node._so101_stop_cli.set_ready(True)
    node._stop_so101_servo("retry now ready")
    assert len(node._so101_stop_cli.futures) == 1
    assert node._so101_stop_inflight is True

    # Confirmed stop clears the latch.
    node._so101_stop_cli.futures[0].resolve_success()
    assert node._so101_stop_pending is False
    assert node._so101_stop_inflight is False


def test_stop_rejected_stays_pending():
    node = _make_node(stop_ready=True)
    node._so101_started = True
    node._stop_so101_servo("stall")
    assert node._so101_stop_inflight is True
    # Servo rejects the stop → still pending, arm not confirmed stopped.
    node._so101_stop_cli.futures[0].resolve_rejected()
    assert node._so101_stop_pending is True
    assert node._so101_stop_inflight is False


def test_stop_exception_stays_pending():
    node = _make_node(stop_ready=True)
    node._so101_started = True
    node._stop_so101_servo("stall")
    node._so101_stop_cli.futures[0].resolve_exception()
    assert node._so101_stop_pending is True


def test_stale_watchdog_retries_until_confirmed():
    node = _make_node(stop_ready=False)
    node._so101_started = True

    # First stale tick: sets stalled, tries stop (not ready) → pending.
    node._handle_so101_stale()
    assert node._so101_stalled is True
    assert node._so101_stop_pending is True
    assert node._so101_stop_cli.futures == []

    # Second stale tick (still stalled): must retry because pending, not return.
    node._so101_stop_cli.set_ready(True)
    node._handle_so101_stale()
    assert len(node._so101_stop_cli.futures) == 1

    node._so101_stop_cli.futures[0].resolve_success()
    assert node._so101_stop_pending is False


# --------------------------------------------------------------------------- #
# Scenario 3: stop racing an inflight recalib/home.
# --------------------------------------------------------------------------- #
def test_stop_during_inflight_recalib_wins_over_success():
    node = _make_node()
    # A recalib (start) is inflight.
    node._so101_recalib_inflight = True
    start_fut = node._so101_start_cli.call_async(Trigger.Request())
    start_fut.add_done_callback(node._on_so101_recalib_response)

    # A stall lands while recalib is inflight: stop must latch pending, not race.
    node._stop_so101_servo("stall during recalib")
    assert node._so101_stop_pending is True
    assert node._so101_started is False

    # The recalib now succeeds. It must NOT set started; it must honor the stop.
    start_fut.resolve_success()
    assert node._so101_started is False
    # And it re-issues a stop (stop_cli ready → dispatched).
    assert len(node._so101_stop_cli.futures) == 1


def test_stop_during_inflight_home_cancels_action_and_wins():
    node = _make_node()
    assert node._go_home_so101() is True
    goal_handle = node._so101_home_cli.accept()

    node._stop_so101_servo("stall during home")
    assert node._so101_stop_pending is True
    assert goal_handle.cancel_calls == 1
    assert len(node._so101_stop_cli.futures) == 1

    goal_handle.finish(success=False, error_code="STOP_REQUESTED")
    assert node._so101_started is False
    assert len(node._so101_stop_cli.futures) == 1


def test_start_success_honors_pending_stop():
    node = _make_node()
    node._so101_start_inflight = True
    start_fut = node._so101_start_cli.call_async(Trigger.Request())
    start_fut.add_done_callback(node._on_so101_start_response)

    node._stop_so101_servo("stall during start")
    assert node._so101_stop_pending is True

    start_fut.resolve_success()
    assert node._so101_started is False
    assert len(node._so101_stop_cli.futures) == 1


def test_recalibrate_refused_while_stop_pending():
    node = _make_node()
    node._so101_stop_pending = True
    assert node._recalibrate_so101_baseline() is False
    assert node._so101_start_cli.futures == []  # no start dispatched


def test_home_refused_while_stop_pending():
    node = _make_node()
    node._so101_stop_pending = True
    assert node._go_home_so101() is False
    assert node._so101_home_cli.futures == []


def test_ensure_started_blocked_while_pending():
    # Stop service not ready, so the timer cannot even dispatch a retry: the key
    # invariant is that it must NOT start the arm while a stop is pending.
    node = _make_node(stop_ready=False)
    node._so101_stop_pending = True
    node._ensure_so101_started()
    assert node._so101_start_cli.futures == []  # no start dispatched
    assert node._so101_started is False


def test_ensure_started_retries_pending_stop_from_timer():
    """A stop rejected while the stream was fresh leaves pending latched with no
    watchdog tick to retry it. The always-on 0.5s timer (_ensure_so101_started)
    must drive the retry so the latch can eventually clear — otherwise the user
    can never re-teleop (safe but livelocked)."""
    node = _make_node(stop_ready=True)
    node._so101_started = True

    # Stop is dispatched then rejected → pending stays, stream is fresh (no stall).
    node._stop_so101_servo("stall")
    node._so101_stop_cli.futures[0].resolve_rejected()
    assert node._so101_stop_pending is True
    assert node._so101_stalled is False  # never entered stall in this path

    # The timer fires: it must retry the stop (not attempt a start).
    node._ensure_so101_started()
    assert len(node._so101_stop_cli.futures) == 2  # retry dispatched
    assert node._so101_start_cli.futures == []  # no start raced

    # This retry confirms → latch clears, and the arm can be re-engaged later.
    node._so101_stop_cli.futures[1].resolve_success()
    assert node._so101_stop_pending is False


# --------------------------------------------------------------------------- #
# Scenario 4: recovered stream while trigger still held must NOT auto-resume.
# --------------------------------------------------------------------------- #
def _ctrl(enabled, secondary=False):
    c = types.SimpleNamespace()
    c.enabled = enabled
    c.secondary_button = secondary
    c.grip_value = 0.0
    import numpy as np

    c.position = np.zeros(3)
    from scipy.spatial.transform import Rotation

    c.rotation = Rotation.identity()
    return c


def test_stall_clears_only_on_trigger_release_not_fresh_frame():
    node = _make_node()
    node._publish_so101_gripper = lambda *_a, **_k: None
    node._publish_so101_pose = lambda *_a, **_k: None

    # Enter a stall (stop confirmed to isolate the stalled-latch behavior).
    node._so101_started = True
    node._handle_so101_stale()
    node._so101_stop_cli.futures[0].resolve_success()
    assert node._so101_stalled is True
    assert node._so101_stop_pending is False

    # A fresh frame with the trigger STILL HELD must not clear the stall and must
    # not re-engage: _recalibrate is refused while stalled, so calib rolls back.
    node._control_so101_pose(_ctrl(enabled=True))
    assert node._so101_stalled is True
    assert node._pose_calib_pos is None  # never engaged
    assert node._so101_start_cli.futures == []  # no re-latch dispatched

    # Trigger RELEASE clears the stall.
    node._control_so101_pose(_ctrl(enabled=False))
    assert node._so101_stalled is False

    # Next real PRESS re-baselines: recalib dispatched now that stall is cleared.
    node._control_so101_pose(_ctrl(enabled=True))
    assert len(node._so101_start_cli.futures) == 1
    assert node._so101_recalib_inflight is True


def test_absent_controller_does_not_clear_stall():
    node = _make_node()
    node._publish_so101_gripper = lambda *_a, **_k: None
    node._so101_started = True
    node._handle_so101_stale()
    node._so101_stop_cli.futures[0].resolve_success()
    assert node._so101_stalled is True

    # ctrl is None (disconnect, not a deliberate release) must NOT clear stall.
    node._control_so101_pose(None)
    assert node._so101_stalled is True


# --------------------------------------------------------------------------- #
# Fix #1: velocity mode must be able to recover from a stall. The stale watchdog
# sets _so101_stalled for ALL so101 modes, but only the pose path cleared it —
# velocity mode was permanently locked out. Recovery is a live trigger release.
# --------------------------------------------------------------------------- #
def _vel_data(side="right", enabled=True):
    """A _DualArmVRData-like object exposing one controller on `side`."""
    ctrl = types.SimpleNamespace()
    ctrl.enabled = enabled
    ctrl.grip_value = 0.0
    import numpy as np

    ctrl.position = np.zeros(3)
    from scipy.spatial.transform import Rotation

    ctrl.rotation = Rotation.identity()
    data = types.SimpleNamespace(left=None, right=None)
    setattr(data, side, ctrl)
    return data


def _make_velocity_node():
    node = _make_node()
    node._so101_input_mode = "velocity"
    # Record publishes so we can assert the arm is held (zero) while stalled.
    node._published = []
    node._publish_so101 = lambda lin, ang, grip: node._published.append((lin, ang, grip))
    # _compute_velocities would need full _arm_state/params; stub it to a nonzero
    # velocity so a bug that lets it through the stall would be visible.
    import numpy as np

    node._compute_velocities = lambda ctrl, state: (np.ones(3), np.ones(3))
    node._arm_state = {"left": object(), "right": object()}
    return node


def test_velocity_stall_clears_on_trigger_release():
    node = _make_velocity_node()
    node._so101_started = True

    # Enter a stall (stop confirmed).
    node._handle_so101_stale()
    node._so101_stop_cli.futures[0].resolve_success()
    assert node._so101_stalled is True

    # Trigger still HELD while stalled: must hold (publish zero), NOT feed the
    # nonzero _compute_velocities through, and must NOT auto-clear the stall.
    node._control_so101(_vel_data(enabled=True))
    assert node._so101_stalled is True
    import numpy as np

    last = node._published[-1]
    np.testing.assert_array_equal(last[0], np.zeros(3))
    np.testing.assert_array_equal(last[1], np.zeros(3))

    # Trigger RELEASE with a live controller clears the stall.
    node._control_so101(_vel_data(enabled=False))
    assert node._so101_stalled is False


def test_velocity_absent_controller_does_not_clear_stall():
    node = _make_velocity_node()
    node._so101_started = True
    node._handle_so101_stale()
    node._so101_stop_cli.futures[0].resolve_success()
    assert node._so101_stalled is True

    # ctrl is None (disconnect) must NOT clear the stall in velocity mode either.
    # _DualArmVRData always exposes both sides; the controlled side is None here.
    data = types.SimpleNamespace(left=None, right=None)
    node._control_so101(data)
    assert node._so101_stalled is True


def test_velocity_disconnect_requests_transactional_stop():
    node = _make_node()
    node._so101_input_mode = "velocity"
    node._so101_started = True

    node._publish_all_zero()

    assert node._so101_stalled is True
    assert node._so101_reengage_required is True
    assert len(node._so101_stop_cli.futures) == 1


def test_stale_release_blocks_auto_start_until_next_press():
    node = _make_velocity_node()
    node._so101_started = True
    node._handle_so101_stale()
    node._so101_stop_cli.futures[0].resolve_success()

    node._control_so101(_vel_data(enabled=False))
    node._ensure_so101_started()
    assert node._so101_start_cli.futures == []

    node._control_so101(_vel_data(enabled=True))
    assert len(node._so101_start_cli.futures) == 1
    node._so101_start_cli.futures[0].resolve_success()
    assert node._so101_reengage_required is False


def test_estop_requires_velocity_release_and_repress():
    node = _make_velocity_node()
    node._so101_started = True

    node._on_estop(types.SimpleNamespace(data=True))
    assert node._estop_active is True
    assert node._so101_stalled is True
    assert node._so101_reengage_required is True
    assert len(node._so101_stop_cli.futures) == 1
    node._so101_stop_cli.futures[0].resolve_success()

    node._on_estop(types.SimpleNamespace(data=False))
    node._control_so101(_vel_data(enabled=False))
    node._control_so101(_vel_data(enabled=True))
    assert len(node._so101_start_cli.futures) == 1


# --------------------------------------------------------------------------- #
# Transactional Home gate. A press during Home cannot count as the post-Home
# re-engagement; the action must finish while released, or the user must release
# once more after the terminal result and then press again.
# --------------------------------------------------------------------------- #
def _pose_ctrl(enabled=False, secondary=False):
    c = types.SimpleNamespace()
    c.enabled = enabled
    c.secondary_button = secondary
    c.grip_value = 0.0
    import numpy as np

    c.position = np.zeros(3)
    from scipy.spatial.transform import Rotation

    c.rotation = Rotation.identity()
    return c


def test_home_gate_uses_action_result_and_requires_post_home_repress():
    node = _make_node()
    node._publish_so101_gripper = lambda *_a, **_k: None
    node._publish_so101_pose = lambda *_a, **_k: None
    held_home = _pose_ctrl(enabled=True, secondary=True)

    assert node._handle_so101_home_input(held_home) is True
    assert node._homing is True
    goal_handle = node._so101_home_cli.accept()

    # Release then press again while Home is still moving. The held state at the
    # action terminal result must keep the gate closed.
    assert node._handle_so101_home_input(_pose_ctrl(enabled=False)) is True
    assert node._handle_so101_home_input(_pose_ctrl(enabled=True)) is True
    goal_handle.finish(success=True)
    assert node._homing is True

    # A real release after completion clears the gate. Only the following press
    # is allowed to run the normal re-latch path.
    assert node._handle_so101_home_input(_pose_ctrl(enabled=False)) is False
    assert node._homing is False
    node._control_so101_pose(_pose_ctrl(enabled=True, secondary=False))
    assert len(node._so101_start_cli.futures) == 1


def test_home_holds_gripper_output_while_action_is_running():
    node = _make_node()
    published_gripper = []
    node._publish_so101_gripper = published_gripper.append
    node._publish_so101_pose = lambda *_args, **_kwargs: None
    data = types.SimpleNamespace(left=None, right=_pose_ctrl(enabled=False, secondary=True))

    node._control_so101(data)
    node._control_so101(types.SimpleNamespace(left=None, right=_pose_ctrl(enabled=False)))

    assert published_gripper == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
