"""Home-service stale-frame test for so101_placo_servo_node.

Regression for: pressing B (go-home) while the trigger is still held and the
hand has moved would make the arm target ``home + last_vr_displacement`` instead
of ``home``. The hot path recomputes ``_p_ref = _ee0_p + _latest_pose.position``
every fresh tick, so the home service must drop the cached pose command (exactly
as the start service does) — otherwise the stale displacement rides onto the
freshly-latched home baseline.

This drives the *real* ``SO101PlacoServoNode._on_home_srv`` via ``__new__``
(skipping the ROS/placo-heavy ``__init__``); only the home/enable state it
touches is set. The heavy top-level import ``so101_placo_kinematics`` and the
ROS message surface are stubbed as infrastructure — the method under test is the
production one.
"""

import importlib.util
import os
import sys
import types

import numpy as np


def _install_stubs():
    def _mod(name):
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            sys.modules[name] = m
        return m

    # rclpy + submodules. Guard per-symbol (not per-package): another test file
    # in the same pytest process may have installed a minimal rclpy stub without
    # these submodules, and sys.modules is shared.
    _mod("rclpy")
    for sub in ("node", "callback_groups", "duration", "executors", "time"):
        _mod(f"rclpy.{sub}")
    nm = sys.modules["rclpy.node"]
    if not hasattr(nm, "Node"):
        nm.Node = type("Node", (), {})
    sys.modules["rclpy"].node = nm
    cg = sys.modules["rclpy.callback_groups"]
    if not hasattr(cg, "MutuallyExclusiveCallbackGroup"):
        cg.MutuallyExclusiveCallbackGroup = type("MECG", (), {})
    du = sys.modules["rclpy.duration"]
    if not hasattr(du, "Duration"):
        du.Duration = type("Duration", (), {})
    ex = sys.modules["rclpy.executors"]
    if not hasattr(ex, "ExternalShutdownException"):
        ex.ExternalShutdownException = type("ESE", (Exception,), {})
        ex.SingleThreadedExecutor = type("STE", (), {})
    tm = sys.modules["rclpy.time"]
    if not hasattr(tm, "Time"):
        tm.Time = type("Time", (), {})

    for pkg, names in (
        ("geometry_msgs.msg", ["PoseStamped", "Vector3Stamped"]),
        ("sensor_msgs.msg", ["JointState"]),
        ("std_msgs.msg", ["Float64MultiArray"]),
    ):
        if pkg not in sys.modules:
            _mod(pkg.split(".")[0])
            m = _mod(pkg)
            for n in names:
                setattr(m, n, type(n, (), {}))
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
        _mod("tf2_ros")
    # Heavy sibling import at module top-level.
    if "so101_placo_kinematics" not in sys.modules:
        k = _mod("so101_placo_kinematics")
        k.SO101PlacoDiffIK = type("SO101PlacoDiffIK", (), {})


_install_stubs()

# Resolve the node path relative to THIS test file (not the CWD): under
# ``colcon test`` the working directory is the build tree, so a workspace-root
# relative path would not exist. ``<pkg>/test/..``/scripts/<node>.py is stable
# both from the source tree (bare pytest) and the installed/build layout.
_NODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "so101_placo_servo_node.py",
)
_spec = importlib.util.spec_from_file_location(
    "so101_placo_servo_node",
    _NODE_PATH,
)
mod = importlib.util.module_from_spec(_spec)
sys.modules["so101_placo_servo_node"] = mod
_spec.loader.exec_module(mod)

from std_srvs.srv import Trigger  # noqa: E402  (our stub)


class _FakeLogger:
    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _make_node():
    node = mod.SO101PlacoServoNode.__new__(mod.SO101PlacoServoNode)
    node._home_enabled = True
    node._home_p = np.array([0.2, 0.0, 0.3], dtype=np.float64)
    node._home_R = np.eye(3, dtype=np.float64)
    node._enabled = True  # already enabled: skip the measured-seed branch
    node._p_ref = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    node._r_ref = np.eye(3)
    node._ee0_p = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    node._ee0_R = np.eye(3)
    node._last_cmd = np.zeros(5)
    # A stale pose command from the previous grip: nonzero displacement.
    stale = types.SimpleNamespace()
    stale.pose = types.SimpleNamespace()
    stale.pose.position = types.SimpleNamespace(x=0.05, y=-0.03, z=0.02)
    stale.pose.orientation = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    node._latest_pose = stale
    node._latest_pose_stamp = 0.0
    node._last_input_time = 0.0
    node._accept_pose_commands = True
    # Stubs for _on_start_srv's measured-seed + IK path (re-latch on re-grip).
    node._measured_arm_joints = lambda: np.zeros(5)
    node.diffik = types.SimpleNamespace(
        ee_position=lambda q: np.array([0.2, 0.0, 0.3], dtype=np.float64),
        ee_rotation=lambda q: np.eye(3, dtype=np.float64),
    )
    node._now = lambda: 123.0
    node.get_logger = lambda: _FakeLogger()
    return node


def test_home_clears_stale_pose_command():
    node = _make_node()
    resp = node._on_home_srv(Trigger.Request(), Trigger.Response())

    assert resp.success is True
    # The cached pose command MUST be dropped so the next tick does not add its
    # displacement onto the home baseline (home + offset regression).
    assert node._latest_pose is None
    assert node._latest_pose_stamp == 123.0
    # Reference and clutch baseline are set to home.
    np.testing.assert_allclose(node._p_ref, node._home_p)
    np.testing.assert_allclose(node._ee0_p, node._home_p)


def test_home_baseline_plus_dropped_command_would_be_home_not_offset():
    """Reproduce the hot-path recompute: with the cache cleared, a fresh tick has
    no stale displacement to add, so the target equals home exactly."""
    node = _make_node()
    node._on_home_srv(Trigger.Request(), Trigger.Response())

    # Emulate the hot path's fresh-command computation guard: pose_stale is True
    # because _latest_pose is None, so _p_ref stays at home (not home+offset).
    pose_stale = node._latest_pose is None
    assert pose_stale is True
    # Had the cache NOT been cleared, this is the wrong target it would have hit:
    wrong = node._ee0_p + np.array([0.05, -0.03, 0.02])
    assert not np.allclose(node._p_ref, wrong)
    np.testing.assert_allclose(node._p_ref, node._home_p)


# --------------------------------------------------------------------------- #
# Fix #3: the _accept_pose_commands gate. Clearing _latest_pose does not stop a
# pose message that was already queued in DDS when a stop/home ran from landing
# in _on_pose AFTERWARD. The gate closes on stop/home and only the next start
# re-opens it, so such a late message is rejected instead of reviving a stale
# displacement.
# --------------------------------------------------------------------------- #
def _fresh_pose():
    p = types.SimpleNamespace()
    p.pose = types.SimpleNamespace()
    p.pose.position = types.SimpleNamespace(x=0.09, y=0.08, z=0.07)
    p.pose.orientation = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    return p


def test_on_pose_rejected_while_gate_closed():
    node = _make_node()
    node._accept_pose_commands = False
    node._latest_pose = None
    node._on_pose(_fresh_pose())
    # Gate closed: the late/queued message must be dropped, not cached.
    assert node._latest_pose is None


def test_on_pose_accepted_while_gate_open():
    node = _make_node()
    node._accept_pose_commands = True
    node._latest_pose = None
    msg = _fresh_pose()
    node._on_pose(msg)
    assert node._latest_pose is msg


def test_stop_closes_pose_gate():
    node = _make_node()
    node._accept_pose_commands = True
    node._on_stop_srv(Trigger.Request(), Trigger.Response())
    assert node._accept_pose_commands is False
    # A pose that lands after the stop is now rejected (not stored).
    node._latest_pose = None
    node._on_pose(_fresh_pose())
    assert node._latest_pose is None


def test_home_keeps_pose_gate_closed_until_next_start():
    node = _make_node()
    node._accept_pose_commands = True  # was enabled/gripping, then B pressed
    node._on_home_srv(Trigger.Request(), Trigger.Response())
    # Home drops the stale cache AND keeps the gate closed: a pose still queued
    # in DDS from before home ran would otherwise re-apply the previous grip's
    # displacement onto the freshly-latched home baseline. The single-threaded
    # executor cannot distinguish that late stale frame from a genuinely new one.
    assert node._accept_pose_commands is False
    assert node._latest_pose is None
    # A late/stale pose arriving after the home callback returned is rejected.
    node._on_pose(_fresh_pose())
    assert node._latest_pose is None
    # Only the next start (which re-latches the clutch baseline) re-opens it.
    node._on_start_srv(Trigger.Request(), Trigger.Response())
    assert node._accept_pose_commands is True
    msg = _fresh_pose()
    node._on_pose(msg)
    assert node._latest_pose is msg


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
