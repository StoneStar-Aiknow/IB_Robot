#!/usr/bin/env python3
"""SO101 Placo Servo node.

In-process Placo QP differential-IK Cartesian teleop backend for the SO-101
5-DOF arm. ``placo_servo`` is the SO-101 Cartesian teleop solver selected by
``robot.teleoperation.cartesian.solver == 'placo_servo'``.

Why an independent node:

* **Does not block the 50 Hz teleop loop** — Placo IK runs on this node's own
  timer, isolated from ``teleop_node``'s 5 ms latency budget.

Design (Jacobian + QP velocity-level differential IK, see
``so101_placo_kinematics.SO101PlacoDiffIK``):

* **Command-side reference, not measured-pose ratchet.** The node maintains a
  virtual Cartesian reference and advances it by ``v * dt``. The QP is seeded
  from the last published command, not from hardware ``/joint_states`` every
  tick: real servos sag/lag under gravity, and feeding that measured lag into a
  5-DOF position-only QP makes the nullspace drift. Measured joints initialise
  enable/reset only; the command path sends the same ideal trajectory to
  hardware that simulation computes.
* **Position primary, orientation soft.** A Placo ``PositionTask`` constrains
  target position and an optional low-weight ``OrientationTask`` follows angular
  joystick input without letting orientation dominate the under-actuated arm.
* **Hard limits in the QP.** Joint + velocity limits are enforced
  inside the QP (``enable_joint_limits`` / ``enable_velocity_limits``), so the
  unreachable velocity component is projected onto the feasible set (correct
  energy decomposition) and the arm yields at the reachable edge. In Cartesian
  mode the arm command chain bypasses ``TeleopNode``'s ``SafetyFilter``, so this
  in-solver clamp is the authoritative joint-limit guard; the node also clamps
  the published command to the configured limits as defense in depth.
* **Hardware-safe seed.** Every step seeds from the command-side joint state
  (last published command). ``/joint_states`` remains a diagnostic/initial latch
  source, not the hot-path seed.
* **Same control point path for gripper & tcp.** ``target_frame`` is
  ``ik_link_name`` (== ``moveit.ee_link``); tcp is a fixed child of gripper, and
  Placo's frame Jacobian propagates the 95 mm lever arm exactly — no special
  case.

Smoothness: a fixed-rate timer (default 50 Hz), in-QP velocity limit as the real
speed ceiling, and an optional first-order low-pass on the joint command.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Vector3Stamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

# The radian-native Placo wrapper lives next to this node (installed into the
# same lib/<pkg> directory). Make the script dir importable so both the
# installed node and a direct source-tree run resolve it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from so101_placo_kinematics import SO101PlacoDiffIK  # noqa: E402


def _require_placo(logger) -> None:
    try:
        import placo  # noqa: F401
    except ImportError as exc:
        logger.fatal(
            "placo is required for solver=placo_servo. Run ./scripts/setup.sh "
            "to install LeRobot with the kinematics extra, or install the "
            "matching LeRobot extra manually with: python3 -m pip install -e "
            "'libs/lerobot[kinematics]'"
        )
        raise RuntimeError("placo is required for solver=placo_servo") from exc


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _rotation_delta(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix for a base-frame angular delta vector."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = np.asarray(rotvec, dtype=np.float64) / theta
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


class SO101PlacoServoNode(Node):
    """In-process Placo QP differential-IK Cartesian servo for SO-101."""

    # ------------------------------------------------------------------ init
    def __init__(self) -> None:
        super().__init__("so101_placo_servo_node")

        self.cb_group = MutuallyExclusiveCallbackGroup()

        # ---- parameters ----
        self.declare_parameter("planning_frame", "base")
        self.declare_parameter("ik_link_name", "gripper")  # target frame in URDF
        self.declare_parameter("arm_joint_names", ["1", "2", "3", "4", "5"])

        # Set true only for diagnostics. Normal SO-101 Cartesian teleop uses
        # position primary + a low-weight orientation task.
        self.declare_parameter("position_only", False)

        # Differential-IK damping: regularization weight giving DLS-like smooth
        # yielding at singularities / reachable boundaries.
        self.declare_parameter("diffik_damping", 1e-3)

        # Orientation tracking: small soft weight, lower than position.
        # Ignored when position_only=true.
        self.declare_parameter("orientation_weight", 0.01)

        # Per-joint velocity ceiling (rad/s) for the QP velocity-limit
        # constraint. The SO-101 URDF declares 10 rad/s (too fast for hand
        # teleop); this is the real speed ceiling of the servo.
        self.declare_parameter("max_joint_speed", 2.0)

        # Smoothness knob: optional first-order low-pass on q_des. The
        # real speed ceiling is the in-QP velocity limit, not a per-tick clip.
        self.declare_parameter("output_lowpass_alpha", 0.0)  # 0 = off; (0,1] = on

        # Arm joint limits (rad) — self-owned safety. Required: the
        # node refuses to start without them so a misconfig fails loudly.
        # dynamic_typing avoids the "declare name only" deprecation while still
        # treating absence as "not set" (validated below).
        from rcl_interfaces.msg import ParameterDescriptor

        _dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("joint_limits_lower", descriptor=_dyn)
        self.declare_parameter("joint_limits_upper", descriptor=_dyn)

        # Topics & services consumed by PlacoServoBackend.
        self.declare_parameter("linear_cmd_topic", "/so101_placo_servo_node/linear_cmd_base")
        self.declare_parameter("angular_cmd_topic", "/so101_placo_servo_node/angular_cmd_base")
        self.declare_parameter("start_service", "/so101_placo_servo_node/start")
        self.declare_parameter("stop_service", "/so101_placo_servo_node/stop")
        self.declare_parameter("command_out_topic", "/arm_position_controller/commands")

        # Timing.
        self.declare_parameter("control_period", 0.02)  # 50 Hz solve/publish
        self.declare_parameter("incoming_command_timeout", 0.5)
        self.declare_parameter("target_reset_timeout_s", 2.0)
        self.declare_parameter("tf_stale_threshold_s", 0.2)

        # Optional pre-expanded URDF path; if absent the node expands the
        # so101 xacro in-memory at runtime (decision: runtime expansion).
        self.declare_parameter("urdf_path", "")

        # ---- snapshot ----
        self.planning_frame = self.get_parameter("planning_frame").value
        self.ik_link_name = self.get_parameter("ik_link_name").value
        self.arm_joint_names: list[str] = list(self.get_parameter("arm_joint_names").value)
        self.position_only = bool(self.get_parameter("position_only").value)
        self.diffik_damping = float(self.get_parameter("diffik_damping").value)
        self.orientation_weight = float(self.get_parameter("orientation_weight").value)
        self.max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self.output_lowpass_alpha = _clamp(float(self.get_parameter("output_lowpass_alpha").value), 0.0, 1.0)

        def _opt_param(name):
            try:
                value = self.get_parameter(name).value
            except Exception:  # noqa: BLE001 — treat unset as None
                return None
            return value

        jl_lower = _opt_param("joint_limits_lower")
        jl_upper = _opt_param("joint_limits_upper")
        if jl_lower is None or jl_upper is None:
            raise RuntimeError(
                "joint_limits_lower / joint_limits_upper are required. "
                "Load so101_placo_servo.yaml (set moveit.so101_placo_servo_config_path)."
            )
        self.joint_lo = np.array(jl_lower, dtype=np.float64)
        self.joint_hi = np.array(jl_upper, dtype=np.float64)
        if len(self.joint_lo) != len(self.arm_joint_names) or len(self.joint_hi) != len(self.arm_joint_names):
            raise RuntimeError(
                f"joint_limits_lower/upper length must match arm_joint_names ({len(self.arm_joint_names)})"
            )

        self.linear_topic = self.get_parameter("linear_cmd_topic").value
        self.angular_topic = self.get_parameter("angular_cmd_topic").value
        self.start_srv_name = self.get_parameter("start_service").value
        self.stop_srv_name = self.get_parameter("stop_service").value
        self.cmd_out_topic = self.get_parameter("command_out_topic").value
        self.control_period = float(self.get_parameter("control_period").value)
        self.input_timeout = float(self.get_parameter("incoming_command_timeout").value)
        self.target_reset_timeout = float(self.get_parameter("target_reset_timeout_s").value)
        self.tf_stale_threshold_s = float(self.get_parameter("tf_stale_threshold_s").value)
        urdf_path = self.get_parameter("urdf_path").value or None

        # ---- Placo differential-IK (in-process, radian-native, IB-Robot URDF) ----
        _require_placo(self.get_logger())
        self.diffik = SO101PlacoDiffIK(
            urdf_path=urdf_path,
            target_frame=self.ik_link_name,
            arm_joint_names=self.arm_joint_names,
            damping=self.diffik_damping,
            control_period=self.control_period,
            max_joint_speed=self.max_joint_speed,
            orientation_weight=0.0 if self.position_only else self.orientation_weight,
        )

        # ---- state ----
        self._enabled: bool = False
        # Command-side IK state: hardware /joint_states are used to initialise
        # and reset, but NOT as the per-tick IK seed. Real servos lag/sag under
        # gravity; if that measured lag is fed back into a 5-DOF position-only
        # QP, the under-constrained nullspace drifts (seen on hardware as J4
        # snapping while simulation stayed correct). Instead, _last_cmd is the
        # seed and _p_ref is the held Cartesian reference. This sends the same
        # ideal command trajectory to hardware that simulation computes, while
        # still letting enable snap cleanly to the real measured state.
        self._p_ref: np.ndarray | None = None  # command-side EE reference (base frame, m)
        self._r_ref: np.ndarray | None = None  # command-side EE orientation (base frame, 3x3)
        self._last_cmd: np.ndarray | None = None  # last published joint command
        self._latest_linear: Vector3Stamped | None = None
        self._latest_linear_stamp: float = 0.0
        self._latest_angular: Vector3Stamped | None = None
        self._latest_angular_stamp: float = 0.0
        self._latest_js: JointState | None = None
        self._last_input_time: float = 0.0

        # ---- diagnostics counters ----
        self._recovery_count: int = 0
        self._dropped_frame_count: int = 0
        self._solve_count: int = 0

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- ROS I/O ----
        self.create_subscription(Vector3Stamped, self.linear_topic, self._on_linear, 10, callback_group=self.cb_group)
        self.create_subscription(Vector3Stamped, self.angular_topic, self._on_angular, 10, callback_group=self.cb_group)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10, callback_group=self.cb_group)
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.cmd_out_topic, 10)
        self.create_service(Trigger, self.start_srv_name, self._on_start_srv, callback_group=self.cb_group)
        self.create_service(Trigger, self.stop_srv_name, self._on_stop_srv, callback_group=self.cb_group)

        # ---- timer (fixed-rate solve+publish for smoothness) ----
        self.create_timer(self.control_period, self._on_control_tick, callback_group=self.cb_group)

        self.get_logger().info(
            f"so101_placo_servo_node up: frame={self.planning_frame} ik_link={self.ik_link_name} "
            f"joints={self.arm_joint_names} rate={1.0 / self.control_period:.0f}Hz "
            f"diffik(damping={self.diffik_damping}, max_speed={self.max_joint_speed}rad/s, "
            f"orientation_weight={0.0 if self.position_only else self.orientation_weight}) "
            f"{'POSITION-ONLY ' if self.position_only else ''}"
            f"lowpass={self.output_lowpass_alpha}"
        )

    # ------------------------------------------------------------------ subs
    def _on_linear(self, msg: Vector3Stamped) -> None:
        self._latest_linear = msg
        self._latest_linear_stamp = self._now()
        self._last_input_time = self._latest_linear_stamp

    def _on_angular(self, msg: Vector3Stamped) -> None:
        self._latest_angular = msg
        self._latest_angular_stamp = self._now()
        self._last_input_time = self._latest_angular_stamp

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_js = msg

    # ------------------------------------------------------------------ srvs
    def _on_start_srv(self, _req, resp: Trigger.Response) -> Trigger.Response:
        # Latch command-side state to the current measured pose on enable so the
        # first hold tick targets where the arm actually is (no jump). After
        # that, the hot path seeds from _last_cmd to avoid feeding hardware
        # sag/lag into the position-only QP nullspace.
        q = self._measured_arm_joints()
        if q is None:
            resp.success = False
            resp.message = "enable refused: no /joint_states seed available yet"
            self.get_logger().error(resp.message)
            return resp
        self._p_ref = self.diffik.ee_position(q)
        self._r_ref = self.diffik.ee_rotation(q)
        self._last_cmd = q.copy()
        self._enabled = True
        self._last_input_time = self._now()
        resp.success = True
        resp.message = "so101_placo_servo_node enabled"
        self.get_logger().info(resp.message)
        return resp

    def _on_stop_srv(self, _req, resp: Trigger.Response) -> Trigger.Response:
        self._enabled = False
        self._p_ref = None
        self._r_ref = None
        self._last_cmd = None
        resp.success = True
        resp.message = "so101_placo_servo_node disabled"
        self.get_logger().info(resp.message)
        return resp

    # ------------------------------------------------------------------ tick
    def _on_control_tick(self) -> None:
        """50 Hz: closed-loop QP step toward the command-side reference, publish.

        Steps:
            1. fetch linear twist (zero if stale / idle) — already in base frame
            2. read measured joints for enable/reset/diagnostics, but seed the
               QP from the command-side joint state (last published command)
            3. advance the command-side reference position by v*dt (only while
               the user commands motion; a zero hold leaves it fixed)
            4. one Placo PositionTask QP step toward that absolute reference;
               joint + velocity limits enforced inside the QP
            5. finite-check, defense-in-depth limit clamp, optional low-pass
            6. publish
        """
        if not self._enabled:
            return

        now = self._now()
        dt = self.control_period

        # Hardware /joint_states are not a stable IK seed under gravity: the
        # servos can lag the command, and feeding that lag back into a
        # position-only 5-DOF QP lets the under-constrained wrist/nullspace drift
        # (observed on hardware as J4 snapping upward while sim stayed correct).
        # Use measured joints only to initialise/reset the command-side state
        # and for diagnostics. The actual IK seed is last_cmd, i.e. the ideal
        # command trajectory — same as simulation and what the user requested
        # (“even sending the sim ideal values to hardware is OK”).
        q_measured = self._measured_arm_joints()
        if q_measured is None:
            self.get_logger().warn("no /joint_states yet; skipping solve", throttle_duration_sec=2.0)
            self._dropped_frame_count += 1
            return
        q_cmd_seed = self._last_cmd.copy() if self._last_cmd is not None else q_measured.copy()

        # --- linear velocity (base frame, m/s) ---
        # Hold to zero when input is stale or after a long idle.
        idle = (now - self._last_input_time) > self.target_reset_timeout
        if idle or self._latest_linear is None or (now - self._latest_linear_stamp) > self.input_timeout:
            v = np.zeros(3)
        else:
            lv = self._latest_linear.vector
            v = np.array([lv.x, lv.y, lv.z], dtype=np.float64)

        # --- angular velocity (base frame, rad/s) ---
        # Integrate a command-side orientation reference. The backend has
        # already converted tool-frame stick semantics into base-frame angular
        # velocity for placo_servo.
        if (
            self.position_only
            or idle
            or self._latest_angular is None
            or (now - self._latest_angular_stamp) > self.input_timeout
        ):
            w = np.zeros(3)
        else:
            av = self._latest_angular.vector
            w = np.array([av.x, av.y, av.z], dtype=np.float64)

        # === Command-side reference (the gravity-ratchet fix) ===
        # _p_ref is the EE position the arm is commanded to hold. It is advanced
        # by v*dt ONLY when the user commands motion; a zero hold leaves it
        # fixed. On a real arm the measured joints sag under gravity, so the
        # measured EE drifts away from _p_ref — and because the QP targets the
        # FIXED _p_ref (not "v*dt from the sagging measured pose"), it actively
        # corrects the sag instead of welding it in (the hardware bug).
        #
        # Do NOT re-snap _p_ref just because the sticks are idle: that would
        # reintroduce the hardware gravity ratchet after target_reset_timeout
        # (the reference would repeatedly accept the sagging measured pose).
        # Re-snap only when the reference is missing (startup/recovery); use the
        # stop/start service if a human physically moved or blocked the arm and
        # wants to accept the new pose as the command baseline.
        if self._p_ref is None:
            self._p_ref = self.diffik.ee_position(q_measured)
            self._r_ref = self.diffik.ee_rotation(q_measured)
            q_cmd_seed = q_measured.copy()
        else:
            self._p_ref = self._p_ref + v * dt
            if not self.position_only:
                if self._r_ref is None:
                    self._r_ref = self.diffik.ee_rotation(q_measured)
                self._r_ref = _rotation_delta(w * dt) @ self._r_ref

        # === QP step toward the absolute reference (Jacobian + QP) ===
        # Seeded from the command-side joints; target is the command-side _p_ref.
        # Joint + velocity limits are enforced INSIDE the QP, so a large
        # reference error converges over a few ticks rather than snapping, and
        # an unreachable reference component is projected onto the feasible set.
        try:
            if self.position_only or self._r_ref is None:
                q_des = self.diffik.solve_to_position(q_cmd_seed, self._p_ref, dt)
            else:
                q_des = self.diffik.solve_to_pose(q_cmd_seed, self._p_ref, self._r_ref, dt)
        except Exception as exc:  # noqa: BLE001
            # The QP is built to stay feasible; a raised exception is
            # unexpected. Skip this tick rather than publish a bad command; the
            # next tick re-seeds from the command-side state again.
            self.get_logger().warn(f"diffik step raised; skipping tick: {exc}", throttle_duration_sec=1.0)
            self._recovery_count += 1
            return

        if not np.all(np.isfinite(q_des)):
            self.get_logger().warn("diffik produced non-finite joints; skipping tick", throttle_duration_sec=1.0)
            self._recovery_count += 1
            return

        # === Defense-in-depth joint-limit clamp ===
        # The QP already enforces limits; this is a redundant guard because the
        # Cartesian arm chain bypasses TeleopNode's SafetyFilter.
        q_des = np.clip(q_des, self.joint_lo, self.joint_hi)

        # === Optional output low-pass for de-jitter ===
        if self.output_lowpass_alpha > 0.0 and self._last_cmd is not None:
            a = self.output_lowpass_alpha
            q_des = a * q_des + (1.0 - a) * self._last_cmd

        out = Float64MultiArray()
        out.data = [float(x) for x in q_des]
        self.cmd_pub.publish(out)
        self._last_cmd = q_des
        self._solve_count += 1

    # ------------------------------------------------------------------ helpers
    def _now(self) -> float:
        return time.monotonic()

    def _measured_arm_joints(self) -> np.ndarray | None:
        """Return measured arm joints (rad) ordered by ``arm_joint_names``."""
        if self._latest_js is None:
            return None
        measured = dict(zip(self._latest_js.name, self._latest_js.position, strict=False))
        try:
            return np.array([measured[n] for n in self.arm_joint_names], dtype=np.float64)
        except KeyError:
            return None

    def _tf_age_ok(self) -> bool:
        """Best-effort TF freshness check (diagnostics; not on the hot path)."""
        try:
            tr = self.tf_buffer.lookup_transform(self.planning_frame, self.ik_link_name, Time(), Duration(seconds=0))
        except Exception:  # noqa: BLE001
            return False
        now_s = self.get_clock().now().nanoseconds * 1e-9
        tf_s = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
        return (now_s - tf_s) <= self.tf_stale_threshold_s

    def destroy_node(self) -> bool:
        if hasattr(self, "diffik"):
            self.diffik.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SO101PlacoServoNode()
    # All callbacks are short (<5 ms) and the solve tick must never overlap
    # itself.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
