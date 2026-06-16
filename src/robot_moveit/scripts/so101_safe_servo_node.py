#!/usr/bin/env python3
"""SO101 Safe Servo node.

Always-solvable Cartesian teleop backend for the SO-101 5-DOF arm. Decomposes
the user twist into two parallel pipelines:

* **Linear (base frame)** → position-only IK at ``gripper`` with pending
    wrist targets injected into ``start_joint_state``. The published wrist
    command blends the local wrist accumulator with the J4/J5 values returned
    by IK, preserving smooth direct wrist control while allowing a small IK
    correction when configured.
* **Angular (tool frame)** → direct integration into joint 4 (pitch) and
  joint 5 (roll). Joint 4 follows ``angular.y``, joint 5 follows
  ``angular.z``. No Jacobian / projection involved — the wrist accumulators
  drive PositionJointInterface targets directly.

This guarantees IK never fails on rotational under-actuation: position is
solvable by 3 joints in a 3-DOF subspace, orientation is bypass-controlled
by 2 dedicated joints.

Safety stack (in input-tick order):

1. **Deadband** before integration (Xbox zero-drift kills wrist creep).
2. **Slew-rate clamp** on per-tick wrist delta (servo + 3D-print safety).
3. **Joint-limit clamp** against URDF limits.
4. **Pending-candidate** state machine: candidates are committed only on IK
    success; on IK failure the virtual target snaps back to ``base→gripper``
   TF and wrist accumulators snap back to ``/joint_states`` — preventing
   the classic "virtual target walks into unreachable space" windup.
5. **Future-state collision check**: ``start_joint_state`` passed to
   ``compute_ik_async`` already contains the pending j4/j5 so MoveIt's
   collision checking evaluates the configuration that will actually be
   commanded.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Vector3Stamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor  # noqa: F401
from rclpy.node import Node
from rclpy.task import Future
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from pymoveit2 import MoveIt2


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SO101SafeServoNode(Node):
    """Position-only gripper IK + wrist-accumulator node for SO-101."""

    # ------------------------------------------------------------------ init
    def __init__(self) -> None:
        super().__init__("so101_safe_servo_node")

        self.cb_group = ReentrantCallbackGroup()
        # The solve tick must never run concurrently with itself.
        # MutuallyExclusiveCallbackGroup is kept even though SingleThreadedExecutor
        # makes re-entrancy impossible: it documents intent and protects against
        # future executor changes (MultiThreadedExecutor + ReentrantCallbackGroup
        # could fire the timer again while the previous invocation is still running,
        # causing two solve ticks to both see _pending_ik is None, both call
        # compute_ik_async, and both mutate __compute_ik_req → IK corruption).
        self.solve_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- parameters ----
        self.declare_parameter("move_group_name", "arm")
        self.declare_parameter("planning_frame", "base")
        self.declare_parameter("ik_link_name", "gripper")  # must be chain tip recognised by MoveIt
        self.declare_parameter("arm_joint_names", ["1", "2", "3", "4", "5"])
        self.declare_parameter("pitch_joint", "4")
        self.declare_parameter("roll_joint", "5")
        self.declare_parameter("pitch_axis_index", 1)  # 0=x,1=y,2=z; int avoids YAML-bool trap
        self.declare_parameter("roll_axis_index", 2)  # 0=x,1=y,2=z
        self.declare_parameter("pitch_scale", 1.0)
        self.declare_parameter("roll_scale", 1.0)

        # --- Safety & Physical Boundaries (SSOT enforced by YAML) ---
        self.declare_parameter("angular_deadband")
        self.declare_parameter("max_wrist_angular_speed")
        self.declare_parameter("wrist_ik_blend")
        self.declare_parameter("wrist_drift_warn_threshold")

        # Joint limits must be provided via YAML (so101_safe_servo.yaml).
        # They mirror the URDF link limits but are declared here as required
        # parameters so the node fails early if the config is not loaded.
        self.declare_parameter("joint_limit_lower.pitch")
        self.declare_parameter("joint_limit_upper.pitch")
        self.declare_parameter("joint_limit_lower.roll")
        self.declare_parameter("joint_limit_upper.roll")
        self.declare_parameter("linear_cmd_topic", "/so101_safe_servo_node/linear_cmd_base")
        self.declare_parameter("angular_cmd_topic", "/so101_safe_servo_node/angular_cmd_tool")
        self.declare_parameter("start_service", "/so101_safe_servo_node/start")
        self.declare_parameter("stop_service", "/so101_safe_servo_node/stop")
        self.declare_parameter("command_out_topic", "/arm_position_controller/commands")
        self.declare_parameter("input_period", 0.02)  # 50 Hz input/integration
        self.declare_parameter("ik_publish_period_s", 0.04)  # 25 Hz IK
        self.declare_parameter("incoming_command_timeout", 0.5)
        self.declare_parameter("target_reset_timeout_s", 2.0)
        self.declare_parameter("ik_wall_timeout_s", 2.5)
        self.declare_parameter("tf_stale_threshold_s", 0.2)

        # ---- snapshot ----
        self.group_name = self.get_parameter("move_group_name").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.ik_link_name = self.get_parameter("ik_link_name").value
        self.arm_joint_names: list[str] = list(self.get_parameter("arm_joint_names").value)
        self.pitch_joint = str(self.get_parameter("pitch_joint").value)
        self.roll_joint = str(self.get_parameter("roll_joint").value)
        self.pitch_axis_index = int(self.get_parameter("pitch_axis_index").value)
        self.roll_axis_index = int(self.get_parameter("roll_axis_index").value)
        self.pitch_scale = float(self.get_parameter("pitch_scale").value)
        self.roll_scale = float(self.get_parameter("roll_scale").value)
        self.angular_deadband = float(self.get_parameter("angular_deadband").value)
        self.max_wrist_angular_speed = float(self.get_parameter("max_wrist_angular_speed").value)
        _jl = {
            k: self.get_parameter(k).value
            for k in (
                "joint_limit_lower.pitch",
                "joint_limit_upper.pitch",
                "joint_limit_lower.roll",
                "joint_limit_upper.roll",
            )
        }
        _missing = [k for k, v in _jl.items() if v is None]
        if _missing:
            raise RuntimeError(
                f"Required parameters not set: {_missing}. "
                "Load so101_safe_servo.yaml (set moveit.so101_safe_servo_config_path)."
            )
        self.pitch_lo = float(_jl["joint_limit_lower.pitch"])
        self.pitch_hi = float(_jl["joint_limit_upper.pitch"])
        self.roll_lo = float(_jl["joint_limit_lower.roll"])
        self.roll_hi = float(_jl["joint_limit_upper.roll"])
        self.linear_topic = self.get_parameter("linear_cmd_topic").value
        self.angular_topic = self.get_parameter("angular_cmd_topic").value
        self.start_srv_name = self.get_parameter("start_service").value
        self.stop_srv_name = self.get_parameter("stop_service").value
        self.cmd_out_topic = self.get_parameter("command_out_topic").value
        self.input_period = float(self.get_parameter("input_period").value)
        self.ik_period = float(self.get_parameter("ik_publish_period_s").value)
        self.input_timeout = float(self.get_parameter("incoming_command_timeout").value)
        self.target_reset_timeout = float(self.get_parameter("target_reset_timeout_s").value)
        self.ik_wall_timeout = float(self.get_parameter("ik_wall_timeout_s").value)
        self.tf_stale_threshold_s = float(self.get_parameter("tf_stale_threshold_s").value)
        self.wrist_ik_blend = _clamp(float(self.get_parameter("wrist_ik_blend").value), 0.0, 1.0)
        self.wrist_drift_warn_threshold = float(self.get_parameter("wrist_drift_warn_threshold").value)

        if self.pitch_joint not in self.arm_joint_names or self.roll_joint not in self.arm_joint_names:
            raise RuntimeError(
                f"pitch_joint={self.pitch_joint!r} / roll_joint={self.roll_joint!r} "
                f"must appear in arm_joint_names={self.arm_joint_names}"
            )

        # ---- state ----
        self._enabled: bool = False
        # Accepted (committed) targets:
        self._target_position: np.ndarray | None = None  # base-frame ik_link position
        self._pitch_target: float | None = None
        self._roll_target: float | None = None
        # Pending (in-flight) targets — only committed on IK success:
        self._pending_target_position: np.ndarray | None = None
        self._pending_pitch_target: float | None = None
        self._pending_roll_target: float | None = None
        # Latest input cache:
        self._latest_linear: Vector3Stamped | None = None
        self._latest_linear_stamp: float = 0.0
        self._latest_angular: Vector3Stamped | None = None
        self._latest_angular_stamp: float = 0.0
        self._latest_js: JointState | None = None
        self._last_input_time: float = 0.0
        self._pending_ik: Future | None = None
        self._pending_ik_deadline: float = 0.0
        # Snapshot of the pending targets captured at IK-issue time.
        # On IK success we commit the SNAPSHOT (what was actually solved), not
        # the CURRENT _pending_* (which may have advanced several ticks while
        # IK was in flight). Committing the current pending causes _target_position
        # to jump ahead, making the next integration start from the wrong baseline.
        self._ik_snapshot_pos: np.ndarray | None = None
        self._ik_snapshot_pitch: float | None = None
        self._ik_snapshot_roll: float | None = None

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- MoveIt2 ----
        # end_effector_name is the SRDF tip used by compute_ik_async.
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name=self.planning_frame,
            end_effector_name=self.ik_link_name,
            group_name=self.group_name,
            callback_group=self.cb_group,
        )

        # ---- ROS I/O ----
        self.create_subscription(
            Vector3Stamped,
            self.linear_topic,
            self._on_linear,
            10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Vector3Stamped,
            self.angular_topic,
            self._on_angular,
            10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
            callback_group=self.cb_group,
        )
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.cmd_out_topic, 10)

        self.create_service(Trigger, self.start_srv_name, self._on_start_srv, callback_group=self.cb_group)
        self.create_service(Trigger, self.stop_srv_name, self._on_stop_srv, callback_group=self.cb_group)

        # ---- timers ----
        self.create_timer(self.input_period, self._on_input_tick, callback_group=self.cb_group)
        # Solve tick uses its own MutuallyExclusiveCallbackGroup (see __init__).
        self.create_timer(self.ik_period, self._on_solve_tick, callback_group=self.solve_cb_group)

        self.get_logger().info(
            f"so101_safe_servo_node up: group={self.group_name} ik_link={self.ik_link_name} "
            f"pitch={self.pitch_joint}(axis{self.pitch_axis_index}) roll={self.roll_joint}(axis{self.roll_axis_index}) "
            f"input={self.input_period * 1000:.0f}ms ik={self.ik_period * 1000:.0f}ms "
            f"deadband={self.angular_deadband} max_wrist_speed={self.max_wrist_angular_speed} rad/s"
        )

    # ------------------------------------------------------------------
    # subscribers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # services
    # ------------------------------------------------------------------

    def _on_start_srv(self, _req, resp: Trigger.Response) -> Trigger.Response:
        try:
            self._reset_all_from_robot_state()
        except Exception as exc:  # noqa: BLE001
            resp.success = False
            resp.message = f"reset_all_from_robot_state failed: {exc}"
            self.get_logger().error(resp.message)
            return resp
        self._enabled = True
        self._last_input_time = self._now()
        resp.success = True
        resp.message = "so101_safe_servo_node enabled"
        self.get_logger().info(resp.message)
        return resp

    def _on_stop_srv(self, _req, resp: Trigger.Response) -> Trigger.Response:
        self._enabled = False
        self._target_position = None
        self._pitch_target = None
        self._roll_target = None
        self._pending_target_position = None
        self._pending_pitch_target = None
        self._pending_roll_target = None
        self._pending_ik = None
        self._ik_snapshot_pos = None
        self._ik_snapshot_pitch = None
        self._ik_snapshot_roll = None
        resp.success = True
        resp.message = "so101_safe_servo_node disabled"
        self.get_logger().info(resp.message)
        return resp

    # ------------------------------------------------------------------
    # timers
    # ------------------------------------------------------------------

    def _on_input_tick(self) -> None:
        """50 Hz: build candidate (position, pitch, roll) from latest input.

        Order (plan §6):
            1. fetch raw twist (zero if stale)
            2. deadband BEFORE integration
            3. integrate to candidate
            4. slew-rate clamp on wrist delta
            5. joint-limit clamp on wrist
            6. store as _pending_* (NOT committed — IK result will commit)
        """
        if not self._enabled or self._target_position is None:
            return

        now = self._now()
        dt = self.input_period

        # Idle reset: long pause → snap targets back to live robot state so the
        # next input does not lurch from a stale virtual target.
        if (now - self._last_input_time) > self.target_reset_timeout:
            try:
                self._reset_all_from_robot_state()
                # Update _last_input_time after a successful reset so the next
                # tick does not immediately re-trigger the idle reset (the
                # condition would still be True without this update because
                # _last_input_time is only set by _on_linear/_on_angular).
                self._last_input_time = self._now()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"idle-timeout reset failed: {exc}",
                    throttle_duration_sec=1.0,
                )
            return

        # --- linear (base frame) ----------------------------------------
        if self._latest_linear is None or (now - self._latest_linear_stamp) > self.input_timeout:
            vx = vy = vz = 0.0
        else:
            v = self._latest_linear.vector
            vx, vy, vz = float(v.x), float(v.y), float(v.z)

        # --- angular (tool frame, raw semantic) -------------------------
        if self._latest_angular is None or (now - self._latest_angular_stamp) > self.input_timeout:
            wx = wy = wz = 0.0
        else:
            w = self._latest_angular.vector
            wx, wy, wz = float(w.x), float(w.y), float(w.z)

        # === Step 1: Deadband BEFORE integration (mandatory, plan §6) ===
        # Floor-noise rejection. Without this, even idle sticks accumulate.
        wx = wx if abs(wx) >= self.angular_deadband else 0.0
        wy = wy if abs(wy) >= self.angular_deadband else 0.0
        wz = wz if abs(wz) >= self.angular_deadband else 0.0

        # Pick the stick axis assigned to each wrist joint.
        w_pitch = self._pick_axis((wx, wy, wz), self.pitch_axis_index)
        w_roll = self._pick_axis((wx, wy, wz), self.roll_axis_index)

        # === Step 2: Integrate candidates ===============================
        cand_pos = self._target_position + np.array([vx, vy, vz], dtype=np.float64) * dt
        cand_pitch = self._pitch_target + w_pitch * dt * self.pitch_scale
        cand_roll = self._roll_target + w_roll * dt * self.roll_scale

        # === Step 3: Slew-rate clamp on wrist delta (mandatory, plan §6) ===
        # J4/J5 bypass MoveIt's time parameterization → guard servos and
        # 3D-printed wrist by capping per-tick angular increment.
        max_delta = self.max_wrist_angular_speed * dt
        cand_pitch = self._pitch_target + _clamp(
            cand_pitch - self._pitch_target,
            -max_delta,
            +max_delta,
        )
        cand_roll = self._roll_target + _clamp(
            cand_roll - self._roll_target,
            -max_delta,
            +max_delta,
        )

        # === Step 4: Joint-limit clamp ==================================
        cand_pitch = _clamp(cand_pitch, self.pitch_lo, self.pitch_hi)
        cand_roll = _clamp(cand_roll, self.roll_lo, self.roll_hi)

        # === Step 5: Stash as pending (commit only on IK success) ======
        self._pending_target_position = cand_pos
        self._pending_pitch_target = cand_pitch
        self._pending_roll_target = cand_roll

    def _on_solve_tick(self) -> None:
        """25 Hz: fire IK request if none pending, process result if ready."""
        if not self._enabled:
            return
        if (
            self._pending_target_position is None
            or self._pending_pitch_target is None
            or self._pending_roll_target is None
        ):
            return

        # In-flight request handling.
        if self._pending_ik is not None:
            if self._pending_ik.done():
                future, self._pending_ik = self._pending_ik, None
                self._on_ik_result(future)
                return
            if self._now() > self._pending_ik_deadline:
                self.get_logger().warn(
                    "IK request exceeded ik_wall_timeout, dropping",
                    throttle_duration_sec=1.0,
                )
                self._pending_ik = None
                # MUST return: re-issuing immediately floods move_group queue.
                return
            return  # still waiting, do not stack requests

        # Build start_joint_state from the latest measured joint state, then
        # overwrite j4/j5 with the *pending* wrist targets so MoveIt's
        # collision check evaluates the configuration we are about to publish
        # (plan §6 future-state collision rule).
        start_js = self._build_start_joint_state()
        if start_js is None:
            # No joint state yet — let MoveIt fall back to its own /joint_states.
            self.get_logger().warn(
                "no /joint_states yet; skipping IK tick",
                throttle_duration_sec=2.0,
            )
            return

        # Position-only target. Quaternion is mathematically required by
        # the MoveIt service but kinematics.yaml has position_only_ik: true,
        # so it is ignored by the plugin.
        p_tgt = self._pending_target_position

        # --- Sub-optimal pymoveit2 handling ---
        # compute_ik_async has an internal wait_for_service(timeout_sec=3.0) hack that
        # overrides our wait_for_server_timeout_sec=0.0 parameter.
        # To avoid blocking the solve loop, we check if the internal client exists and is ready.
        if hasattr(self.moveit2, "_MoveIt2__compute_ik_client"):
            ik_client = self.moveit2._MoveIt2__compute_ik_client
            if not ik_client.service_is_ready():
                self.get_logger().warn(
                    "IK service not ready; skipping solve loop.",
                    throttle_duration_sec=1.0,
                )
                return

        future = self.moveit2.compute_ik_async(
            position=(float(p_tgt[0]), float(p_tgt[1]), float(p_tgt[2])),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            ik_link_name=self.ik_link_name,
            start_joint_state=start_js,
            wait_for_server_timeout_sec=0.0,  # non-blocking: service is always up
        )
        if future is None:
            self.get_logger().warn(
                "compute_ik_async returned None",
                throttle_duration_sec=1.0,
            )
            return
        # Snapshot what we actually solved for; _on_ik_result must commit this,
        # not the current _pending_target_position which may have advanced while
        # IK was in flight.
        self._ik_snapshot_pos = p_tgt.copy()
        self._ik_snapshot_pitch = float(self._pending_pitch_target)
        self._ik_snapshot_roll = float(self._pending_roll_target)
        self._pending_ik = future
        self._pending_ik_deadline = self._now() + self.ik_wall_timeout

    # ------------------------------------------------------------------
    # IK result
    # ------------------------------------------------------------------

    def _on_ik_result(self, future: Future) -> None:
        js = self.moveit2.get_compute_ik_result(future)
        if js is None:
            self.get_logger().warn(
                "IK failed; resetting pending state from robot",
                throttle_duration_sec=1.0,
            )
            # Anti-windup: snap accepted targets back to physical robot state
            # and discard pending candidates. The next input tick will re-
            # integrate from the fresh baseline, so reversing the joystick
            # recovers immediately.
            try:
                self._reset_all_from_robot_state()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"post-failure reset failed: {exc}",
                    throttle_duration_sec=1.0,
                )
            return

        # Extract joints 1-3 from solver. J4/J5 are blended between the direct
        # wrist accumulator snapshot and the IK result, so hardware control
        # remains smooth while IK can contribute a small consistency correction.
        name_to_pos: dict[str, float] = dict(zip(js.name, js.position, strict=False))
        snap_pitch = self._ik_snapshot_pitch if self._ik_snapshot_pitch is not None else self._pending_pitch_target
        snap_roll = self._ik_snapshot_roll if self._ik_snapshot_roll is not None else self._pending_roll_target
        out_pitch, out_roll = self._blend_wrist_solution(name_to_pos, snap_pitch, snap_roll)
        out_positions: list[float] = []
        try:
            for n in self.arm_joint_names:
                if n == self.pitch_joint:
                    out_positions.append(float(out_pitch))
                elif n == self.roll_joint:
                    out_positions.append(float(out_roll))
                else:
                    out_positions.append(float(name_to_pos[n]))
        except KeyError as exc:
            self.get_logger().error(f"IK result missing joint {exc}; configured arm_joint_names={self.arm_joint_names}")
            return

        out = Float64MultiArray()
        out.data = out_positions
        self.cmd_pub.publish(out)

        # Commit the SNAPSHOT (position that was actually solved), not the
        # current _pending_target_position which may have advanced several
        # input_tick cycles while IK was in flight. Committing the current
        # pending causes _target_position to jump ahead, making the next
        # integration start from a wrong baseline → jerky / lurch motion.
        if self._ik_snapshot_pos is not None:
            self._target_position = self._ik_snapshot_pos
            self._pitch_target = out_pitch
            self._roll_target = out_roll
        else:
            # Should not happen; fall back to keep behaviour correct.
            self.get_logger().error("IK result arrived without snapshot — falling back to pending")
            self._target_position = self._pending_target_position
            self._pitch_target = self._pending_pitch_target
            self._roll_target = self._pending_roll_target

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_axis(v: tuple[float, float, float], index: int) -> float:
        """Return v[index]; index 0=x, 1=y, 2=z."""
        if 0 <= index <= 2:
            return v[index]
        return 0.0

    def _now(self) -> float:
        return time.monotonic()

    def _build_start_joint_state(self) -> JointState | None:
        """Latest measured joint state with pending wrist targets injected.

        Returns ``None`` if no ``/joint_states`` has arrived yet.
        """
        if self._latest_js is None:
            return None
        measured: dict[str, float] = dict(zip(self._latest_js.name, self._latest_js.position, strict=False))

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(self.arm_joint_names)
        positions: list[float] = []
        for n in self.arm_joint_names:
            if n == self.pitch_joint and self._pending_pitch_target is not None:
                positions.append(float(self._pending_pitch_target))
            elif n == self.roll_joint and self._pending_roll_target is not None:
                positions.append(float(self._pending_roll_target))
            else:
                # measured may not have all 5 names yet (e.g. very early
                # startup); fall back to 0.0 — MoveIt will refine.
                positions.append(float(measured.get(n, 0.0)))
        out.position = positions
        return out

    def _blend_wrist_solution(
        self,
        name_to_pos: dict[str, float],
        snap_pitch: float | None,
        snap_roll: float | None,
    ) -> tuple[float, float]:
        """Blend direct wrist targets with the IK-returned wrist joints."""
        if snap_pitch is None or snap_roll is None:
            raise RuntimeError("missing wrist snapshot for IK result")
        try:
            ik_pitch = float(name_to_pos[self.pitch_joint])
            ik_roll = float(name_to_pos[self.roll_joint])
        except KeyError as exc:
            self.get_logger().warn(
                f"IK result missing wrist joint {exc}; using direct wrist snapshot",
                throttle_duration_sec=1.0,
            )
            return float(snap_pitch), float(snap_roll)

        pitch_err = abs(ik_pitch - float(snap_pitch))
        roll_err = abs(ik_roll - float(snap_roll))
        if max(pitch_err, roll_err) > self.wrist_drift_warn_threshold:
            self.get_logger().warn(
                "IK wrist result differs from direct target; blending correction "
                f"pitch_err={pitch_err:.4f} roll_err={roll_err:.4f} "
                f"ik_weight={self.wrist_ik_blend:.2f}",
                throttle_duration_sec=1.0,
            )

        direct_weight = 1.0 - self.wrist_ik_blend
        out_pitch = direct_weight * float(snap_pitch) + self.wrist_ik_blend * ik_pitch
        out_roll = direct_weight * float(snap_roll) + self.wrist_ik_blend * ik_roll
        return (
            _clamp(out_pitch, self.pitch_lo, self.pitch_hi),
            _clamp(out_roll, self.roll_lo, self.roll_hi),
        )

    def _lookup_ee_position(self) -> np.ndarray:
        """Return base→ik_link translation without blocking the executor.

        Uses timeout=0 (returns latest cached TF immediately or raises
        tf2_ros.LookupException/ExtrapolationException if not yet available).
        After a successful lookup the transform's header stamp is compared with
        the current ROS clock; if the difference exceeds tf_stale_threshold_s
        a RuntimeError is raised so callers can refuse to reset to a stale
        EE pose.  Callers must handle both TF and RuntimeError exceptions.
        """
        tr = self.tf_buffer.lookup_transform(
            self.planning_frame,
            self.ik_link_name,
            Time(),  # latest available
            Duration(seconds=0),  # non-blocking: never wait
        )
        # Staleness guard: tf_buffer caches the last received transform
        # indefinitely; if TF publishing stops the cached value becomes stale.
        # ExtrapolationException is NOT thrown for Time() lookups, so we must
        # check the header stamp explicitly.
        now_s = self.get_clock().now().nanoseconds * 1e-9
        tf_stamp_s = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
        age_s = now_s - tf_stamp_s
        if age_s > self.tf_stale_threshold_s:
            raise RuntimeError(
                f"TF {self.planning_frame}<-{self.ik_link_name} is stale "
                f"({age_s:.3f}s > threshold {self.tf_stale_threshold_s}s); "
                "refusing EE pose reset to avoid windup from stale baseline"
            )
        t = tr.transform.translation
        return np.array([t.x, t.y, t.z], dtype=np.float64)

    def _reset_all_from_robot_state(self) -> None:
        """Snap accepted + pending targets back to live robot state.

        Used on enable, idle timeout, and IK failure (plan §6 anti-windup).
        Raises if TF is unavailable so the caller decides how to react.
        """
        pos = self._lookup_ee_position()
        self._target_position = pos
        self._pending_target_position = pos.copy()

        # Wrist accumulators: prefer measured joint state. If unavailable,
        # fall back to 0.0 (safe centre).
        pitch = 0.0
        roll = 0.0
        if self._latest_js is not None:
            for n, p in zip(self._latest_js.name, self._latest_js.position, strict=False):
                if n == self.pitch_joint:
                    pitch = float(p)
                elif n == self.roll_joint:
                    roll = float(p)
        self._pitch_target = pitch
        self._roll_target = roll
        self._pending_pitch_target = pitch
        self._pending_roll_target = roll
        self.get_logger().info(
            f"snapped to robot state: pos={pos.tolist()} pitch={pitch:.4f} roll={roll:.4f}",
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SO101SafeServoNode()
    # Use SingleThreadedExecutor: MultiThreadedExecutor was confirmed to stop
    # dispatching all callbacks for up to 7 s (EXEC_STALL active_count=0).
    # All callbacks complete in <5 ms; single-threaded dispatch is sufficient.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
