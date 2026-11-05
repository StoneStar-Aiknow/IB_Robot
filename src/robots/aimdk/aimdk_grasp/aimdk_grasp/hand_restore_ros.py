"""The runtime's public interfaces as :class:`aimdk_grasp.hand_restore.RestorePorts`.

Everything is bound by interface ID from the runtime description, like the
grasp pipeline's own inputs: the three joint views and the runtime status
(:func:`aimdk_grasp.feedback_ros.subscribe_feedback`), the arm command board,
and the hand stream (:class:`aimdk_grasp.omnihand.RosHandIO`, whose
``refusal`` is the runtime's own take-over check). The Gateway is asked over
its status service. The node needs no executor for the views: they are taken
when read (:mod:`robot_runtime.on_demand`).
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from .feedback import FeedbackHistory
from .feedback_ros import subscribe_feedback
from .hand_restore import GATEWAY_STATUS_MAX_AGE_S, CommandBoard, GatewayState, Observation, RestoreError, RuntimeState

GATEWAY_REQUEST_PERIOD_S = 0.2


class RuntimeRestorePorts:
    """Call :meth:`close` before ``node`` is destroyed."""

    def __init__(self, node, description: Mapping, *, gateway_service: str, callback_group=None):
        from ibrobot_msgs.msg import JointCommandState, RuntimeStatus
        from ibrobot_msgs.srv import GetSkillGatewayStatus
        from robot_config.contract_utils import qos_profile_from_dict
        from robot_runtime.on_demand import OnDemandSubscriptions

        from .omnihand import RosHandIO

        self._node = node
        interfaces = description["interfaces"]
        # Execution consumes only latest(), never image-time history. Keeping
        # 120 samples per view makes a post-sweep drain decode stale backlog
        # before the freshness gate; let DDS discard superseded samples.
        self.history = FeedbackHistory(history=1)
        self._subscriptions = OnDemandSubscriptions(node.context, f"{node.get_name()}_hand_restore")
        subscribe_feedback(self._subscriptions, description, self.history)
        self._status = None
        self._status_at = float("-inf")
        self._board = None
        self._board_at = float("-inf")
        for kind, name, handler in (
            (RuntimeStatus, "runtime.status", self._on_status),
            (JointCommandState, "joint.arm_command_state", self._on_board),
        ):
            interface = interfaces[name]
            self._subscriptions.subscribe(
                kind, str(interface["endpoint"]), qos_profile_from_dict(interface.get("qos")), 1, handler
            )
        self._hands = RosHandIO(node, description)
        self._client = node.create_client(GetSkillGatewayStatus, gateway_service, callback_group=callback_group)
        self._request = GetSkillGatewayStatus.Request
        self._pending = None
        self._requested_at = float("-inf")
        self._gateway = None
        self._gateway_at = float("-inf")

    # -- subscriptions --

    def _on_status(self, message, received: float | None) -> None:
        self._status, self._status_at = message, float("-inf") if received is None else received

    def _on_board(self, message, received: float | None) -> None:
        self._board, self._board_at = message, float("-inf") if received is None else received

    def _poll_gateway(self) -> None:
        now = time.monotonic()
        if self._pending is not None and self._pending.done():
            try:
                result = self._pending.result()
            except Exception:
                result = None
                self._gateway = None
            if result is not None:
                # A delayed response must not refresh an old ownership verdict.
                self._gateway, self._gateway_at = result, self._requested_at
            self._pending = None
        if self._pending is not None and now - self._requested_at > 1.0:
            self._client.remove_pending_request(self._pending)
            self._pending = None
            self._gateway = None
        if (
            self._pending is None
            and now - self._requested_at >= GATEWAY_REQUEST_PERIOD_S
            and self._client.service_is_ready()
        ):
            self._pending = self._client.call_async(self._request(schema_version=1))
            self._requested_at = now

    # -- RestorePorts --

    def prepare(self, ctx) -> None:
        """Allow lazy subscriptions one startup window, without buffering output."""
        started = time.monotonic()
        until = min(ctx.deadline, started + 3.0)
        # A previous task may have left a completed RPC or a cached ownership
        # verdict behind. Even a recent answer may name that previous task.
        # Require a new RPC made after this preparation began; never re-date
        # the old verdict or let a late old response satisfy this handshake.
        if self._pending is not None:
            self._client.remove_pending_request(self._pending)
        self._pending = None
        self._gateway = None
        self._gateway_at = self._requested_at = float("-inf")
        while True:
            if ctx.cancelled():
                raise RestoreError("POSE_CANCELLED", "canceled while waiting for hand feedback")
            if time.monotonic() >= ctx.deadline:
                raise RestoreError("TASK_POSE_TIMEOUT", "hand feedback startup exhausted the shared budget")
            self._subscriptions.drain()
            self._poll_gateway()
            try:
                self.history.latest()
                available = (
                    all(value is not None for value in (self._status, self._board, self._gateway))
                    and self._gateway_at >= started
                    and 0 <= time.monotonic() - self._gateway_at <= GATEWAY_STATUS_MAX_AGE_S
                )
            except ValueError:
                available = False
            if available:
                # RosHandIO owns an independent lazy status reader. Wait for
                # that first heartbeat too; unhealthy live status still fails
                # immediately in the restorer rather than being retried.
                status = self._status
                unhealthy = status.lifecycle != "ACTIVE" or status.faults or status.stop_latched
                if unhealthy or status.active_mode != "policy_stream" or self._hands.refusal() is None:
                    return
            if time.monotonic() >= until:
                raise RestoreError("HAND_FEEDBACK_STALE", "hand feedback, runtime or Gateway status unavailable")
            self.wait(0.02)

    def observe(self) -> Observation:
        self._subscriptions.drain()
        self._poll_gateway()
        try:
            feedback = self.history.latest()
        except ValueError as exc:
            raise RestoreError("HAND_FEEDBACK_STALE", f"body/hand feedback unavailable: {exc}") from None
        now = time.monotonic()
        stamp_age = self._node.get_clock().now().nanoseconds * 1e-9 - feedback.received_at
        status = self._status
        runtime = None
        if status is not None:
            refusal = self._hands.refusal()
            ok = status.lifecycle == "ACTIVE" and not status.faults and not status.stop_latched and not refusal
            detail = f"lifecycle={status.lifecycle} faults={list(status.faults)} stop_latched={status.stop_latched}"
            if refusal:
                detail += f"; {refusal}"
            runtime = RuntimeState(ok, detail, str(status.active_mode), now - self._status_at)
        board = self._board
        command = None
        if board is not None:
            command = CommandBoard(
                tuple(board.name),
                np.asarray(board.position, float),
                np.asarray(board.stiffness, float),
                np.asarray(board.damping, float),
                now - self._board_at,
            )
        gateway = None
        if self._gateway is not None:
            reply = self._gateway
            gateway = GatewayState(
                reply.schema_version == 1 and bool(reply.control_plane_ready),
                bool(reply.motion_authorized),
                bool(reply.busy),
                str(reply.active_task_id),
                now - self._gateway_at,
            )
        return Observation(feedback, stamp_age, command, runtime, gateway)

    def refusal(self) -> str | None:
        return self._hands.refusal()

    def publish(self, left, right) -> None:
        """Both hands in one command: the stream carries the left hand's ten joints, then the right's."""
        self._hands.publish([*np.asarray(left, float).tolist(), *np.asarray(right, float).tolist()])

    def release(self) -> None:
        self._hands.release()

    def monotonic(self) -> float:
        return time.monotonic()

    def wait(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))

    def close(self) -> None:
        if self._pending is not None:
            self._client.remove_pending_request(self._pending)
            self._pending = None
        self._node.destroy_client(self._client)
        self._hands.close()
        self._subscriptions.close()
