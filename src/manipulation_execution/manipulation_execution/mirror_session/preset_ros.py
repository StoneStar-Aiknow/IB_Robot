"""Nonblocking ROS transport for the mirror's UUID-bound named sub-action.

No ROS objects are created merely by importing this module. The session must
construct the port only for an explicitly enabled feature, using its bound
runtime description. Asset and path authorization remain the runtime's job.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT, SessionBindingError
from manipulation_execution.mirror_session.hooks import UpperBodyCommand
from manipulation_execution.mirror_session.preset import PresetEpoch, PresetSnapshot

_SHAPES = {
    "motion.named": ("action", "serve", "ibrobot_msgs/action/ExecuteNamedMotion"),
    "motion.named_hold_lease": ("topic", "subscribe", "std_msgs/msg/String"),
    "motion.named_hold_lease_ack": ("topic", "publish", "std_msgs/msg/String"),
}
_ABSOLUTE_NAME = re.compile(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*\Z")
_MAX_SUBMITTED_TARGETS = 8


def _ros_dependencies():
    """Load the real transport only when the enabled session constructs a port."""
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient
    from std_msgs.msg import String
    from unique_identifier_msgs.msg import UUID

    from ibrobot_msgs.action import ExecuteNamedMotion
    from robot_config.qos_profiles import qos_profile_from_dict

    return SimpleNamespace(
        ActionClient=ActionClient,
        ExecuteNamedMotion=ExecuteNamedMotion,
        GoalStatus=GoalStatus,
        String=String,
        UUID=UUID,
        qos=qos_profile_from_dict,
    )


def _resolve(description: Mapping) -> dict:
    interfaces = description.get("interfaces") if isinstance(description, Mapping) else None
    if not isinstance(interfaces, Mapping):
        raise SessionBindingError("description_required", "runtime.interface_description", "bind the live description")
    result = {}
    for interface_id, shape in _SHAPES.items():
        item = interfaces.get(interface_id)
        if not isinstance(item, Mapping):
            raise SessionBindingError("unknown_interface", interface_id, "ID is not in the runtime description")
        if (item.get("kind"), item.get("direction"), item.get("message_type")) != shape:
            raise SessionBindingError("invalid_interface", interface_id, f"requires {shape}")
        endpoint = item.get("endpoint")
        if not isinstance(endpoint, str) or _ABSOLUTE_NAME.fullmatch(endpoint) is None:
            raise SessionBindingError("invalid_interface", interface_id, "endpoint must be an absolute ROS name")
        qos = item.get("qos", {})
        if not isinstance(qos, Mapping):
            raise SessionBindingError("invalid_interface", interface_id, "qos must be an object")
        result[interface_id] = (endpoint, dict(qos))
    return result


def _target(command: UpperBodyCommand) -> tuple[UpperBodyCommand, list[float], str]:
    if not isinstance(command, UpperBodyCommand):
        raise ValueError("exit_target must be an UpperBodyCommand")
    try:
        arm, hand = tuple(command.arm), tuple(command.hand)
    except TypeError as exc:
        raise ValueError("exit_target must contain all 34 finite joint values") from exc
    values = arm + hand
    if (
        len(arm) != ARM_JOINT_COUNT
        or len(hand) != HAND_JOINT_COUNT
        or any(
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
            for value in values
        )
    ):
        raise ValueError("exit_target must contain all 34 finite joint values")
    values = list(map(float, values))
    encoded = json.dumps(values, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return UpperBodyCommand(tuple(values[:ARM_JOINT_COUNT]), tuple(values[ARM_JOINT_COUNT:])), values, digest


@dataclass
class _Flight:
    epoch: PresetEpoch
    name: str
    goal_id: str
    goal_uuid: Any
    dispatched: bool = False
    cancel_requested: bool = False
    cancel_sent: bool = False
    transport_uncertain: bool = False
    handle: Any = None
    submitted: OrderedDict[str, UpperBodyCommand] = field(default_factory=OrderedDict)


class RosPresetPort:
    """One action in flight; no waits, spins, services, or executor ownership.

    Callbacks close over both the epoch and its preallocated UUID. A cancel
    before lease acknowledgement prevents dispatch; a cancel while acceptance
    is pending is delivered to the late accepted handle. A cancel response is
    never treated as owner-release proof: only the runtime's terminal result
    can provide that proof after a goal has been dispatched.
    """

    def __init__(
        self,
        node,
        runtime_description: Mapping,
        callback_group=None,
        clock: Callable[[], float] = time.monotonic,
    ):
        endpoints = _resolve(runtime_description)
        self._ros = _ros_dependencies()
        self._clock = clock
        self._lock = threading.RLock()
        self._flight: _Flight | None = None
        self._state = PresetSnapshot()
        lease, lease_qos = endpoints["motion.named_hold_lease"]
        action, _ = endpoints["motion.named"]
        ack, ack_qos = endpoints["motion.named_hold_lease_ack"]
        lease_profile = self._ros.qos(lease_qos)
        ack_profile = self._ros.qos(ack_qos)
        self._publisher = node.create_publisher(self._ros.String, lease, 10 if lease_profile is None else lease_profile)
        self._client = self._ros.ActionClient(node, self._ros.ExecuteNamedMotion, action, callback_group=callback_group)
        self._subscription = node.create_subscription(
            self._ros.String,
            ack,
            self._on_lease_ack,
            10 if ack_profile is None else ack_profile,
            callback_group=callback_group,
        )

    def begin(self, name: str, epoch: PresetEpoch) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty configured motion name")
        if not isinstance(epoch, PresetEpoch):
            raise ValueError("epoch must be a PresetEpoch")
        with self._lock:
            if self._flight is not None and not (self._state.terminal and self._state.release_confirmed):
                raise RuntimeError("a named goal is still pending or lacks confirmed release")
            identifier = uuid.uuid4()
            flight = _Flight(epoch, name, identifier.hex, self._ros.UUID(uuid=list(identifier.bytes)))
            self._flight = flight
            self._state = PresetSnapshot(
                epoch=epoch,
                stage="waiting_lease_ack",
                pending=True,
                goal_id=flight.goal_id,
                observed_at=self._clock(),
            )
            self._publish(flight)

    def refresh(self, exit_target: UpperBodyCommand | None = None) -> None:
        # Validate before publishing or altering any in-flight evidence.
        submitted = _target(exit_target) if exit_target is not None else None
        with self._lock:
            flight = self._flight
            if flight is None or flight.cancel_requested or self._state.terminal:
                return
            values = None
            if submitted is not None:
                command, values, digest = submitted
                flight.submitted[digest] = command
                flight.submitted.move_to_end(digest)
                while len(flight.submitted) > _MAX_SUBMITTED_TARGETS:
                    flight.submitted.popitem(last=False)
            self._publish(flight, values)

    def cancel(self, reason: str) -> None:
        del reason  # The standard action cancel request carries only the UUID.
        with self._lock:
            flight = self._flight
            if flight is None or self._state.terminal:
                return
            flight.cancel_requested = True
            if not flight.dispatched and not flight.transport_uncertain:
                # A lease alone cannot claim the runtime's arm owner.
                self._state = replace(
                    self._state, stage="cancelled", pending=False, terminal=True, release_confirmed=True
                )
                return
            self._state = replace(self._state, stage="cancelling")
            if flight.handle is not None:
                self._cancel_handle(flight, flight.handle)

    def snapshot(self) -> PresetSnapshot:
        with self._lock:
            return self._state

    def _current(self, flight: _Flight) -> bool:
        return self._flight is flight and self._state.epoch == flight.epoch and self._state.goal_id == flight.goal_id

    def _publish(self, flight: _Flight, values: list[float] | None = None) -> None:
        payload = {"goal_id": flight.goal_id, "motion": flight.name}
        if values is not None:
            payload["exit_target"] = values
        try:
            self._publisher.publish(self._ros.String(data=json.dumps(payload, separators=(",", ":"), allow_nan=False)))
        except Exception as exc:
            flight.cancel_requested = True
            self._transport_fault(flight, "lease_publish", exc)

    def _transport_fault(self, flight: _Flight, operation: str, error: Exception) -> None:
        if not self._current(flight):
            return
        flight.transport_uncertain = True
        self._state = replace(
            self._state,
            stage="fault",
            success=False,
            release_confirmed=False,
            final_target=None,
            final_target_at=None,
            final_target_verified=False,
            fault=self._state.fault or f"preset_{operation}:{type(error).__name__}",
            observed_at=self._clock(),
        )

    def _on_lease_ack(self, message) -> None:
        try:
            if not isinstance(message.data, str) or len(message.data) > 2048:
                return
            payload = json.loads(message.data)
        except (AttributeError, TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            flight = self._flight
            if (
                flight is None
                or flight.cancel_requested
                or self._state.terminal
                or payload.get("goal_id") != flight.goal_id
                or payload.get("motion") != flight.name
            ):
                return
            if flight.dispatched:
                return
            self._state = replace(self._state, observed_at=self._clock())
            flight.dispatched = True
            self._state = replace(self._state, stage="pending")
            goal = self._ros.ExecuteNamedMotion.Goal(name=flight.name, target="both", interrupt=False, resume_token="")
            try:
                future = self._client.send_goal_async(
                    goal, feedback_callback=lambda event: self._on_feedback(flight, event), goal_uuid=flight.goal_uuid
                )
                future.add_done_callback(lambda done: self._on_goal_response(flight, done))
            except Exception as exc:
                flight.cancel_requested = True
                self._transport_fault(flight, "goal_send", exc)

    def _on_goal_response(self, flight: _Flight, future) -> None:
        with self._lock:
            try:
                handle = future.result()
                if handle is None or not isinstance(handle.accepted, bool):
                    raise ValueError("goal response did not contain an acceptance decision")
            except Exception as exc:
                self._transport_fault(flight, "goal_response", exc)
                return
            if not self._current(flight) or self._state.terminal:
                if handle.accepted:
                    self._cancel_handle(flight, handle)
                return
            if not handle.accepted:
                self._state = replace(
                    self._state,
                    stage="rejected",
                    pending=False,
                    terminal=True,
                    release_confirmed=True,
                    fault=self._state.fault or "preset_goal_rejected",
                    observed_at=self._clock(),
                )
                return
            flight.handle = handle
            self._state = replace(
                self._state,
                stage="cancelling" if flight.cancel_requested else "entering",
                pending=False,
                accepted=True,
                observed_at=self._clock(),
            )
            try:
                if bytes(handle.goal_id.uuid).hex() != flight.goal_id:
                    raise ValueError("accepted goal UUID differs from the preallocated UUID")
                result = handle.get_result_async()
                result.add_done_callback(lambda done: self._on_result(flight, done))
            except Exception as exc:
                flight.cancel_requested = True
                self._transport_fault(flight, "result_subscribe", exc)
            if flight.cancel_requested:
                self._cancel_handle(flight, handle)

    def _cancel_handle(self, flight: _Flight, handle) -> None:
        # Stale handles still have to be cancelled, without touching a new epoch.
        if flight.cancel_sent:
            return
        flight.cancel_sent = True
        try:
            future = handle.cancel_goal_async()
            future.add_done_callback(lambda done: self._on_cancel_response(flight, done))
        except Exception as exc:
            self._transport_fault(flight, "cancel_send", exc)

    def _on_cancel_response(self, flight: _Flight, future) -> None:
        with self._lock:
            if not self._current(flight) or self._state.terminal:
                return
            try:
                response = future.result()
                if response is None:
                    raise ValueError("cancel response is unavailable")
            except Exception as exc:
                self._transport_fault(flight, "cancel_response", exc)
                return
            self._state = replace(self._state, observed_at=self._clock())

    def _on_feedback(self, flight: _Flight, event) -> None:
        with self._lock:
            if not self._current(flight) or self._state.terminal or not self._state.accepted:
                return
            try:
                if bytes(event.goal_id.uuid).hex() != flight.goal_id:
                    return
                stage = event.feedback.state
                if stage not in {"accepted", "running", "completing", "entering", "holding", "exiting", "exit_blocked"}:
                    raise ValueError("unknown named hold stage")
            except Exception as exc:
                self._transport_fault(flight, "feedback", exc)
                return
            self._state = replace(
                self._state, stage="cancelling" if flight.cancel_requested else stage, observed_at=self._clock()
            )

    def _on_result(self, flight: _Flight, future) -> None:
        with self._lock:
            if not self._current(flight) or self._state.terminal:
                return
            try:
                wrapper = future.result()
                result = wrapper.result
                status = wrapper.status
                code = result.error_code
                succeeded = status == self._ros.GoalStatus.STATUS_SUCCEEDED and result.success is True
                cancelled = (
                    status == self._ros.GoalStatus.STATUS_CANCELED
                    and result.success is False
                    and code == self._ros.ExecuteNamedMotion.Result.CANCELLED
                )
                target = None
                if succeeded and code == self._ros.ExecuteNamedMotion.Result.NONE:
                    payload = json.loads(result.message)
                    if isinstance(payload, dict) and payload.get("motion") == flight.name:
                        digest = payload.get("exit_target_sha256")
                        if isinstance(digest, str):
                            target = flight.submitted.get(digest)
                verified = target is not None
                released = verified or cancelled
            except Exception as exc:
                self._transport_fault(flight, "result", exc)
                return
            now = self._clock()
            self._state = replace(
                self._state,
                stage="succeeded" if verified else "cancelled" if cancelled else "fault",
                pending=False,
                terminal=True,
                success=verified,
                release_confirmed=released,
                final_target=target,
                final_target_at=now if verified else None,
                final_target_verified=verified,
                fault=self._state.fault or ("" if released else "preset_result_unverified"),
                observed_at=now,
            )
