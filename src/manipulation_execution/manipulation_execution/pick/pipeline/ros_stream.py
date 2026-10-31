"""The robot runtime's joint position stream as a ``StreamIO``, bound by interface ID.

Endpoints are never configured: they come from the runtime's interface
description, and each must have the expected kind, direction and message type.
Feedback merges one or more published JointState views (e.g. the arms on one,
the head on another), renaming joints a view reports under other names to the
stream's names (``feedback_names``); its age is that of the oldest view, by
when each was received; a sample the middleware did not time is dated
infinitely old, so it never passes a freshness gate. Feedback, the command
echo and the status are taken when read
(:mod:`robot_runtime.on_demand`): an idle pipeline runs no Python for them.
While the executor holds the stream, the last command is re-sent every
``keepalive_s`` so the runtime never sees it go stale; release() stops that
and the robot holds the last command. The keepalive's send, a new command and
release serialize on one lock: none of them returns while an older command is
still being sent. Output is refused unless the runtime is ACTIVE, fault-free,
not stop latched and in the mode the stream was taken over in, by a fresh
status.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .streaming import CommandSample, JointSample, received_at_or_stale

FLOAT_ARRAY = "std_msgs/msg/Float64MultiArray"
JOINT_STATE = "sensor_msgs/msg/JointState"
JOINT_COMMAND_STATE = "ibrobot_msgs/msg/JointCommandState"
RUNTIME_STATUS = "ibrobot_msgs/msg/RuntimeStatus"


class StreamBindingError(ValueError):
    pass


@dataclass(frozen=True)
class BoundTopic:
    interface_id: str
    endpoint: str
    qos: Mapping
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamBinding:
    stream: BoundTopic
    feedback: tuple[BoundTopic, ...]
    status: BoundTopic
    command: BoundTopic | None
    identity: Mapping
    mode: str
    #: A feedback name as the views report it -> the stream joint it is, so the
    #: merged feedback speaks the stream's joint names.
    feedback_aliases: Mapping[str, str] = field(default_factory=dict)


def _topic(interfaces: Mapping, interface_id: str, direction: str, message_type: str) -> BoundTopic:
    item = interfaces.get(interface_id)
    if not isinstance(item, Mapping):
        raise StreamBindingError(f"{interface_id}: not in the runtime description")
    if (item.get("kind"), item.get("direction"), item.get("message_type")) != ("topic", direction, message_type):
        raise StreamBindingError(f"{interface_id}: requires a {message_type} topic the runtime {direction}s")
    endpoint = item.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("/"):
        raise StreamBindingError(f"{interface_id}: endpoint must be an absolute ROS name")
    names = tuple(str(name) for name in item.get("joint_names") or ())
    if len(set(names)) != len(names):
        raise StreamBindingError(f"{interface_id}: joint_names repeat a joint")
    return BoundTopic(interface_id, endpoint, dict(item.get("qos") or {}), names)


def bind_stream(
    description: Mapping,
    *,
    stream: str,
    feedback: Sequence[str],
    command: str | None = None,
    status: str = "runtime.status",
    mode: str = "policy_stream",
    feedback_names: Mapping[str, str] | None = None,
) -> StreamBinding:
    """Resolve the named interfaces, or raise StreamBindingError naming the first bad one.

    ``feedback_names`` maps a stream joint to the name its feedback view reports it
    under, where the runtime names the two differently (the X2 hand stream's
    ``left_thumb_roll`` is ``L_thumb_roll_joint`` in its state view). The merged
    feedback is renamed through it, so a ``StreamIO`` reads the stream's own
    joint names.
    """
    interfaces = description.get("interfaces") if isinstance(description, Mapping) else None
    identity = description.get("robot") if isinstance(description, Mapping) else None
    if not isinstance(interfaces, Mapping) or not isinstance(identity, Mapping):
        raise StreamBindingError("bind the live runtime description first")
    bound_stream = _topic(interfaces, stream, "subscribe", FLOAT_ARRAY)
    if not bound_stream.joint_names:
        raise StreamBindingError(f"{stream}: describes no joint_names")
    if mode not in (interfaces[stream].get("modes") or ()):
        raise StreamBindingError(f"{stream}: must accept runtime mode {mode}")
    views = tuple(_topic(interfaces, view, "publish", JOINT_STATE) for view in feedback)
    reported = {name for view in views for name in view.joint_names}
    aliases = dict(feedback_names or {})
    if len(set(aliases.values())) != len(aliases):
        raise StreamBindingError("feedback_names maps two stream joints to one feedback name")
    expected = {aliases.get(name, name) for name in bound_stream.joint_names}
    if not expected <= reported:
        missing = sorted(expected - reported)
        raise StreamBindingError(f"feedback views do not report {missing}")
    bound_command = None
    if command is not None:
        bound_command = _topic(interfaces, command, "publish", JOINT_COMMAND_STATE)
        if not set(bound_stream.joint_names) <= set(bound_command.joint_names):
            raise StreamBindingError(f"{command}: does not report every joint of {stream}")
    return StreamBinding(
        bound_stream,
        views,
        _topic(interfaces, status, "publish", RUNTIME_STATUS),
        bound_command,
        dict(identity),
        mode,
        {reported_name: stream_name for stream_name, reported_name in aliases.items()},
    )


class RosStreamIO:
    """StreamIO over ROS topics; call :meth:`close` before ``node`` is destroyed."""

    def __init__(self, node, binding: StreamBinding, *, keepalive_s: float = 0.1):
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        from ibrobot_msgs.msg import JointCommandState, RuntimeStatus
        from robot_config.contract_utils import qos_profile_from_dict
        from robot_runtime.on_demand import OnDemandSubscriptions

        self._node = node
        self.binding = binding
        self.channel_joints = binding.stream.joint_names
        self._array = Float64MultiArray
        self._keepalive_s = float(keepalive_s)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._views: dict[str, tuple[float, dict[str, float]]] = {}
        self._command: CommandSample | None = None
        self._status = None
        self._last: list[float] | None = None
        self._sent_at = 0.0
        self._closed = False
        self._publisher = node.create_publisher(
            Float64MultiArray, binding.stream.endpoint, qos_profile_from_dict(binding.stream.qos)
        )
        # Only the newest of each is ever read.
        self._subscriptions = OnDemandSubscriptions(
            node.context, f"{node.get_name()}_{binding.stream.interface_id.replace('.', '_')}"
        )
        for view in binding.feedback:
            self._subscriptions.subscribe(
                JointState,
                view.endpoint,
                qos_profile_from_dict(view.qos),
                1,
                lambda message, received, key=view.interface_id: self._on_view(key, message, received),
            )
        if binding.command is not None:
            self._subscriptions.subscribe(
                JointCommandState,
                binding.command.endpoint,
                qos_profile_from_dict(binding.command.qos),
                1,
                self._on_command,
            )
        self._subscriptions.subscribe(
            RuntimeStatus, binding.status.endpoint, qos_profile_from_dict(binding.status.qos), 1, self._on_status
        )
        self._keeper = threading.Thread(
            target=self._keep_alive, daemon=True, name=f"keepalive_{binding.stream.interface_id}"
        )
        self._keeper.start()

    # -- subscriptions --------------------------------------------------------------

    def _on_view(self, key: str, message, received: float | None) -> None:
        positions = {str(name): float(value) for name, value in zip(message.name, message.position, strict=False)}
        with self._lock:
            self._views[key] = (received_at_or_stale(received), positions)

    def _on_command(self, message, received: float | None) -> None:
        names = [str(name) for name in message.name]

        def by_name(values):
            return {name: float(value) for name, value in zip(names, values, strict=False)}

        sample = CommandSample(
            received_at_or_stale(received),
            by_name(message.position),
            by_name(message.stiffness),
            by_name(message.damping),
        )
        with self._lock:
            self._command = sample

    def _on_status(self, message, _received: float | None) -> None:
        with self._lock:
            self._status = message

    def _keep_alive(self) -> None:
        """Re-send the last command once it is ``keepalive_s`` old; sleep while nothing streams.

        The send happens inside the lock, in the same critical section as the
        copy: a ``release`` or a newer ``publish`` cannot slip in between the
        two, so nothing stale leaves after either of them returns.
        """
        while True:
            with self._wake:
                while not self._closed and (self._last is None or time.monotonic() - self._sent_at < self._keepalive_s):
                    self._wake.wait(
                        None if self._last is None else self._sent_at + self._keepalive_s - time.monotonic()
                    )
                if self._closed:
                    return
                data, self._sent_at = list(self._last), time.monotonic()
                self._publisher.publish(self._array(data=data))

    # -- StreamIO -------------------------------------------------------------------

    def feedback(self) -> JointSample:
        self._subscriptions.drain()
        with self._lock:
            views = dict(self._views)
        if len(views) < len(self.binding.feedback):
            return JointSample(float("-inf"), {}, healthy=False)
        aliases = self.binding.feedback_aliases
        positions: dict[str, float] = {}
        for _, values in views.values():
            positions.update({aliases.get(name, name): value for name, value in values.items()})
        return JointSample(min(received for received, _ in views.values()), positions)

    def command(self) -> CommandSample | None:
        """The controller's last reported command; None only when the runtime reports none.

        A bound report that has not arrived yet is an infinitely old sample, so
        the executor refuses it as stale instead of running as if the robot had
        no report: that would skip the board and gain checks and take over from
        the sagging measured pose.
        """
        self._subscriptions.drain()
        with self._lock:
            if self._command is None and self.binding.command is not None:
                return CommandSample(float("-inf"), {})
            return self._command

    def publish(self, positions: Sequence[float]) -> None:
        data = [float(value) for value in positions]
        if len(data) != len(self.channel_joints):
            raise ValueError(f"a command carries all {len(self.channel_joints)} stream joints")
        with self._wake:
            self._last, self._sent_at = data, time.monotonic()
            self._publisher.publish(self._array(data=data))
            self._wake.notify()

    def release(self) -> None:
        with self._lock:
            self._last = None

    def close(self) -> None:
        with self._wake:
            self._closed, self._last = True, None
            self._wake.notify()
        self._keeper.join(timeout=1.0)
        self._subscriptions.close()

    def refusal(self) -> str | None:
        from manipulation_execution.hri.mirror_session.node import observation_from_status
        from manipulation_execution.hri.mirror_session.runtime import takeover_refusal

        self._subscriptions.drain()
        with self._lock:
            status = self._status
        if status is None:
            return "no runtime status received"
        observation = observation_from_status(status, self._node.get_clock().now().nanoseconds)
        return takeover_refusal(observation, self.binding.identity, self.binding.mode)

    def monotonic(self) -> float:
        return time.monotonic()

    def wait(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))
