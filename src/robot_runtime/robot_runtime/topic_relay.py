"""Minimal topic relay for message-compatible topic renames (runtime side).

robot_runtime copy used by runtime peripheral composition (D12): the runtime
must not depend on robot_config, so the relay executable shipped by the
contract layer keeps the camera contract stable when upstream drivers publish
under driver-specific topic trees. robot_config keeps its own copy for
provider-less legacy configurations.
"""

import importlib
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def _load_message_type(type_name: str):
    parts = type_name.split("/")
    if len(parts) != 3 or parts[1] != "msg":
        raise ValueError("message type must be in the form '<package>/msg/<MessageName>'")

    package_name, _, message_name = parts
    module = importlib.import_module(f"{package_name}.msg")
    return getattr(module, message_name)


class TopicRelay(Node):
    """Relays messages from one topic to another, optionally normalizing frame_id.

    ``reliable`` upgrades both the SUBSCRIPTION and the PUBLICATION to
    reliable QoS. The subscription side matters for sources whose publishers
    are reliable and whose messages are large enough to suffer best-effort
    fragment loss (e.g. the mujoco_ros2_control camera streams); the
    publication side matters when the relay fills an endpoint whose advertised
    contract is reliable. Relays without the flag keep the historical
    best-effort sensor-data behaviour.
    """

    def __init__(
        self,
        source_topic: str,
        target_topic: str,
        message_type: str,
        target_frame_id: str | None = None,
        reliable: bool = False,
    ):
        super().__init__("topic_relay")

        self._target_frame_id = target_frame_id
        msg_cls = _load_message_type(message_type)
        if reliable:
            from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

            reliable_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            )
            publisher_qos = subscription_qos = reliable_qos
        else:
            publisher_qos = subscription_qos = qos_profile_sensor_data
        self._publisher = self.create_publisher(msg_cls, target_topic, publisher_qos)
        self._subscription = self.create_subscription(
            msg_cls,
            source_topic,
            self._relay,
            subscription_qos,
        )

        frame_suffix = f" (frame_id -> {target_frame_id})" if target_frame_id else ""
        reliable_suffix = " (reliable source)" if reliable else ""
        self.get_logger().info(
            f"Relaying {message_type}: {source_topic} -> {target_topic}{frame_suffix}{reliable_suffix}"
        )

    def _relay(self, msg):
        if self._target_frame_id and hasattr(msg, "header"):
            msg.header.frame_id = self._target_frame_id
        self._publisher.publish(msg)


def main(argv=None):
    raw_args = argv if argv is not None else sys.argv[1:]
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    args = raw_args[:ros_args_index]

    if len(args) not in (3, 4, 5):
        raise SystemExit(
            "usage: topic_relay <source_topic> <target_topic> <package/msg/MessageName> [target_frame_id] [--reliable]"
        )

    source_topic, target_topic, message_type = args[:3]
    target_frame_id = args[3] if len(args) >= 4 and args[3] != "--reliable" else None
    reliable = "--reliable" in args

    rclpy.init(args=raw_args[ros_args_index:])
    node = TopicRelay(source_topic, target_topic, message_type, target_frame_id, reliable=reliable)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
