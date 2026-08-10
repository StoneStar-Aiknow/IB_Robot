from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from action_dispatch.topic_executor import TopicExecutor


def test_execute_channel_publishes_only_selected_contract_topic() -> None:
    publishers: dict[str, MagicMock] = {}
    node = MagicMock()

    def create_publisher(_message_type, topic, _qos):
        publisher = MagicMock()
        publishers[topic] = publisher
        return publisher

    node.create_publisher.side_effect = create_publisher
    specs = [
        SimpleNamespace(
            topic="/arm_controller/commands",
            ros_type="std_msgs/msg/Float64MultiArray",
            names=["action.0", "action.1"],
        ),
        SimpleNamespace(
            topic="/base_controller/commands",
            ros_type="std_msgs/msg/Float64MultiArray",
            names=["action.2", "action.3"],
        ),
    ]
    executor = TopicExecutor(node, {"action_specs": specs})
    assert executor.initialize()

    executor.execute_channel("/base_controller/commands", np.array([0.0, 0.0]))

    publishers["/arm_controller/commands"].publish.assert_not_called()
    message = publishers["/base_controller/commands"].publish.call_args.args[0]
    assert list(message.data) == [0.0, 0.0]


def test_execute_channel_rejects_wrong_vector_size() -> None:
    node = MagicMock()
    node.create_publisher.return_value = MagicMock()
    spec = SimpleNamespace(
        topic="/arm_controller/commands",
        ros_type="std_msgs/msg/Float64MultiArray",
        names=["action.0", "action.1"],
    )
    executor = TopicExecutor(node, {"action_specs": [spec]})
    executor.initialize()

    try:
        executor.execute_channel("/arm_controller/commands", np.array([0.0]))
    except ValueError as exc:
        assert "expects 2 values" in str(exc)
    else:
        raise AssertionError("execute_channel accepted a partial safety command")
