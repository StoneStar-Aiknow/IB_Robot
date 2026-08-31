from types import SimpleNamespace

from robot_teleop.teleop_groups import parse_publish_groups
from robot_teleop.teleop_node import TeleopNode


def _publisher(messages):
    return SimpleNamespace(publish=messages.append)


def test_publish_targets_gates_each_group_independently():
    arm_messages = []
    hand_messages = []
    node = SimpleNamespace(
        publish_groups=parse_publish_groups(
            [
                {"name": "arm", "joint_names": ["1", "2"], "topic": "/arm/commands"},
                {"name": "hand", "joint_names": ["thumb", "index"], "topic": "/hand/commands"},
            ]
        ),
        command_publishers={"arm": _publisher(arm_messages), "hand": _publisher(hand_messages)},
    )

    TeleopNode._publish_targets(node, {"1": 0.1, "2": 0.2, "thumb": 0.3})

    assert [list(message.data) for message in arm_messages] == [[0.1, 0.2]]
    assert hand_messages == []


def test_publish_targets_preserves_configured_joint_order():
    messages = []
    node = SimpleNamespace(
        publish_groups=parse_publish_groups(
            [{"name": "hand", "joint_names": ["index", "thumb"], "topic": "/hand/commands"}]
        ),
        command_publishers={"hand": _publisher(messages)},
    )

    TeleopNode._publish_targets(node, {"thumb": 0.1, "index": 0.2})

    assert [list(message.data) for message in messages] == [[0.2, 0.1]]


def test_runtime_calibration_is_not_owned_by_teleop_node():
    assert not hasattr(TeleopNode, "_runtime_calibration_callback")
