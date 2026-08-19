import pytest

from robot_config.teleop_groups import (
    legacy_publish_groups,
    parse_publish_groups,
    resolve_target_publish_groups,
)


def test_parse_publish_groups_preserves_group_and_joint_order():
    groups = parse_publish_groups('[{"name":"hand","joint_names":["thumb","index"],"topic":"/hand/commands"}]')

    assert [group.name for group in groups] == ["hand"]
    assert groups[0].joint_names == ("thumb", "index")
    assert groups[0].topic == "/hand/commands"


def test_legacy_publish_groups_preserve_existing_topics_and_order():
    groups = legacy_publish_groups(
        ["1", "2"],
        ["6"],
        "/arm_position_controller/commands",
        "/gripper_position_controller/commands",
    )

    assert [group.to_dict() for group in groups] == [
        {
            "name": "arm",
            "joint_names": ["1", "2"],
            "topic": "/arm_position_controller/commands",
        },
        {
            "name": "gripper",
            "joint_names": ["6"],
            "topic": "/gripper_position_controller/commands",
        },
    ]


def test_explicit_publish_groups_cannot_mix_with_legacy_target_keys():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_target_publish_groups(
            {
                "publish_groups": [{"name": "hand", "joint_names": ["thumb"], "topic": "/hand/commands"}],
                "arm_command_topic": "/arm/commands",
            },
            {"arm": ["1"]},
        )


@pytest.mark.parametrize(
    "groups,error",
    [
        ([], None),
        ([{"name": "hand", "joint_names": [], "topic": "/hand"}], "joint_names"),
        ([{"name": "hand", "joint_names": ["thumb"], "topic": ""}], "topic"),
        (
            [
                {"name": "first", "joint_names": ["thumb"], "topic": "/hand"},
                {"name": "second", "joint_names": ["index"], "topic": "/hand"},
            ],
            "duplicate topic",
        ),
    ],
)
def test_publish_group_validation(groups, error):
    if error is None:
        assert parse_publish_groups(groups) == []
    else:
        with pytest.raises(ValueError, match=error):
            parse_publish_groups(groups)
