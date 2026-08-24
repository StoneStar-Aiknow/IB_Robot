from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

NAVIGATION_FIELDS = (
    "string direction",
    "float64 distance",
    "float64 degree",
    "bool has_x",
    "float64 x",
    "bool has_y",
    "float64 y",
    "bool has_yaw",
    "float64 yaw",
)


def test_navigation_parameters_are_consistent_across_public_skill_interfaces():
    interfaces = (
        ROOT / "src/ibrobot_msgs/action/SkillCommand.action",
        ROOT / "src/ibrobot_msgs/msg/WorkflowStep.msg",
        ROOT / "src/ibrobot_msgs/srv/ValidateSkill.srv",
    )

    for interface in interfaces:
        request_contract = interface.read_text(encoding="utf-8").partition("\n---\n")[0]
        for field in NAVIGATION_FIELDS:
            assert field in request_contract, f"{interface.name} is missing {field}"


def test_resolved_navigation_payload_is_consistent_across_primitive_interfaces():
    interfaces = (
        ROOT / "src/ibrobot_msgs/action/PrimitiveCommand.action",
        ROOT / "src/ibrobot_msgs/srv/ValidatePrimitive.srv",
    )
    navigation_fields = (
        "uint8 navigation_command_type",
        "geometry_msgs/PoseStamped navigation_target_pose",
        "float64 navigation_value",
    )

    for interface in interfaces:
        request_contract = interface.read_text(encoding="utf-8").partition("\n---\n")[0]
        for field in navigation_fields:
            assert field in request_contract, f"{interface.name} is missing {field}"
