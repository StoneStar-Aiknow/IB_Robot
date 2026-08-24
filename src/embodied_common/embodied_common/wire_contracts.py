"""Startup checks for generated public request wire contracts."""

from __future__ import annotations


def validate_public_request_wire_contracts() -> None:
    """Reject stale generated interfaces that do not expose the version prefix."""
    from ibrobot_msgs.action import PrimitiveCommand, SkillCommand
    from ibrobot_msgs.srv import ValidatePrimitive, ValidateSkill

    request_types = (
        ("SkillCommand.Goal", SkillCommand.Goal),
        ("PrimitiveCommand.Goal", PrimitiveCommand.Goal),
        ("ValidateSkill.Request", ValidateSkill.Request),
        ("ValidatePrimitive.Request", ValidatePrimitive.Request),
    )
    expected_prefix = ("schema_version", "uint32")
    for type_name, request_type in request_types:
        try:
            fields = list(request_type.get_fields_and_field_types().items())
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"public request wire contract mismatch for {type_name}: first field must be uint32 schema_version"
            ) from exc
        if not fields or fields[0] != expected_prefix:
            raise RuntimeError(
                f"public request wire contract mismatch for {type_name}: first field must be uint32 schema_version"
            )
