import operator
from collections.abc import Mapping

import pytest
from jsonschema import Draft202012Validator

from embodied_common.primitive_contracts import (
    PRIMITIVE_CONTRACT_DIGEST,
    PRIMITIVE_CONTRACT_V1,
    PRIMITIVE_CONTRACT_V2,
    PRIMITIVE_DESCRIPTORS,
    SUPPORTED_PRIMITIVES,
    canonical_json,
    primitive_contract_for_version,
    primitive_contract_preimage,
)

V1_PRIMITIVE_DIGEST = "537210f86c9d8b7fa70f063235e7b9f10226de62c64a4b38376bbe5c53b42700"
V1_PRIMITIVE_NAMES = (
    "close_gripper",
    "move_relative_ee",
    "move_through_joint_positions",
    "move_to_configuration",
    "move_to_joint_positions",
    "move_to_named_pose",
    "move_to_pose",
    "open_gripper",
    "rotate_gripper_ccw",
    "rotate_gripper_cw",
)
V2_NAVIGATION_PRIMITIVE_NAMES = ("nav_abs_coordinate", "nav_straight", "nav_turn")


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def test_primitive_registry_is_complete_and_canonical():
    assert frozenset(PRIMITIVE_DESCRIPTORS) == SUPPORTED_PRIMITIVES
    assert tuple(sorted(PRIMITIVE_DESCRIPTORS)) == V1_PRIMITIVE_NAMES
    assert PRIMITIVE_CONTRACT_DIGEST == V1_PRIMITIVE_DIGEST
    assert canonical_json(primitive_contract_preimage()).startswith('{"primitives":')


def test_v1_primitive_preimage_and_digest_are_the_exact_base_contract():
    expected_preimage = {
        "schema_version": 1,
        "primitives": [
            {
                "schema_version": descriptor.schema_version,
                "name": descriptor.name,
                "parameter_contract": _json_value(descriptor.parameter_contract),
                "required_runtime_capabilities": list(descriptor.required_runtime_capabilities),
                "dispatch_kind": descriptor.dispatch_kind,
            }
            for descriptor in (PRIMITIVE_CONTRACT_V1.descriptors[name] for name in V1_PRIMITIVE_NAMES)
        ],
    }

    assert primitive_contract_preimage(PRIMITIVE_CONTRACT_V1) == expected_preimage
    assert PRIMITIVE_CONTRACT_V1.digest == V1_PRIMITIVE_DIGEST


def test_primitive_contract_selector_returns_immutable_v1_and_v2_sets():
    assert primitive_contract_for_version(1) is PRIMITIVE_CONTRACT_V1
    assert primitive_contract_for_version(2) is PRIMITIVE_CONTRACT_V2
    assert tuple(sorted(PRIMITIVE_CONTRACT_V1.descriptors)) == V1_PRIMITIVE_NAMES
    assert tuple(sorted(set(PRIMITIVE_CONTRACT_V2.descriptors) - set(PRIMITIVE_CONTRACT_V1.descriptors))) == (
        V2_NAVIGATION_PRIMITIVE_NAMES
    )
    assert PRIMITIVE_CONTRACT_V2.digest != PRIMITIVE_CONTRACT_V1.digest

    for contract in (PRIMITIVE_CONTRACT_V1, PRIMITIVE_CONTRACT_V2):
        with pytest.raises(TypeError):
            operator.setitem(contract.descriptors, "new", PRIMITIVE_CONTRACT_V1.descriptors["open_gripper"])


@pytest.mark.parametrize("version", [0, -1, 3, 99])
def test_primitive_contract_selector_rejects_unknown_versions(version):
    with pytest.raises(ValueError, match="SKILL_SCHEMA_INVALID"):
        primitive_contract_for_version(version)


def test_legacy_primitive_aliases_are_exactly_v1():
    assert PRIMITIVE_DESCRIPTORS is PRIMITIVE_CONTRACT_V1.descriptors
    assert frozenset(PRIMITIVE_CONTRACT_V1.descriptors) == SUPPORTED_PRIMITIVES
    assert PRIMITIVE_CONTRACT_V1.digest == PRIMITIVE_CONTRACT_DIGEST


def test_descriptor_binds_primitive_name_and_dispatch_capability():
    descriptor = PRIMITIVE_DESCRIPTORS["open_gripper"]
    assert descriptor.parameter_contract["properties"]["primitive_name"] == {
        "type": "string",
        "const": "open_gripper",
    }
    assert "task_executor" in descriptor.required_runtime_capabilities
    assert descriptor.dispatch_kind == "task_executor_action"


def test_registry_and_descriptors_are_immutable():
    try:
        PRIMITIVE_DESCRIPTORS["new"] = PRIMITIVE_DESCRIPTORS["open_gripper"]
    except TypeError:
        pass
    else:
        raise AssertionError("registry must be immutable")


def test_rotate_descriptors_require_fresh_ee_pose():
    for name in ("rotate_gripper_cw", "rotate_gripper_ccw"):
        assert "fresh_ee_pose" in PRIMITIVE_DESCRIPTORS[name].required_runtime_capabilities


@pytest.mark.parametrize("name", ["nav_straight", "nav_turn", "nav_abs_coordinate"])
def test_navigation_descriptors_use_navigation_action_and_readiness(name):
    descriptor = PRIMITIVE_CONTRACT_V2.descriptors[name]

    assert descriptor.schema_version == 2
    assert descriptor.dispatch_kind == "navigation_action"
    assert descriptor.required_runtime_capabilities == ("navigation", "validate_skill")


@pytest.mark.parametrize(
    ("name", "valid_step"),
    [
        (
            "nav_straight",
            {
                "primitive_name": "nav_straight",
                "direction_from_request": True,
                "distance_from_request": True,
            },
        ),
        (
            "nav_turn",
            {
                "primitive_name": "nav_turn",
                "direction_from_request": True,
                "degree_from_request": True,
            },
        ),
        (
            "nav_abs_coordinate",
            {
                "primitive_name": "nav_abs_coordinate",
                "x_from_request": True,
                "y_from_request": True,
                "yaw_from_request": True,
            },
        ),
    ],
)
def test_navigation_descriptors_accept_request_bound_parameters(name, valid_step):
    validator = Draft202012Validator(PRIMITIVE_CONTRACT_V2.descriptors[name].parameter_contract)

    assert list(validator.iter_errors(valid_step)) == []


@pytest.mark.parametrize(
    ("name", "invalid_step"),
    [
        ("nav_straight", {"primitive_name": "nav_straight", "direction": "up", "distance": 1.0}),
        ("nav_straight", {"primitive_name": "nav_straight", "direction": "forward", "distance": 0.0}),
        ("nav_turn", {"primitive_name": "nav_turn", "direction": "forward", "degree": 90.0}),
        ("nav_turn", {"primitive_name": "nav_turn", "direction": "left", "degree": -1.0}),
    ],
)
def test_navigation_descriptors_reject_invalid_direction_or_magnitude(name, invalid_step):
    validator = Draft202012Validator(PRIMITIVE_CONTRACT_V2.descriptors[name].parameter_contract)

    assert list(validator.iter_errors(invalid_step))


def test_absolute_coordinate_descriptor_accepts_signed_values_and_explicit_zero():
    validator = Draft202012Validator(PRIMITIVE_CONTRACT_V2.descriptors["nav_abs_coordinate"].parameter_contract)
    step = {"primitive_name": "nav_abs_coordinate", "x": 0.0, "y": -2.5, "yaw": -180.0}

    assert list(validator.iter_errors(step)) == []
