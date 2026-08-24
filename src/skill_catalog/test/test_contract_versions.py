from dataclasses import replace

import pytest

from embodied_common.primitive_contracts import primitive_contract_for_version
from skill_catalog.models import SkillRobotContext
from skill_catalog.validator import validate_implementation, validate_manifest

V2_PARAMETER_SCHEMAS = {
    "direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]},
    "distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
    "degree": {"type": "number", "exclusiveMinimum": 0, "unit": "degrees"},
    "x": {"type": "number", "unit": "meters"},
    "y": {"type": "number", "unit": "meters"},
    "yaw": {"type": "number", "unit": "degrees"},
}


def _parameters(properties=None):
    properties = {} if properties is None else properties
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _manifest(version, *, parameters=None, control_mode=None):
    return {
        "schema_version": version,
        "name": "contract_skill",
        "version": "1.0.0",
        "semantic_level": "atomic_operator",
        "description": {
            "summary": "Exercise one versioned primitive.",
            "category": "contract",
            "when_to_use": ["verify the version matrix"],
            "motion_scope": ["base" if version == 2 else "gripper"],
            "intensity": "subtle",
        },
        "capability": {
            "schema_version": version,
            "summary": "Exercise one versioned primitive.",
            "domain": "navigation" if version == 2 else "manipulation",
            "moves_robot": True,
            "required_control_mode": control_mode or ("base_navigation" if version == 2 else "moveit_planning"),
            "parameters": _parameters(parameters),
            "recovery_policy": "never_retry",
        },
        "implementations": {"test_robot": "implementations/test_robot.yaml"},
    }


def _implementation(version, step):
    return {
        "schema_version": version,
        "kind": "primitive_sequence",
        "robot": "test_robot",
        "initial_gripper_state": "none",
        "timeout_sec": 5.0,
        "primitive_sequence": [step],
    }


def _robot_context(version):
    endpoints = {
        "skill_action": "/embodied/execute_skill",
        "primitive_action": "/embodied/execute_primitive",
        "validate_skill_service": "/embodied/validate_skill",
        "validate_primitive_service": "/embodied/validate_primitive",
        "gateway_status_service": "/embodied/get_skill_gateway_status",
        "begin_workflow_service": "/embodied/begin_workflow_execution",
        "finalize_workflow_service": "/embodied/finalize_workflow_execution",
        "task_executor_action": "/embodied/execute_task",
        "arm_trajectory_action": "/arm/execute_trajectory",
        "move_configuration_service": "/arm/move_configuration",
    }
    if version == 2:
        endpoints["navigation_action"] = "/navigation/execute"
    return SkillRobotContext(
        robot_name="test_robot",
        context_schema_version=version,
        robot_config_digest="a" * 64,
        named_poses={},
        named_targets={},
        arm_joint_names=(),
        joint_limits={},
        workspace_limits={},
        required_control_mode="base_navigation" if version == 2 else "moveit_planning",
        timeout_policy={"task_budget_sec": 60.0},
        relative_motion_reference_frame="base",
        relative_motion_step_m=0.03,
        relative_motion_direction_mapping={},
        gripper_open_position=1.0,
        gripper_closed_position=0.0,
        execution_endpoints=endpoints,
    )


@pytest.mark.parametrize(
    ("version", "manifest", "implementation"),
    [
        (1, _manifest(1), _implementation(1, {"primitive_name": "open_gripper"})),
        (
            2,
            _manifest(
                2,
                parameters={
                    "direction": V2_PARAMETER_SCHEMAS["direction"],
                    "distance": V2_PARAMETER_SCHEMAS["distance"],
                },
            ),
            _implementation(
                2,
                {
                    "primitive_name": "nav_straight",
                    "direction_from_request": True,
                    "distance_from_request": True,
                },
            ),
        ),
    ],
)
def test_closed_contract_matrix_accepts_matching_v1_and_v2_documents(version, manifest, implementation):
    contract = primitive_contract_for_version(version)

    assert validate_manifest(manifest, package_name="contract_skill", source_relative_path="manifest.yaml") == []
    diagnostics, _, _ = validate_implementation(
        implementation,
        manifest=manifest,
        implementation_name="test_robot",
        context=_robot_context(version),
        delegated_executors={},
        primitive_contracts=contract.descriptors,
        source_relative_path="implementation.yaml",
    )

    assert diagnostics == []
    assert contract.descriptors[implementation["primitive_sequence"][0]["primitive_name"]].schema_version == version


@pytest.mark.parametrize(
    ("manifest_version", "capability_version", "implementation_version", "contract_version"),
    [(1, 2, 1, 1), (2, 1, 2, 2), (1, 1, 2, 1), (2, 2, 1, 2)],
)
def test_contract_matrix_rejects_mixed_source_versions(
    manifest_version, capability_version, implementation_version, contract_version
):
    contract = primitive_contract_for_version(contract_version)
    if contract_version == 1:
        step = {"primitive_name": "open_gripper"}
        parameters = None
    else:
        step = {"primitive_name": "nav_turn", "direction_from_request": True, "degree_from_request": True}
        parameters = {
            "direction": V2_PARAMETER_SCHEMAS["direction"],
            "degree": V2_PARAMETER_SCHEMAS["degree"],
        }
    manifest = _manifest(manifest_version, parameters=parameters)
    manifest["capability"]["schema_version"] = capability_version
    implementation = _implementation(implementation_version, step)

    diagnostics = validate_manifest(manifest, package_name="contract_skill", source_relative_path="manifest.yaml")
    implementation_diagnostics, _, _ = validate_implementation(
        implementation,
        manifest=manifest,
        implementation_name="test_robot",
        context=_robot_context(contract_version),
        delegated_executors={},
        primitive_contracts=contract.descriptors,
        source_relative_path="implementation.yaml",
    )

    assert diagnostics or implementation_diagnostics
    assert all(item.error_code == "SKILL_SCHEMA_INVALID" for item in diagnostics + implementation_diagnostics)


@pytest.mark.parametrize("version,primitive_name", [(1, "open_gripper"), (2, "nav_straight")])
def test_contract_matrix_rejects_descriptor_version_mismatch(version, primitive_name):
    contract = primitive_contract_for_version(version)
    descriptor = replace(contract.descriptors[primitive_name], schema_version=2 if version == 1 else 1)
    parameters = (
        None
        if version == 1
        else {
            "direction": V2_PARAMETER_SCHEMAS["direction"],
            "distance": V2_PARAMETER_SCHEMAS["distance"],
        }
    )
    step = (
        {"primitive_name": primitive_name}
        if version == 1
        else {
            "primitive_name": primitive_name,
            "direction_from_request": True,
            "distance_from_request": True,
        }
    )

    diagnostics, _, _ = validate_implementation(
        _implementation(version, step),
        manifest=_manifest(version, parameters=parameters),
        implementation_name="test_robot",
        context=_robot_context(version),
        delegated_executors={},
        primitive_contracts={primitive_name: descriptor},
        source_relative_path="implementation.yaml",
    )

    assert any("version" in item.message for item in diagnostics)


@pytest.mark.parametrize("parameter_name", sorted(V2_PARAMETER_SCHEMAS))
def test_v1_capability_rejects_each_v2_only_parameter(parameter_name):
    manifest = _manifest(1, parameters={parameter_name: V2_PARAMETER_SCHEMAS[parameter_name]})

    diagnostics = validate_manifest(manifest, package_name="contract_skill", source_relative_path="manifest.yaml")

    assert any(item.field_path.startswith("capability.parameters") for item in diagnostics)


def test_v1_capability_rejects_v2_control_mode():
    manifest = _manifest(1, control_mode="base_navigation")

    diagnostics = validate_manifest(manifest, package_name="contract_skill", source_relative_path="manifest.yaml")

    assert any(item.field_path == "capability.required_control_mode" for item in diagnostics)


@pytest.mark.parametrize("primitive_name", ["nav_straight", "nav_turn", "nav_abs_coordinate"])
def test_v1_implementation_rejects_each_v2_descriptor(primitive_name):
    diagnostics, _, _ = validate_implementation(
        _implementation(1, {"primitive_name": primitive_name}),
        manifest=_manifest(1),
        implementation_name="test_robot",
        context=_robot_context(1),
        delegated_executors={},
        primitive_contracts=primitive_contract_for_version(1).descriptors,
        source_relative_path="implementation.yaml",
    )

    assert any(item.error_code == "SKILL_REFERENCE_MISSING" for item in diagnostics)


@pytest.mark.parametrize("document", ["manifest", "capability", "implementation"])
def test_contract_documents_reject_unknown_fields(document):
    manifest = _manifest(1)
    implementation = _implementation(1, {"primitive_name": "open_gripper"})
    if document == "implementation":
        target = implementation
    elif document == "capability":
        target = manifest["capability"]
    else:
        target = manifest
    target["unknown"] = True

    diagnostics = validate_manifest(manifest, package_name="contract_skill", source_relative_path="manifest.yaml")
    implementation_diagnostics, _, _ = validate_implementation(
        implementation,
        manifest=manifest,
        implementation_name="test_robot",
        context=_robot_context(1),
        delegated_executors={},
        primitive_contracts=primitive_contract_for_version(1).descriptors,
        source_relative_path="implementation.yaml",
    )

    assert any(item.field_path.endswith("unknown") for item in diagnostics + implementation_diagnostics)


def test_primitive_descriptor_shape_is_closed_for_both_versions():
    expected_fields = {
        "schema_version",
        "name",
        "parameter_contract",
        "required_runtime_capabilities",
        "dispatch_kind",
    }

    for version in (1, 2):
        for descriptor in primitive_contract_for_version(version).descriptors.values():
            assert set(descriptor.__dataclass_fields__) == expected_fields
