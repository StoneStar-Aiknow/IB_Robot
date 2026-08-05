from embodied_common.primitive_contracts import (
    PRIMITIVE_CONTRACT_DIGEST,
    PRIMITIVE_DESCRIPTORS,
    SUPPORTED_PRIMITIVES,
    canonical_json,
    primitive_contract_preimage,
)


def test_primitive_registry_is_complete_and_canonical():
    assert frozenset(PRIMITIVE_DESCRIPTORS) == SUPPORTED_PRIMITIVES
    assert len(PRIMITIVE_DESCRIPTORS) == 10
    assert len(PRIMITIVE_CONTRACT_DIGEST) == 64
    assert canonical_json(primitive_contract_preimage()).startswith('{"primitives":')


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
