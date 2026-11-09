"""SO-101 grasp providers satisfy the generic pick pipeline's provider contract.

The contract tuples belong to the consumer (``manipulation_execution.pick.providers``)
and the providers to the robot suite (``so101_suite``). Neither package may
depend on the other, so the conformance check lives here, in the composition
root that already test-depends on both.
"""

from manipulation_execution.pick.providers import (
    GRASP_GEOMETRY_FUNCTIONS,
    WRIST_GUARD_FUNCTIONS,
    load_provider,
)


def test_so101_suite_gripper_geometry_satisfies_grasp_geometry_provider_contract():
    module = load_provider(
        "target_geometry.grasp_geometry_provider",
        "so101_suite.gripper_geometry",
        GRASP_GEOMETRY_FUNCTIONS,
    )

    assert module.__name__ == "so101_suite.gripper_geometry"


def test_so101_suite_wrist_guard_satisfies_wrist_guard_provider_contract():
    module = load_provider(
        "target_gripper.ik_orientation_guard.wrist_guard_provider",
        "so101_suite.wrist_guard",
        WRIST_GUARD_FUNCTIONS,
    )

    assert module.__name__ == "so101_suite.wrist_guard"
