"""A pick retains only for its adjacent same-target place in a Program."""

from types import SimpleNamespace

import pytest

from skill_library.carry_handover import retains_for_place


def step(skill, target="a red apple", container="the green bowl"):
    return SimpleNamespace(skill_name=skill, target_name=target, container_name=container)


def test_same_target_place_retains_the_pick():
    assert retains_for_place([step("pick_object"), step("place_in_container")], 0)


@pytest.mark.parametrize(
    "steps",
    [
        [step("pick_object")],
        [step("pick_object"), step("place_in_container", "a green apple")],
        [step("pick_object"), step("wave"), step("place_in_container")],
        [step("pick_object"), step("place_in_container", container="")],
    ],
)
def test_other_sequences_preserve_the_standalone_pick_policy(steps):
    assert not retains_for_place(steps, 0)


@pytest.mark.parametrize("version", [1, 2])
def test_stop_unknown_in_supported_detail_versions_stays_unsafe(version):
    from skill_library.carry_handover import motion_stop_is_unknown

    result = SimpleNamespace(
        error_code="", execution_detail=SimpleNamespace(schema_version=version, recovery_state="UNKNOWN")
    )
    assert motion_stop_is_unknown(result)


def test_unknown_release_is_not_unknown_controller_stop():
    from skill_library.carry_handover import motion_stop_is_unknown

    result = SimpleNamespace(
        error_code="",
        execution_detail=SimpleNamespace(schema_version=2, recovery_state="NOT_NEEDED", release_status="UNKNOWN"),
    )
    assert not motion_stop_is_unknown(result)
