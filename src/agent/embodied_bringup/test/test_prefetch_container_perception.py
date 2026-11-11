"""A pick that prefetches perception detects the place's container with the placement perception settings."""

import pytest

from embodied_bringup.launch_builders.embodied import delegated_grasp_execution


def config(prefetch, perception=None):
    placement = {"enabled": True, "motion": {}}
    if perception is not None:
        placement["perception"] = perception
    return {
        "grasp_execution": {"enabled": True, "pipeline": {"prefetch_perception": prefetch}},
        "placement_execution": placement,
    }


def test_the_placement_perception_reaches_a_prefetching_pick():
    grasp = delegated_grasp_execution(config(True, {"box_threshold": 0.5}))
    assert grasp["container_perception"] == {"box_threshold": 0.5}


@pytest.mark.parametrize("prefetch,perception", [(False, {"box_threshold": 0.5}), (True, None)])
def test_otherwise_the_pick_configuration_is_unchanged(prefetch, perception):
    assert "container_perception" not in delegated_grasp_execution(config(prefetch, perception))


def test_the_agent_gets_the_pick_prefetch_service_only_when_the_pick_prefetches():
    from embodied_bringup.launch_builders.embodied import _pick_prefetch_service

    def robot(prefetch, **grasp):
        return {
            "grasp_execution": {"enabled": True, "backend": "x", "pipeline": {"prefetch_perception": prefetch}, **grasp}
        }

    assert _pick_prefetch_service(robot(True)) == "/manipulation/execute_pick/prefetch"
    assert _pick_prefetch_service(robot(True, action_name="/x2/pick/")) == "/x2/pick/prefetch"
    assert _pick_prefetch_service(robot(False)) == ""
    assert (
        _pick_prefetch_service({"grasp_execution": {"enabled": True, "pipeline": {"prefetch_perception": True}}}) == ""
    )
