"""Without the simulated-time hold extension, a hold is refused everywhere (fail closed)."""

from __future__ import annotations

import pytest

from robot_runtime.capabilities import is_valid
from robot_runtime_sim.direct_control.ports import COMMAND_KINDS, DescriptorError, descriptor_from_profile
from robot_runtime_sim.testing.direct_control_fakes import dual_arm_descriptor


def test_a_descriptor_that_declares_hold_commands_is_rejected():
    assert "hold" not in COMMAND_KINDS
    with pytest.raises(DescriptorError, match="unknown command kinds"):
        dual_arm_descriptor(supported_commands=frozenset({"joint", "hold"}))
    profile = {
        "motion_groups": {"arm": {"joints": ["j1"]}},
        "simulation": {"p": {"dt_s": 0.04, "supported_commands": ["joint", "hold"]}},
        "pose_frame": "world",
    }
    with pytest.raises(DescriptorError, match="unknown command kinds"):
        descriptor_from_profile(profile, platform="p", clock_mode="platform_pull")


def test_a_runtime_profile_cannot_publish_a_hold_capability():
    assert not is_valid("simulation.hold")


def test_the_runtime_serves_no_hold_service():
    from robot_runtime_sim.direct_control import node as runtime_node

    assert not hasattr(runtime_node, "HOLD_SERVICE")
    assert not hasattr(runtime_node.SimulationRuntime, "_on_hold")
