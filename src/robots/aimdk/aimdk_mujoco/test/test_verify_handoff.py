"""Acceptance guards reject false positives without requiring licensed weights or a ROS graph."""

from types import SimpleNamespace

import pytest
from aimdk_mujoco.verify_handoff import validate_result


def test_invalid_terminal_result_is_rejected():
    response = SimpleNamespace(
        status=4,
        result=SimpleNamespace(
            success=True,
            completed_steps=249,
            message="terminal result",
        ),
    )
    with pytest.raises(RuntimeError, match="Premature or invalid"):
        validate_result(response, "mc_rl_velocity", 250, canceled=False)
