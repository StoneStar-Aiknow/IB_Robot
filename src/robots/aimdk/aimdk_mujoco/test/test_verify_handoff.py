"""Acceptance guards reject false positives without requiring licensed weights or a ROS graph."""

from types import SimpleNamespace

import pytest
from aimdk_mujoco.verify_handoff import Telemetry, validate_result
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus


def test_streams_are_monotonic_independently_and_diagnostics_can_be_slow():
    telemetry = Telemetry()
    telemetry.observe("clock", Time(sec=2))
    telemetry.observe("odom", Time(sec=1))
    telemetry.observe("clock", Time(sec=2))  # Paused simulation still publishes.
    telemetry.observe("diagnostics", Time(sec=1))
    telemetry.observe("diagnostics", Time(sec=2))
    with pytest.raises(RuntimeError, match="discontinuity"):
        telemetry.observe("odom", Time())


@pytest.mark.parametrize(
    "stamp,values,error",
    [
        (Time(sec=1, nanosec=1_000_000_000), (), "timestamp"),
        (Time(sec=-1), (), "timestamp"),
        (Time(sec=1), (float("nan"),), "non-finite"),
        (Time(sec=2), (), "discontinuity"),
    ],
)
def test_invalid_telemetry_fails(stamp, values, error):
    telemetry = Telemetry()
    telemetry.observe("clock", Time(sec=1))
    with pytest.raises(RuntimeError, match=error):
        telemetry.observe("clock", stamp, values)


def test_teleport_at_unchanged_sim_time_fails():
    telemetry = Telemetry()
    telemetry.observe("odom", Time(sec=1), (1.5, 0, 0.6))
    with pytest.raises(RuntimeError, match="position discontinuity"):
        telemetry.observe("odom", Time(sec=1), (0, 0, 0.6))


def test_missing_telemetry_and_diagnostic_faults_fail(monkeypatch):
    monkeypatch.setattr("aimdk_mujoco.verify_handoff.time.monotonic", lambda: 1.0)
    telemetry = Telemetry()
    telemetry.observe("clock", Time())
    monkeypatch.setattr("aimdk_mujoco.verify_handoff.time.monotonic", lambda: 4.1)
    with pytest.raises(RuntimeError, match="telemetry stopped"):
        telemetry.check()
    telemetry.diagnostics(
        DiagnosticArray(
            status=[
                DiagnosticStatus(
                    name="aimdk_mujoco/physics",
                    level=DiagnosticStatus.ERROR,
                    message="Physics fault",
                )
            ]
        )
    )
    with pytest.raises(RuntimeError, match="Physics fault"):
        telemetry.check()


@pytest.mark.parametrize(
    "controller,total,status,success,steps,canceled,valid",
    [
        ("bmimic", 2820, 5, False, 501, True, True),
        ("bmimic", 2820, 5, False, 499, True, False),
        ("bmimic", 2820, 4, True, 2820, True, False),
        ("bmimic", 1171, 4, True, 1171, False, True),
        ("bmimic", 1171, 4, True, 1170, False, False),
        ("recover_stand", 500, 4, True, 50, False, True),
        ("recover_stand", 500, 4, True, 49, False, False),
        ("recover_stand", 500, 6, False, 500, False, False),
        ("mc_rl_velocity", 250, 4, True, 250, False, True),
        ("mc_rl_velocity", 250, 4, True, 249, False, False),
        ("mc_rl_velocity", 250, 4, True, 251, False, False),
        ("bmimic", 0, 4, True, 0, False, False),
    ],
)
def test_terminal_result_contract(controller, total, status, success, steps, canceled, valid):
    response = SimpleNamespace(
        status=status,
        result=SimpleNamespace(
            success=success,
            completed_steps=steps,
            message="terminal result",
        ),
    )
    if valid:
        validate_result(response, controller, total, canceled=canceled)
    else:
        with pytest.raises(RuntimeError, match="Premature or invalid"):
            validate_result(response, controller, total, canceled=canceled)
