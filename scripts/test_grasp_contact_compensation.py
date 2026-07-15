from __future__ import annotations

import math

from grasp_contact_compensation import ContactPrediction, compensate_contact_xy


def test_compensation_moves_command_more_negative_for_positive_y_underreach() -> None:
    calls: list[float] = []

    def predict(command_xyz, previous_payload):
        _ = previous_payload
        calls.append(command_xyz[1])
        return ContactPrediction(
            contact_base=(command_xyz[0], command_xyz[1] + 0.004, 0.0),
            payload=len(calls),
        )

    result = compensate_contact_xy(
        (0.1, -0.2, 0.05),
        (0.1, -0.2, -0.02),
        predict,
        tolerance_m=0.001,
        max_iterations=3,
        max_correction_m=0.03,
    )

    assert result.converged
    assert result.solve_count == 2
    assert math.isclose(result.command_xyz[1], -0.204, abs_tol=1e-12)
    assert abs(result.residual_y) < 1e-12
    assert result.correction_y < 0.0
    assert result.correction_x == 0.0


def test_compensation_reuses_previous_prediction_payload_as_seed() -> None:
    previous_payloads: list[int | None] = []

    def predict(command_xyz, previous_payload):
        previous_payloads.append(previous_payload)
        return ContactPrediction(
            contact_base=(command_xyz[0], command_xyz[1] + 0.002, 0.0),
            payload=len(previous_payloads),
        )

    result = compensate_contact_xy(
        (0.0, -0.1, 0.0),
        (0.0, -0.1, 0.0),
        predict,
        tolerance_m=0.0005,
        max_iterations=2,
        max_correction_m=0.01,
    )

    assert result.converged
    assert previous_payloads == [None, 1]


def test_compensation_stops_before_exceeding_correction_limit() -> None:
    def predict(command_xyz, previous_payload):
        _ = previous_payload
        return ContactPrediction(
            contact_base=(command_xyz[0], command_xyz[1] + 0.04, 0.0),
            payload=None,
        )

    result = compensate_contact_xy(
        (0.0, -0.2, 0.0),
        (0.0, -0.2, 0.0),
        predict,
        tolerance_m=0.001,
        max_iterations=3,
        max_correction_m=0.03,
    )

    assert not result.converged
    assert result.solve_count == 1
    assert result.command_xyz == (0.0, -0.2, 0.0)
    assert "exceeds_limit" in result.reason


def test_compensation_corrects_both_x_and_y() -> None:
    def predict(command_xyz, previous_payload):
        _ = previous_payload
        return ContactPrediction(
            contact_base=(command_xyz[0] + 0.003, command_xyz[1] - 0.005, 0.0),
            payload=None,
        )

    result = compensate_contact_xy(
        (0.2, -0.15, 0.05),
        (0.2, -0.15, 0.0),
        predict,
        tolerance_m=0.001,
        max_iterations=3,
        max_correction_m=0.03,
    )

    assert result.converged
    assert abs(result.residual_x) < 1e-12
    assert abs(result.residual_y) < 1e-12
    assert math.isclose(result.correction_x, -0.003, abs_tol=1e-12)
    assert math.isclose(result.correction_y, 0.005, abs_tol=1e-12)
