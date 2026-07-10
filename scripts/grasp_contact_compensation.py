"""Pure contact-point compensation helpers for grasp execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

Vector3 = tuple[float, float, float]
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class ContactPrediction(Generic[PayloadT]):
    """Predicted base-frame contact point and solver-specific payload."""

    contact_base: Vector3
    payload: PayloadT


@dataclass(frozen=True)
class ContactCompensationResult(Generic[PayloadT]):
    """Result of iteratively correcting a command along base-frame X and Y."""

    command_xyz: Vector3
    prediction: ContactPrediction[PayloadT]
    initial_residual_x: float
    initial_residual_y: float
    residual_x: float
    residual_y: float
    correction_x: float
    correction_y: float
    solve_count: int
    converged: bool
    reason: str


class ContactPredictor(Protocol[PayloadT]):
    def __call__(
        self,
        command_xyz: Vector3,
        previous_payload: PayloadT | None,
    ) -> ContactPrediction[PayloadT]: ...


def compensate_contact_xy(
    initial_command_xyz: Vector3,
    target_contact_base: Vector3,
    predict: ContactPredictor[PayloadT],
    *,
    tolerance_m: float,
    max_iterations: int,
    max_correction_m: float,
) -> ContactCompensationResult[PayloadT]:
    """Iteratively shift command X and Y until predicted contact reaches the target.

    Both base-frame X and Y are corrected; Z is left untouched. ``max_iterations``
    counts correction updates after the initial prediction, so a value of 3
    performs at most four IK/FK predictions.
    """

    tolerance = max(0.0, float(tolerance_m))
    correction_limit = max(0.0, float(max_correction_m))
    correction_iterations = max(0, int(max_iterations))
    initial_x = float(initial_command_xyz[0])
    initial_y = float(initial_command_xyz[1])
    command_xyz: Vector3 = (
        float(initial_command_xyz[0]),
        float(initial_command_xyz[1]),
        float(initial_command_xyz[2]),
    )
    previous_payload: PayloadT | None = None
    initial_residual_x = 0.0
    initial_residual_y = 0.0

    def _result(
        prediction: ContactPrediction[PayloadT],
        converged: bool,
        reason: str,
        solve_index: int,
        residual_x: float,
        residual_y: float,
    ) -> ContactCompensationResult[PayloadT]:
        return ContactCompensationResult(
            command_xyz=command_xyz,
            prediction=prediction,
            initial_residual_x=initial_residual_x,
            initial_residual_y=initial_residual_y,
            residual_x=residual_x,
            residual_y=residual_y,
            correction_x=command_xyz[0] - initial_x,
            correction_y=command_xyz[1] - initial_y,
            solve_count=solve_index + 1,
            converged=converged,
            reason=reason,
        )

    for solve_index in range(correction_iterations + 1):
        prediction = predict(command_xyz, previous_payload)
        residual_x = float(target_contact_base[0]) - float(prediction.contact_base[0])
        residual_y = float(target_contact_base[1]) - float(prediction.contact_base[1])
        if solve_index == 0:
            initial_residual_x = residual_x
            initial_residual_y = residual_y

        if abs(residual_x) <= tolerance and abs(residual_y) <= tolerance:
            return _result(prediction, True, "contact_xy_within_tolerance", solve_index, residual_x, residual_y)

        if solve_index >= correction_iterations:
            return _result(prediction, False, "max_iterations_exceeded", solve_index, residual_x, residual_y)

        next_correction_x = (command_xyz[0] - initial_x) + residual_x
        next_correction_y = (command_xyz[1] - initial_y) + residual_y
        exceeded = []
        if abs(next_correction_x) > correction_limit:
            exceeded.append(f"x_{next_correction_x:.6f}")
        if abs(next_correction_y) > correction_limit:
            exceeded.append(f"y_{next_correction_y:.6f}")
        if exceeded:
            reason = "required_correction_" + "_".join(exceeded) + f"_exceeds_limit_{correction_limit:.6f}"
            return _result(prediction, False, reason, solve_index, residual_x, residual_y)

        command_xyz = (initial_x + next_correction_x, initial_y + next_correction_y, command_xyz[2])
        previous_payload = prediction.payload

    raise AssertionError("contact compensation loop exited unexpectedly")
