"""Variable-dt constant-velocity target filter."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterUpdate:
    accepted: bool
    innovation_distance: float | None
    reason: str


class ConstantVelocityFilter:
    """Kalman filter over [x, y, vx, vy] with innovation gating."""

    def __init__(
        self,
        position: tuple[float, float],
        *,
        initial_position_variance: float = 0.04,
        initial_velocity_variance: float = 0.25,
        acceleration_variance: float = 1.0,
    ):
        self.state = np.array([position[0], position[1], 0.0, 0.0], dtype=np.float64)
        self.covariance = np.diag(
            [initial_position_variance, initial_position_variance, initial_velocity_variance, initial_velocity_variance]
        )
        self.acceleration_variance = float(acceleration_variance)

    def predict(self, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        noise_map = np.array([[0.5 * dt**2, 0.0], [0.0, 0.5 * dt**2], [dt, 0.0], [0.0, dt]])
        process_noise = self.acceleration_variance * noise_map @ noise_map.T
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def update(
        self,
        measurement: tuple[float, float],
        measurement_covariance: np.ndarray,
        *,
        innovation_gate: float = 9.21,
    ) -> FilterUpdate:
        observation = np.asarray(measurement, dtype=np.float64)
        covariance = np.asarray(measurement_covariance, dtype=np.float64)
        if observation.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("measurement and covariance must have shapes (2,) and (2, 2)")
        measurement_model = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        innovation = observation - measurement_model @ self.state
        innovation_covariance = measurement_model @ self.covariance @ measurement_model.T + covariance
        distance = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
        if distance > innovation_gate:
            return FilterUpdate(False, distance, "innovation_gate")
        gain = np.linalg.solve(innovation_covariance, measurement_model @ self.covariance).T
        self.state = self.state + gain @ innovation
        identity = np.eye(4)
        residual = identity - gain @ measurement_model
        self.covariance = residual @ self.covariance @ residual.T + gain @ covariance @ gain.T
        return FilterUpdate(True, distance, "measured")

    @property
    def position_variance(self) -> float:
        return float(self.covariance[0, 0] + self.covariance[1, 1])
