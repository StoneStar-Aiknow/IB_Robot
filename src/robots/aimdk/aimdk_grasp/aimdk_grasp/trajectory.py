"""Time-preserving position trajectory contract for the 50 Hz executor."""

from dataclasses import dataclass

import numpy as np

from .joints import LEFT_ARM, RIGHT_ARM


@dataclass(frozen=True)
class TimedTrajectory:
    times: np.ndarray
    positions: np.ndarray
    joint_names: tuple = LEFT_ARM

    def __post_init__(self):
        times = np.array(self.times, dtype=np.float64, copy=True)
        positions = np.array(self.positions, dtype=np.float64, copy=True)
        if (
            times.ndim != 1
            or tuple(self.joint_names) not in (LEFT_ARM, RIGHT_ARM)
            or positions.shape != (len(times), 7)
        ):
            raise ValueError("Trajectory must command only the seven joints of one arm")
        if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("Times must start at zero and strictly increase")
        if not np.isfinite(times).all() or not np.isfinite(positions).all():
            raise ValueError("Nonfinite trajectory")
        times.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)

    def sample(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError("Finite elapsed time required")
        return np.array([np.interp(elapsed, self.times, self.positions[:, i]) for i in range(7)])

    def max_velocity(self):
        return float(np.max(np.abs(np.diff(self.positions, axis=0) / np.diff(self.times)[:, None])))

    def within_velocity(self, limit):
        """Uniformly stretch time so the same path fits under an execution limit.

        Geometry and waypoint order are unchanged, so path collision validity
        carries over; only duration grows. Never speeds a trajectory up.
        """
        scale = max(1.0, self.max_velocity() / limit * 1.001)
        return TimedTrajectory(self.times * scale, self.positions, self.joint_names), scale

    def clip_velocity(self, limit):
        """Stretch only the intervals faster than the limit (same path and waypoints).

        Unlike within_velocity, the slow parts keep their timing: the speed
        profile is clipped at the limit instead of the whole trajectory being
        slowed, so the start (which accelerates from rest) is not stretched.
        Velocities never increase. Returns (trajectory, added seconds).
        """
        steps = np.diff(self.times)
        needed = np.max(np.abs(np.diff(self.positions, axis=0)), axis=1) / limit * 1.001
        times = np.concatenate(([0.0], np.cumsum(np.maximum(steps, needed))))
        return TimedTrajectory(times, self.positions, self.joint_names), float(times[-1] - self.times[-1])

    def trimmed(self, tolerance=1e-6):
        """Drop the stationary padding at both ends (cuRobo pads 0.1-0.4 s of
        identical waypoints); keeps the last resting sample before motion and
        the first one after it. A segment without motion is returned as is."""
        moving = np.flatnonzero(np.max(np.abs(np.diff(self.positions, axis=0)), axis=1) > tolerance)
        if not len(moving):
            return self
        first, last = moving[0], moving[-1] + 1
        return TimedTrajectory(
            self.times[first : last + 1] - self.times[first], self.positions[first : last + 1], self.joint_names
        )

    def reversed(self):
        """Reverse the same timed path; callers must check reverse motion limits."""
        return TimedTrajectory(self.times[-1] - self.times[::-1], self.positions[::-1], self.joint_names)

    @classmethod
    def from_curobo(cls, result, interpolation_dt, *, locked_positions=None, arm=LEFT_ARM):
        """The planned ``arm`` (seven joints) of a cuRobo result; every other joint must stay at its lock."""
        arm = tuple(arm)
        if (
            result is None
            or result.success is None
            or not bool(result.success.all())
            or not np.isfinite(interpolation_dt)
            or interpolation_dt <= 0
        ):
            raise ValueError("Unsuccessful plan or invalid interpolation timestep")
        plan = result.get_interpolated_plan()
        names = tuple(plan.joint_names)
        locked_positions = {} if locked_positions is None else locked_positions
        if len(set(names)) != len(names) or not set(arm) <= set(names):
            raise ValueError("Unexpected cuRobo planned joints")
        q = plan.position.detach().cpu().numpy()
        while q.ndim > 2 and q.shape[0] == 1:
            q = q[0]
        if q.ndim != 2 or q.shape[1] != len(names) or not np.isfinite(q).all():
            raise ValueError("Invalid cuRobo trajectory dimensions or values")
        for name in set(names) - set(arm):
            if name not in locked_positions or not np.allclose(
                q[:, names.index(name)], locked_positions[name], atol=1e-6, rtol=0
            ):
                raise ValueError(f"Unexpected motion or unknown locked joint: {name}")
        q = q[:, [names.index(n) for n in arm]]
        return cls(np.arange(len(q)) * interpolation_dt, q, arm)
