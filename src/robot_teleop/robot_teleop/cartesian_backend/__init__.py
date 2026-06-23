"""Cartesian teleoperation backend abstraction.

Backends share a common ``servo(linear, angular)`` interface so device-side
code (``xbox_controller.py``, ``phone_device.py``) is solver-agnostic:

- :class:`VelocityServoBackend` — wraps the existing :class:`pymoveit2.MoveIt2Servo`.
- :class:`SO101SafeServoBackend` — publishes split linear/angular Vector3Stamped
  to the ``so101_safe_servo_node``. For SO101 only.
- :class:`PlacoServoBackend` — publishes base-frame linear/angular Vector3Stamped
  to the ``so101_placo_servo_node`` (in-process Placo QP differential IK). For
  SO101 only.

``velocity_servo`` and ``placo_servo`` honour the project convention::

    linear  ∈ base
    angular ∈ tool_frame

and convert ``angular`` into the base frame via :class:`ToolAngularAdapter`.
``so101_safe_servo`` is a *different* contract: ``angular`` is forwarded **raw**
in tool-frame semantic form because joints 4/5 integrate it directly, not
via differential kinematics.

Factory entry point: :func:`make_cartesian_backend`.
"""

from .base import CartesianBackend  # noqa: F401
from .frame_adapter import ToolAngularAdapter  # noqa: F401
from .placo_servo import PlacoServoBackend  # noqa: F401
from .so101_safe_servo import SO101SafeServoBackend  # noqa: F401
from .velocity_servo import VelocityServoBackend  # noqa: F401


def make_cartesian_backend(solver: str, **kwargs) -> CartesianBackend:
    """Construct a Cartesian backend by name.

    Args:
        solver: One of ``'servo'`` (alias ``'velocity_servo'``),
            ``'safe_servo'`` (alias ``'so101_safe_servo'``) or
            ``'placo_servo'``.
        **kwargs: Forwarded to the concrete backend constructor. Common
            arguments: ``node``, ``tf_buffer``, ``base_link``, ``tool_frame``,
            ``linear_speed``, ``angular_speed``.

    Raises:
        ValueError: If ``solver`` is not a known solver name.
    """
    if solver == "servo" or solver == "velocity_servo":
        return VelocityServoBackend(**kwargs)
    if solver == "so101_safe_servo" or solver == "safe_servo":
        return SO101SafeServoBackend(**kwargs)
    if solver == "placo_servo":
        return PlacoServoBackend(**kwargs)
    raise ValueError(f"unknown cartesian solver: {solver!r}")
