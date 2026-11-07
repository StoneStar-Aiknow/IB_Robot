"""Lazy selection keeps the fixed executor importable without MuJoCo."""


class GeneratedBackend:
    def __init__(self):
        from aimdk_robot.pose_backend import model, planner, safety
        from aimdk_robot.pose_backend.motion_limits import check_displacement

        from pose_composer import spec

        self.model, self.safety, self.spec = model, safety, spec
        self.plan_safe = planner.plan_safe
        self.prepare_execution = planner.prepare_execution
        self.check_displacement = check_displacement


def backend_for(owner):
    backend = getattr(owner, "_selected_backend", None)
    if backend is None:
        backend = owner._selected_backend = GeneratedBackend()
    return backend
