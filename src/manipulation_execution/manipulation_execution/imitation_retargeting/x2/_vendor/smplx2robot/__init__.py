"""smplx2robot -- SMPL-X body motion to Lingxi/Zhiyuan X2 arm commands.

    from smplx2robot import SourceAnimation, X2Kinematics, retarget, safety

    animation = SourceAnimation.load("clip.npz")
    trajectory = retarget(animation)          # [N,14] radians, arm_pos order
    print(safety.check(trajectory).text())

The value mapping this applies is not a fit. Every sign, zero and readout in
`data/x2_retarget_ledger.json` was confirmed by an operator watching a rendered
side-by-side of the real body and the robot, one channel at a time; see
`docs/method.md`. This package replays those seven decisions over a whole clip.

OFFLINE ONLY. The output is a position trajectory with joint limits respected
and nothing else -- no velocity limit, no acceleration limit, no comms timeout,
no e-stop, no collision check. `safety.check` reports on the first three and
clears nothing.
"""

__version__ = "0.1.0"

# Imported on demand, not at package import time. `robot.py` imports mujoco at
# module scope and retarget / export / safety all reach it, so an eager import
# here makes `import smplx2robot.postprocess` fail anywhere mujoco is missing --
# which is exactly the 310P board this pipeline has to run on. The names below
# still resolve normally on a workstation; they just cost their import when
# first touched. `tests/test_postprocess.py` holds this line.

_LAZY = {
    "CHANNEL_ORDER": "ledger", "Ledger": "ledger", "LedgerError": "ledger",
    "EVALUATION_ORDER": "retarget", "Trajectory": "retarget", "retarget": "retarget",
    "ARM_JOINT_NAMES": "robot", "X2Kinematics": "robot",
    "SourceAnimation": "source",
    "export": None, "paths": None, "readouts": None, "safety": None,
    "postprocess": None,
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = _LAZY[name]
    if module is None:
        return importlib.import_module(f".{name}", __name__)
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

__all__ = [
    "ARM_JOINT_NAMES", "CHANNEL_ORDER", "EVALUATION_ORDER", "Ledger", "LedgerError",
    "SourceAnimation", "Trajectory", "X2Kinematics", "export", "paths",
    "postprocess", "readouts", "retarget", "safety", "__version__",
]
