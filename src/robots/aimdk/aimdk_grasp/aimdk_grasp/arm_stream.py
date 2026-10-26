"""How the X2's left arm is streamed during a grasp.

The robot-agnostic streaming executor with the X2's settings: the grasp
sequence's tracking and arrival tolerances, MC's kp 40 / kd 2 arm gains, the
MC-enforced left-arm limits of the robot configuration, the right arm and head
locked into the planning model, the gravity model as command shaper, and a
fixed home command.
"""

from __future__ import annotations

from manipulation_execution.grasp_pipeline.streaming import StreamConfig, StreamExecutor, StreamIO

from .arm_gravity import LIMIT_MARGIN_RAD, ArmGravity, GravityShaper
from .joints import HEAD, LEFT_ARM, RIGHT_ARM

# Where a sequence leaves the arm: this left-arm COMMAND, MC's own arm pose
# on entering UPPERBODY_REMOTE_SPLIT (the board restore step 4 echoes,
# boot20261014urscmd1). A home aimed at the plan's start -- the measured pose
# the previous sequence ended at -- ratcheted: each landed a few mrad lower
# (joint friction leaves the sag at kp 40 anywhere in 0.017-0.033 rad of
# shoulder roll) and the next plan started there (live 2026-10-20 q1..o2:
# roll -0.033 -> -0.067, past the -0.061 limit, o3 refused). A fixed command
# gives the same sag every time, friction scatter included, nothing adds up.
HOME_COMMAND = dict(zip(LEFT_ARM, (0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0), strict=True))
# The board is moved to HOME_COMMAND when any joint is further off than this,
# and left alone beyond HOME_MAX_RAD (an operator's recovery, not a correction).
HOME_COMMAND_TOL_RAD = 0.005
HOME_MAX_RAD = 0.1
# Home and other target moves: a straight joint line at this speed, at least HOME_MIN_S.
HOME_SPEED_RAD_S = 0.05
HOME_MIN_S = 1.0
# MC's arm gains in UPPERBODY_REMOTE_SPLIT, which the gravity model was fitted at.
MC_STIFFNESS = 40.0
MC_DAMPING = 2.0


def x2_stream_config(limits=None) -> StreamConfig:
    """``limits``: left-arm MC limits; by default the robot configuration's."""
    if limits is None:
        from .robot_model import arm_limits

        limits = arm_limits()
    return StreamConfig(
        tracking_error_rad=0.3,
        arrival_error_rad=0.2,
        locked_joints=RIGHT_ARM + HEAD,
        limits={name: tuple(limits[name]) for name in LEFT_ARM},
        limit_margin_rad=LIMIT_MARGIN_RAD,
        expected_stiffness=MC_STIFFNESS,
        expected_damping=MC_DAMPING,
        target_speed_rad_s=HOME_SPEED_RAD_S,
        target_min_s=HOME_MIN_S,
        home_command=HOME_COMMAND,
        home_tolerance_rad=HOME_COMMAND_TOL_RAD,
        home_max_rad=HOME_MAX_RAD,
    )


def x2_arm_executor(io: StreamIO, *, gravity: ArmGravity | None = None, limits=None) -> StreamExecutor:
    """The X2 left arm's executor over ``io``, the runtime's ``joint.arm_stream``."""
    missing = set(LEFT_ARM + RIGHT_ARM) - set(io.channel_joints)
    if missing:
        raise ValueError(f"the arm stream does not carry {sorted(missing)}")
    return StreamExecutor(io, x2_stream_config(limits), GravityShaper(gravity))
