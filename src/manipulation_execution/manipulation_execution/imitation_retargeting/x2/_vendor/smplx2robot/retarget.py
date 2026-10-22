"""The conversion itself: one SMPL-X clip in, one [N,14] arm_pos trajectory out.

    x2_arm_pos[index] = sign * (source_rad - offset_rad) * scale

applied per channel per side, with the signs, zeros and scales taken from the
ledger. Six of the seven channels are independent and could be evaluated in any
order. One is not, and the whole shape of this module is that exception:

    ORDERING CONSTRAINT. The X2 shoulder is pitch -> roll -> yaw with yaw
    DISTAL, so R_arm = R_p R_r R_y and the product R_p R_r already carries twist
    about the bone. Along the trajectory the confirmed pitch and roll mappings
    actually produce, that built-in twist sweeps 88.6 deg (left) and 92.2 deg
    (right). It is not a constant, so it cannot be an offset: the best single
    number is wrong by up to 47 / 52 deg. It goes inside the READOUT instead --

        source := wrap(psi_robot_at_yaw_zero(pitch_f, roll_f, elbow_f) - psi_her_f)

    -- which means `arm_pos[2]` can only be computed after `arm_pos[0,1,3]`, and
    `arm_pos[9]` after `arm_pos[7,8,10]`, by running robot forward kinematics on
    every frame. `EVALUATION_ORDER` encodes that and `retarget()` obeys it.

The result is VERIFIED BY CLOSURE, not by correlation: the finished command is
pushed back through forward kinematics, the twist is measured again, and the
difference from her own twist is reported. That is a kinematic proof that the
sign and the subtraction order are right, and it is the number to quote.

SAFETY. The output of this module is a position trajectory that respects the
model's joint limits and nothing else. It has no velocity limit, no acceleration
limit, no continuity guarantee across the clip's real timestamp gaps, no comms
timeout and no e-stop. `safety.py` reports on the first three; none of them are
enforced here, and none of them are a substitute for the rest. Do not send this
to a robot until that list is closed and the first run is low-speed, unloaded,
and within reach of the stop button.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from . import readouts
from .ledger import ARM_DOF, CHANNEL_ORDER, SIDES, Channel, Ledger
from .robot import (ARM_JOINT_NAMES, MIRROR_SIGNS, X2Kinematics, mirror_residual_mm,
                    wrap_deg)
from .source import SourceAnimation

# The six channels whose source depends on the SMPL-X clip alone.
INDEPENDENT_CHANNELS: tuple[str, ...] = (
    "shoulder_pitch",
    "shoulder_roll",
    "elbow",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)
# The one whose source depends on commands already issued. Must come last.
COUPLED_CHANNEL = "shoulder_yaw"
EVALUATION_ORDER: tuple[str, ...] = INDEPENDENT_CHANNELS + (COUPLED_CHANNEL,)


# A mirror residual can only be two things: the robot's own left/right
# asymmetry, measured at 2.8 mm at worst, or a wrong sign, measured at 94 mm at
# best. 20 mm is the middle of an empty range, not a tuned tolerance.
MIRROR_CHECK_LIMIT_MM = 20.0


def mirror_offsets_rad(ledger: Ledger) -> np.ndarray:
    """[14] constant the mirrored command needs on top of the sign flip.

    Take mirroring at the SOURCE as the definition -- her right-arm readout,
    reflected, drives the left arm -- and push it through the ledger formula:

        cmd_L = sign_L * (mu * src_R - off_L)          mu = reflection of the readout
        src_R = sign_R * cmd_R + off_R                 (sign^2 = 1)
        =>  cmd_L = (sign_L * sign_R * mu) * cmd_R + sign_L * (mu * off_R - off_L)
                  = S * cmd_R + b_left

    which is where both `MIRROR_SIGNS` and `b` come from, and it also gives mu
    back as `S * sign_L * sign_R` without the source frame ever being named.

    Only the three channels whose two sides have different zeros contribute:
    the left shoulder is built with 4.2655 deg of forward hang against the
    right's 3.9142, so commanding the two arms to the same ANATOMICAL angle
    means commanding two numbers 0.3513 deg apart. Small, and worth having --
    it cuts the measured mirror residual to under a third.
    """
    bias = np.zeros(ARM_DOF)
    for index, name in enumerate(CHANNEL_ORDER):
        channel = ledger[name]
        left, right = channel.sign["left"], channel.sign["right"]
        zero_left, zero_right = channel.offset_rad["left"], channel.offset_rad["right"]
        mu = MIRROR_SIGNS[index] * left * right
        bias[channel.index("left")] = left * (mu * zero_right - zero_left)
        bias[channel.index("right")] = right * (mu * zero_left - zero_right)
    return bias

# Channels whose command must already be in the pose before the coupling is
# measured. Wrist channels are included even though the wrist_yaw ANCHOR is
# fixed in the forearm link and cannot move with them -- `test_retarget`
# asserts that invariance rather than this module assuming it.
READOUT_B = "plane"
READOUT_A = "spin"


@dataclass(frozen=True)
class Trajectory:
    """A finished 14-channel command sequence, with everything used to make it."""

    arm_pos: np.ndarray               # [N,14] radians, clipped to joint limits
    raw: np.ndarray                   # [N,14] radians, before clipping
    clipped: np.ndarray               # [N,14] bool, where the limit bit
    frame_id: np.ndarray              # [N]
    timestamp_ns: np.ndarray          # [N]
    time_s: np.ndarray                # [N] seconds from the first frame
    limits_rad: np.ndarray            # [14,2] read from the robot model
    source_deg: Mapping[str, Mapping[str, np.ndarray]]   # channel -> side -> [N]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(self.arm_pos.shape[0])

    @property
    def arm_pos_deg(self) -> np.ndarray:
        return np.degrees(self.arm_pos)

    def channel(self, index: int) -> np.ndarray:
        return self.arm_pos[:, index]

    def retimed(self, scale: float) -> "Trajectory":
        """The same poses, played `scale` times slower. Nothing else changes.

        The only mitigation for an over-speed trajectory that does not touch the
        confirmed mapping: it divides every velocity and quarters every
        acceleration at scale 2, and the robot passes through exactly the same
        poses in exactly the same order. Filtering the angles would be the other
        option and is deliberately not offered here -- a filter changes what the
        operator confirmed, so it belongs in a stage that can be reviewed the
        same way the mapping was.
        """
        if not scale > 0:
            raise ValueError(f"time scale must be positive, got {scale}")
        base = self.timestamp_ns[0]
        offsets = (self.timestamp_ns - base).astype(np.float64) * scale
        diagnostics = dict(self.diagnostics)
        diagnostics["time_scale"] = float(scale) * float(diagnostics.get("time_scale", 1.0))
        return Trajectory(
            arm_pos=self.arm_pos, raw=self.raw, clipped=self.clipped,
            frame_id=self.frame_id,
            timestamp_ns=(base + np.rint(offsets)).astype(np.int64),
            time_s=self.time_s * scale,
            limits_rad=self.limits_rad, source_deg=self.source_deg,
            diagnostics=diagnostics,
        )

    def mirrored(self, ledger: Ledger, kinematics: X2Kinematics | None = None,
                 offsets: bool = True) -> "Trajectory":
        """Mirror imitation: her right arm drives the left arm, and the reverse.

        One sign flip, one left/right swap of the whole block, and a constant of
        a third of a degree:

            mirror_raw[0:7]  = S * raw[7:14] + b_left
            mirror_raw[7:14] = S * raw[0:7]  + b_right

        with `S = robot.MIRROR_SIGNS` and `b` computed here from the ledger's own
        zeros. Nothing in the ledger changes: mirroring is not an eighth mapping,
        it is one coordinate reflection laid on top of the confirmed seven, and
        it belongs to the robot's geometry rather than to her body.

        Done AFTER `retarget()` rather than inside it, on purpose. The closure
        check stays a proof about those seven mappings, which mirroring does not
        touch, and the mirror's own proof -- `mirror_fk_check_mm` -- is a
        separate number about a separate claim. A reader who sees both knows
        which one certifies what.

        `kinematics` turns on that self-check: every frame is posed both ways and
        the two arms compared through forward kinematics. Over
        `MIRROR_CHECK_LIMIT_MM` it raises rather than returns, because the only
        thing that produces a residual in that range is a wrong sign, and a
        wrong sign is a full-speed reversal of one joint on the robot.
        """
        signs = np.asarray(MIRROR_SIGNS, dtype=np.float64)
        bias = mirror_offsets_rad(ledger) if offsets else np.zeros(ARM_DOF)
        half = ARM_DOF // 2

        raw = np.empty_like(self.raw)
        raw[:, :half] = signs * self.raw[:, half:] + bias[:half]
        raw[:, half:] = signs * self.raw[:, :half] + bias[half:]
        arm_pos = np.clip(raw, self.limits_rad[:, 0], self.limits_rad[:, 1])

        diagnostics = dict(self.diagnostics)
        # Mirroring is an involution, so mirroring a mirrored clip gives the
        # original back and the flag has to say so.
        diagnostics["mirror"] = not bool(self.diagnostics.get("mirror", False))
        diagnostics["mirror_signs"] = [int(sign) for sign in MIRROR_SIGNS]
        diagnostics["mirror_offset_deg"] = {
            "left": np.degrees(bias[:half]).tolist(),
            "right": np.degrees(bias[half:]).tolist(),
        }
        # `source_deg[channel]["left"]` has always meant "the readout driving the
        # left arm". Swapping the keys keeps that true; the note says where the
        # numbers now come from, since they are her other side.
        diagnostics["mirror_source_deg"] = (
            "source_deg keys follow the ARM they drive, so they were swapped: "
            "source_deg[channel]['left'] is now read off her RIGHT side."
        )

        if kinematics is not None:
            check = {
                "raw": mirror_residual_mm(kinematics, self.raw, raw),
                "clipped": mirror_residual_mm(kinematics, self.arm_pos, arm_pos),
            }
            diagnostics["mirror_fk_check_mm"] = check
            worst = max(check["raw"]["max"], check["clipped"]["max"])
            if worst > MIRROR_CHECK_LIMIT_MM:
                raise ValueError(
                    f"mirror self-check failed: the mirrored pose misses the "
                    f"reflection of the original by up to {worst:.1f} mm, over "
                    f"the {MIRROR_CHECK_LIMIT_MM:.0f} mm limit. The robot's own "
                    f"left/right asymmetry measures 2.8 mm at worst and the "
                    f"mildest single wrong sign measures 94 mm, so this is a "
                    f"wrong sign, not a tolerance. Nothing was written."
                )

        return Trajectory(
            arm_pos=arm_pos, raw=raw, clipped=np.abs(arm_pos - raw) > 1e-12,
            frame_id=self.frame_id, timestamp_ns=self.timestamp_ns,
            time_s=self.time_s, limits_rad=self.limits_rad,
            source_deg={channel: {"left": sides["right"], "right": sides["left"]}
                        for channel, sides in self.source_deg.items()},
            diagnostics=diagnostics,
        )

    def clipped_counts(self) -> dict[str, int]:
        return {ARM_JOINT_NAMES[i]: int(self.clipped[:, i].sum())
                for i in range(ARM_DOF) if self.clipped[:, i].any()}


def _radians(values: np.ndarray) -> np.ndarray:
    return np.radians(np.asarray(values, dtype=np.float64))


def _independent_source_deg(animation: SourceAnimation, channel: Channel,
                            side: str, pitch_axis: np.ndarray) -> np.ndarray:
    """The source readout for one non-coupled channel, in degrees."""
    kind = channel.source_type
    points = animation.points
    if kind == "euler":
        if channel.body_index is None or channel.euler_axis is None:
            raise ValueError(f"{channel.name} claims euler but carries no index/axis")
        return readouts.wrist_euler_deg(animation, channel.body_index[side], channel.euler_axis)
    if kind == "geometric_flexion":
        return readouts.elbow_flexion_deg(points, side)
    if kind == "geometric_abduction":
        return readouts.abduction_deg(points, side)
    if kind == "geometric_hinge_pitch":
        return readouts.pitch_deg(points, side, pitch_axis)
    raise ValueError(f"{channel.name} has unsupported source_type {kind!r}")


def _measure_coupling_deg(kinematics: X2Kinematics, pose: np.ndarray) -> dict[str, np.ndarray]:
    """The robot's own plane twist per frame, at whatever pose it is given.

    One forward-kinematics pass per frame serves both arms, because a pose holds
    both. Call it with the yaw channels at zero to get the coupling term; call
    it with the finished trajectory to close the loop.
    """
    frames = pose.shape[0]
    out = {side: np.empty(frames) for side in SIDES}
    for index in range(frames):
        kinematics.set_arm(pose[index])
        for side in SIDES:
            # plane_twist_deg re-poses; avoid that by inlining the read here.
            out[side][index] = kinematics.plane_twist_deg(side, pose[index])
    return out


def retarget(animation: SourceAnimation,
             kinematics: X2Kinematics | None = None,
             ledger: Ledger | None = None,
             readout: str = READOUT_B,
             verify: bool = True) -> Trajectory:
    """Convert one clip into a 14-channel arm_pos trajectory in radians.

    `readout` selects the confirmed `shoulder_yaw` source: "plane" is readout B,
    the operator's choice, which makes the robot's forearm point where hers
    points (3.3 / 2.8 deg rms) at the cost of the upper arm's own spin. "spin"
    is readout A, kept because B's conditioning is |sin(elbow flexion)| and
    degenerates on a straight arm, which this clip never reaches but a future
    one might.
    """
    if readout not in (READOUT_A, READOUT_B):
        raise ValueError(f"readout must be {READOUT_B!r} (B) or {READOUT_A!r} (A), got {readout!r}")
    ledger = Ledger.load() if ledger is None else ledger
    ledger.require_confirmed()
    kinematics = X2Kinematics() if kinematics is None else kinematics

    frames = len(animation)
    raw = np.zeros((frames, ARM_DOF), dtype=np.float64)
    held = np.zeros_like(raw)
    source_deg: dict[str, dict[str, np.ndarray]] = {}

    pitch_axes = kinematics.pitch_axes()
    lower, upper = kinematics.limits_rad[:, 0], kinematics.limits_rad[:, 1]

    # --- the six independent channels -----------------------------------
    for name in INDEPENDENT_CHANNELS:
        channel = ledger[name]
        source_deg[name] = {}
        for side in SIDES:
            degrees = _independent_source_deg(animation, channel, side, pitch_axes[side])
            index = channel.index(side)
            source_deg[name][side] = degrees
            raw[:, index] = channel.command_rad(side, _radians(degrees))
            held[:, index] = np.clip(raw[:, index], lower[index], upper[index])

    # --- the coupled channel, on top of the six -------------------------
    # The coupling is measured at the CLIPPED commands, not the raw ones: the
    # robot will be standing at the clipped pose, so that is the twist it will
    # actually have. The elbow clips on 39 of 368 left frames in the reference
    # clip and reading the coupling off an unreachable pose would be wrong there.
    coupling = _measure_coupling_deg(kinematics, held)

    channel = ledger[COUPLED_CHANNEL]
    source_deg[COUPLED_CHANNEL] = {}
    indicators: dict[str, dict[str, np.ndarray]] = {}
    for side in SIDES:
        indicators[side] = readouts.twist_indicators(animation, side)
        her_twist = readouts.upper_arm_twist_deg(indicators[side], readout)
        degrees = wrap_deg(coupling[side] - her_twist)
        index = channel.index(side)
        source_deg[COUPLED_CHANNEL][side] = degrees
        raw[:, index] = channel.command_rad(side, _radians(degrees))
        held[:, index] = np.clip(raw[:, index], lower[index], upper[index])

    clipped = np.abs(held - raw) > 1e-12

    diagnostics: dict[str, object] = {
        "readout": readout,
        "ledger": str(ledger.path),
        "model": str(kinematics.model_file),
        "animation": str(animation.path),
        "pitch_axes": {side: pitch_axes[side].tolist() for side in SIDES},
        "yaw_axes": {side: axis.tolist() for side, axis in kinematics.yaw_axes().items()},
        "built_in_pitch_deg": kinematics.built_in_pitch_deg(pitch_axes),
        "built_in_splay_deg": kinematics.built_in_splay_deg(),
        "built_in_bend_deg": kinematics.built_in_bend_deg(),
        "coupling_span_deg": {side: float(np.ptp(coupling[side])) for side in SIDES},
        "yaw_conditioning_min": {side: float(indicators[side]["conditioning"].min())
                                 for side in SIDES},
        "yaw_reference_min": {side: float(readouts.reference_norm(indicators[side]).min())
                              for side in SIDES},
    }

    trajectory = Trajectory(
        arm_pos=held,
        raw=raw,
        clipped=clipped,
        frame_id=animation.frame_id.copy(),
        timestamp_ns=animation.timestamp_ns.copy(),
        time_s=animation.time_s,
        limits_rad=np.array(kinematics.limits_rad, dtype=np.float64),
        source_deg=source_deg,
        diagnostics=diagnostics,
    )

    if verify:
        diagnostics["closure_deg"] = closure(trajectory, animation, kinematics, readout)
    return trajectory


def closure(trajectory: Trajectory, animation: SourceAnimation,
            kinematics: X2Kinematics, readout: str = READOUT_B) -> dict[str, dict[str, float]]:
    """Push the finished command back through FK and re-measure the twist.

    This is the verification the channel rests on. Correlation and fitted slopes
    say a readout is related to a joint; closure says the robot commanded this
    way ends up where she was. It should land near the archived figures --
    rms 0.51 / 0.60 deg, worst about 2.0 deg, for readout B on the reference
    clip -- and a large number here means the mapping, not the clip, is wrong.
    """
    measured = _measure_coupling_deg(kinematics, trajectory.arm_pos)
    out: dict[str, dict[str, float]] = {}
    for side in SIDES:
        her = readouts.upper_arm_twist_deg(readouts.twist_indicators(animation, side), readout)
        error = np.abs(wrap_deg(measured[side] - her))
        out[side] = {
            "rms": float(np.sqrt((error ** 2).mean())),
            "max": float(error.max()),
            "mean": float(error.mean()),
        }
    return out
