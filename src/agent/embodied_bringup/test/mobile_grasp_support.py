"""Mobile-grasp E2E support (lekiwi-nav-grasp revision 2, R8, test-only).

The pure logic shared by the production-pick-pipeline end-to-end test in this
package: the MoveIt planning-scene construction (the chassis AABB cut at the
arm mount, fully transformed into the arm ``base`` frame — the table is
deliberately NOT in MoveIt's planning scene, maintainer decision 2026-11-02),
the motion-sample validity rule (the passive ``/check_state_validity`` audit
with independent endpoint coverage), the ground-truth judge (the frozen rest
height over a continuous recent window, the pre-lift window bounded by the
layout's ``lift_m``, the grasp region at the model TCP, the base stillness and
roll/pitch), the continuous-run invariants (the timestamped runtime-status
coverage/freshness rule, the steady-wall sim-clock watchdog with reset
detection) and the pick-feedback candidate census. The ROS-facing wrappers at
the bottom stay thin: they only move messages into arrays and call the pure
functions.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shared pose/quaternion math (pure)
# ---------------------------------------------------------------------------


def apply_transform(points: np.ndarray, matrix44: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to an Nx3 point set."""
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix44.T)[:, :3]


def quat_xyzw_to_matrix(quat) -> np.ndarray:
    """An xyzw quaternion -> its 3x3 rotation matrix."""
    qx, qy, qz, qw = (float(value) for value in quat)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def matrix_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    """A URDF origin (xyz + roll/pitch/yaw) -> its homogeneous 4x4 matrix."""
    roll, pitch, yaw = rpy
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [
                np.cos(yaw) * np.cos(pitch),
                np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll),
            ],
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll),
            ],
            [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)],
        ]
    )
    matrix[:3, 3] = xyz
    return matrix


def transform_matrix(transform) -> np.ndarray:
    """A geometry_msgs/TransformStamped (its ``.transform``) -> a 4x4 matrix."""
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = quat_xyzw_to_matrix((rotation.x, rotation.y, rotation.z, rotation.w))
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def point_in_frame(origin_xyz, origin_quat_xyzw, point_xyz) -> tuple[float, float, float]:
    """``point_xyz`` expressed in the frame at ``origin_xyz``/``origin_quat``."""
    rotation = quat_xyzw_to_matrix(origin_quat_xyzw)
    return tuple(rotation.T @ (np.array(point_xyz) - np.array(origin_xyz)))


def quat_in_frame(origin_quat_xyzw, quat_xyzw) -> tuple[float, float, float, float]:
    """The orientation ``quat_xyzw`` expressed IN the ``origin_quat_xyzw``
    frame: the Hamilton product of the frame's conjugate with the quat."""
    ox, oy, oz, ow = (float(v) for v in origin_quat_xyzw)
    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    # conj(origin) = (-x, -y, -z, w)
    ax, ay, az, aw = -ox, -oy, -oz, ow
    return (
        aw * qx + ax * qw + ay * qz - az * qy,
        aw * qy - ax * qz + ay * qw + az * qx,
        aw * qz + ax * qy - ay * qx + az * qw,
        aw * qw - ax * qx - ay * qy - az * qz,
    )


def quat_roll_pitch_deg(quat_xyzw) -> tuple[float, float]:
    """The roll/pitch of an xyzw quaternion (the base tilt invariants)."""
    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    roll = math.degrees(math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
    pitch = math.degrees(math.asin(min(1.0, max(-1.0, 2.0 * (qw * qy - qz * qx)))))
    return roll, pitch


# ---------------------------------------------------------------------------
# The planning-scene construction (pure, R6: the chassis only — the table is
# deliberately NOT in MoveIt's planning scene, maintainer decision 2026-11-02)
# ---------------------------------------------------------------------------


@dataclass
class SceneBox:
    """One collision object of the grasp planning scene.

    ``centre``/``dims`` are expressed in ``frame`` — review r1 #1: each box
    carries its OWN true frame (the chassis in the arm ``base`` frame);
    MoveIt transforms the stamped pose, so the frames must never be mixed.
    """

    object_id: str
    centre: tuple[float, float, float]
    dims: tuple[float, float, float]
    frame: str = "base"


def chassis_box(vertices: np.ndarray, arm_mount_underside_z: float) -> SceneBox:
    """The D6 chassis box: the AABB of the base_link collision mesh's
    vertices, cut at the arm mount's underside (in the vertices' frame)
    so the stowed arm is valid against it."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    hi[2] = min(hi[2], float(arm_mount_underside_z))
    centre = (lo + hi) / 2.0
    dims = np.maximum(hi - lo, 1e-3)
    return SceneBox("chassis", tuple(float(v) for v in centre), tuple(float(v) for v in dims))


def transform_scene_box(box: SceneBox, matrix44: np.ndarray, frame: str) -> SceneBox:
    """A SceneBox fully transformed into ``frame`` (review r1 item 4): the
    box's eight corners go through the homogeneous transform and are
    re-AABB'd in the target frame — never a centre-only translation that
    would silently drop the mount rotation."""
    half = np.array(box.dims) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    corners = apply_transform(np.array(box.centre) + signs * half, matrix44)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    dims = np.maximum(hi - lo, 1e-3)
    return SceneBox(
        box.object_id,
        tuple(float(v) for v in (lo + hi) / 2.0),
        tuple(float(v) for v in dims),
        frame=frame,
    )


# ---------------------------------------------------------------------------
# Motion-sample validity (pure, R6)
# ---------------------------------------------------------------------------

#: The motion-recording gates (D6): >= 20 Hz, gaps > 0.1 s sim time fail.
MIN_SAMPLE_RATE_HZ = 20.0
MAX_SAMPLE_GAP_SEC = 0.1


def motion_sample_verdict(
    stamps: Sequence[float],
    validity_results: Sequence[bool],
    min_rate_hz: float = MIN_SAMPLE_RATE_HZ,
    max_gap_sec: float = MAX_SAMPLE_GAP_SEC,
) -> tuple[bool, str]:
    """The D6 motion-validity verdict: the sample rate, the gap rule and
    every sample's collision check (group ``arm`` against the same scene)."""
    if not stamps:
        return False, "no samples recorded"
    if len(stamps) != len(validity_results):
        return False, "validity results do not match the sample count"
    if len(stamps) < 2:
        return False, f"insufficient samples ({len(stamps)}) — an executed motion must leave a stream, not a point"
    for previous, following in zip(stamps, stamps[1:], strict=False):
        if following - previous > max_gap_sec + 1e-9:
            return False, f"sample gap {following - previous:.3f} s > {max_gap_sec} s"
        if not following > previous:
            return False, f"sample stamps must strictly increase: {following:.3f} s <= {previous:.3f} s"
    span = float(stamps[-1]) - float(stamps[0])
    if len(stamps) / span < min_rate_hz:
        return False, f"sample rate {len(stamps) / span:.1f} Hz < {min_rate_hz} Hz"
    invalid = int(sum(1 for result in validity_results if not result))
    if invalid:
        return False, f"{invalid} invalid samples against the planning scene"
    return True, f"{len(stamps)} samples, all valid"


def motion_audit_window(
    samples: Sequence[tuple[float, Sequence[float]]],
    start_s: float,
    end_s: float,
    epsilon_rad: float = 1e-3,
) -> tuple[list[tuple[float, list[float]]], str]:
    """The R6 audit window with independent endpoint coverage (review r2 #5).

    ``start_s``/``end_s`` are INDEPENDENT markers — the sim stamps of the
    recording's begin and end — never inferred from the surviving samples.
    The window runs from the LAST AT-REST sample before the first arm motion
    (the onset transition is retained: a buffer that starts mid-motion cannot
    masquerade as the onset) to the buffer's end. The buffer itself must
    cover both markers within the unchanged 0.1 s bound: a lost head or tail
    truncates the audit and fails instead of passing a short prefix.

    Returns ``(window, "")`` when the arm moved over a covered buffer;
    ``([], reason)`` otherwise — ``no arm motion over the covered window``
    is the explicit no-motion evidence a failed pick needs (review r2 #2),
    every other reason is a coverage failure.
    """
    if not samples:
        return [], "no arm samples recorded — the recording itself is missing"
    if float(samples[0][0]) > float(start_s) + MAX_SAMPLE_GAP_SEC:
        return [], (
            f"the recording lost its head: the first sample is at {float(samples[0][0]):.2f} s, "
            f"more than {MAX_SAMPLE_GAP_SEC} s after the recording started at {float(start_s):.2f} s"
        )
    if float(samples[-1][0]) < float(end_s) - MAX_SAMPLE_GAP_SEC:
        return [], (
            f"the recording lost its tail: the last sample is at {float(samples[-1][0]):.2f} s, "
            f"more than {MAX_SAMPLE_GAP_SEC} s before the recording ended at {float(end_s):.2f} s"
        )
    rest = np.asarray(samples[0][1], dtype=float)
    for index, (_stamp, positions) in enumerate(samples):
        if float(np.max(np.abs(np.asarray(positions, dtype=float) - rest))) > epsilon_rad:
            return [(stamp, list(positions)) for stamp, positions in samples[max(0, index - 1) :]], ""
    return [], "no arm motion over the covered window"


# ---------------------------------------------------------------------------
# The ground-truth judge (pure, R7)
# ---------------------------------------------------------------------------

#: The judge tolerances (D6).
JUDGE_MAX_SAMPLE_GAP_SEC = 0.1
JUDGE_GRASP_REGION_M = 0.03
JUDGE_REGION_MOTION_M = 0.01

#: The frozen-rest window rules (design R7: "1 s of fresh samples"): the
#: trailing window must be CONTINUOUS (no gap over the streams' own 0.1 s
#: bound) and DENSE (the GT streams run 13-20 Hz; two samples a second
#: apart are not a second of fresh data — review r2 #7).
REST_WINDOW_MIN_SAMPLES = 10
REST_WINDOW_MAX_GAP_SEC = 0.1


def _quat_angle_deg(a, b) -> float:
    """The rotation angle between two xyzw quaternions (shortest arc)."""
    dot = abs(sum(x * y for x, y in zip(a, b, strict=False)))
    return math.degrees(2.0 * math.acos(min(1.0, float(dot))))


def frozen_rest_height(samples: Sequence[tuple[float, float]], window_s: float = 1.0) -> float | None:
    """The frozen rest height (review r1 item 7, tightened r2 #7): the MEDIAN
    z of the trailing ``window_s`` of samples, frozen BEFORE any arm motion.

    Frozen means frozen: a later drop below the rest height can never lower
    it. The trailing window must be continuous and dense (see
    :data:`REST_WINDOW_MIN_SAMPLES`/:data:`REST_WINDOW_MAX_GAP_SEC`);
    ``None`` when it is empty, gapped or sparse — a one-point or gapped
    median is not a second of fresh pre-motion data."""
    if not samples:
        return None
    last = float(samples[-1][0])
    window = [float(z) for stamp, z in samples if last - window_s <= float(stamp) <= last]
    if len(window) < REST_WINDOW_MIN_SAMPLES:
        return None
    window_stamps = [float(stamp) for stamp, _z in samples if last - window_s <= float(stamp) <= last]
    for previous, following in zip(window_stamps, window_stamps[1:], strict=False):
        if following - previous > REST_WINDOW_MAX_GAP_SEC + 1e-9:
            return None
    return float(np.median(window))


def first_lift_stamp(
    object_samples: Sequence[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]],
    z_rest: float,
    *,
    detect_m: float,
) -> float | None:
    """The sim stamp of the first lift: the first object sample whose z
    rises ``detect_m`` above the frozen rest height. ``detect_m`` is the
    LAYOUT's ``lift_m`` (review r2 #7): the pre-lift footprint window closes
    exactly where the legitimate carry begins — a small bounce below the
    required lift height never disables the knocked-off monitoring (``None``
    when the object never rises that far)."""
    for stamp, position, _quat in object_samples:
        if position[2] >= z_rest + detect_m:
            return float(stamp)
    return None


@dataclass
class JudgeInput:
    """The synthetic-stream input of the judge (all sim-time stamped)."""

    z_rest: float
    """The FROZEN rest height: the median of a continuous, dense 1 s window
    of fresh samples taken BEFORE any arm motion — never a whole-run
    minimum, so an object dropped below its rest height can never lower it."""
    hold_seconds: float
    lift_m: float
    """The layout's lift requirement — ALSO the pre-lift window's boundary
    (the knocked-off monitoring ends exactly when the object legitimately
    rises this far)."""
    max_object_rotation_in_gripper_deg: float
    """The hold rigidity bound (Revision-2 corrections): the object's
    orientation IN THE GRIPPER LINK frame may change by at most this angle
    over the hold window."""
    object_samples: list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]]
    """(stamp, position, quat_xyzw) of the object in the WORLD/map frame."""
    object_in_gripper_samples: list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]]
    """(stamp, position of the object centre in the GT gripper frame, the
    object orientation in the gripper frame) — EVERY paired observation, in
    or out of the grasp region (review r2 #6): the judge's region and gap
    rules do the rejecting, never a wrapper-side filter."""
    hold_point_in_gripper: tuple[float, float, float]
    """The grasp-region centre in the gripper LINK frame: the MODEL TCP —
    the ``gripper -> tcp`` origin from the URDF/TF (review r2 #7), never an
    empirical constant."""
    object_table_region: tuple[tuple[float, float], tuple[float, float]]
    """The table's (x range, y range) in the MAP frame — the object must not
    leave it before the lift."""
    base_planar_speeds: list[tuple[float, float]]
    """(stamp, speed) of the base from the observe step to the end — the
    judge's base-stillness rule (the spec's planar speed <= 0.02 m/s)."""
    base_roll_pitch_deg: list[tuple[float, float]]
    """(roll, pitch) in degrees of every recorded base sample — RECORD ONLY
    (maintainer decision 2026-11-03: the max over the hold window goes to
    metrics.json; the judge applies no tilt bound — the spec judges the
    base by planar speed alone)."""


def judge_hold(data: JudgeInput) -> tuple[bool, str]:
    """The D6 judge: success iff over a continuous hold window the object is
    lifted, upright, carried in the grasp region and not moving within it;
    plus the pre-lift table-region, base-stillness and base-tilt rules.

    The pre-lift table-region (knocked-off) rule applies only to the samples
    BEFORE the first lift, and the first lift is the object rising
    ``lift_m`` (review r2 #7): the transport legitimately carries the object
    out of the table footprint, and a bounce below the required lift height
    never closes the monitoring early."""
    if not data.object_samples:
        return False, "no ground-truth object samples"
    # the object never left the table region before the lift
    lift_stamp = first_lift_stamp(data.object_samples, data.z_rest, detect_m=data.lift_m)
    for stamp, position, _quat in data.object_samples:
        if lift_stamp is not None and stamp >= lift_stamp:
            break  # the pre-lift window closed at the first lift (that sample is already lifted)
        (x_lo, x_hi), (y_lo, y_hi) = data.object_table_region
        if not (x_lo <= position[0] <= x_hi and y_lo <= position[1] <= y_hi):
            return False, f"the object left the table region at ({position[0]:.3f}, {position[1]:.3f}) — knocked off"
    # the base stays still from the observe step to the end
    for stamp, speed in data.base_planar_speeds:
        if speed > 0.02:
            return False, f"the base moved at {speed:.3f} m/s at t={stamp:.2f} during the attempt"
    # (the base's roll/pitch is RECORD ONLY — maintainer decision 2026-11-03:
    # the max over the hold window goes to metrics.json; no tilt bound)
    # the hold window: continuous, lifted, upright, in the region, carried.
    # The design's hold is "after the result, hold_s of sim time" — the
    # TRAILING contiguous run of the carried history, not the whole span,
    # with the 0.1 s gap rule unchanged INSIDE the window.
    if not data.object_in_gripper_samples:
        return False, "no object samples in the gripper frame"
    runs: list[list[tuple[float, tuple[float, float, float]]]] = []
    for sample in data.object_in_gripper_samples:
        if runs and sample[0] - runs[-1][-1][0] > JUDGE_MAX_SAMPLE_GAP_SEC + 1e-9:
            runs.append([])
        if not runs:
            runs.append([])
        runs[-1].append(sample)
    window = runs[-1]
    stamps = [sample[0] for sample in window]
    span = stamps[-1] - stamps[0]
    if span < data.hold_seconds:
        if len(runs) > 1:
            split_gap = window[0][0] - runs[-2][-1][0]
            return False, (
                f"the trailing hold window is {span:.2f} s < {data.hold_seconds} s — a sample gap of "
                f"{split_gap:.3f} s > {JUDGE_MAX_SAMPLE_GAP_SEC} s split it off the carried history"
            )
        return False, f"the hold window is {span:.2f} s < {data.hold_seconds} s"
    first = np.array(window[0][1])
    region_motion = max(float(np.linalg.norm(np.array(sample[1]) - first)) for sample in window)
    if region_motion > JUDGE_REGION_MOTION_M:
        return False, f"the object moved {region_motion * 1000:.1f} mm within the gripper frame — slipping"
    hold_point = np.array(data.hold_point_in_gripper)
    outside = [
        position
        for _stamp, position, _quat in window
        if float(np.linalg.norm(np.array(position) - hold_point)) > JUDGE_GRASP_REGION_M
    ]
    if outside:
        return False, "the object is outside the gripper's grasp region — resting on the wrist or arm"
    # the rigidity rule (Revision-2 corrections): the object's orientation
    # IN THE GRIPPER frame changes <= max_object_rotation_in_gripper_deg
    # over the window
    first_quat = window[0][2]
    max_rotation = 0.0
    for _stamp, _position, quat in window:
        angle = _quat_angle_deg(first_quat, quat)
        max_rotation = max(max_rotation, angle)
    if max_rotation > data.max_object_rotation_in_gripper_deg:
        return False, (
            f"the object rotated {max_rotation:.1f} deg in the gripper frame > "
            f"{data.max_object_rotation_in_gripper_deg} — not rigidly held"
        )
    # the world-frame lift over the window
    window_samples = [
        (stamp, position, quat)
        for stamp, position, quat in data.object_samples
        if stamps[0] - 0.1 <= stamp <= stamps[-1] + 0.1
    ]
    if not window_samples:
        return False, "no world-frame object samples over the hold window"
    heights = [position[2] for _stamp, position, _quat in window_samples]
    if min(heights) < data.z_rest + data.lift_m - 1e-9:
        return False, f"the object z {min(heights):.4f} < z_rest + lift ({data.z_rest + data.lift_m:.4f})"
    return True, (
        f"held {stamps[-1] - stamps[0]:.2f} s, rotated {max_rotation:.1f} deg in the gripper, "
        f"drifted {region_motion * 1000:.1f} mm, lifted {min(heights) - data.z_rest:.3f} m"
    )


# ---------------------------------------------------------------------------
# The continuous invariants (pure, review r1 item 8, tightened r2 #4) and the
# feedback census (review r1 item 15)
# ---------------------------------------------------------------------------

#: The runtime-status coverage/freshness bounds: the facade publishes on a
#: 1 s ROS-clock (sim) timer, so adjacent samples are ~1 s of sim time
#: apart; a gap or end-staleness over three publication periods is a
#: coverage hole (review r2 #4).
STATUS_MAX_GAP_SIM_S = 3.0


def runtime_status_verdict(
    samples: Sequence[tuple[float, str, bool]],
    end_sim_s: float | None = None,
    max_gap_s: float = STATUS_MAX_GAP_SIM_S,
) -> tuple[bool, str]:
    """The pick window's runtime-status invariant: EVERY recorded sample
    must be lifecycle ACTIVE with the stop latch open, and the samples —
    timestamped (review r2 #4) — must COVER the window: an empty stream, a
    coverage hole, or a stream gone stale before the window's end each fail
    the run (never a vacuous "0 status samples, all ACTIVE" pass)."""
    if not samples:
        return False, "no runtime-status samples in the window — the status evidence is missing"
    for _sim, lifecycle, stop_latched in samples:
        if stop_latched:
            return False, f"the runtime stop latch engaged during the attempt (lifecycle {lifecycle})"
        if str(lifecycle) != "ACTIVE":
            return False, f"the runtime lifecycle left ACTIVE during the attempt: {lifecycle}"
    for (sim_a, _la, _sa), (sim_b, _lb, _sb) in zip(samples, samples[1:], strict=False):
        if float(sim_b) - float(sim_a) > max_gap_s:
            return False, (
                f"runtime-status coverage gap: {float(sim_b) - float(sim_a):.1f} s of sim time "
                f"between samples ({float(sim_a):.1f} -> {float(sim_b):.1f} s)"
            )
    if end_sim_s is not None:
        stale = float(end_sim_s) - float(samples[-1][0])
        if stale > max_gap_s:
            return False, f"the last runtime-status sample is {stale:.1f} s stale at the window's end"
    return True, f"{len(samples)} status samples, all ACTIVE with the latch open and covered"


def clock_stall_verdict(
    samples: Sequence[tuple[float, float]],
    max_stall_wall_s: float = 2.0,
    end_wall_s: float | None = None,
) -> tuple[bool, str]:
    """The steady-wall sim-clock watchdog (review r2 #4): the simulation
    must keep advancing. Each sample is a ``(wall_s, sim_s)`` receipt on the
    clock stream. Fails on:

    - a rewind (a later receipt's sim time earlier — a reset detected
      directly on /clock, no tolerance),
    - a receipt dropout (more than ``max_stall_wall_s`` wall time between
      adjacent receipts — silence is not progress),
    - a stall (more than ``max_stall_wall_s`` wall time since the sim time
      last ADVANCED — five seconds of repeated sim time at 10 Hz fails),
    - terminal silence (``end_wall_s`` past the last advance — the window's
      end needs a final observation).
    """
    if not samples:
        return False, "no /clock samples in the window — the clock evidence is missing"
    last_advance_wall = float(samples[0][0])
    for (wall_a, sim_a), (wall_b, sim_b) in zip(samples, samples[1:], strict=False):
        if float(sim_b) < float(sim_a):
            return False, f"the sim clock rewound from {float(sim_a):.2f} to {float(sim_b):.2f} (a reset)"
        if float(wall_b) - float(wall_a) > max_stall_wall_s:
            return False, (
                f"the /clock stream dropped out: {float(wall_b) - float(wall_a):.1f} s of wall time "
                f"between receipts (sim {float(sim_a):.2f} -> {float(sim_b):.2f})"
            )
        if float(sim_b) > float(sim_a):
            last_advance_wall = float(wall_b)
        elif float(wall_b) - last_advance_wall > max_stall_wall_s:
            return False, (
                f"the sim clock stalled: {float(wall_b) - last_advance_wall:.1f} s of wall time with no "
                f"sim advance (sim {float(sim_a):.2f} -> {float(sim_b):.2f})"
            )
    if end_wall_s is not None and float(end_wall_s) - last_advance_wall > max_stall_wall_s:
        return False, (
            f"terminal clock silence: {float(end_wall_s) - last_advance_wall:.1f} s of wall time with no "
            "sim advance before the window's end"
        )
    return True, f"{len(samples) - 1} clock intervals, no stall, dropout or rewind over {max_stall_wall_s:.0f} s"


def pick_feedback_candidate_rounds(feedback_lines: Sequence[str]) -> list[int] | None:
    """The candidate census per selection round (review r1 item 15): the
    count from the executor's own "evaluating N candidates" feedback, one
    entry per selection round. ``None`` — "unavailable" — when the captured
    feedback carries no evaluating line at all."""
    counts: list[int] = []
    for line in feedback_lines:
        match = re.search(r"detail=evaluating (\d+) candidates", str(line))
        if match:
            counts.append(int(match.group(1)))
    return counts or None


# ---------------------------------------------------------------------------
# The thin ROS-facing wrappers (not unit-tested; the logic is above)
# ---------------------------------------------------------------------------


def collision_object_message(box, operation) -> object:
    """The moveit_msgs CollisionObject of a :class:`SceneBox` (review r1 #1).

    The message stamps the box's OWN frame (``box.frame``) and carries the
    box centre/dims unchanged — MoveIt transforms the stamped pose into the
    planning frame, so mixing frames here displaces the box by the full
    mount transform.
    """
    from moveit_msgs.msg import CollisionObject  # noqa: PLC0415  (lazy: no ROS import at module load)
    from shape_msgs.msg import SolidPrimitive  # noqa: PLC0415

    obj = CollisionObject()
    obj.header.frame_id = str(box.frame)
    obj.id = box.object_id
    if operation == CollisionObject.ADD:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(value) for value in box.dims]
        obj.primitives.append(primitive)
        from geometry_msgs.msg import Pose  # noqa: PLC0415

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in box.centre)
        pose.orientation.w = 1.0
        obj.primitive_poses = [pose]
    obj.operation = operation
    return obj


def _stl_vertices(filename: str | None) -> list[np.ndarray]:
    """The vertices of a binary STL file (a test-side reader, no new dep).

    The filename is a ``package://`` URI; the package share directory is
    resolved through the ament index. An ASCII STL or a missing file raises
    ValueError — the mesh the chassis derivation reads is a binary STL by
    construction.
    """
    if not filename:
        raise ValueError("no mesh filename to read")
    path = Path(_package_mesh_path(filename))
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path.name}: too short for a binary STL")
    count = struct.unpack("<I", data[80:84])[0]
    if len(data) < 84 + count * 50:
        raise ValueError(f"{path.name}: truncated binary STL")
    vertices: list[np.ndarray] = []
    for index in range(count):
        base = 84 + index * 50 + 12  # skip the normal
        for corner in range(3):
            x, y, z = struct.unpack("<3f", data[base + corner * 12 : base + corner * 12 + 12])
            vertices.append(np.array([x, y, z]))
    return vertices


def _package_mesh_path(filename: str) -> str:
    match = re.match(r"package://([^/]+)/(.*)", filename)
    if not match:
        return filename
    from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

    return f"{get_package_share_directory(match.group(1))}/{match.group(2)}"
