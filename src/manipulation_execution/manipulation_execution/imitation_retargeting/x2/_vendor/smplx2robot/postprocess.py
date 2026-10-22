"""Offline post-processing for a converted trajectory. Nothing here touches mapping.

`retarget` turns a clip into joint angles. That is a faithful execution of the
ledger and it is finished. What comes out of it is still not a trajectory a
robot can be asked to follow: it carries the fitter's mistakes, it moves several
times faster than the arm is rated for, and it starts wherever the actor's first
frame happened to be.

This module is the part that makes it followable:

    de-spike        the fitter flips a limb for three frames and comes back;
                    a low-pass filter does not remove that, it SMEARS it
    smooth          trade source fidelity for continuity, on purpose, by an
                    amount the operator chooses by watching video
    limit repair    a filter overshoots at corners and pushes angles out of
                    range; put them back without re-smoothing the whole clip
    ramps           lead in from the robot's held pose and lead back to it,
                    because the player publishes frame 0 the instant it starts
    retime          slow down until every channel is inside its speed budget
    resample        land exactly on the player's 50 Hz grid

ALL OF THE ABOVE IS IMPLEMENTED. `run()` applies the chain in the order
section 4 of `docs/postprocess_plan.md` fixes -- resample_uniform, despike,
smooth, repair_limits, retime, add_ramps -- and writes an npz plus a sidecar
accounting for every stage. `report()` is still the measure-only path: it
prints and writes nothing.

RUNS ON THE BOARD. The whole toolkit is going to 310P eventually, and this
module is the part that gets there first, so it imports numpy and scipy and
NOTHING ELSE from this package's dependency set. In particular it must never
reach `robot.py`, which imports mujoco at module scope -- mujoco is not
installed on the board, and today that single import is enough to make
`export`, `safety` and `retarget` all fail to load there.
`tests/test_postprocess.py` asserts this with mujoco blocked out of
`sys.meta_path`, because a rule like this survives as a test and not as a
comment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_DOF = 14

# The base pose. Given by the operator on 2026-09-09 as the vendor's own
# required value -- shoulder_pitch 0.4, elbow -1.2, every other channel exactly
# zero, left and right identical. Every animation starts here and returns here.
#
# It is NOT zero: zero is both elbows straight, which is a pose and not a
# restful one. Seen from the front it reads as "arms hanging down", because the
# forearms point at the camera and foreshorten away; the side view is the one
# that shows what it is -- upper arms down, forearms horizontal and forward.
# Renders that a person is asked to judge must include a side view. That
# mistake cost a round trip once already (plan 6.3.1).
BASE_POSE_RAD = np.array(
    [0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0,
     0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0]
)

# What the arm was MEASURED holding in the vendor's acceptance animation
# (aimdk/.../animations/acceptance_left_elbow.json `base.arm_pos`, the numbers
# asserted in that package's test/test_ros_integration.py). Kept because it is
# the only recorded observation of a real X2 at rest, and it is worth knowing
# how far one robot on one day sat from nominal: 4.06 deg at the elbow, under
# 1.7 deg everywhere else. It is NOT the default -- nominal is what the vendor
# asks for and what the operator ruled on. Use --base-pose to select it.
MEASURED_BASE_RAD = np.array(
    [0.392598, -0.025215, -0.013518, -1.129090, 0.006807, 0.011635, 0.0,
     0.390680, 0.028282, 0.022147, -1.147114, 0.006999, 0.000191, 0.005150]
)

# Rated speeds, rad/s, in ARM_JOINT_NAMES order. These belong to the robot
# model, not to this file -- the copy is here only because the model needs
# mujoco and this module must load without it. `convert` will write them into
# the npz as `velocity_limits_rad_s` (plan 6.7 (2)); once it does, a trajectory
# is self-sufficient and this fallback stops being read. Until then the report
# says out loud which of the two it used, because a stale copy that nobody
# notices is exactly the failure this repo already guards against for the
# ledger.
# Confirmed by the operator on 2026-09-09, watching the four-column comparison
# in output/postprocess_*.mp4: 4 Hz and 2 Hz look nearly the same on screen and
# 2 Hz looks slightly smoother. The numbers agree and say more than the eye
# could: peak jerk falls 4-7x (clip06 224707 -> 45209 deg/s^3) for about 1.5 deg
# of extra rms deviation. Cheap at the price, so 2 Hz is the default.
DEFAULT_CUTOFF_HZ = 2.0

# What the robot's own model says an arm joint can do, before any margin:
# (tau_max - |gravity term|) / M_ii evaluated on every real pose in the four
# clips, tightest channel shoulder_roll ~= 5874 deg/s^2 (plan 6.6; the rule of
# thumb "rated speed / 0.1 s" gives 7499 for the shoulder, same order).
#
# The acceleration ceiling is deliberately NOT an independent constant any
# more. It used to be a flat 1500, which is this number times 0.25 -- so asking
# for a 100% velocity budget while still measuring acceleration against a 25%
# margin reported four clips as "OVER" that were in fact at half the model's
# limit. One dial, one margin: both budgets are `velocity_fraction` of rated.
# --max-acceleration still overrides it outright.
RATED_ACCELERATION_DEG_S2 = 5874.0

# The ramps keep the conservative 25% margin whatever the clip's budget is.
# Linking the reporting ceiling to velocity_fraction had a side effect worth
# naming: `quintic_ramp_seconds` sizes the ramp against the ceiling, so raising
# it made the lead-in shorter and snappier, and the trajectory's peak
# acceleration stopped being a property of the motion at all -- it was the
# entrance, at 84% of the torque limit, every time. The ramp is invented motion
# whose only job is to get from base to the first pose; there is nothing to be
# gained by spending the budget on it, and a person judging "is this too fast"
# should be judging the clip, not my approach segment.
RAMP_ACCELERATION_DEG_S2 = 0.25 * RATED_ACCELERATION_DEG_S2

# The speed budget, as a fraction of the vendor's rated joint speeds. Ruled by
# the operator on 2026-09-09 after watching the 100/90/80 comparison of clip03
# and clip06 side by side: "我觉得也就 1.0 能看，0.8 就有点慢了." At 1.0 three of
# the four clips are not stretched at all and clip06 is stretched x1.34, which
# is not a margin choice -- it is wrist_roll being rated at 240 deg/s against
# 863.8 everywhere else, so that stretch is the hardware talking, not me.
#
# This is a number a person chose by looking, not one this file derived. It is
# the whole margin: nothing downstream adds more. Lower it and everything gets
# proportionally slower; there is no other effect.
DEFAULT_VELOCITY_FRACTION = 1.0

# The vendor's rated arm speeds, in arm_pos order. Not a stopgap for reading
# them out of the URDF: it is the BETTER source. The URDF carries these same
# figures converted to rad/s and kept to three decimals -- 13.088 / 15.077 /
# 4.188 -- and not consistently rounded to nearest either (863.8 deg/s is
# 15.076154 and got written 15.077; 240.0 is 4.18879 and got written 4.188).
# Reading them back out and converting to degrees therefore lands up to
# 0.048 deg/s (0.019%) off the round numbers the vendor actually specified.
# `tests/test_postprocess.py` pins the two together at the URDF's precision.
RATED_SPEED_DEG_S = np.array(
    [749.9, 749.9, 863.8, 863.8, 863.8, 240.0, 240.0] * 2
)


@dataclass(frozen=True)
class Loaded:
    """A converted trajectory, as far as this module needs to understand one."""

    path: Path
    time_s: np.ndarray          # [N]
    arm_pos: np.ndarray         # [N, 14] rad
    limits_rad: np.ndarray      # [14, 2] lo, hi
    joint_names: list[str]
    clipped: np.ndarray         # [N, 14] bool
    rated_rad_s: np.ndarray     # [14]
    rated_source: str           # where rated_rad_s came from, for the report
    # True when convert was run with --mirror. Carried, never acted on: this
    # module filters a curve and does not care which arm it belongs to. It is
    # here so the sidecar can say it, because a mirrored clip and its baseline
    # differ in no other way an eye can catch at this stage.
    mirror: bool = False


def load(path: str | Path) -> Loaded:
    """Read a convert npz. Position limits and joint names ride along in it."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arm_pos = np.asarray(z["arm_pos"], dtype=float)
        time_s = np.asarray(z["time_s"], dtype=float)
        limits = np.asarray(z["limits_rad"], dtype=float)
        names = [str(n) for n in z["joint_names"]]
        clipped = np.asarray(z["clipped"], dtype=bool)
        if "velocity_limits_rad_s" in z.files:
            rated = np.asarray(z["velocity_limits_rad_s"], dtype=float)
            source = "npz velocity_limits_rad_s"
        else:
            rated = np.radians(RATED_SPEED_DEG_S)
            source = "postprocess.py rated-speed table (deg/s, vendor spec)"
        mirror = bool(z["mirror"]) if "mirror" in z.files else False
    if arm_pos.shape[1] != ARM_DOF:
        raise ValueError(f"{path}: expected {ARM_DOF} channels, got {arm_pos.shape[1]}")
    return Loaded(path, time_s, arm_pos, limits, names, clipped, rated, source,
                  mirror)


def timing(t: np.ndarray) -> dict:
    """Frame spacing, and where the clip's real gaps are.

    A filter assumes even spacing. Filtering across a 66.67 ms hole treats two
    frames of motion as one, so the gaps have to be found before anything else
    is believed.
    """

    dt = np.diff(t)
    if dt.size == 0:
        return {"count": t.size, "dt_median": float("nan"), "gaps": []}
    median = float(np.median(dt))
    gaps = [(int(i), float(d)) for i, d in enumerate(dt) if d > 1.5 * median]
    return {"count": int(t.size), "dt_median": median,
            "fps_nominal": 1.0 / median if median else float("nan"),
            "dt_min": float(dt.min()), "dt_max": float(dt.max()), "gaps": gaps}


def derivatives(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and acceleration from the REAL timestamps, not from a nominal
    frame rate. The clip has gaps; pretending it does not understates both."""

    v = np.gradient(x, t, axis=0)
    a = np.gradient(v, t, axis=0)
    return v, a


def hampel_flags(x: np.ndarray, half_window: int = 5, threshold: float = 3.0,
                 mad_floor_rad: float = np.radians(0.05)) -> np.ndarray:
    """Per-channel Hampel filter: |x - rolling median| > threshold * rolling MAD.

    `half_window` must be WIDER than the spike, which is why the default is 5
    and not 2: clip06's left wrist_yaw flips for three consecutive frames, and a
    window of five cannot see past it -- the median is the flip. That is the
    same reason a 5-point median filter barely moves the peak velocity.

    `mad_floor_rad` exists because MAD goes to zero on a still section, and
    without a floor every trembling digit there becomes an outlier.

    This marks candidates. It does not remove anything, and on its own it will
    mark real fast motion -- the plan's intersection rule (4.1) is what keeps
    those. M0 only reports.
    """

    n, c = x.shape
    pad = np.pad(x, ((half_window, half_window), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        pad, 2 * half_window + 1, axis=0
    )                                                    # [n, c, w]
    med = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - med[..., None]), axis=-1)
    scale = np.maximum(1.4826 * mad, mad_floor_rad)
    return np.abs(x - med) > threshold * scale


def classify_jumps(x: np.ndarray, min_step_rad: float,
                   bounce_within: int = 6) -> tuple[list, list]:
    """Split large single-frame changes into spikes and steps.

    A spike goes out and comes back within a few frames -- the fitter lost the
    limb and found it again. A step goes out and stays -- either a real fast
    movement or a fit that changed its mind for good. They need opposite
    treatment, so counting "jumps" as one number hides the only distinction
    that matters.
    """

    spikes, steps = [], []
    d = np.diff(x, axis=0)
    consumed = set()
    for frame, joint in zip(*np.where(np.abs(d) > min_step_rad)):
        frame, joint = int(frame), int(joint)
        if (frame, joint) in consumed:
            continue
        after = x[frame + 1: frame + 1 + bounce_within, joint]
        back = np.where(np.abs(after - x[frame, joint]) < 0.5 * abs(d[frame, joint]))[0]
        if back.size:
            # The excursion and the return are one event, not two. Charging the
            # return leg separately would file it under `steps`, since after
            # coming back there is nothing left to come back to -- which is how
            # a spike gets counted once in each column.
            spikes.append((frame, joint, float(d[frame, joint])))
            consumed.update((f, joint) for f in range(frame, frame + 1 + int(back[0]) + 1))
        else:
            steps.append((frame, joint, float(d[frame, joint])))
    return spikes, steps


def quintic_ramp_seconds(delta_rad: np.ndarray, vmax_rad_s: np.ndarray,
                         amax_rad_s2: float) -> tuple[float, int, str]:
    """How long a rest-to-rest quintic needs to cover `delta`, per channel.

    A quintic with zero end velocity and acceleration peaks at 1.875*d/T and
    5.7735*d/T^2, so each channel's minimum T falls straight out of its own
    budget. All fourteen run together, so the ramp takes the largest.
    """

    d = np.abs(delta_rad)
    t_v = np.divide(1.875 * d, vmax_rad_s, out=np.zeros_like(d),
                    where=vmax_rad_s > 0)
    t_a = np.sqrt(5.7735 * d / amax_rad_s2) if amax_rad_s2 > 0 else np.zeros_like(d)
    t = np.maximum(t_v, t_a)
    j = int(np.argmax(t))
    return float(t[j]), j, "velocity" if t_v[j] >= t_a[j] else "acceleration"


# ---------------------------------------------------------------------------
# M2 / M3 / M4 -- the steps that actually change numbers.
# ---------------------------------------------------------------------------


def resample_uniform(t: np.ndarray, x: np.ndarray, dt: float | None = None):
    """Put samples on an even grid, filling the dropped frames.

    Every filter downstream assumes even spacing. Filtering across a 66.67 ms
    hole treats two frames of motion as one, which is a fabricated velocity
    spike at exactly the place the data is already weakest. Linear, because a
    spline would overshoot into the gap and we would then be smoothing our own
    invention.
    """

    t = np.asarray(t, dtype=np.float64)
    if dt is None:
        dt = float(np.median(np.diff(t)))
    grid = t[0] + np.arange(int(round((t[-1] - t[0]) / dt)) + 1) * dt
    out = np.empty((grid.size, x.shape[1]), dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(grid, t, x[:, j])
    return grid, out


def despike(x: np.ndarray, dt: float, rated_rad_s: np.ndarray,
            half_window: int = 5, threshold: float = 3.0,
            mad_floor_rad: float = np.radians(0.05),
            impossible_factor: float = 3.0):
    """Flag outliers by the INTERSECTION rule, then rebuild them by interpolation.

    Two independent statistics have to agree (plan 4.1). The union was the
    earlier plan and it is the wrong operator here: false deletion is risk #1,
    because a wrongly deleted movement leaves a result that is smooth, plausible
    and simply missing something nobody can see is gone. A real fast wave trips
    the level test but not the step test; a fit that flipped the limb trips both.

    The one thing allowed to convict on its own is physical impossibility -- a
    single-frame change the machine could not perform at full rated speed. That
    is not a statistical claim, so it does not need a second opinion.
    """

    level = hampel_flags(x, half_window, threshold, mad_floor_rad)

    d = np.diff(x, axis=0, prepend=x[:1])
    median = np.median(d, axis=0, keepdims=True)
    mad = np.median(np.abs(d - median), axis=0, keepdims=True) * 1.4826
    step = np.abs(d - median) > threshold * np.maximum(mad, mad_floor_rad)

    # NOT `> rated * dt`. At source frame rate the clips already run at 361-482%
    # of rated, so a 1x test convicts 184 samples on clip06 -- overwhelmingly
    # real fast motion, whose answer is the slowdown in step 6, not deletion.
    # The factor has to sit above what a person can do, not above what the arm
    # can do. It is narrow: the known fit flip on clip06 measures 4.08x, so 3x
    # catches it and 5x misses it. Anything relying on this number should say so
    # out loud rather than trust it.
    impossible = np.abs(d) > impossible_factor * rated_rad_s[None, :] * dt

    # The intersection decides WHETHER there is an outlier event here. It cannot
    # decide how WIDE it is: the step test fires on entering and leaving an
    # excursion and is silent inside it, so on its own the intersection removes
    # only a spike's first frame and leaves the peak standing -- and the rebuild
    # below then interpolates the removed frames against that surviving peak,
    # which is worse than doing nothing. Measured on clip06 left_wrist_yaw: the
    # 3-frame flip at 260-262 kept frame 261 at 79.0 deg, and the low-pass then
    # smeared it across 15 frames.
    #
    # So: convict on the intersection, then grow the verdict along the level
    # mask to cover the whole excursion. Growth needs a seed, so a channel of
    # pure Hampel flags with no agreeing step test still gets deleted nothing --
    # the guard against false deletion is intact.
    seeds = (level & step) | impossible
    flags = seeds.copy()
    for j in range(x.shape[1]):
        column = level[:, j]
        for i in np.where(seeds[:, j])[0]:
            k = i
            while k + 1 < len(column) and column[k + 1]:
                k += 1
                flags[k, j] = True
            k = i
            while k - 1 >= 0 and column[k - 1]:
                k -= 1
                flags[k, j] = True

    out = x.copy()
    for j in range(x.shape[1]):
        bad = flags[:, j]
        if not bad.any() or bad.all():
            continue
        index = np.arange(x.shape[0])
        out[bad, j] = np.interp(index[bad], index[~bad], x[~bad, j])
    return out, flags, {"level": level, "step": step, "impossible": impossible}


def smooth(x: np.ndarray, dt: float, cutoff_hz: float = DEFAULT_CUTOFF_HZ,
           order: int = 4) -> np.ndarray:
    """Zero-phase low-pass, with the ends held out flat before filtering.

    `filtfilt`'s own padding reflects the signal, which on a trajectory that
    starts mid-motion invents a mirror-image motion before frame 0 and lets it
    ring back in. Holding the endpoint value flat says the only honest thing we
    know about the time before the clip: nothing was moving that we recorded.
    """

    from scipy.signal import butter, filtfilt

    nyquist = 0.5 / dt
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be inside (0, {nyquist})")
    b, a = butter(order, cutoff_hz / nyquist)

    pad = max(3 * (max(len(a), len(b)) - 1), int(round(1.0 / dt)))
    padded = np.vstack([np.repeat(x[:1], pad, axis=0), x,
                        np.repeat(x[-1:], pad, axis=0)])
    return filtfilt(b, a, padded, axis=0, padlen=0)[pad:pad + len(x)]


def repair_limits(x: np.ndarray, dt: float, lo: np.ndarray, hi: np.ndarray,
                  weight: float = 2e-4, margin_rad: float = np.radians(2.0)):
    """Pull the curve back inside the limits with a min-jerk QP, one solve.

    Not iterative clipping. Violations sit in a handful of frames, but clipping
    can only push back by smoothing the whole channel again, which moves frames
    that were never in violation -- measured at up to 16.4 deg across 290 clean
    frames -- and that extra smoothing bypasses the cutoff frequency the
    operator chose. It also failed to converge on clip02. This keeps the
    deviation where the problem is: strictly convex, one unique answer.

        min ||y - x||^2 + w^2 ||d3y/dt3||^2   s.t.  lo+m <= y <= hi-m
    """

    from scipy import sparse
    from scipy.optimize import lsq_linear

    n = x.shape[0]
    low = lo + margin_rad
    high = hi - margin_rad
    bad = low > high                      # margin wider than the limit itself
    low[bad] = lo[bad]
    high[bad] = hi[bad]

    rows = np.arange(n - 3)
    d3 = sparse.csr_matrix(
        (np.tile([-1.0, 3.0, -3.0, 1.0], n - 3),
         (np.repeat(rows, 4), (rows[:, None] + np.arange(4)).ravel())),
        shape=(n - 3, n),
    ) / dt ** 3
    A = sparse.vstack([sparse.identity(n, format="csr"), weight * d3], format="csr")

    out = x.copy()
    touched = np.zeros(x.shape[1], dtype=bool)
    for j in range(x.shape[1]):
        if x[:, j].min() >= low[j] and x[:, j].max() <= high[j]:
            continue
        b = np.concatenate([x[:, j], np.zeros(n - 3)])
        start = np.clip(x[:, j], low[j], high[j])
        out[:, j] = lsq_linear(A, b, bounds=(low[j], high[j]),
                               lsq_solver="lsmr", max_iter=200,
                               tol=1e-10).x if start.size else x[:, j]
        touched[j] = True
    return out, touched


def quintic_blend(a: np.ndarray, b: np.ndarray, steps: int) -> np.ndarray:
    """[steps,DOF] going a -> b, zero velocity and acceleration at both ends.

    Excludes both endpoints, so it splices between existing frames without
    duplicating either.
    """

    s = (np.arange(1, steps + 1) / (steps + 1))[:, None]
    return a[None, :] + (b - a)[None, :] * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def add_ramps(t: np.ndarray, x: np.ndarray, base: np.ndarray,
              vmax: np.ndarray, amax_rad_s2: float, dwell_s: float = 0.0):
    """Hold pose -> clip -> hold pose, as one continuous trajectory.

    The player publishes frame 0 the moment it starts -- no blend, no reading of
    where the arms actually are -- and holds the final frame forever after. Both
    ends are ours to build; neither is inherited from anywhere.

    `dwell_s` holds the base pose still at each end before the ramp moves. It
    costs nothing on the wire and buys two things: the arm is demonstrably at
    base before any motion starts, and -- the reason it exists -- the approach
    and the return become visible to a person watching. The operator read a
    0.43 s return as "you never wrote the return"; a still pose at each end is
    what makes the arc legible.
    """

    dt = float(np.median(np.diff(t)))
    amax = np.radians(amax_rad_s2)

    lead_T, _, _ = quintic_ramp_seconds(x[0] - base, vmax, amax)
    tail_T, _, _ = quintic_ramp_seconds(base - x[-1], vmax, amax)
    lead = max(int(np.ceil(lead_T / dt)), 1)
    tail = max(int(np.ceil(tail_T / dt)), 1)
    dwell = max(int(round(dwell_s / dt)), 0)

    hold = np.repeat(base[None, :], dwell, axis=0)
    stacked = np.vstack([
        hold,
        base[None, :], quintic_blend(base, x[0], lead - 1) if lead > 1 else np.empty((0, x.shape[1])),
        x,
        quintic_blend(x[-1], base, tail - 1) if tail > 1 else np.empty((0, x.shape[1])), base[None, :],
        hold,
    ])
    grid = np.arange(len(stacked)) * dt
    # Where each output frame came from: index into `x`, or -1 for the frames
    # this function invented. Downstream has no other way to tell a measured
    # sample from a ramp, and pairing a ramp frame with a source frame's bbox or
    # readout is exactly the kind of quiet mistake this repo keeps guarding.
    origin = np.full(len(stacked), -1, dtype=np.int64)
    origin[dwell + lead: dwell + lead + len(x)] = np.arange(len(x))
    return grid, stacked, {"lead_frames": lead, "tail_frames": tail,
                           "lead_s": lead * dt, "tail_s": tail * dt,
                           "dwell_frames": dwell, "dwell_s": dwell * dt,
                           "clip_first_frame": dwell + lead,
                           "clip_last_frame": dwell + lead + len(x) - 1,
                           "origin": origin}


# ---------------------------------------------------------------------------
# M5 -- the time axis. The only step that changes when, not what.
# ---------------------------------------------------------------------------


def retime(t: np.ndarray, x: np.ndarray, factor: float):
    """Stretch the clock by `factor`. Same poses, same order, longer duration.

    This is the one lever that lowers angular velocity, and it is worth being
    precise about why, because the obvious guess is wrong. Angular velocity is
    degrees per second. Re-sampling to a higher rate adds samples but keeps the
    duration, so it divides both the degrees and the seconds and leaves the
    quotient alone -- 50 Hz output plays at exactly the speed 30 Hz output did.
    The only way to make a joint sweep fewer degrees per second, without
    touching a single joint angle, is to give it more seconds. That is this.

    The curve is re-sampled onto a grid of the same dt, so a 6 s clip stretched
    3x comes back as an 18 s clip at the same frame rate, not as the same frames
    with sparser timestamps. Downstream keeps a sane grid, and `max_frame_step`
    stays a meaningful number.

    Endpoints are exact: the spline knots include both ends, so the first and
    last poses come back bit-for-bit. `add_ramps` reads exactly those two.
    """

    factor = float(factor)
    if not factor > 0:
        raise ValueError(f"time factor must be positive, got {factor}")
    span = float(t[-1] - t[0])
    if abs(factor - 1.0) < 1e-12 or span <= 0:
        return t - t[0], x.copy(), {
            "factor": 1.0, "input_frames": int(len(t)), "output_frames": int(len(t)),
            "input_duration_s": span, "output_duration_s": span,
            "dt_s": float(np.median(np.diff(t))) if len(t) > 1 else 0.0,
            "spline_vs_linear_deg": 0.0,
        }

    from scipy.interpolate import CubicSpline

    dt = float(np.median(np.diff(t)))
    stretched = (t - t[0]) * factor
    n = max(int(round(stretched[-1] / dt)) + 1, 2)
    grid = np.linspace(0.0, stretched[-1], n)
    y = CubicSpline(stretched, x, axis=0, bc_type="natural")(grid)
    # How far the spline bulges off the straight line through the same samples.
    # It is not error -- the polyline is not truth either -- but it is the only
    # honest number for "how much did re-gridding move the path", and if it ever
    # grows past a degree or so the interpolant, not the stretch, is the story.
    linear = np.column_stack([np.interp(grid, stretched, x[:, j])
                              for j in range(x.shape[1])])
    return grid, y, {
        "factor": factor, "input_frames": int(len(t)), "output_frames": int(n),
        "input_duration_s": span, "output_duration_s": float(grid[-1]),
        "dt_s": float(grid[1] - grid[0]),
        "spline_vs_linear_deg": float(np.degrees(np.abs(y - linear)).max()),
    }


def fit_velocity_budget(t: np.ndarray, x: np.ndarray, vmax: np.ndarray,
                        max_passes: int = 6, safety: float = 1.001):
    """The smallest stretch that puts every channel inside `vmax`.

    Solved by iteration rather than by dividing once, because the measurement
    is a finite difference on a grid: stretching changes how densely the grid
    samples the motion, so the peak the second pass measures is not exactly the
    first pass's peak over the factor. Two passes is the normal answer. The
    alternative -- divide once and trust it -- would leave the result a few
    percent over budget while reporting that it was inside, which is the single
    worst failure this module could have.
    """

    factor, over = 1.0, np.inf
    for i in range(max_passes):
        tt, xx, _ = retime(t, x, factor)
        v = np.gradient(xx, tt, axis=0)
        over = float((np.abs(v).max(axis=0) / np.maximum(vmax, 1e-12)).max())
        if over <= 1.0:
            return factor, i + 1, over
        factor *= over * safety
    return factor, max_passes, over


@dataclass(frozen=True)
class Processed:
    """The chain's output, plus everything needed to explain how it got there."""

    time_s: np.ndarray          # [M] even grid starting at 0
    arm_pos: np.ndarray         # [M, 14] rad
    origin: np.ndarray          # [M] index into the source clip, -1 where invented
    accounting: dict


def process(loaded: Loaded, base_pose_rad: np.ndarray = BASE_POSE_RAD,
            cutoff_hz: float | None = DEFAULT_CUTOFF_HZ,
            velocity_fraction: float = DEFAULT_VELOCITY_FRACTION,
            acceleration_ceiling_deg_s2: float | None = None,
            dwell_s: float = 0.5, trim_head: int = 0,
            trim_tail: int = 0, identity: bool = False,
            time_scale: float | None = None,
            fit_budget: bool = True) -> Processed:
    """The whole chain, in the order section 4 of the plan fixes.

    resample -> despike -> smooth -> repair limits -> retime -> ramp in and out.
    The order is not a preference: filtering before de-spiking smears an outlier
    across its neighbours instead of removing it, repairing limits before
    filtering lets the filter walk the result back outside them, and retiming
    before the ramps is what lets the ramps be built against the speed the clip
    will actually be played at.

    `velocity_fraction` is the headroom dial and the one number a human should
    be choosing here: the budget is that fraction of the vendor's rated joint
    speeds, and the required stretch is inversely proportional to it. At 0.25 a
    clip that peaks at rated speed is stretched 4x; at 1.0 it is not stretched
    at all and the motors run flat out. Nothing in this file knows what the real
    X2 tolerates, so nothing in this file should be the thing that decides.

    By default the stretch is applied (`fit_budget`), because output that is
    silently over budget is the failure mode worth designing against. Pass
    `time_scale` to force a factor, or `fit_budget=False` to only measure.
    """

    if acceleration_ceiling_deg_s2 is None:
        acceleration_ceiling_deg_s2 = velocity_fraction * RATED_ACCELERATION_DEG_S2

    if identity:
        # Every step off. The point is not that this is useful output -- it is
        # that the reader, the writer and the schema can be tested apart from
        # the filters. If this does not come back bit-for-bit, a later
        # difference cannot be blamed on the filter that seems to have caused it.
        return Processed(
            loaded.time_s.copy(), loaded.arm_pos.copy(),
            np.arange(len(loaded.arm_pos), dtype=np.int64),
            {"parameters": {"identity": True},
             "resample": {"input_frames": int(len(loaded.arm_pos)),
                          "grid_frames": int(len(loaded.arm_pos)),
                          "frames_filled": 0,
                          "dt_s": float(np.median(np.diff(loaded.time_s))),
                          "rate_hz": 1.0 / float(np.median(np.diff(loaded.time_s)))},
             "despike": {"samples_flagged": 0, "per_channel": {},
                         "seeds_level_and_step": 0, "seeds_impossible": 0},
             "smooth": {"max_deviation_deg": 0.0, "rms_deviation_deg": 0.0},
             "limits": {"channels_repaired": [], "outside_after_repair": 0},
             "ramps": {"lead_frames": 0, "tail_frames": 0, "dwell_frames": 0},
             "result": {"frames": int(len(loaded.arm_pos)),
                        "duration_s": float(loaded.time_s[-1] - loaded.time_s[0]),
                        "identity_passthrough": True}})

    x0 = loaded.arm_pos[trim_head: len(loaded.arm_pos) - trim_tail or None]
    t0 = loaded.time_s[trim_head: len(loaded.time_s) - trim_tail or None]
    if len(x0) < 8:
        raise ValueError(f"{loaded.path}: {len(x0)} frames left after trimming, "
                         "too few to filter")

    t, x = resample_uniform(t0, x0)
    dt = float(np.median(np.diff(t)))
    filled = len(t) - len(t0)

    xd, flags, parts = despike(x, dt, loaded.rated_rad_s)
    # `cutoff_hz=None` runs the chain without the low-pass. De-spiking still
    # happens: removing a fit that lost the limb and smoothing a curve are two
    # different jobs, and only the second one is optional.
    xs = xd if cutoff_hz is None else smooth(xd, dt, cutoff_hz)
    xr, touched = repair_limits(xs, dt, loaded.limits_rad[:, 0], loaded.limits_rad[:, 1])

    vmax = loaded.rated_rad_s * velocity_fraction
    peak_before = float(np.degrees(np.abs(np.gradient(xr, t, axis=0))).max())
    if time_scale is not None:
        factor, passes, mode = float(time_scale), 0, "explicit"
    elif fit_budget:
        factor, passes, _ = fit_velocity_budget(t, xr, vmax)
        mode = "fit_velocity_budget"
    else:
        factor, passes, mode = 1.0, 0, "off"
    tq, xq, rinfo = retime(t, xr, factor)

    grid, out, info = add_ramps(tq, xq, base_pose_rad, vmax,
                                min(acceleration_ceiling_deg_s2,
                                    RAMP_ACCELERATION_DEG_S2), dwell_s)

    v, a = derivatives(grid, out)
    peak_v = np.abs(v).max(axis=0)
    over = peak_v / np.maximum(vmax, 1e-12)
    jerk = np.gradient(a, grid, axis=0)

    accounting = {
        "parameters": {
            "cutoff_hz": cutoff_hz, "velocity_fraction": velocity_fraction,
            "acceleration_ceiling_deg_s2": acceleration_ceiling_deg_s2,
            "dwell_s": dwell_s, "trim_head": trim_head, "trim_tail": trim_tail,
            "time_scale": time_scale, "fit_budget": fit_budget,
            "base_pose_deg": np.degrees(base_pose_rad).round(4).tolist(),
            "rated_source": loaded.rated_source,
        },
        "resample": {
            "input_frames": int(len(t0)), "grid_frames": int(len(t)),
            "frames_filled": int(filled), "dt_s": dt, "rate_hz": 1.0 / dt,
        },
        "despike": {
            "samples_flagged": int(flags.sum()),
            "per_channel": {n: int(c) for n, c in
                            zip(loaded.joint_names, flags.sum(axis=0)) if c},
            "seeds_level_and_step": int((parts["level"] & parts["step"]).sum()),
            "seeds_impossible": int(parts["impossible"].sum()),
        },
        "smooth": {
            "max_deviation_deg": float(np.degrees(np.abs(xs - xd)).max()),
            "rms_deviation_deg": float(np.degrees(np.sqrt(((xs - xd) ** 2).mean()))),
        },
        "limits": {
            "channels_repaired": [n for n, c in zip(loaded.joint_names, touched) if c],
            "outside_after_repair": int(((xr < loaded.limits_rad[:, 0]) |
                                         (xr > loaded.limits_rad[:, 1])).sum()),
        },
        "retime": {
            "mode": mode, "factor": float(factor), "passes": int(passes),
            "peak_velocity_before_deg_s": peak_before,
            "budget_deg_s": np.degrees(vmax).round(2).tolist(),
            **rinfo,
        },
        "ramps": {k: v for k, v in info.items() if k != "origin"},
        "result": {
            "frames": int(len(out)), "duration_s": float(grid[-1]),
            "first_frame_max_deg_from_base":
                float(np.degrees(np.abs(out[0] - base_pose_rad)).max()),
            "last_frame_max_deg_from_base":
                float(np.degrees(np.abs(out[-1] - base_pose_rad)).max()),
            "peak_velocity_deg_s": float(np.degrees(peak_v.max())),
            "peak_velocity_fraction_of_budget": float(over.max()),
            # 1.0 means "in budget". Above 1.0 means the file in your hand is
            # faster than the budget it claims, which is only possible with
            # --time-scale or --no-slowdown; it is kept as a key rather than an
            # exception because measuring an over-speed clip is a legitimate
            # thing to want, and refusing to write it would just move the
            # measurement into someone's private script.
            "still_needs_slowdown": float(max(over.max(), 1.0)),
            # The channel that decides the stretch. It is rarely the fastest one
            # in deg/s -- wrist_roll and wrist_pitch are rated at 240 deg/s
            # against 863.8 elsewhere, so they hit their ceiling while moving at
            # a third of another joint's speed. Quoting the trajectory's overall
            # peak next to "100% of budget" invites exactly that confusion, so
            # the channel's own peak and its own budget are carried too.
            "worst_channel": loaded.joint_names[int(np.argmax(over))],
            "worst_channel_peak_deg_s": float(np.degrees(peak_v[int(np.argmax(over))])),
            "worst_channel_budget_deg_s": float(np.degrees(vmax[int(np.argmax(over))])),
            "peak_acceleration_deg_s2": float(np.degrees(np.abs(a).max())),
            "acceleration_ceiling_deg_s2": float(acceleration_ceiling_deg_s2),
            "acceleration_fraction_of_ceiling":
                float(np.degrees(np.abs(a).max()) / acceleration_ceiling_deg_s2),
            # Not enforced. Slowing down divides acceleration by the square of
            # the factor, so fitting velocity usually carries it -- but "usually"
            # is not "always", and a number nobody prints is a number nobody checks.
            "ramp_acceleration_ceiling_deg_s2":
                float(min(acceleration_ceiling_deg_s2, RAMP_ACCELERATION_DEG_S2)),
            "acceleration_within_ceiling":
                bool(np.degrees(np.abs(a).max()) <= acceleration_ceiling_deg_s2),
            "peak_jerk_deg_s3": float(np.degrees(np.abs(jerk).max())),
            "max_frame_step_deg": float(np.degrees(np.abs(np.diff(out, axis=0)).max())),
        },
    }
    return Processed(grid, out, info["origin"], accounting)


def write(result: Processed, loaded: Loaded, out_npz: str | Path) -> tuple[Path, Path]:
    """Write `*_smoothed.npz` and its sidecar json. Never over the input.

    The npz keeps convert's schema so `check` and `preview` read it unchanged.
    What it deliberately does NOT carry forward is every per-frame array whose
    row no longer means the same thing: `arm_pos_raw`, `frame_id`,
    `timestamp_ns`, and the `source_deg__*` readouts are all indexed by SOURCE
    frame, and this trajectory has a different length, filled gaps and invented
    ramp frames. Copying them across would line row i of a readout up against a
    pose that is not row i. `origin` is the honest replacement: it says which
    source frame each output frame came from, and -1 where the answer is "none".
    """

    import hashlib, json

    out_npz = Path(out_npz)
    if out_npz.resolve() == Path(loaded.path).resolve():
        raise ValueError(
            f"refusing to write over the input {loaded.path}. The convert output "
            "is the ledger's direct product and the root of every trace; a "
            "filtered curve on top of it is indistinguishable from it afterwards."
        )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = loaded.limits_rad[:, 0], loaded.limits_rad[:, 1]

    np.savez(
        out_npz,
        arm_pos=result.arm_pos,
        arm_pos_deg=np.degrees(result.arm_pos),
        time_s=result.time_s,
        origin=result.origin,
        limits_rad=loaded.limits_rad,
        joint_names=np.array(loaded.joint_names),
        clipped=(result.arm_pos <= lo) | (result.arm_pos >= hi),
        velocity_limits_rad_s=loaded.rated_rad_s,
        # Only when it is true, so an ordinary npz keeps exactly the keys it has
        # always had and a reader cannot be handed a mirrored file that looks
        # ordinary.
        **({"mirror": np.array(True)} if loaded.mirror else {}),
    )

    source = Path(loaded.path)
    sidecar = {
        "tool": "smplx2robot.postprocess",
        "source_npz": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        # Written from what ran, not from what the module can do. A sidecar
        # that lists `smooth` for a file that was never smoothed is worse than
        # no sidecar: it is a provenance record that lies.
        "chain": (["resample_uniform", "despike"]
                  + (["smooth"] if result.accounting["parameters"].get("cutoff_hz")
                     else [])
                  + ["repair_limits", "retime", "add_ramps"]),
        "joint_names": loaded.joint_names,
        **({"mirror": True} if loaded.mirror else {}),
        **result.accounting,
        "NOT A HARDWARE CLEARANCE": (
            "Positions and joint speed only, and the speed budget is whatever "
            "parameters.velocity_fraction says -- a number chosen offline, not "
            "measured on a robot. result.still_needs_slowdown must read 1.0. "
            "Nothing here checks collision, comms timeout or e-stop, and "
            "nothing here has been near an X2. First run on the real robot: "
            "low speed, unloaded, hand on the e-stop."
        ),
    }
    out_json = out_npz.with_suffix(".json")
    out_json.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    return out_npz, out_json


def report(loaded: Loaded, base_pose_rad: np.ndarray = BASE_POSE_RAD,
           velocity_fraction: float = DEFAULT_VELOCITY_FRACTION,
           acceleration_ceiling_deg_s2: float | None = None,
           step_threshold_deg: float = 15.0) -> list[str]:
    """Measure a trajectory and say what is wrong with it. Changes nothing.

    Every line is a number with its threshold beside it, in the same spirit as
    `safety.py`: the point is to show the margin, not to print a tick.
    """

    if acceleration_ceiling_deg_s2 is None:
        acceleration_ceiling_deg_s2 = velocity_fraction * RATED_ACCELERATION_DEG_S2
    t, x = loaded.time_s, loaded.arm_pos
    rated_deg = np.degrees(loaded.rated_rad_s)
    budget_deg = velocity_fraction * rated_deg
    v, a = derivatives(t, x)
    v_deg, a_deg = np.degrees(np.abs(v)), np.degrees(np.abs(a))
    x_deg = np.degrees(x)
    lo_deg, hi_deg = np.degrees(loaded.limits_rad[:, 0]), np.degrees(loaded.limits_rad[:, 1])

    out: list[str] = []
    add = out.append
    add(f"M0 PRE-FLIGHT  {loaded.path.name}")
    add("measures only -- nothing here changes, fixes or clears anything")
    add("")

    tm = timing(t)
    add(f"timing     {tm['count']} frames, {t[-1] - t[0]:.2f} s, "
        f"dt median {tm['dt_median'] * 1000:.2f} ms "
        f"({tm['fps_nominal']:.1f} fps nominal)")
    if tm["gaps"]:
        shown = ", ".join(f"after frame {i} ({d * 1000:.2f} ms)" for i, d in tm["gaps"][:6])
        add(f"           {len(tm['gaps'])} gap(s): {shown}"
            + ("" if len(tm["gaps"]) <= 6 else " ..."))
        add("           a filter assumes even spacing -- resample before filtering")
    else:
        add("           evenly spaced, no gaps")
    add(f"rated      {loaded.rated_source}")
    add("")

    add(f"{'joint':28s} {'peak vel':>10s} {'% rated':>8s} {'max step':>9s} "
        f"{'peak acc':>10s} {'clipped':>8s} {'margin':>8s}")
    worst_ratio, worst_j = 0.0, 0
    for j, name in enumerate(loaded.joint_names):
        ratio = v_deg[:, j].max() / rated_deg[j] * 100 if rated_deg[j] else float("nan")
        if ratio > worst_ratio:
            worst_ratio, worst_j = ratio, j
        step = np.abs(np.diff(x_deg[:, j])).max() if x_deg.shape[0] > 1 else 0.0
        margin = min(x_deg[:, j].min() - lo_deg[j], hi_deg[j] - x_deg[:, j].max())
        add(f"{name:28s} {v_deg[:, j].max():9.1f}° {ratio:7.0f}% {step:8.1f}° "
            f"{a_deg[:, j].max():9.0f}° {int(loaded.clipped[:, j].sum()):8d} "
            f"{margin:7.2f}°")
    add("")
    add(f"velocity   worst {loaded.joint_names[worst_j]} at {worst_ratio:.0f}% of rated; "
        f"budget is {velocity_fraction * 100:.0f}%")
    need = worst_ratio / (velocity_fraction * 100)
    add(f"           a uniform slowdown of {need:.2f}x would fit the budget "
        f"({(t[-1] - t[0]) * need:.1f} s instead of {t[-1] - t[0]:.1f} s)")
    acc_ratio = a_deg.max() / acceleration_ceiling_deg_s2
    add(f"accel      peak {a_deg.max():.0f} deg/s^2 against a {acceleration_ceiling_deg_s2:.0f} "
        f"ceiling ({acc_ratio * 100:.0f}%); slowdown divides it by {need:.2f}^2")
    add(f"           the ceiling is {velocity_fraction * 100:.0f}% of "
        f"{RATED_ACCELERATION_DEG_S2:.0f} deg/s^2, the model's own torque limit -- "
        f"same margin as the speed budget")
    add("")

    spikes, steps = classify_jumps(x, np.radians(step_threshold_deg))
    add(f"jumps      {len(spikes) + len(steps)} single-frame changes over "
        f"{step_threshold_deg:.0f}°: {len(spikes)} spike, {len(steps)} step")
    add("           spike = out and back within 6 frames (a fit that lost the limb);")
    add("           step  = out and stays (real fast motion, or a fit that changed its mind)")
    for label, items in (("spike", spikes), ("step", steps)):
        by_joint: dict[int, list[int]] = {}
        for frame, joint, _ in items:
            by_joint.setdefault(joint, []).append(frame)
        for joint, frames in sorted(by_joint.items()):
            shown = ", ".join(str(f) for f in frames[:14])
            add(f"           {label:5s} {loaded.joint_names[joint]:28s} "
                f"{len(frames):3d}: {shown}" + ("" if len(frames) <= 14 else " ..."))
    add("")

    flags = hampel_flags(x)
    add(f"hampel     {int(flags.sum())} sample(s) flagged "
        f"(half-window 5, threshold 3, MAD floor 0.05°)")
    add("           CANDIDATES ONLY. Real fast motion gets flagged too; the")
    add("           intersection rule in plan 4.1 is what keeps it. Nothing is removed.")
    for joint in np.where(flags.any(axis=0))[0]:
        frames = np.where(flags[:, joint])[0]
        shown = ", ".join(str(int(f)) for f in frames[:14])
        add(f"           {loaded.joint_names[joint]:28s} {frames.size:3d}: {shown}"
            + ("" if frames.size <= 14 else " ..."))
    add("")

    amax_rad = np.radians(acceleration_ceiling_deg_s2)
    vmax_rad = np.radians(budget_deg)
    add("ramps      the player publishes frame 0 the instant it starts and holds the")
    add("           last frame forever, so both ends have to be built, not inherited")
    for label, delta in (("lead-in ", x[0] - base_pose_rad),
                         ("lead-out", x[-1] - base_pose_rad)):
        T, j, limiter = quintic_ramp_seconds(delta, vmax_rad, amax_rad)
        far = int(np.argmax(np.abs(delta)))
        add(f"           {label} {np.degrees(np.abs(delta)).max():6.1f}° at "
            f"{loaded.joint_names[far]:28s} -> T = {T:.2f} s "
            f"(set by {loaded.joint_names[j]}, {limiter})")
    add("")
    add("NOT A HARDWARE CLEARANCE. This is a measurement of a file.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m smplx2robot.postprocess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Turn a converted clip into a playable one: drop the outliers, "
            "optionally smooth, fit the speed budget, and build both ends back "
            "to the base pose."),
        epilog=(
            "Output goes next to the input file:\n"
            "  <clip>_post.npz + .json                always\n"
            "  <clip>_smoothed.npz + .json            with --smooth\n"
            "  <clip>_*.mp4                           with --visualize\n"
            "\n"
            "The .npz holds the joint angles; the .json beside it is the record "
            "of\nwhat was done to them. Both belong to the same output -- keep "
            "the pair.\n"
            "\n"
            "There is a second row of switches for unusual cases (--cutoff, "
            "--time-scale,\n--no-slowdown, --identity, --trim-head/--trim-tail, "
            "--max-acceleration,\n--base-pose, --output-dir). They still work; "
            "they are just not what you\nchoose day to day. See the README."),
    )
    parser.add_argument("npz", nargs="+", help="the clip: convert's *_arm_pos.npz")
    parser.add_argument("--smooth", action="store_true",
                        help="also write a smoothed version. You get two of "
                             "everything: the un-smoothed file stays, so the two "
                             "can be compared and the original is never lost.")
    parser.add_argument("--vel_fraction", "--velocity-fraction", dest="vel_fraction",
                        type=float, default=DEFAULT_VELOCITY_FRACTION,
                        help=f"speed budget as a fraction of the vendor's rated "
                             f"joint speeds (default {DEFAULT_VELOCITY_FRACTION}). "
                             "Any decimal: 0.9 is 90%%. The clip is stretched "
                             "until every joint fits inside it, so halving this "
                             "roughly doubles the slowdown. Only the clock "
                             "changes -- no joint angle moves.")
    parser.add_argument("--visualize", action="store_true",
                        help="render an mp4 of each file written, front over "
                             "side, at the speed it will actually play. Needs "
                             "mujoco, so it is off by default.")

    # Everything below is deliberately absent from --help. Not deprecated, not
    # unsupported -- just not a daily choice, and a help screen that lists
    # fourteen switches makes the four that matter harder to see.
    hidden = parser.add_argument_group("advanced")
    def _hidden(*names, **kw):
        hidden.add_argument(*names, help=argparse.SUPPRESS, **kw)
    _hidden("-o", "--output-dir", type=str, default=None)
    _hidden("--cutoff", type=float, default=DEFAULT_CUTOFF_HZ)
    _hidden("--dwell", type=float, default=0.5)
    _hidden("--trim-head", type=int, default=0)
    _hidden("--trim-tail", type=int, default=0)
    _hidden("--identity", action="store_true")
    _hidden("--time-scale", type=float, default=None)
    _hidden("--no-slowdown", action="store_true")
    _hidden("--max-acceleration", type=float, default=None)
    _hidden("--step-threshold", type=float, default=15.0)
    _hidden("--base-pose", type=str, default=None)
    args = parser.parse_args(argv)

    base = BASE_POSE_RAD
    if args.base_pose is not None:
        base = np.array([float(v) for v in args.base_pose.split(",")], dtype=float)
        if base.size != ARM_DOF:
            parser.error(f"--base-pose needs {ARM_DOF} values, got {base.size}")

    for i, path in enumerate(args.npz):
        if i:
            print("\n" + "=" * 78 + "\n")
        loaded = load(path)
        for line in report(loaded, base, args.vel_fraction,
                           args.max_acceleration, args.step_threshold):
            print(line)

        out_dir = Path(args.output_dir) if args.output_dir else Path(path).parent
        stem = Path(path).stem.replace("_arm_pos", "")

        # Un-smoothed first, always. `--smooth` adds a file, it does not replace
        # one: the operator asked for the smoothed animation to be a new json
        # that never overwrites the original, and the cheapest way to keep that
        # promise is to make the original one of the outputs.
        jobs = [("post", None)]
        if args.smooth:
            jobs.append(("smoothed", args.cutoff))

        for tag, cutoff in jobs:
            result = process(loaded, base, cutoff, args.vel_fraction,
                             args.max_acceleration, args.dwell,
                             args.trim_head, args.trim_tail, args.identity,
                             args.time_scale, not args.no_slowdown)
            npz, js = write(result, loaded, out_dir / f"{stem}_{tag}.npz")
            for line in _summarize(result, tag, npz, js, args):
                print(line)
            if args.visualize:
                from . import preview
                fps = 1.0 / float(np.median(np.diff(result.time_s)))
                mp4 = preview.render_video(result.arm_pos,
                                           out_dir / f"{stem}_{tag}.mp4", fps=fps)
                print(f"  video     {mp4}  ({fps:.1f} fps, real time)")
    return 0


def _summarize(result: Processed, tag: str, npz: Path, js: Path, args) -> list[str]:
    """The lines printed for one written file. Numbers beside their thresholds."""
    r = result.accounting["result"]
    out = ["", f"  {'identity' if args.identity else tag} -> {npz}",
           f"              {js}",
           f"  {r['frames']} frames, {r['duration_s']:.2f} s at "
           f"{result.accounting['resample']['rate_hz']:.1f} Hz"]
    if args.identity:
        out.append("  identity passthrough -- no filtering, no ramps")
        return out
    out.append(f"  first frame {r['first_frame_max_deg_from_base']:.6f} deg from base,"
               f"  last frame {r['last_frame_max_deg_from_base']:.6f} deg from base")
    out.append(f"  de-spiked {result.accounting['despike']['samples_flagged']} samples,"
               + (" no low-pass (run again with --smooth to add one)"
                  if result.accounting["parameters"]["cutoff_hz"] is None else
                  f" smoothing moved at most "
                  f"{result.accounting['smooth']['max_deviation_deg']:.1f} deg"))
    rt = result.accounting["retime"]
    out.append(f"  budget {args.vel_fraction * 100:.0f}% of rated"
               f"  ->  time x{rt['factor']:.2f} ({rt['mode']}), "
               f"{rt['input_duration_s']:.2f} s becomes {rt['output_duration_s']:.2f} s"
               + (f", spline moved the path {rt['spline_vs_linear_deg']:.2f} deg"
                  if rt['factor'] != 1.0 else ""))
    out.append(f"  peak acceleration {r['peak_acceleration_deg_s2']:.0f} of "
               f"{r['acceleration_ceiling_deg_s2']:.0f} deg/s^2 allowed = "
               f"{r['acceleration_fraction_of_ceiling'] * 100:.0f}%"
               + ("" if r["acceleration_within_ceiling"] else "  -- OVER, not enforced"))
    out.append(f"  tightest channel {r['worst_channel']}: "
               f"{r['worst_channel_peak_deg_s']:.1f} of "
               f"{r['worst_channel_budget_deg_s']:.1f} deg/s allowed = "
               f"{r['peak_velocity_fraction_of_budget'] * 100:.0f}%"
               f"   (fastest anywhere: {r['peak_velocity_deg_s']:.1f} deg/s)"
               + (f"  -- OVER BUDGET by {r['still_needs_slowdown']:.2f}x"
                  if r['still_needs_slowdown'] > 1.0 else "  -- inside budget"))
    # An un-smoothed clip is over the acceleration ceiling on all four of these
    # clips, and it is stretched MORE than the smoothed one, not less: the
    # jitter the low-pass would have removed is itself fast, so it eats the
    # speed budget and drags the whole clip out with it. Worth saying out loud
    # next to the numbers rather than leaving as an inference.
    if (result.accounting["parameters"]["cutoff_hz"] is None
            and not r["acceleration_within_ceiling"]):
        out.append("  NOTE      this file was not smoothed, and its acceleration is "
                   f"{r['acceleration_fraction_of_ceiling']:.1f}x the ceiling.")
        out.append("            the jitter is fast as well as ugly, so it also ate the "
                   "speed budget")
        out.append(f"            and stretched the clip to x{rt['factor']:.2f}. "
                   "--smooth is the file to play.")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
