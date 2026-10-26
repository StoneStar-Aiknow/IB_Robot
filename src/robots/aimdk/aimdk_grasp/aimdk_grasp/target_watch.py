"""Watch the planned target while the arm moves: has it clearly moved?

The pre-motion recheck (revalidation.require_stable_patch) compares the
target region of two images pixel by pixel. During execution the moving arm
and hand enter that region, so here pixels covered by the robot in either
image are left out. The robot cover is the collision-sphere model at each
image's own image-time feedback, projected into the (undistorted) RGB image.
The same per-pixel thresholds as the recheck apply to what stays visible.
Only the moving left arm and hand are projected: the torso, shoulder and
head spheres sit at or behind the camera plane and would cover the image.

While the arm moves the camera sways and its exposure changes; target_change
corrects both before comparing. A real move of the target (centimetres) is
far beyond what the correction absorbs.

A frame is only a vote. TargetWatch decides "moved" when at least
moved_frames of the last window_frames voted moved; too little visible target
votes "unknown", which neither counts as moved nor as still.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .revalidation import RevalidationConfig


@dataclass(frozen=True)
class WatchConfig:
    # Per-pixel change limits, as the pre-motion recheck.
    depth_change_m: float = RevalidationConfig.depth_change_m
    rgb_change: float = RevalidationConfig.rgb_change
    # Fractions of the visible target region, wider than the recheck's
    # (2 % / 5 %). 7 armed approach runs on a still bottle (2026-10-12, 195
    # compared frames, after target_change's corrections) reached 19 % (RGB)
    # and 6 % (depth); the same frames with the bottle pasted 2 cm (20 px)
    # away read at least 37 % and 25 %, 1 cm away 7-40 % and 6-27 %.
    changed_fraction: float = 0.3
    depth_changed_fraction: float = 0.15
    # Robot spheres are grown by this much before projection (feedback lag,
    # sphere cover error); pixels within it are never compared.
    robot_margin_m: float = 0.02
    # Fewer visible target pixels than this fraction of the planned ones: unknown.
    min_visible_fraction: float = 0.3
    # Largest whole-image shift taken out before comparing. It is estimated
    # on the target itself, so it needs no background (people may walk past).
    # The price: a target moved by less than about this reads still. 12 px is
    # ~1 cm at 0.5 m. Live 2026-10-13: a still bottle drifted 9 px in the image
    # during one approach (the standing body shifts as the arm extends); with
    # 12, a pasted move of 25 px (~2.5 cm) or more always reads moved, 15 px
    # almost never.
    shift_px: int = 12
    # With the chest IMU (camera_shift) the predicted shift is taken out first
    # and only this much is searched around it. 8 still approach runs on
    # 2026-10-13: the image drift (up to 22 px, mostly the body rotating ~1 deg
    # as the arm extends) minus the IMU prediction left at most 7 px. Replayed
    # with 6: worst still frame at 0.22 of the limits (1.07 without the IMU);
    # a pasted move of 16 px (~1.2 cm) always reads moved, 12 px in 20 of 49.
    imu_shift_px: int = 6
    # The camera's image shift measured on the background around the target
    # (background_shift): patches of this size on a grid of half that step,
    # searched this far around the IMU prediction (or no shift), kept at a
    # normalized correlation of at least background_min_score; the median of at
    # least background_min_patches is taken out like the IMU prediction, with
    # background_search_px left around it. The IMU sees the body's rotation,
    # not its translation: live 2026-10-15 p_stop st6, bottle at 0.475 m, the
    # image drifted 36 px as the arm reached, the IMU predicted 16 and a still
    # bottle read moved. Over 694 still frames (33 runs with an IMU, target
    # shift found by template) the background left at most 8 px (p99 5),
    # the IMU 17 (p99 9). None disables it.
    background_patch_px: int | None = 24
    background_search_px: int = 20
    background_min_score: float = 0.8
    background_min_patches: int = 4
    background_shift_px: int = 6
    moved_frames: int = 3
    window_frames: int = 4

    def __post_init__(self):
        if not (
            0 < self.moved_frames <= self.window_frames
            and self.shift_px >= 0
            and 0 < self.min_visible_fraction <= 1
            and self.robot_margin_m >= 0
        ):
            raise ValueError("Invalid target watch configuration")


@lru_cache(maxsize=1)
def moving_sphere_indices():
    """Rows of RobotSphereModel.spheres for the left arm below the shoulder and the left hand."""
    from .robot_spheres import RobotSphereModel

    moving = ("left_elbow_link", "left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link")
    links = RobotSphereModel().sphere_links
    # R_* are the left OmniHand's links (the hand model keeps its own names).
    return np.array([i for i, link in enumerate(links) if link in moving or link.startswith("R_")])


# Spheres reaching closer than this to the camera plane are bounded by a box.
NEAR_M = 0.05


def robot_pixel_mask(spheres_torso, torso_from_camera, intrinsics, shape, margin_m=0.0):
    """Pixels covered by any robot sphere (Nx4 torso-frame centre/radius) in a pinhole image."""
    height, width = shape
    covered = np.zeros(shape, dtype=bool)
    spheres = np.asarray(spheres_torso, dtype=float).reshape(-1, 4)
    spheres = spheres[spheres[:, 3] > 0]
    camera_from_torso = np.linalg.inv(np.asarray(torso_from_camera, dtype=float))
    centres = spheres[:, :3] @ camera_from_torso[:3, :3].T + camera_from_torso[:3, 3]
    radii = spheres[:, 3] + margin_m
    k = intrinsics
    for (x, y, z), radius in zip(centres, radii, strict=True):
        if z + radius <= NEAR_M:
            continue  # wholly behind the camera or at it
        if z - radius <= NEAR_M:
            # The sphere crosses the near plane, usually beside the camera (the
            # elbow passing the head), not in view. Bound its part in front by
            # the box x+-r, y+-r, z in [NEAR_M, z+r]: projected, a rectangle.
            depths = (NEAR_M, z + radius)
            us = [k.fx * (x + sx) / d + k.cx for sx in (-radius, radius) for d in depths]
            vs = [k.fy * (y + sy) / d + k.cy for sy in (-radius, radius) for d in depths]
            u0, u1 = max(int(np.floor(min(us))), 0), min(int(np.ceil(max(us))) + 1, width)
            v0, v1 = max(int(np.floor(min(vs))), 0), min(int(np.ceil(max(vs))) + 1, height)
            if u0 < u1 and v0 < v1:
                covered[v0:v1, u0:u1] = True
            continue
        # Exact horizontal and vertical tangent rays bound the projected
        # ellipse. Filling their axis-aligned box is conservative and avoids
        # the centre-shift/radial-stretch missed by a centre-projected disc.
        denominator = z * z - radius * radius
        u_root = radius * np.sqrt(max(x * x + z * z - radius * radius, 0.0))
        v_root = radius * np.sqrt(max(y * y + z * z - radius * radius, 0.0))
        us = k.fx * np.array([x * z - u_root, x * z + u_root]) / denominator + k.cx
        vs = k.fy * np.array([y * z - v_root, y * z + v_root]) / denominator + k.cy
        u0, u1 = max(int(np.floor(us.min())), 0), min(int(np.ceil(us.max())) + 1, width)
        v0, v1 = max(int(np.floor(vs.min())), 0), min(int(np.ceil(vs.max())) + 1, height)
        if u0 < u1 and v0 < v1:
            covered[v0:v1, u0:u1] = True
    return covered


def clear_of_target(spheres_torso, torso_from_camera, intrinsics, mask, margin_px, margin_m=0.0):
    """True when no projected robot sphere reaches the target mask's bounding box grown by margin_px."""
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(rows):
        return True
    height, width = np.shape(mask)
    top, bottom = max(rows.min() - margin_px, 0), min(rows.max() + 1 + margin_px, height)
    left, right = max(cols.min() - margin_px, 0), min(cols.max() + 1 + margin_px, width)
    robot = robot_pixel_mask(spheres_torso, torso_from_camera, intrinsics, (height, width), margin_m)
    return not robot[top:bottom, left:right].any()


def _roi(mask, iterations=3):
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=iterations)


def target_change(
    reference_rgb,
    reference_depth,
    current_rgb,
    current_depth,
    mask,
    reference_robot,
    current_robot,
    config=WatchConfig(),
    predicted_shift=None,
):
    """Change of the planned target region, robot pixels excluded.

    mask is the planned target mask (reference image). reference_robot and
    current_robot are robot_pixel_mask results for the two images. Returns a
    dict with the vote ("still", "moved" or "unknown"), the visible fraction
    and, unless unknown, the depth/RGB changed fractions and the shift used.

    Before comparing, the current image is corrected for what the moving arm
    does to a still scene (live 2026-10-12): the camera's exposure (each
    channel fitted as gain * current + offset over the visible region) and
    the camera's motion (a whole-pixel shift from phase correlation, at most
    shift_px, kept if it reads less changed; then every pixel may match
    any current pixel within 1 px). With predicted_shift (du, dv in pixels,
    from camera_shift) that shift is applied first and at most imu_shift_px
    is searched around it.
    """
    import cv2

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != reference_depth.shape or not mask.any():
        raise ValueError("A planned target mask of the image shape is required")
    base = (0, 0) if predicted_shift is None else tuple(int(round(float(value))) for value in predicted_shift)
    search = config.shift_px if predicted_shift is None else config.imu_shift_px
    s = max(abs(base[0]), abs(base[1])) + search + 1
    height, width = mask.shape
    rows, cols = np.nonzero(mask)
    # Region of interest (the mask grown by 3 px) inside a crop that leaves
    # room for the shift; the target is never at the image border.
    r0, r1 = max(rows.min() - 3, s), min(rows.max() + 4, height - s)
    c0, c1 = max(cols.min() - 3, s), min(cols.max() + 4, width - s)
    p0, q0 = max(r0 - 3, 0), max(c0 - 3, 0)
    roi = _roi(mask[p0 : r1 + 3, q0 : c1 + 3])[r0 - p0 : r1 - p0, c0 - q0 : c1 - q0]
    planned = int(roi.sum())
    compare = roi & ~reference_robot[r0:r1, c0:c1]
    a_rgb, a_depth = reference_rgb[r0:r1, c0:c1], reference_depth[r0:r1, c0:c1]
    a_valid = np.isfinite(a_depth) & (a_depth > 0)
    kernel = np.ones((3, 3), np.uint8)

    def at(dv, du):
        window = slice(r0 + dv, r1 + dv), slice(c0 + du, c1 + du)
        visible = compare & ~current_robot[window]
        count = int(visible.sum())
        if not count:
            return None
        b_depth = current_depth[window]
        valid = a_valid & np.isfinite(b_depth) & (b_depth > 0)
        depth = float((visible & valid & (np.abs(a_depth - b_depth) > config.depth_change_m)).sum()) / count
        b_rgb = current_rgb[window]
        x, y = b_rgb[visible].astype(np.float32), a_rgb[visible].astype(np.float32)
        gain = np.clip(((x - x.mean(0)) * (y - y.mean(0))).mean(0) / np.maximum(x.var(0), 1.0), 0.5, 2.0)
        offset = y.mean(0) - gain * x.mean(0)
        adjusted = cv2.transform(b_rgb, np.hstack([np.diag(gain), offset[:, None]]).astype(np.float32))
        low, high = cv2.erode(adjusted, kernel).astype(np.int16), cv2.dilate(adjusted, kernel).astype(np.int16)
        a = a_rgb.astype(np.int16)
        changed = np.any((a < low - config.rgb_change) | (a > high + config.rgb_change), axis=2)
        rgb = float((visible & changed).sum()) / count
        # Rank by how far over its limit each part is.
        score = max(depth / config.depth_changed_fraction, rgb / config.changed_fraction)
        return dict(
            score=score,
            depth_changed_fraction=depth,
            rgb_changed_fraction=rgb,
            shift_px=[du, dv],
            visible_fraction=count / planned,
            rgb_gain=gain.round(3).tolist(),
            rgb_offset=offset.round(1).tolist(),
        )

    if r1 <= r0 or c1 <= c0:
        return dict(vote="unknown", visible_fraction=0.0)  # the shift leaves no room in this image
    best = at(base[1], base[0])
    if best is not None and search:
        grey = [
            cv2.cvtColor(np.ascontiguousarray(image[window]), cv2.COLOR_RGB2GRAY).astype(np.float32)
            for image, window in (
                (reference_rgb, (slice(r0, r1), slice(c0, c1))),
                (current_rgb, (slice(r0 + base[1], r1 + base[1]), slice(c0 + base[0], c1 + base[0]))),
            )
        ]
        (du, dv), _ = cv2.phaseCorrelate(*grey)
        du, dv = (int(np.clip(round(value), -search, search)) for value in (du, dv))
        if du or dv:
            shifted = at(base[1] + dv, base[0] + du)
            if shifted is not None and shifted["score"] < best["score"]:
                best = shifted
    if best is None or best["visible_fraction"] < config.min_visible_fraction:
        return dict(vote="unknown", visible_fraction=0.0 if best is None else best["visible_fraction"])
    best["vote"] = "moved" if best["score"] > 1.0 else "still"
    return best


class OrientationBuffer:
    """Recent body orientations (an IMU's fused quaternion) by header stamp, for lookup at image times."""

    def __init__(self, seconds=30.0, max_gap_s=0.1):
        self.seconds, self.max_gap_s = seconds, max_gap_s
        self._rows = deque()
        self._lock = threading.Lock()

    def add(self, stamp_ns, quaternion_xyzw):
        quaternion = np.asarray(quaternion_xyzw, dtype=float)
        if (
            quaternion.shape != (4,)
            or not np.isfinite(quaternion).all()
            or abs(np.linalg.norm(quaternion) - 1.0) > 1e-3
        ):
            return
        with self._lock:
            if self._rows and stamp_ns <= self._rows[-1][0]:
                return
            self._rows.append((int(stamp_ns), quaternion))
            while self._rows[0][0] < stamp_ns - self.seconds * 1e9:
                self._rows.popleft()

    def at(self, stamp_ns):
        """Quaternion (xyzw) at stamp_ns, or None unless samples within max_gap_s lie on both sides."""
        from scipy.spatial.transform import Rotation, Slerp

        with self._lock:
            rows = list(self._rows)
        stamps = [row[0] for row in rows]
        index = int(np.searchsorted(stamps, stamp_ns))
        if not 0 < index < len(rows):
            return None
        (t0, q0), (t1, q1) = rows[index - 1], rows[index]
        if max(stamp_ns - t0, t1 - stamp_ns) > self.max_gap_s * 1e9:
            return None
        return Slerp([0.0, 1.0], Rotation.from_quat([q0, q1]))((stamp_ns - t0) / (t1 - t0)).as_quat()


def camera_shift(reference_quaternion, current_quaternion, torso_from_camera, intrinsics, pixel):
    """Predicted image shift (du, dv) of a still point at pixel (u, v) from the body's rotation.

    The quaternions (xyzw) are a torso-fixed IMU's orientation at the planned
    image and now; its axes are taken as torso_link's (nominal mounting). The
    head is still relative to the torso (the executor checks), so the camera
    turned by the same rotation. Translation of the camera is not measured:
    for a target at 0.5 m it was the smaller part (live 2026-10-13).
    """
    from scipy.spatial.transform import Rotation

    turned = Rotation.from_quat(reference_quaternion).inv() * Rotation.from_quat(current_quaternion)
    camera = np.asarray(torso_from_camera, dtype=float)[:3, :3]
    reference_from_current = camera.T @ turned.as_matrix() @ camera
    k = intrinsics
    ray = reference_from_current.T @ np.array([(pixel[0] - k.cx) / k.fx, (pixel[1] - k.cy) / k.fy, 1.0])
    return k.fx * ray[0] / ray[2] + k.cx - pixel[0], k.fy * ray[1] / ray[2] + k.cy - pixel[1]


# Background patches never come closer to the target mask than this.
BACKGROUND_CLEARANCE_PX = 10
# Their texture: mean absolute grey gradient along each image axis.
BACKGROUND_MIN_GRADIENT = 2.0


def background_patches(reference_grey, mask, reference_robot, config=WatchConfig()):
    """Top-left corners (row, col) of the textured background patches of an image (grey, mask, robot alike)."""
    import cv2

    size = config.background_patch_px
    near = (
        cv2.dilate(np.asarray(mask, dtype=np.uint8), np.ones((2 * BACKGROUND_CLEARANCE_PX + 1,) * 2, np.uint8))
        | reference_robot
    )
    # Per patch: covered pixels and mean absolute gradients, all at once (box sums).
    gy, gx = np.gradient(reference_grey)
    sums = [
        cv2.integral(np.ascontiguousarray(image, dtype=np.float64))
        for image in (near.astype(np.float64), np.abs(gx), np.abs(gy))
    ]
    height, width = reference_grey.shape
    rows = np.arange(0, height - size + 1, size // 2)[:, None]
    cols = np.arange(0, width - size + 1, size // 2)[None, :]
    covered, texture_x, texture_y = (
        (total[rows + size, cols + size] - total[rows, cols + size] - total[rows + size, cols] + total[rows, cols])
        for total in sums
    )
    keep = (covered == 0) & (np.minimum(texture_x, texture_y) >= BACKGROUND_MIN_GRADIENT * size * size)
    return [(int(rows[r, 0]), int(cols[0, c])) for r, c in zip(*np.nonzero(keep), strict=True)]


def background_shift(reference_grey, current_grey, corners, current_robot, base=(0, 0), config=WatchConfig()):
    """The current image's shift (du, dv) from the background patches' matches, or None.

    Each patch is searched within background_search_px of base. A match below
    background_min_score or onto robot pixels is dropped; the median of the
    rest is returned with their count when at least background_min_patches
    remain. People walking past move only some patches; the median ignores them.
    """
    import cv2

    size, radius = config.background_patch_px, config.background_search_px
    height, width = current_grey.shape
    du0, dv0 = (int(round(float(value))) for value in base)
    found = []
    for row, col in corners:
        a0, a1 = max(row + dv0 - radius, 0), min(row + dv0 + radius + size, height)
        b0, b1 = max(col + du0 - radius, 0), min(col + du0 + radius + size, width)
        if a1 - a0 < size or b1 - b0 < size:
            continue
        scores = cv2.matchTemplate(
            current_grey[a0:a1, b0:b1], reference_grey[row : row + size, col : col + size], cv2.TM_CCOEFF_NORMED
        )
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < config.background_min_score:
            continue
        if current_robot[a0 + y : a0 + y + size, b0 + x : b0 + x + size].mean() > 0.05:
            continue
        found.append((b0 + x - col, a0 + y - row))
    if len(found) < config.background_min_patches:
        return None, len(found)
    return tuple(float(value) for value in np.median(np.array(found), axis=0)), len(found)


class TargetWatch:
    """Votes of successive frames against one planned image; decides "moved".

    region (r0, r1, c0, c1) bounds the background patches (default: the whole
    image); the live watch passes its saved crop, so a replay of the crops
    finds the same patches.
    """

    def __init__(self, reference_rgb, reference_depth, mask, reference_robot, config=WatchConfig(), region=None):
        self.reference = (np.asarray(reference_rgb), np.asarray(reference_depth, dtype=np.float32))
        self.mask = np.asarray(mask, dtype=bool)
        self.reference_robot = np.asarray(reference_robot, dtype=bool)
        self.config = config
        self.votes = deque(maxlen=config.window_frames)
        self.moved = False
        self.corners = []
        if config.background_patch_px:
            # Patches inside region; frames are compared in it grown by the search (grey, that part only).
            height, width = self.mask.shape
            r0, r1, c0, c1 = region or (0, height, 0, width)
            grow = config.background_search_px + config.shift_px
            self.window = (
                slice(max(r0 - grow, 0), min(r1 + grow, height)),
                slice(max(c0 - grow, 0), min(c1 + grow, width)),
            )
            self.reference_grey = self._grey(self.reference[0])
            inside = np.zeros(self.reference_grey.shape, dtype=bool)
            inside[
                r0 - self.window[0].start : r1 - self.window[0].start,
                c0 - self.window[1].start : c1 - self.window[1].start,
            ] = True
            self.corners = background_patches(
                self.reference_grey, self.mask[self.window], self.reference_robot[self.window] | ~inside, config
            )

    def _grey(self, rgb):
        import cv2

        return cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb)[self.window]), cv2.COLOR_RGB2GRAY).astype(np.float32)

    def update(self, rgb, depth, current_robot, predicted_shift=None, require_shift=False):
        """Vote one frame. The background's shift, when measured, replaces predicted_shift (the IMU's).

        With require_shift a frame with neither votes "unknown".
        """
        shift, search, extra = predicted_shift, self.config.imu_shift_px, {}
        if self.corners:
            measured, count = background_shift(
                self.reference_grey,
                self._grey(rgb),
                self.corners,
                np.asarray(current_robot, dtype=bool)[self.window],
                (0, 0) if predicted_shift is None else predicted_shift,
                self.config,
            )
            extra = dict(background_patches=count, background_candidates=len(self.corners))
            if measured is not None:
                shift, search = measured, self.config.background_shift_px
                extra["background_shift_px"] = [round(value, 1) for value in measured]
        if shift is None and require_shift:
            result = dict(vote="unknown", unknown="no camera shift (background or IMU)")
        else:
            result = target_change(
                *self.reference,
                rgb,
                depth,
                self.mask,
                self.reference_robot,
                current_robot,
                self.config if shift is predicted_shift else replace(self.config, imu_shift_px=search),
                shift,
            )
        if predicted_shift is not None:
            result["predicted_shift_px"] = [round(float(value), 1) for value in predicted_shift]
        result.update(extra)
        self.votes.append(result["vote"])
        if sum(vote == "moved" for vote in self.votes) >= self.config.moved_frames:
            self.moved = True
        result["moved"] = self.moved
        return result


WATCHED_PHASES = ("approach", "grasp")


class ExecutionWatcher:
    """Capture, mask and vote on its own thread while SOC0 runs the arm segments.

    capture() returns a LiveState whose observation is newer than the last
    one; spheres(feedback) returns the torso-frame robot spheres at that
    feedback; phase() names the SOC0 phase now running ("" before the first
    one). Frames are taken only during WATCHED_PHASES; the first later phase
    (close, a retreat, ...) ends the watch. on_moved(record) is called once,
    when the votes decide "moved"; with on_moved=None nothing is ever sent.
    Crops of every frame around the target are kept for offline replay.

    orientation(stamp_ns) returns a torso-fixed IMU's quaternion (xyzw) at an
    image time, or None (OrientationBuffer.at). With it, each frame's
    predicted image shift from the body's rotation since the planned image
    (camera_shift) is taken out before comparing. A frame without an
    orientation is compared as without an IMU, unless require_orientation:
    then it votes "unknown" (kept for replay). Without the IMU the shift
    search (shift_px) is too small for the grasp segment: live 2026-10-13 the
    image drifted 16-21 px as the hand reached the bottle, and 3 of 6 still
    grasp runs recorded without an IMU read moved there.
    """

    def __init__(
        self,
        request,
        capture,
        spheres,
        phase,
        *,
        config=WatchConfig(),
        on_moved=None,
        period_s=0.0,
        keep_frames=False,
        crop_margin_px=40,
        clock=time.monotonic,
        orientation=None,
        require_orientation=False,
        source_clock=None,
    ):
        """``source_clock``: the clock observations are stamped on (``received_at``), when it is not ``clock``."""
        self.request = request
        self.capture, self.spheres, self.phase = capture, spheres, phase
        self.config, self.on_moved, self.period_s = config, on_moved, period_s
        self.clock = clock
        self.source_clock = clock if source_clock is None else source_clock
        mask = np.asarray(request.mask, dtype=bool)
        rows, cols = np.nonzero(mask)
        pad = crop_margin_px + config.shift_px
        height, width = mask.shape
        self.crop = (
            max(rows.min() - pad, 0),
            min(rows.max() + pad + 1, height),
            max(cols.min() - pad, 0),
            min(cols.max() + pad + 1, width),
        )
        self.moving = moving_sphere_indices()
        reference_spheres = spheres(request.image_feedback or request.feedback)[self.moving]
        reference_robot = robot_pixel_mask(
            reference_spheres, request.transform.base_from_camera, request.intrinsics, mask.shape, config.robot_margin_m
        )
        self.watch = TargetWatch(request.rgb, request.depth, mask, reference_robot, config, self.crop)
        self.orientation = orientation
        self.require_orientation = require_orientation
        self.target_pixel = (float(cols.mean()), float(rows.mean()))
        self.reference_quaternion = None if orientation is None else orientation(request.transform.stamp_ns)
        self.records, self.frames = [], [] if keep_frames else None
        if keep_frames:
            self.frames.append(
                self._frame(
                    request.rgb,
                    request.depth,
                    reference_spheres,
                    request.transform,
                    request.transform.stamp_ns,
                    self.reference_quaternion,
                )
            )
        self.error = None
        self.notified = None
        self._last_stamp = request.transform.stamp_ns
        self._started = self.clock()
        self._stop = threading.Event()
        self._dispatch_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="target_watch")

    def _frame(self, rgb, depth, spheres, transform, stamp_ns, quaternion=None):
        r0, r1, c0, c1 = self.crop
        return dict(
            stamp_ns=int(stamp_ns),
            rgb=np.array(rgb[r0:r1, c0:c1]),
            depth=np.array(depth[r0:r1, c0:c1]),
            imu_quaternion=np.full(4, np.nan) if quaternion is None else np.asarray(quaternion, dtype=float),
            spheres=np.asarray(spheres, dtype=np.float32),
            torso_from_camera=np.array(transform.base_from_camera),
        )

    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout=2.0):
        self._stop.set()
        # A dispatch that already owns the lock began before stop(). A later
        # one sees _stop while holding the same lock and cannot call back.
        if self._dispatch_lock.acquire(timeout=max(0.0, timeout)):
            self._dispatch_lock.release()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self):
        try:
            seen_watched = False
            while not self._stop.is_set():
                phase = self.phase()
                if phase in WATCHED_PHASES:
                    seen_watched = True
                elif seen_watched:
                    break  # close or later: the hand is meant to touch the target now
                else:
                    self._stop.wait(0.02)  # before the first segment starts
                    continue
                began = self.clock()
                self._notify_if_moved(self._observe(phase))
                self._stop.wait(max(0.0, began + self.period_s - self.clock()))
        except Exception as exc:  # noqa: BLE001 - a failed watch must never touch the execution
            self.error = f"{type(exc).__name__}: {exc}"

    def _observe(self, phase):
        began = self.clock()
        observation = self.capture().observation
        stamp = observation.transform.stamp_ns
        record = dict(
            t_s=began - self._started,
            phase=phase,
            stamp_ns=int(stamp),
            age_s=self.source_clock() - observation.received_at,
        )
        if stamp <= self._last_stamp:
            record["skipped"] = "not newer"
        elif not observation.image_feedback.healthy:
            record["skipped"] = "image-time feedback unhealthy"
        else:
            self._last_stamp = stamp
            spheres = self.spheres(observation.image_feedback)[self.moving]
            frame = observation.frame
            robot = robot_pixel_mask(
                spheres,
                observation.transform.base_from_camera,
                frame.intrinsics,
                frame.depth.shape,
                self.config.robot_margin_m,
            )
            quaternion = None if self.reference_quaternion is None else self.orientation(stamp)
            shift = (
                None
                if quaternion is None
                else camera_shift(
                    self.reference_quaternion,
                    quaternion,
                    self.request.transform.base_from_camera,
                    frame.intrinsics,
                    self.target_pixel,
                )
            )
            record.update(self.watch.update(frame.rgb, frame.depth, robot, shift, self.require_orientation))
            if self.frames is not None:
                self.frames.append(
                    self._frame(frame.rgb, frame.depth, spheres, observation.transform, stamp, quaternion)
                )
        record["wall_s"] = self.clock() - began
        self.records.append(record)
        return record

    def _notify_if_moved(self, record):
        """Dispatch once only while execution is still in a watched phase."""
        if self.watch.moved and self.notified is None and "skipped" not in record:
            with self._dispatch_lock:
                if self._stop.is_set() or self.phase() not in WATCHED_PHASES:
                    return
                self.notified = record
                if self.on_moved is not None:
                    self.on_moved(record)

    def report(self):
        votes = [record.get("vote") for record in self.records]
        return dict(
            config=dict(vars(self.config)),
            crop=list(map(int, self.crop)),
            error=self.error,
            imu=self.reference_quaternion is not None,
            moved=self.watch.moved,
            moved_at=self.notified,
            frames=len(self.records),
            votes={vote: votes.count(vote) for vote in ("still", "moved", "unknown")},
            skipped=sum("skipped" in record for record in self.records),
            records=self.records,
        )
