"""Registered depth to timestamped torso-frame collision geometry.

Only observed surfaces are reconstructed. Camera calibration, synchronized robot
feedback and collision-sphere coverage remain caller responsibilities.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from aimdk_grasp.frames import ImageTransform, rigid
from aimdk_grasp.joints import check_side
from aimdk_grasp.leg_obstacle import leg_obstacle
from aimdk_grasp.rgbd_to_cloud import mask_depth_to_points
from aimdk_grasp.types import CameraIntrinsics


def left_arm_reach():
    """Cached torso-frame (center, radius) of the left arm/hand collision spheres."""
    from aimdk_grasp.robot_spheres import left_reach_bounds

    center, radius, _ = left_reach_bounds()
    return center, radius


def _config_directory(side: str):
    from aimdk_grasp.robot_model import config_directory

    return config_directory(side)


def arm_reach(side: str = "left"):
    """Cached torso-frame (center, radius) of ``side``'s arm/hand collision spheres."""
    if side == "left":
        return left_arm_reach()
    from aimdk_grasp.robot_spheres import reach_bounds

    center, radius, _ = reach_bounds(side)
    return center, radius


@dataclass(frozen=True)
class SceneEstimationConfig:
    min_depth_m: float = 0.05
    max_depth_m: float = 4.0
    support_max_depth_m: float = 1.0
    plane_voxel_m: float = 0.02
    plane_threshold_m: float = 0.01
    plane_iterations: int = 384
    max_planes: int = 5
    min_plane_points: int = 60
    max_tilt_deg: float = 25.0
    max_support_gap_m: float = 0.15
    table_thickness_m: float = 0.04
    table_margin_m: float = 0.02
    # Coarse environment for optimization; target and final checks stay fine.
    mesh_pitch_m: float = 0.03
    self_filter_margin_m: float = 0.052
    target_mesh_pitch_m: float = 0.005
    validation_mesh_pitch_m: float = 0.005
    validation_self_filter_margin_m: float = 0.01
    # The segmentation mask stops a few pixels inside the target outline: on
    # the 2026-09-30 bottle captures 74-90% of the pixels 2-6 px outside it lay
    # within 1 cm of the adjacent bottle depth (real surface, not edge noise).
    # Left in the environment they stand as an obstacle hugging the target
    # (800-990 fine-mesh vertices within 1 cm of the bottle), where side-grasp
    # fingertips have to pass. After the support fit, environment points more
    # than target_edge_min_height_m above it join the target when they lie
    # within target_edge_radius_m of it, repeated target_edge_grow_steps times.
    # Growing in the image instead leaked onto the table at the bottle's base
    # and around stray mask islands.
    target_edge_grow_steps: int = 4
    target_edge_radius_m: float = 0.005
    target_edge_min_height_m: float = 0.01
    # The arm the scene is estimated for (its planning model's side): the
    # reach crop below uses that arm's bound.
    arm_side: str = "left"
    # Points the planning arm/hand cannot reach are left out of the environment
    # mesh; target points are always kept, and the near field the support fit
    # uses is still self-filtered and fitted (see estimate_scene). The radius
    # is a kinematic upper bound (RobotSphereModel.left_arm_reach,
    # 0.698 m for the official model); the margin covers calibration error and
    # sphere-cover gaps. Valid only while the robot does not walk and the
    # waist, right arm and head stay locked.
    crop_to_left_arm_reach: bool = True
    reach_margin_m: float = 0.10
    # The coarse planning mesh leaves points inside the table cuboid to the
    # cuboid (see estimate_scene); the fine validation mesh always keeps them.
    leave_table_to_cuboid: bool = True
    # The planning model has no legs. The fine validation scene gets the
    # pelvis and left leg at the standing pose as a fixed mesh (leg_obstacle):
    # live 2026-10-15 (move_A1) an approach swung the palm behind the torso
    # into the left hip. Not the coarse scene: the hanging arm at the start is
    # 5-10 mm from the hip, inside the optimizer's 2 cm activation, and every
    # trajectory paid for it (replay of 149 inputs: median planning 0.18 ->
    # 0.32 s, one input lost).
    leg_obstacle: bool = True
    # The coarse scene can carry the pelvis and left leg as oriented boxes
    # instead, each side pulled in by coarse_leg_shrink_m (leg_boxes): a box
    # has a distance inside it, a mesh query only near its surface, so the
    # optimizer pushes a path out of a box but went deeper through the mesh.
    # Live 2026-10-26 every approach to a target mid-table swung the wrist
    # through the pelvis (20-107 mm) and only the fine check refused it.
    # Replay of those 14 plan calls x2 on the 3090 (plan_replay): planned
    # 13 -> 21 of 28 (apple 5 -> 9 of 12, orange 2 -> 6 of 10), the cup 6 of 6
    # either way at 0.18 -> 0.20 s; shrink 2.5 / 1.5 / 1.0 cm planned 18 / 20 / 20.
    coarse_leg_boxes: bool = True
    coarse_leg_shrink_m: float = 0.02

    def __post_init__(self):
        check_side(self.arm_side)
        values = [value for value in vars(self).values() if not isinstance(value, bool | str)]
        if not np.isfinite(values).all() or any(v <= 0 for v in values):
            raise ValueError("Scene estimation parameters must be finite and positive")
        for name in ("plane_iterations", "max_planes", "min_plane_points", "target_edge_grow_steps"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if (
            not self.min_depth_m < self.support_max_depth_m <= self.max_depth_m
            or self.max_tilt_deg >= 90
            or self.min_plane_points < 3
        ):
            raise ValueError("Invalid depth, tilt or plane support limits")
        # A planning target coarser than validation is only checked against
        # the planning margin; validation always rebuilds it at its own pitch.
        target_margin = (
            self.validation_self_filter_margin_m
            if self.target_mesh_pitch_m <= self.validation_mesh_pitch_m
            else self.self_filter_margin_m
        )
        for pitch, margin in (
            (self.mesh_pitch_m, self.self_filter_margin_m),
            (self.target_mesh_pitch_m, target_margin),
            (self.validation_mesh_pitch_m, self.validation_self_filter_margin_m),
        ):
            if np.sqrt(3) * pitch > margin:
                raise ValueError("Voxel diagonal must not exceed its self-filter margin")
        if self.self_filter_margin_m < self.validation_self_filter_margin_m:
            raise ValueError("Planning self-filter margin must cover the validation margin")


LEG_OBSTACLE = "left_leg"


def leg_obstacle_name(side: str = "left") -> str:
    """The scene entry name of the planning arm's leg obstacle (``LEG_OBSTACLE`` for the left)."""
    return f"{check_side(side)}_leg"


def leg_boxes(shrink_m: float, side: str = "left") -> dict:
    """{name: cuboid}: one oriented box per pelvis/leg hull (its principal axes), each side pulled in by shrink_m.

    A box's corners can stand out of its hull, so these are for the coarse
    planning scene only; the fine check keeps the exact hulls (leg_obstacle).
    """
    from aimdk_grasp.robot_model import config_directory

    mesh = leg_obstacle(config_directory(side))
    vertices, faces = np.asarray(mesh["vertices"]), np.asarray(mesh["faces"]).reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    _, labels = connected_components(adjacency, directed=False)
    boxes = {}
    for label in np.unique(labels):
        points = vertices[labels == label]
        centre = points.mean(axis=0)
        axes = np.linalg.svd(points - centre, full_matrices=False)[2]
        if np.linalg.det(axes) < 0:
            axes[2] *= -1
        local = (points - centre) @ axes.T
        low, high = local.min(axis=0), local.max(axis=0)
        dims = high - low - 2 * shrink_m
        if (dims <= 0).any():
            continue  # thinner than twice the shrink: nothing left of it
        quat = Rotation.from_matrix(axes.T).as_quat()[[3, 0, 1, 2]]
        boxes[f"{leg_obstacle_name(side)}_{label}"] = {
            "dims": dims.tolist(),
            "pose": [*(centre + axes.T @ ((low + high) / 2)).tolist(), *quat.tolist()],
        }
    return boxes


class NoSupportError(RuntimeError):
    """No near-field support plane beneath the target: usually not the target at all
    (live 2026-10-14: a second "bottle" detection on the floor 0.66 m below the table)."""


@dataclass(frozen=True)
class SupportPlane:
    point: np.ndarray
    normal: np.ndarray
    inliers: np.ndarray
    target_heights_m: dict = field(default_factory=dict)

    def _box(self, config):
        basis = plane_basis(self.normal)
        local = (self.inliers - self.point) @ basis
        lower, upper = local.min(axis=0), local.max(axis=0)
        lower[:2] -= config.table_margin_m
        upper[:2] += config.table_margin_m
        lower[2] -= config.table_thickness_m
        upper[2] = max(upper[2], 0.0)
        return basis, lower, upper

    def contains(self, points, config):
        """Torso-frame points inside the table cuboid (already a planning obstacle)."""
        basis, lower, upper = self._box(config)
        local = (np.asarray(points) - self.point) @ basis
        return np.all((local >= lower) & (local <= upper), axis=1)

    def cuboid(self, config):
        basis, lower, upper = self._box(config)
        center = self.point + basis @ ((lower + upper) / 2)
        quat = Rotation.from_matrix(basis).as_quat()[[3, 0, 1, 2]]
        return {"dims": (upper - lower).tolist(), "pose": [*center.tolist(), *quat.tolist()]}


@dataclass(frozen=True)
class SceneEstimate:
    """One timestamp shared by coarse planning and fine validation geometry.

    Pass the estimate to LeftArmPlanner to use both resolutions. The target and
    support are shared; only the environment mesh and its self-filter differ.
    """

    scene: dict
    received_at: float
    stamp_ns: int
    camera_frame: str
    calibration_verified: bool
    support: SupportPlane
    diagnostics: dict
    validation_scene: Mapping
    frame_id: str = "torso_link"


class LazyScene(Mapping):
    """Materialize once, including concurrent first readers; retain failed builds.

    The factory owns filtered point arrays rather than caller RGB-D buffers.
    Receipt time lives on SceneEstimate and is never renewed by this operation.
    """

    def __init__(self, factory):
        self._factory = factory
        self._value = None
        self._error = None
        self._lock = RLock()

    @property
    def built(self):
        return self._value is not None

    def materialize(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Fine scene construction failed") from self._error
            if self._value is None:
                try:
                    assert self._factory is not None
                    self._value = self._factory()
                except BaseException as exc:
                    self._error = exc
                    raise
                finally:
                    self._factory = None
            return self._value

    def __getitem__(self, key):
        return self.materialize()[key]

    def __iter__(self):
        return iter(self.materialize())

    def __len__(self):
        return len(self.materialize())

    def __bool__(self):
        return True

    def __deepcopy__(self, memo):
        import copy

        return copy.deepcopy(self.materialize(), memo)


def _points(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("Expected finite Nx3 points")
    return value


def plane_basis(normal):
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(axis, normal)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(normal, u), normal))


def _sphere_membership_cpu(points, centers, radii, voxel_m=0.12):
    """Exact inside-any-sphere test; voxel buckets bound the per-sphere work."""
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / voxel_m).astype(np.int64)
    dims = keys.max(axis=0) + 1
    packed = keys @ np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    order = np.argsort(packed, kind="stable")
    sorted_keys, sorted_points = packed[order], points[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    voxel_keys = sorted_keys[starts]
    keep_sorted = np.ones(len(points), dtype=bool)
    for center, radius in zip(centers, radii, strict=True):
        low = np.maximum(np.floor((center - radius - origin) / voxel_m).astype(np.int64), 0)
        high = np.minimum(np.floor((center + radius - origin) / voxel_m).astype(np.int64), dims - 1)
        # Quantization is conservative: every voxel the closed ball can touch
        # is enumerated, and candidates get the exact distance test.
        for ix in range(int(low[0]), int(high[0]) + 1):
            for iy in range(int(low[1]), int(high[1]) + 1):
                base = ix * dims[1] * dims[2] + iy * dims[2]
                for iz in range(int(low[2]), int(high[2]) + 1):
                    at = np.searchsorted(voxel_keys, base + iz)
                    if at == len(voxel_keys) or voxel_keys[at] != base + iz:
                        continue
                    begin = starts[at]
                    end = starts[at + 1] if at + 1 < len(starts) else len(sorted_keys)
                    block = sorted_points[begin:end]
                    keep_sorted[begin:end] &= ((block - center) ** 2).sum(axis=1) > radius * radius
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    return keep_sorted[inverse]


def _sphere_membership_many(points, centers, radii):
    """One distance matrix for several radius sets, with the same float64 test."""
    try:
        import torch
    except ImportError:
        return np.stack([_sphere_membership_cpu(points, centers, value) for value in radii])
    if not torch.cuda.is_available():
        return np.stack([_sphere_membership_cpu(points, centers, value) for value in radii])
    device = torch.device("cuda")
    keep = np.ones((len(radii), len(points)), dtype=bool)
    # Broad phase in float32: group spheres in 10 cm cells, each cell bounded
    # by one ball containing its spheres at the largest radius plus 1 mm. A
    # point outside every bound is outside every sphere; float32 error at
    # metre scale (~1e-7 m) is far below the 1 mm slack, so the exact float64
    # test below decides every other point and the mask is unchanged. Orin's
    # float64 cdist of ~4.5e5 points x 795 spheres took 0.66 s with no
    # robot point in view (2026-10-09).
    reach = np.asarray(radii, dtype=np.float64).max(axis=0)
    _, cell = np.unique(np.floor(centers / 0.1).astype(np.int64), axis=0, return_inverse=True)
    cell = cell.reshape(-1)
    bound_centers = np.stack([centers[cell == k].mean(axis=0) for k in range(cell.max() + 1)])
    bounds = (
        np.array(
            [
                np.max(np.linalg.norm(centers[cell == k] - bound_centers[k], axis=1) + reach[cell == k])
                for k in range(len(bound_centers))
            ]
        )
        + 1e-3
    )
    pts32 = torch.as_tensor(points, dtype=torch.float32, device=device)
    near = (
        torch.cdist(pts32, torch.as_tensor(bound_centers, dtype=torch.float32, device=device))
        <= torch.as_tensor(bounds, dtype=torch.float32, device=device)
    ).any(dim=1)
    candidates = torch.nonzero(near).reshape(-1)
    if not len(candidates):
        return keep
    pts = torch.as_tensor(points, dtype=torch.float64, device=device)[candidates]
    ctr = torch.as_tensor(centers, dtype=torch.float64, device=device)
    rad = torch.as_tensor(radii, dtype=torch.float64, device=device)
    exact = torch.ones((len(radii), len(pts)), dtype=torch.bool, device=device)
    block = max(1, 2**24 // max(1, len(centers)))
    for start in range(0, len(pts), block):
        chunk = pts[start : start + block]
        distances = torch.cdist(chunk, ctr)
        for index in range(len(radii)):
            exact[index, start : start + block] = ~(distances <= rad[index]).any(dim=1)
    keep[:, candidates.cpu().numpy()] = exact.cpu().numpy()
    return keep


def _sphere_membership(points, centers, radii):
    """Exact sphere membership, float64 on GPU when available; CPU otherwise."""
    return _sphere_membership_many(points, centers, np.asarray(radii)[None])[0]


def _self_filter_masks(points_camera, spheres_torso, torso_from_camera, margins, *, cpu_only=False):
    points = _points(points_camera)
    spheres = np.asarray(spheres_torso, dtype=np.float64)
    if spheres.ndim != 2 or spheres.shape[1] != 4 or not np.isfinite(spheres).all():
        raise ValueError("Expected finite Nx4 torso-frame robot spheres")
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all() or (margins < 0).any():
        raise ValueError("Self-filter margin must be finite and nonnegative")
    transform = rigid(torso_from_camera)
    keep = np.ones((len(margins), len(points)), dtype=bool)
    active = spheres[spheres[:, 3] > 0]
    centers = (active[:, :3] - transform[:3, 3]) @ transform[:3, :3]
    if len(points) and len(active):
        radii = active[None, :, 3] + margins[:, None]
        keep = (
            np.stack([_sphere_membership_cpu(points, centers, value) for value in radii])
            if cpu_only
            else _sphere_membership_many(points, centers, radii)
        )
    return keep


def self_filter_mask(points_camera, spheres_torso, torso_from_camera, margin_m=0.01, *, cpu_only=False):
    """Keep points outside FK spheres, including objects on the same camera ray.

    cuRobo uses nonpositive radii for disabled spheres. These are ignored.
    Filtering is depth-aware sphere membership, not a silhouette erasure.
    Both backends compute exact float64 distances and return the same mask.
    cpu_only avoids CUDA selection and torch initialization in execution workers.
    """
    return _self_filter_masks(points_camera, spheres_torso, torso_from_camera, [margin_m], cpu_only=cpu_only)[0]


def select_support_plane(points, target_points, up, config=SceneEstimationConfig()):
    """Select a near-field support under the target, not the largest plane.

    Points and gravity/up must share a frame. With no target, choose the nearest
    sufficiently horizontal plane. Fit disconnected coplanar patches separately
    so unrelated surfaces cannot stretch a single solid support cuboid across
    the gap. The caller retains every measured point in the environment mesh
    and applies the camera depth ROI.
    """
    points, target = _points(points), _points(target_points)
    up = np.asarray(up, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("A finite nonzero up vector is required")
    up = up / np.linalg.norm(up)
    if len(points):
        cells = np.floor(points / config.plane_voxel_m).astype(np.int64)
        cells -= cells.min(axis=0)
        dims = cells.max(axis=0) + 1
        _, indices = np.unique(cells @ np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64), return_index=True)
        remaining = points[indices]
    else:
        remaining = points
    rng = np.random.default_rng(0)
    candidates = []
    rejected = []  # why each horizontal patch under consideration was not the support
    tilt_limit = np.cos(np.deg2rad(config.max_tilt_deg))
    for _ in range(config.max_planes):
        if len(remaining) < config.min_plane_points:
            break
        samples = remaining[rng.integers(len(remaining), size=(config.plane_iterations, 3))]
        normals = np.cross(samples[:, 1] - samples[:, 0], samples[:, 2] - samples[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        good = lengths > 1e-8
        normals[good] /= lengths[good, None]
        good &= np.abs(normals @ up) >= tilt_limit
        normals, origins = normals[good], samples[good, 0]
        if not len(normals):
            break
        # Bound temporary allocations even with high-resolution or large ROIs.
        best_count, best_mask = 0, None
        for offset in range(0, len(normals), 32):
            ns, ps = normals[offset : offset + 32], origins[offset : offset + 32]
            distances = np.abs(remaining @ ns.T - np.sum(ps * ns, axis=1))
            masks = distances <= config.plane_threshold_m
            counts = masks.sum(axis=0)
            best = int(counts.argmax())
            if counts[best] > best_count:
                best_count, best_mask = int(counts[best]), masks[:, best]
        if best_mask is None or best_count < config.min_plane_points:
            break
        inliers = remaining[best_mask]
        remaining = remaining[~best_mask]
        # Three downsample voxels tolerate local sampling gaps, including
        # diagonal neighbors, without joining distant coplanar objects. Check
        # support size, tilt and target coverage independently for each patch.
        pairs = cKDTree(inliers).query_pairs(3 * config.plane_voxel_m, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(inliers), len(inliers))
        )
        count, labels = connected_components(graph, directed=False)
        for component in range(count):
            patch = inliers[labels == component]
            if len(patch) < config.min_plane_points:
                continue
            point = patch.mean(axis=0)
            _, singular, vh = np.linalg.svd(patch - point, full_matrices=False)
            if singular[1] < config.plane_voxel_m:
                continue
            normal = vh[-1]
            normal *= 1 if normal @ up >= 0 else -1
            if normal @ up < tilt_limit:
                continue
            basis = plane_basis(normal)
            heights = {}
            if len(target):
                values = (target - point) @ normal
                quantiles = np.quantile(values, [0, 0.05, 0.5, 0.95, 1])
                heights = dict(zip(("min", "p05", "median", "p95", "max"), quantiles.tolist(), strict=True))
                summary = dict(
                    points=len(patch),
                    p05_m=round(heights["p05"], 3),
                    median_m=round(heights["median"], 3),
                    below_1cm=round(float(np.mean(values < -config.plane_threshold_m)), 3),
                )
                if heights["p05"] < -config.plane_threshold_m or not heights["median"] >= 0:
                    rejected.append(dict(summary, reason="target below the plane"))
                    continue
                if heights["p05"] > config.max_support_gap_m:
                    rejected.append(dict(summary, reason="target too far above the plane"))
                    continue
                # The target must project onto this observed support patch.
                footprint = (patch - point) @ basis[:, :2]
                hull = ConvexHull(footprint)
                center = (np.median(target, axis=0) - point) @ basis[:, :2]
                outside = float(np.max(hull.equations[:, :2] @ center + hull.equations[:, 2]))
                if outside > config.table_margin_m:
                    rejected.append(
                        dict(summary, reason="target centre outside the patch", outside_m=round(outside, 3))
                    )
                    continue
                rank = (max(0.0, heights["p05"]), -len(patch))
            else:
                rank = (float(np.linalg.norm(point)), -len(patch))
            candidates.append((rank, SupportPlane(point, normal, patch, heights)))
    if not candidates:
        raise NoSupportError(f"No near-field support plane beneath the target; patches: {rejected[:5]}")
    return min(candidates, key=lambda item: item[0])[1]


def _voxel_cells(pts, pitch):
    """Origin and occupied voxel indices; GPU-unique when available.

    Large clouds are heavily redundant per voxel; unique cells give the same
    occupied grid (and therefore the same mesh) with float64-identical
    floor mapping. Small clouds and CUDA-less hosts use the direct path.

    The origin lies on the global pitch lattice of the mesh frame, so a point's
    voxel does not depend on the rest of the cloud. Anchored at the cloud
    minimum, any change far away (or the reach crop) shifted every voxel by up
    to one pitch and changed the obstacle next to the target (r4, 2026-10-08).
    """
    if len(pts) >= 50_000:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            device = torch.device("cuda")
            cloud = torch.as_tensor(pts, dtype=torch.float64, device=device)
            origin = _lattice_origin(cloud.min(dim=0).values.cpu().numpy(), pitch)
            cells = torch.floor((cloud - torch.as_tensor(origin, dtype=torch.float64, device=device)) / pitch).to(
                torch.int64
            )
            bounds = (cells.max(dim=0).values + 1).tolist()
            keys = (
                torch.unique(cells[:, 0] * (bounds[1] * bounds[2]) + cells[:, 1] * bounds[2] + cells[:, 2])
                .cpu()
                .numpy()
            )
            unpacked = np.column_stack(
                (keys // (bounds[1] * bounds[2]), (keys // bounds[2]) % bounds[1], keys % bounds[2])
            )
            return origin, unpacked
    origin = _lattice_origin(pts.min(axis=0), pitch)
    return origin, np.floor((pts - origin) / pitch).astype(np.int64)


def _lattice_origin(minimum, pitch):
    return (np.floor(np.asarray(minimum, dtype=np.float64) / pitch) - 1.0) * pitch


_VOXEL_FACE_TEMPLATES = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
    ],
    dtype=np.float64,
)
_VOXEL_DIRECTIONS = ((0, 1, 1), (0, -1, 0), (1, 1, 3), (1, -1, 2), (2, 1, 5), (2, -1, 4))


def _voxel_boundary_gpu(origin, cells, pitch):
    """Same sorted boundary faces and float64 vertices, built in CUDA batches."""
    import torch

    dims = cells.max(axis=0) + 3
    strides = torch.tensor([dims[1] * dims[2], dims[2], 1], dtype=torch.int64, device="cuda")
    cells = torch.as_tensor(cells, dtype=torch.int64, device="cuda")
    keys = torch.unique((cells * strides).sum(dim=1), sorted=True)
    cells = torch.stack((keys // strides[0], (keys // strides[1]) % int(dims[1]), keys % int(dims[2])), dim=1)
    templates = torch.as_tensor(_VOXEL_FACE_TEMPLATES, dtype=torch.float64, device="cuda")
    origin = torch.as_tensor(origin, dtype=torch.float64, device="cuda")
    vertex_blocks, face_blocks, base = [], [], 0
    for axis, direction, template in _VOXEL_DIRECTIONS:
        neighbors = keys + direction * strides[axis]
        at = torch.searchsorted(keys, neighbors)
        present = (at < len(keys)) & (keys[at.clamp(max=len(keys) - 1)] == neighbors)
        coords = cells[~present]
        if not len(coords):
            continue
        quads = (coords[:, None, :] + templates[template][None]).reshape(-1, 3) * pitch + origin
        index = torch.arange(len(coords), dtype=torch.int64, device="cuda") * 4 + base
        vertex_blocks.append(quads)
        face_blocks.append(
            torch.stack((index, index + 1, index + 2, index, index + 2, index + 3), dim=1).reshape(-1, 3)
        )
        base += len(coords) * 4
    if not vertex_blocks:
        raise RuntimeError("Voxelization produced no boundary faces")
    return torch.cat(vertex_blocks).cpu().numpy(), torch.cat(face_blocks).cpu().numpy(), len(keys)


def _voxel_surface_data(points, pitch):
    """Voxel-grid surface mesh, bit-identical to cuRobo 0.8.0 Mesh.from_pointcloud.

    cuRobo serializes its result through Python lists (and its callers convert
    straight back to NumPy); this local copy keeps arrays end to end. cuRobo
    emits inward-facing voxel faces; the caller corrects the winding.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("Cannot voxelize an empty point cloud")
    origin, cells = _voxel_cells(pts, pitch)
    if len(cells) >= 50_000:
        try:
            import torch
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                return _voxel_boundary_gpu(origin, cells, pitch)
    # Sparse occupancy avoids scanning the entire bounding volume six times.
    # Packed-key order is the same C order as the old dense np.argwhere path,
    # preserving face/vertex order as well as geometry. Two empty padding cells
    # prevent +/-1 neighbor keys from wrapping into occupied adjacent rows.
    dims = cells.max(axis=0) + 3
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    keys = np.unique(cells @ strides)
    cells = np.column_stack((keys // strides[0], (keys // strides[1]) % dims[1], keys % dims[2]))
    vertex_blocks, face_blocks, base = [], [], 0
    for axis, direction, template in _VOXEL_DIRECTIONS:
        neighbors = keys + direction * strides[axis]
        at = np.searchsorted(keys, neighbors)
        present = (at < len(keys)) & (keys[np.minimum(at, len(keys) - 1)] == neighbors)
        coords = cells[~present]
        if not len(coords):
            continue
        quads = (coords[:, None, :] + _VOXEL_FACE_TEMPLATES[template][None]).reshape(-1, 3) * pitch + origin
        count = len(coords)
        index = np.arange(count) * 4 + base
        vertex_blocks.append(quads)
        face_blocks.append(np.column_stack((index, index + 1, index + 2, index, index + 2, index + 3)).reshape(-1, 3))
        base += count * 4
    if not vertex_blocks:
        raise RuntimeError("Voxelization produced no boundary faces")
    return np.concatenate(vertex_blocks), np.concatenate(face_blocks), len(keys)


def _voxel_surface(points, pitch):
    vertices, faces, _ = _voxel_surface_data(points, pitch)
    return vertices, faces


def _surface_mesh(cloud, pitch, pose):
    vertices, faces, occupied = _voxel_surface_data(cloud, pitch)
    # The six fixed templates enclose occupied grid cells with inward winding.
    # Their exact union volume is the cell count times pitch cubed, including
    # disconnected components and cavities. Avoid materializing every triangle
    # just to rediscover this known orientation and volume.
    faces = faces[:, ::-1]
    volume = occupied * pitch**3
    if not np.isfinite(volume) or volume <= 0:
        raise RuntimeError("Scene mesh must enclose positive occupied volume")
    return (
        {"vertices": vertices, "faces": faces.reshape(-1), "pose": pose},
        {
            "points": len(cloud),
            "vertices": len(vertices),
            "faces": len(faces),
            "pitch_m": pitch,
            "winding_corrected": True,
            "occupied_volume_m3": float(volume),
        },
    )


def grow_target_edge(points, target, plane, config):
    """Move environment points continuing the target surface into the target.

    Returns the grown target mask and the number of points moved. Only points
    above the support plane are eligible, so the table under the target stays
    environment (it is also the table cuboid).
    """
    grown = np.array(target, dtype=bool, copy=True)
    if not grown.any():
        return grown, 0
    reach = config.target_edge_grow_steps * config.target_edge_radius_m
    lower, upper = points[grown].min(0) - reach, points[grown].max(0) + reach
    candidates = np.flatnonzero(~grown & np.all((points >= lower) & (points <= upper), axis=1))
    candidates = candidates[(points[candidates] - plane.point) @ plane.normal > config.target_edge_min_height_m]
    moved = 0
    for _ in range(config.target_edge_grow_steps):
        if not len(candidates):
            break
        near = cKDTree(points[grown]).query(points[candidates], distance_upper_bound=config.target_edge_radius_m)[0]
        take = np.isfinite(near)
        if not take.any():
            break
        grown[candidates[take]] = True
        moved += int(take.sum())
        candidates = candidates[~take]
    return grown, moved


@dataclass
class PreparedScene:
    """Mask-independent scene stage: unprojected, reach-cropped, self-filtered.

    prepare_scene produces it without the target mask so it can run while
    detection and segmentation are still round-tripping to the model services;
    complete_scene finishes with the mask. estimate_scene remains the unchanged
    prepare + complete pair for callers without an overlap window.
    """

    points: np.ndarray
    pixel: np.ndarray
    near: np.ndarray
    reachable: np.ndarray
    keep: np.ndarray
    planning_all: np.ndarray
    dropped_points: np.ndarray
    dropped_pixel: np.ndarray
    depth_shape: tuple
    spheres: np.ndarray
    torso_from_camera: np.ndarray
    stamp_ns: int
    camera_frame: str
    received_at: float
    calibration_verified: bool
    config: SceneEstimationConfig
    depth_points: int
    unproject_s: float
    self_filter_s: float


def prepare_scene(
    depth,
    intrinsics: CameraIntrinsics,
    transform: ImageTransform,
    *,
    stamp_ns,
    camera_frame,
    received_at,
    robot_spheres_torso,
    depth_scale=1.0,
    config=SceneEstimationConfig(),
):
    """The mask-independent half of estimate_scene; see PreparedScene."""
    started = time.perf_counter()
    if (
        stamp_ns <= 0
        or stamp_ns != transform.stamp_ns
        or camera_frame != transform.camera_frame
        or transform.base_frame != "torso_link"
    ):
        raise ValueError("Exact image timestamp/frame and torso_link transform required")
    if not np.isfinite(received_at) or received_at < 0:
        raise ValueError("Finite nonnegative depth receipt time required")
    if (
        not np.isfinite([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, depth_scale]).all()
        or min(intrinsics.fx, intrinsics.fy, depth_scale) <= 0
    ):
        raise ValueError("Finite valid intrinsics and depth scale required")
    torso_from_camera = rigid(transform.base_from_camera)
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("Expected registered 2D depth")
    depth_m = depth.astype(np.float32) * depth_scale
    valid = np.isfinite(depth_m) & (depth_m >= config.min_depth_m) & (depth_m <= config.max_depth_m)
    points = mask_depth_to_points(
        valid, depth_m, intrinsics, min_depth_m=config.min_depth_m, max_depth_m=config.max_depth_m
    )
    pixel = np.flatnonzero(valid)
    depth_points = len(points)
    near = points[:, 2] <= config.support_max_depth_m
    reachable = np.ones(len(points), dtype=bool)
    base = np.ones(len(points), dtype=bool)
    if config.crop_to_left_arm_reach:
        center, radius = arm_reach(config.arm_side)
        center_camera = torso_from_camera[:3, :3].T @ (center - torso_from_camera[:3, 3])
        offset = points - center_camera.astype(np.float32)
        reachable = np.einsum("ij,ij->i", offset, offset) <= (radius + config.reach_margin_m) ** 2
        # The support fit and target-edge growth keep their full near-field input:
        # cropping the table changed the fitted plane and left target-edge points
        # as an obstacle on r4. Only the meshed environment is limited to reach.
        # Target pixels outside both sets survive in dropped_* and rejoin through
        # complete_scene, which estimate_scene admitted via `| target`.
        base = reachable | near
    unprojected = time.perf_counter()
    keep, planning_all = _self_filter_masks(
        points[base],
        robot_spheres_torso,
        torso_from_camera,
        [config.validation_self_filter_margin_m, config.self_filter_margin_m],
    )
    filtered = time.perf_counter()
    return PreparedScene(
        points=points[base],
        pixel=pixel[base],
        near=near[base],
        reachable=reachable[base],
        keep=keep,
        planning_all=planning_all,
        dropped_points=points[~base],
        dropped_pixel=pixel[~base],
        depth_shape=depth.shape,
        spheres=robot_spheres_torso,
        torso_from_camera=torso_from_camera,
        stamp_ns=stamp_ns,
        camera_frame=camera_frame,
        received_at=received_at,
        calibration_verified=transform.calibration_verified,
        config=config,
        depth_points=depth_points,
        unproject_s=unprojected - started,
        self_filter_s=filtered - unprojected,
    )


def complete_scene(prepared: PreparedScene, target_mask=None, *, lazy_validation=True):
    """The mask-dependent half of estimate_scene; see prepare_scene."""
    config = prepared.config
    started = time.perf_counter()
    if target_mask is None:
        mask = np.zeros(prepared.depth_shape, dtype=bool)
    else:
        mask = np.asarray(target_mask, dtype=bool)
        if mask.shape != prepared.depth_shape:
            raise ValueError("Target mask and registered depth must match")
    flat = mask.reshape(-1)
    target = flat[prepared.pixel]
    points, reachable, near = prepared.points, prepared.reachable, prepared.near
    keep, planning_all = prepared.keep, prepared.planning_all
    dropped = flat[prepared.dropped_pixel]
    if dropped.any():
        # Target pixels the mask-independent crop dropped rejoin here: self-filter
        # just those, then restore the original pixel order of estimate_scene.
        extra = prepared.dropped_points[dropped]
        extra_keep, extra_planning = _self_filter_masks(
            extra,
            prepared.spheres,
            prepared.torso_from_camera,
            [config.validation_self_filter_margin_m, config.self_filter_margin_m],
        )
        pixel = np.concatenate([prepared.pixel, prepared.dropped_pixel[dropped]])
        points = np.concatenate([points, extra])
        target = np.concatenate([target, np.ones(len(extra), dtype=bool)])
        reachable = np.concatenate([reachable, np.zeros(len(extra), dtype=bool)])
        near = np.concatenate([near, np.zeros(len(extra), dtype=bool)])
        keep = np.concatenate([keep, extra_keep])
        planning_all = np.concatenate([planning_all, extra_planning])
        order = np.argsort(pixel, kind="stable")
        points, target, reachable, near = points[order], target[order], reachable[order], near[order]
        keep, planning_all = keep[order], planning_all[order]
    input_count = len(points)
    points, target, reachable = points[keep], target[keep], reachable[keep]
    planning_all, near = planning_all[keep], near[keep]
    filtered = time.perf_counter()
    if target_mask is not None and target.sum() < 3:
        raise RuntimeError("Target mask has insufficient valid depth after self-filtering")
    if not len(points):
        raise RuntimeError("No scene points after depth and self filtering")
    near = near & ~target
    torso_from_camera = prepared.torso_from_camera
    up_camera = torso_from_camera[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    plane_camera = select_support_plane(points[near], points[target], up_camera, config)
    rotation, translation = torso_from_camera[:3, :3], torso_from_camera[:3, 3]
    support = SupportPlane(
        rotation @ plane_camera.point + translation,
        rotation @ plane_camera.normal,
        plane_camera.inliers @ rotation.T + translation,
        plane_camera.target_heights_m,
    )
    grown_points = 0
    if target_mask is not None:
        target, grown_points = grow_target_edge(points, target, plane_camera, config)
    environment = points[~target & reachable]
    # The table cuboid already covers its own surface. Voxelized at the coarse
    # pitch, a tilted table becomes a staircase whose teeth stood 4 cm above the
    # plane, 1 cm from the target (r4, 29 deg tilt), and blocked low side
    # approaches. Leave those points to the cuboid in the planning mesh; the fine
    # validation mesh still contains them.
    on_table = (
        support.contains(environment @ rotation.T + translation, config)
        if config.leave_table_to_cuboid
        else np.zeros(len(environment), dtype=bool)
    )
    self_keep = planning_all[~target & reachable]
    planning_keep = self_keep & ~on_table
    # A coarser planning target is filtered with the planning margin its voxels need.
    target_keep = (
        planning_all[target]
        if config.target_mesh_pitch_m > config.validation_mesh_pitch_m
        else np.ones(int(target.sum()), dtype=bool)
    )
    fitted = time.perf_counter()
    pose = [*translation.tolist(), *Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist()]
    meshes, mesh_counts, validation_meshes, validation_counts = {}, {}, {}, {}
    if config.leg_obstacle:
        validation_meshes[leg_obstacle_name(config.arm_side)] = leg_obstacle(_config_directory(config.arm_side))
    for name, cloud, planning in (("environment", environment, planning_keep), ("target", points[target], target_keep)):
        if not planning.any():
            continue
        pitch = config.target_mesh_pitch_m if name == "target" else config.mesh_pitch_m
        mesh, counts = _surface_mesh(cloud[planning], pitch, pose)
        if pitch == config.validation_mesh_pitch_m and planning.all():
            validation_meshes[name], validation_counts[name] = mesh, counts
        meshes[name], mesh_counts[name] = mesh, counts
    finished = time.perf_counter()
    diagnostics = {
        "depth_points": prepared.depth_points,
        "reach_cropped_points": prepared.depth_points - input_count,
        "unreachable_near_points": int((~reachable & ~target).sum()),
        "planning_table_points": int(on_table.sum()),
        "input_points": input_count,
        "self_filtered_points": int((~keep).sum() + (~self_keep).sum()),
        "validation_self_filtered_points": int((~keep).sum()),
        "planning_extra_filtered_points": int((~self_keep).sum() + (~target_keep).sum()),
        "active_robot_spheres": int((np.asarray(prepared.spheres)[:, 3] > 0).sum()),
        "support_inliers": len(support.inliers),
        "target_heights_m": support.target_heights_m,
        "target_edge_grown_points": grown_points,
        "mesh_pitch_m": config.mesh_pitch_m,
        "self_filter_margin_m": config.self_filter_margin_m,
        "target_mesh_pitch_m": config.target_mesh_pitch_m,
        "validation_mesh_pitch_m": config.validation_mesh_pitch_m,
        "validation_self_filter_margin_m": config.validation_self_filter_margin_m,
        "voxel_diagonal_m": float(np.sqrt(3) * config.mesh_pitch_m),
        "voxel_expansion_within_self_filter_margin": bool(
            np.sqrt(3) * config.mesh_pitch_m <= config.self_filter_margin_m
        ),
        "meshes": mesh_counts,
        "validation_meshes": validation_counts,
        "timings_s": {
            "unproject": prepared.unproject_s,
            "self_filter": prepared.self_filter_s + (filtered - started),
            "support_plane": fitted - filtered,
            "mesh": finished - fitted,
            "total": prepared.unproject_s + prepared.self_filter_s + (finished - started),
        },
    }
    table = {"table": support.cuboid(config)}
    scene = {
        "cuboid": {
            **table,
            **(leg_boxes(config.coarse_leg_shrink_m, config.arm_side) if config.coarse_leg_boxes else {}),
        }
    }
    scene["mesh"] = meshes
    # Both clouds are private advanced-indexing copies; freeze them before deferral.
    target_points = points[target]
    environment.setflags(write=False)
    target_points.setflags(write=False)

    def build_validation():
        before = time.perf_counter()
        for name, cloud in (("environment", environment), ("target", target_points)):
            if len(cloud) and name not in validation_meshes:
                mesh, counts = _surface_mesh(cloud, config.validation_mesh_pitch_m, pose)
                validation_meshes[name], validation_counts[name] = mesh, counts
        diagnostics["timings_s"]["validation_mesh"] = time.perf_counter() - before
        diagnostics["validation_materialized"] = True
        return {"cuboid": table, "mesh": validation_meshes}

    diagnostics["validation_materialized"] = False
    validation_scene = LazyScene(build_validation)
    if not lazy_validation:
        validation_scene = validation_scene.materialize()
        diagnostics["timings_s"]["total"] = (
            prepared.unproject_s + prepared.self_filter_s + (time.perf_counter() - started)
        )
    return SceneEstimate(
        scene,
        prepared.received_at,
        prepared.stamp_ns,
        prepared.camera_frame,
        prepared.calibration_verified,
        support,
        diagnostics,
        validation_scene,
    )


def estimate_scene(
    depth,
    intrinsics: CameraIntrinsics,
    transform: ImageTransform,
    *,
    stamp_ns,
    camera_frame,
    received_at,
    robot_spheres_torso,
    target_mask=None,
    depth_scale=1.0,
    config=SceneEstimationConfig(),
    lazy_validation=True,
):
    """Build coarse planning and fine validation scenes at one image timestamp.

    received_at is the depth receipt time in the planner's monotonic clock; it is
    preserved through reconstruction. An empty sphere array explicitly means
    no robot geometry is available, and is recorded in diagnostics.
    """
    return complete_scene(
        prepare_scene(
            depth,
            intrinsics,
            transform,
            stamp_ns=stamp_ns,
            camera_frame=camera_frame,
            received_at=received_at,
            robot_spheres_torso=robot_spheres_torso,
            depth_scale=depth_scale,
            config=config,
        ),
        target_mask,
        lazy_validation=lazy_validation,
    )
