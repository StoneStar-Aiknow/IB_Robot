"""CUDA-only planning with locked-joint snapshots and replaceable scene buffers."""

import copy
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock

import numpy as np

from .frames import rigid
from .hand import HandMapping, hand_links
from .joints import HEAD, LEFT_ARM, RIGHT_ARM
from .mesh_loading import use_gpu_validation_meshes
from .robot_model import collision_robot_config, planning_model
from .scene_estimation import LazyScene, SceneEstimate
from .trajectory import TimedTrajectory


@dataclass(frozen=True)
class GraspPlan:
    """The three arm-only segments of one offline grasp plan."""

    approach: TimedTrajectory
    grasp: TimedTrajectory
    lift: TimedTrajectory
    goalset_index: int = 0
    planning_method: str = "approach_first"
    approach_offset_m: float | None = None
    lift_method: str = "optimized"


def _mesh_sphere_cover(mesh, max_spheres):
    """A conservative fixed-size sphere cover of every triangle in a posed scene mesh."""
    from scipy.spatial.transform import Rotation

    if mesh is None:
        return None
    vertices = np.asarray(mesh.get("vertices"), dtype=np.float64)
    faces = np.asarray(mesh.get("faces"), dtype=np.int64).reshape(-1, 3)
    pose = np.asarray(mesh.get("pose"), dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or not len(faces)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or pose.shape != (7,)
        or not np.isfinite(pose).all()
        or not isinstance(max_spheres, int)
        or max_spheres < 1
    ):
        raise ValueError("Finite nonempty mesh and positive sphere capacity required")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-5):
        raise ValueError("Scene mesh pose quaternion must be unit length")

    centers = vertices[faces].mean(axis=1)
    clusters = [np.arange(len(faces))]
    while len(clusters) < max_spheres:
        spans = [np.ptp(centers[index], axis=0) if len(index) > 1 else np.zeros(3) for index in clusters]
        selected = int(np.argmax([np.linalg.norm(span) for span in spans]))
        axis = int(np.argmax(spans[selected]))
        if spans[selected][axis] <= 0:
            break
        index = clusters[selected]
        order = index[np.argsort(centers[index, axis], kind="stable")]
        middle = len(order) // 2
        if middle == 0:
            break
        clusters[selected : selected + 1] = [order[:middle], order[middle:]]

    spheres = []
    for index in clusters:
        points = vertices[np.unique(faces[index])]
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = np.nextafter(np.linalg.norm(points - center, axis=1).max(), np.inf)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Scene mesh triangles must span a positive volume")
        spheres.append([*center, radius])

    base_from_mesh = np.eye(4)
    base_from_mesh[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    base_from_mesh[:3, 3] = pose[:3]
    return rigid(base_from_mesh), np.asarray(spheres, dtype=np.float64)


def _carried_spheres(cover, base_from_palm_at_contact, base_from_palms):
    """Move a mesh-local sphere cover rigidly with the palm along a trajectory."""
    base_from_mesh, spheres = cover
    contact = rigid(base_from_palm_at_contact)
    palms = np.asarray(base_from_palms, dtype=np.float64)
    if palms.ndim != 3 or palms.shape[1:] != (4, 4) or not np.isfinite(palms).all():
        raise ValueError("Finite Nx4x4 palm poses required")
    base_centers = spheres[:, :3] @ base_from_mesh[:3, :3].T + base_from_mesh[:3, 3]
    palm_centers = (base_centers - contact[:3, 3]) @ contact[:3, :3]
    centers = np.einsum("mj,nkj->nmk", palm_centers, palms[:, :3, :3]) + palms[:, None, :3, 3]
    radii = np.broadcast_to(spheres[None, :, 3:4], (*centers.shape[:2], 1))
    return np.concatenate((centers, radii), axis=2)


def _carried_robot_clear(carried, robot, robot_mask, tolerance_m=1e-6):
    """Per-state clearance of carried spheres from non-contact robot spheres."""
    carried = np.asarray(carried, dtype=np.float64)
    robot = np.asarray(robot, dtype=np.float64)
    mask = np.asarray(robot_mask, dtype=bool)
    if carried.ndim != 3 or carried.shape[2] != 4 or robot.ndim != 3 or robot.shape[2] != 4:
        raise ValueError("Expected NxMx4 carried and robot sphere arrays")
    if len(carried) != len(robot) or mask.shape != (robot.shape[1],):
        raise ValueError("Carried and robot sphere batches must align")
    active = robot[:, mask]
    active = np.where(active[..., 3:4] > 0, active, np.array([0.0, 0.0, 0.0, -np.inf]))
    distance = np.linalg.norm(carried[:, :, None, :3] - active[:, None, :, :3], axis=3)
    required = carried[:, :, None, 3] + active[:, None, :, 3]
    return (distance + tolerance_m >= required).all(axis=(1, 2))


def _carried_support_clear(carried, support, tolerance_m=1e-6):
    """The payload sphere cover must never move deeper into its initial support contact."""
    point, normal = support
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    values = np.asarray(carried, dtype=np.float64)
    clearance = (values[..., :3] - np.asarray(point, dtype=np.float64)) @ normal - values[..., 3]
    return (clearance + tolerance_m >= clearance[:1]).all(axis=1)


def _palm_twist(target, poses):
    """Kx6 (translation, rotation vector) from Kx4x4 poses to Kx4x4 targets."""
    from scipy.spatial.transform import Rotation

    rotation = target[:, :3, :3] @ np.transpose(poses[:, :3, :3], (0, 2, 1))
    return np.concatenate((target[:, :3, 3] - poses[:, :3, 3], Rotation.from_matrix(rotation).as_rotvec()), axis=1)


class GraspPlanningError(RuntimeError):
    """Failed segment with an optional index into the caller's goalset."""

    def __init__(self, message, goalset_index=None):
        super().__init__(message)
        self.goalset_index = goalset_index


class PlannerFault(Exception):
    """The GPU planner itself failed (CUDA/Warp error), not a candidate.

    Not a RuntimeError: candidate loops that catch planning failures must not
    try the remaining candidates on a poisoned context. The owner closes the
    planner and rebuilds it for the next request.
    """


def device_fault(exc):
    """Whether a RuntimeError reports a CUDA/Warp failure instead of a planning result."""
    import torch

    kinds = tuple(
        kind
        for kind in (getattr(torch, "OutOfMemoryError", None), getattr(torch, "AcceleratorError", None))
        if kind is not None
    )
    text = str(exc)
    return isinstance(exc, kinds) or any(
        marker in text for marker in ("CUDA error", "CUDA out of memory", "cuda runtime error", "Warp CUDA error")
    )


def grasp_to_palm_transform(gripper: Mapping):
    """Return the GraspGenX gripper-base to mounted ``R_palm`` transform.

    GraspGenX poses describe its canonical gripper base. The gripper
    description's ``base_rotation`` (``hand.load_gripper_description``) is the
    measured/derived assembly relation from that base frame to the robot palm
    frame; keeping it here avoids silently treating a GraspGen pose as an
    R_palm pose.
    """
    return rigid(np.asarray(gripper["base_rotation"], dtype=float))


def palm_goal_from_camera_grasp(camera_from_grasp, base_from_camera, grasp_to_palm):
    """Convert a camera-frame GraspGenX pose into a torso-frame R_palm goal."""
    return palm_goals_from_camera_grasps(np.asarray(camera_from_grasp)[None], base_from_camera, grasp_to_palm)[0]


def palm_goals_from_camera_grasps(camera_from_grasps, base_from_camera, grasp_to_palm):
    """Kx4x4 camera-frame GraspGenX poses as torso-frame R_palm goals, in one product.

    Each pose gets rigid()'s checks, vectorized. ``grasp_to_palm`` is
    grasp_to_palm_transform's result. Per-pose conversion re-read the gripper
    asset every call: 400 live candidates took 0.59 s on the Orin (2026-10-09).
    """
    grasps = np.asarray(camera_from_grasps, dtype=np.float64)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or not np.isfinite(grasps).all():
        raise ValueError("Expected finite Kx4x4 transforms")
    rotation = grasps[:, :3, :3]
    gram = np.einsum("kji,kjl->kil", rotation, rotation)
    if (
        not np.allclose(gram, np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)
        or not np.allclose(grasps[:, 3], [0, 0, 0, 1])
    ):
        raise ValueError("Transform rotation is not proper or homogeneous row invalid")
    return rigid(base_from_camera) @ grasps @ rigid(grasp_to_palm)


@lru_cache(maxsize=8)
def urdf_joints(path):
    """(name, type, parent link, child link, is_mimic) of every URDF joint, cached by path."""
    return tuple(
        (
            j.get("name"),
            j.get("type"),
            j.find("parent").get("link"),
            j.find("child").get("link"),
            j.find("mimic") is not None,
        )
        for j in ET.parse(path).findall("joint")
    )


def robot_config(
    feedback,
    *,
    reference_hand_collisions=True,
    torso_merge_radius_m=None,
    torso_merge_protected_spheres=None,
    hand_merge_radius_m=None,
):
    config = collision_robot_config(
        reference_hand_collisions=reference_hand_collisions,
        torso_merge_radius_m=torso_merge_radius_m,
        torso_merge_protected_spheres=torso_merge_protected_spheres,
        hand_merge_radius_m=hand_merge_radius_m,
    )
    kin = config["robot_cfg"]["kinematics"]
    joints = urdf_joints(kin["urdf_path"])
    links = {kin["base_link"]}
    while True:
        descendants = {child for _, _, parent, child, _ in joints if parent in links}
        if descendants <= links:
            break
        links |= descendants
    movable = {
        name
        for name, kind, parent, _, mimic in joints
        if parent in links and kind not in ("fixed", "floating") and not mimic
    }
    locks = {n: v for n, v in kin["lock_joints"].items() if n in movable}
    locks.update(zip(RIGHT_ARM + HEAD, feedback.vector(RIGHT_ARM + HEAD), strict=True))
    locks.update(HandMapping(planning_model()).kinematic_alias(feedback.left_hand))
    if movable - set(locks) != set(LEFT_ARM):
        raise ValueError("Robot must expose exactly the seven left arm joints")
    kin["lock_joints"] = {n: float(v) for n, v in locks.items()}
    kin["cspace"]["default_joint_position"] = feedback.vector(LEFT_ARM).tolist()
    return config


def palm_goal_pose(transforms, *, batched=False):
    """CUDA R_palm goals from Kx4x4 torso-frame transforms.

    One goalset of K alternatives (1x1x1xK), or K independent problems
    (Kx1x1x1) when batched. Contiguous wxyz quaternions.
    """
    import torch
    from curobo.types import GoalToolPose
    from scipy.spatial.transform import Rotation

    quaternions = np.ascontiguousarray(Rotation.from_matrix(transforms[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]])
    shape = (-1, 1, 1, 1) if batched else (1, 1, 1, -1)
    return GoalToolPose(
        tool_frames=["R_palm"],
        position=torch.as_tensor(transforms[:, :3, 3].copy(), device="cuda", dtype=torch.float32).reshape(*shape, 3),
        quaternion=torch.as_tensor(quaternions, device="cuda", dtype=torch.float32).reshape(*shape, 4),
    )


def screen_robot_config(config, excluded_links):
    """The left-arm moving collision links of ``config`` minus ``excluded_links``.

    The contact screen disables the hand links anyway, and the static body
    cannot move toward a candidate, so a batched screen over 100 candidates
    needs only the few arm spheres. The kinematic tree is trimmed to those
    links and the R_palm tool frame; locks outside it are dropped.
    """
    config = copy.deepcopy(config)
    kin = config["robot_cfg"]["kinematics"]
    joints = {child: (name, parent) for name, _, parent, child, _ in urdf_joints(kin["urdf_path"])}
    shoulder = next(child for child, (name, _) in joints.items() if name == LEFT_ARM[0])
    moving, pending = set(), [shoulder]
    while pending:
        link = pending.pop()
        moving.add(link)
        pending += [child for child, (_, parent) in joints.items() if parent == link]
    kin["collision_link_names"] = [
        link for link in kin["collision_link_names"] if link in moving and link not in set(excluded_links)
    ]
    ancestors = set()
    for link in kin["collision_link_names"] + ["R_palm"]:
        while link in joints:
            ancestors.add(joints[link][0])
            link = joints[link][1]
    kin["lock_joints"] = {name: value for name, value in kin["lock_joints"].items() if name in ancestors}
    for key in ("mesh_link_names", "extra_links"):
        if isinstance(kin.get(key), list):
            kin[key] = [link for link in kin[key] if link in kin["collision_link_names"] or link == "R_palm"]
    return config


def configure_runtime(runtime):
    runtime.kernel_backend = "cuda_core"
    # Jetson NVRTC fails writing Warp's precompiled headers ("error while writing
    # PCH file: Bad address"); the Orin container sets AIMDK_WARP_PCH=0.
    if os.environ.get("AIMDK_WARP_PCH", "1") == "0":
        import warp

        warp.config.use_precompiled_headers = False
    retain_replaced_warp_modules()


# Warp module executables replaced by a reload, kept loaded (see below).
_retained_warp_modules = []


def retain_replaced_warp_modules():
    """Keep CUDA code that captured graphs may still launch loaded.

    cuRobo defines Warp kernel variants lazily in shared modules (e.g. the
    tool-pose cost specialises on num_goalset at its first forward pass). A
    new variant changes the module hash; Warp's next eager launch reloads the
    module and drops the old executable, whose __del__ unloads its CUDA code
    while CUDA graphs captured earlier still launch those kernels. Replay then
    faults (illegal memory access; a segfault under the profiler). Seen once
    the contact screen (num_goalset=1) ran after the motion graphs
    (num_goalset=100) were captured, 2026-10-09; reproduced with Warp and
    torch alone (tests/test_warp_module_retention.py). A replaced executable
    costs one loaded module; variants are few. Idempotent, process-wide.
    """
    from warp._src import context

    if getattr(context.Module.load, "aimdk_retains_replaced", False):
        return
    load = context.Module.load

    def retaining_load(self, *args, **options):
        previous = list(self.execs.values())
        result = load(self, *args, **options)
        current = list(self.execs.values())
        _retained_warp_modules.extend(e for e in previous if all(e is not c for c in current))
        return result

    retaining_load.aimdk_retains_replaced = True
    context.Module.load = retaining_load


def _ensure_mesh_query_distance(scene_collision, distance):
    """Adapt cuRobo 0.8.0's mesh SDF search bound, without changing geometry."""
    import torch

    meshes = scene_collision.data.meshes
    if meshes is None:
        return
    # A missed query returns half the mesh bounding-box diagonal as distance.
    # For small targets that can be less than a torso sphere radius even when
    # metres apart. Extend this search bound beyond radius + activation margin.
    dims = meshes.dims[..., :3]
    diagonal = torch.linalg.vector_norm(dims, dim=-1, keepdim=True)
    dims.mul_(torch.clamp(2 * distance / diagonal.clamp_min(1e-8), min=1.0))


def _set_mesh_query_distance(scene_collision, distance):
    """Set the mesh SDF search bound to exactly ``distance`` (may shrink it).

    Points farther than this from every surface, including points deep inside a
    closed mesh, read as free. Use only for optimizer guidance; validation must
    keep the full bound.
    """
    import torch

    meshes = scene_collision.data.meshes
    if meshes is None:
        return
    dims = meshes.dims[..., :3]
    diagonal = torch.linalg.vector_norm(dims, dim=-1, keepdim=True)
    dims.mul_(2 * distance / diagonal.clamp_min(1e-8))


def _skip_static_scene_spheres(scene_collision, static):
    """Leave spheres of links that never move out of scene-collision queries.

    With the torso locked its scene distance is the same at every waypoint and
    has no gradient in the arm joints, yet its spheres dominated the swept mesh
    SDF time (trajopt_batch_20260929). A negative radius is skipped by the
    kernels. Only the queries of this scene model change: self-collision keeps
    every sphere, and the independent fine checker still checks all links.
    ``static`` is a per-sphere bool mask; elementwise so CUDA graphs capture it.
    """
    import torch

    def masked(state):
        spheres = state.robot_spheres.clone()
        spheres[..., 3] = torch.where(static, -100.0, spheres[..., 3])
        state = copy.copy(state)
        state.robot_spheres = spheres
        return state

    for name in (
        "get_sphere_distance",
        "get_sphere_collision",
        "get_swept_sphere_distance",
        "get_swept_sphere_collision",
    ):
        query = getattr(scene_collision, name)
        setattr(scene_collision, name, lambda state, *a, _query=query, **k: _query(masked(state), *a, **k))


class LeftArmPlanner:
    """Plans and checks only left-arm motion; hardware transport is separate.

    Scene poses and palm goals are in torso_link. update_scene replaces both
    geometry resolutions in place; locked-joint changes need a new instance.
    Sampled collision checks are
    not a continuous collision guarantee or validation of physical geometry.
    The source hand/mount exclusions and mesh-cover torso model are the default.
    Torso-sphere merging is opt-in because its larger cover can block reachable goals.
    Physical coverage remains uncalibrated. ``gripper`` is the deployed grasp
    model's gripper description (hand.load_gripper_description): candidates are
    converted to palm goals through its base_rotation.
    planning_profile="latency" uses _trajopt_seeds parallel trajectory seeds, a
    25-iteration pass and a duration-bounded interpolation buffer; final
    collision criteria are shared. approach_seed_policy="clear_path" ranks
    existing IK seeds for approaches up to 5 cm in latency mode, with one ranked
    retry after a longer approach fails.
    "native" retains cuRobo's original seed ordering.
    """

    _ik_validity_cache_capacity = 4096
    # Fine checks query each sphere pose shared with the batch's first state
    # once (see _scene_cost_once_per_sphere_pose); False queries every sphere.
    _query_shared_spheres_once = True
    # Grasp segment (approach end -> contact, a short palm-axis line) as a
    # joint-space line to a local contact IK state, minimum-jerk timed within
    # the solver's velocity/acceleration/jerk limits, instead of TrajOpt. Used
    # only when every sample keeps the palm within these bounds of the
    # straight Cartesian line and passes the fine check; otherwise TrajOpt.
    _linear_grasp = True
    _linear_grasp_max_deviation_m = 0.005
    _linear_grasp_max_rotation_rad = np.deg2rad(3.0)
    _linear_grasp_waypoints = 6
    _linear_grasp_corrections = 3
    _linear_grasp_limit_fraction = 0.9
    # The offset lift is not planned (2026-09-30): it retreated along the
    # approach axis, which for side grasps dragged the object 6 cm across the
    # table and 0.5-0.9 cm down. select_plan reads this to drop strategies that
    # differ only in lift offset.
    _plan_lift = False
    # Instead the lift raises the palm straight up by vertical_lift_m along
    # lift_up (base frame; the planner service sets the support plane normal
    # per request), keeping its orientation, timed as the linear grasp
    # segment. Its robot fine check leaves out the target at its old world pose;
    # a conservative sphere cover moves with the palm and is checked against
    # the environment and non-hand robot links. The support is omitted only
    # from that payload query because the whole segment moves away along its
    # normal; the robot itself remains checked against the support.
    # When the line cannot be planned (IK, limits, deviation, fine check) the
    # lift is a hold at the grasp pose, which the executor refuses to run
    # until lift. 0 always holds.
    vertical_lift_m = 0.05
    lift_up = (0.0, 0.0, 1.0)
    # The fine target mesh is compressed into this many conservative spheres.
    # They are queried in one batch only for the short lift segment. More
    # spheres reduce false refusals without changing TrajOpt or IK cost.
    _carried_target_max_spheres = 32
    # Experimental latency knobs; defaults keep the measured production path.
    _short_approach_retry = False
    _approach_iters = 25
    # Latency-profile TrajOpt seeds per solve (one IK branch each). Orin replay
    # of 40 live inputs with the merged hand (2026-10-09): 2 seeds raised the
    # attempt success rate 38 -> 48 % and planning success 39 -> 40 of 40, cut
    # the slowest request 7.6 -> 3.3 s, and cost 0.17 s at the median.
    _trajopt_seeds = 2
    # Fine-check rescue: TrajOpt rejects any solution its coarse optimizer
    # model (3 cm environment mesh, merged hand, _optimizer_activation_m) finds too
    # close. A rejected result that reached the goal within the solver's
    # tolerances and keeps velocity/acceleration/jerk within the joint limits
    # goes to the same full-model fine check as an accepted one. On 100 live
    # inputs 22 of 53 rejected approach solutions passed that check (2026-10-09).
    _fine_rescue = True
    # Optimizer collision activation distance (m): IK/TrajOpt costs start this
    # far from obstacles; fine checks use 0. Most rejected approaches graze
    # with a fingertip (thumb distal mostly, 1-15 mm). 2 cm vs 1 cm, replays
    # of live inputs (2026-10-09): first-try 56 -> 64 of 100 (3090), 25 -> 28
    # of 40 and slowest 5.1 -> 3.2 s (Orin).
    _optimizer_activation_m = 0.02
    # No hand sphere centre behind the torso's frontal plane (torso x, m):
    # behind it are the hip, back and hoist, none of them in the model. Live
    # 2026-10-15 (move_A1) an approach swung the palm to x -0.10 and hit the
    # left hip; with the leg in the fine scene the replan still went round it
    # at x -0.09. The 103 recorded live paths stay at x >= 0.034 (the hanging
    # thumb). None disables.
    _hand_min_x_m = 0.0
    # Experimental: margin (m) beyond mesh_query_distance for the motion
    # model's mesh search bound. None keeps the full bounding-box bound.
    _motion_mesh_query_margin = None
    # Approach TrajOpt seeding: None keeps joint-space straight lines;
    # "escape" seeds the first of the parallel TrajOpt seeds from a fixed
    # up-and-outward escape (_escape_seeded_solve). Orin replay of 70 live
    # approaches (2026-10-10): first-try 62 -> 67, attempts 80 -> 74,
    # planning p90 1.11 -> 0.93 s.
    # "over" keeps the escape and leads its joint line into the approach pose
    # from _over_m above it (along lift_up, at the approach orientation)
    # instead of straight in. Live 2026-10-26 on targets 7-9 cm tall mid-table,
    # 80 of 82 refused approaches were TrajOpt's own failures: the line from
    # the escape swept the fingertips or the palm through the target on its
    # way up. Replay of those 15 calls x2: Orin 11 -> 24 of 30 planned (hard
    # calls 5 -> 18 of 24, their median 3.75 -> 1.60 s; the 18 cm cup 1.12 ->
    # 1.18 s), 3090 17 -> 28; 12 cm planned 22 and 28.
    _approach_graph_seed = "over"
    _over_m = 0.18
    _over_cartesian_steps = 5
    # Locked links whose spheres the optimizer's scene queries skip (see
    # _skip_static_scene_spheres); self-collision and fine checks keep them.
    _static_scene_links = ("torso_link",)
    # Contact screen: batched IK (one problem per candidate) over the moving
    # arm spheres only, without self-collision; every solution is fine-checked
    # with the complete model. On 10 recorded live inputs (3090) it found 76
    # of the 78 candidates per-candidate full IK finds, against 60 for the
    # goalset screen, in 10-55 ms instead of 0.1-0.35 s. None restores the
    # goalset screen over the full optimizer model.
    _screen_batch_size = 100
    _screen_ik_seeds = 16
    # IKSolverCfg.exit_early_batch_success_threshold of the screen IK. cuRobo
    # skips the L-BFGS stage when the LM seed stage's successful seeds number
    # at least this fraction of the batch (all seeds of all problems count).
    # At 1.0 a batch of mostly unreachable candidates (< 100 successful seeds)
    # ran L-BFGS: 4 of 13 live inputs of 2026-10-10, screen +0.17-0.24 s on
    # the Orin. 0 always returns the LM seeds; every state is still fine-
    # checked. Replay of 52 live inputs: same selected candidates, one screen
    # set one candidate larger, screen p90 0.30 -> 0.18 s (Orin).
    _screen_exit_early_fraction = 0.0
    # Optimizer hand model: merge each hand link's spheres up to this radius
    # (m) for the motion planner's IK/TrajOpt only (conservative: the merged
    # cover contains the original). Fine checks keep the original 446-sphere
    # hand. 20 mm leaves 71 hand spheres; Orin replay of 28 live inputs
    # (2026-10-09): TrajOpt 355 -> 158 ms per call, planning success 25 -> 27
    # of 28, median 2.40 -> 1.51 s, p90 7.53 -> 4.26 s. None keeps the
    # original cover in the optimizer too.
    _motion_hand_merge_radius_m = 0.02

    def __init__(
        self,
        feedback,
        scene,
        scene_received_at,
        *,
        gripper,
        max_scene_age_s=0.5,
        reference_hand_collisions=True,
        max_goalset=1,
        max_batch_size=1,
        validation_scene=None,
        torso_merge_radius_m=None,
        torso_merge_protected_spheres=None,
        planning_profile="default",
        approach_seed_policy="clear_path",
    ):
        import curobo.runtime as runtime
        import torch
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        carried_target_support = None
        if isinstance(scene, SceneEstimate):
            if validation_scene is not None or scene_received_at != scene.received_at:
                raise ValueError("SceneEstimate owns both geometry resolutions and their receipt time")
            carried_target_support = (np.array(scene.support.point), np.array(scene.support.normal))
            validation_scene, scene = scene.validation_scene, scene.scene
        if not scene or not np.isfinite(scene_received_at) or not np.isfinite(max_scene_age_s) or max_scene_age_s <= 0:
            raise ValueError("Timestamped nonempty torso-frame collision scene required")
        if validation_scene is None:
            validation_scene = scene
        if not validation_scene:
            raise ValueError("Nonempty validation scene required")
        if not isinstance(max_goalset, int) or max_goalset < 1:
            raise ValueError("max_goalset must be a positive integer")
        if not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if planning_profile not in ("default", "latency"):
            raise ValueError("Unknown planning profile")
        if approach_seed_policy not in ("native", "clear_path"):
            raise ValueError("Unknown approach seed policy")
        if planning_profile == "latency" and max_batch_size != 1:
            raise ValueError("The latency profile requires max_batch_size=1 (direct goalsets)")
        configure_runtime(runtime)
        self.right = feedback.vector(RIGHT_ARM)
        self.head = feedback.vector(HEAD)
        model = planning_model()
        self.hand_mapping = HandMapping(model)
        self.hand_links = list(hand_links(model))
        self.grasp_to_palm = grasp_to_palm_transform(gripper)
        self.hand = self.hand_mapping.check(feedback.left_hand).copy()
        self.scene_received_at = scene_received_at
        self.max_scene_age_s = max_scene_age_s
        self.reference_hand_collisions = reference_hand_collisions
        self.max_goalset = max_goalset
        self.max_batch_size = max_batch_size
        self.planning_profile = planning_profile
        self.approach_seed_policy = approach_seed_policy
        self._closed = False
        self._warmed = False
        self._checker_lock = RLock()
        self._checker_batch_size = None
        self._pending_validation_scene = validation_scene if isinstance(validation_scene, LazyScene) else None
        self._carried_target_support = carried_target_support
        self._carried_target_cover = (
            None
            if self._pending_validation_scene is not None
            else _mesh_sphere_cover(validation_scene.get("mesh", {}).get("target"), self._carried_target_max_spheres)
        )
        config = robot_config(feedback, reference_hand_collisions=reference_hand_collisions)
        kin = config["robot_cfg"]["kinematics"]
        self.collision_sphere_links = [
            name for name in kin["collision_link_names"] for _ in kin["collision_spheres"][name]
        ]
        hand = set(self.hand_links)
        self._carried_target_robot_mask = np.asarray(
            [name not in hand for name in self.collision_sphere_links], dtype=bool
        )
        self.locked_positions = config["robot_cfg"]["kinematics"]["lock_joints"].copy()
        activation_distance = self._optimizer_activation_m
        motion_robot = (
            robot_config(
                feedback,
                reference_hand_collisions=reference_hand_collisions,
                torso_merge_radius_m=torso_merge_radius_m,
                torso_merge_protected_spheres=torso_merge_protected_spheres,
                hand_merge_radius_m=self._motion_hand_merge_radius_m,
            )
            if torso_merge_radius_m is not None or self._motion_hand_merge_radius_m is not None
            else copy.deepcopy(config)
        )
        self.planning_sphere_count = sum(
            len(v) for v in motion_robot["robot_cfg"]["kinematics"]["collision_spheres"].values()
        )
        motion_config = MotionPlannerCfg.create(
            robot=motion_robot,
            scene_model=copy.deepcopy(scene),
            self_collision_check=True,
            optimizer_collision_activation_distance=activation_distance,
            max_goalset=max_goalset,
            max_batch_size=1,
            num_trajopt_seeds=self._trajopt_seeds if planning_profile == "latency" else 4,
        )
        if planning_profile == "latency":
            # Preallocate enough for the configured maximum spline duration,
            # rather than evaluating 5000 padded states on every solver pass.
            from curobo._src.types.control_space import ControlSpace

            trajectory = motion_config.trajopt_solver_config
            transition = trajectory.metrics_rollout_config.transition_model_cfg
            knots = ControlSpace.spline_total_knots(transition.control_space, transition.n_knots)
            steps = (
                int(
                    np.ceil(
                        trajectory.maximum_trajectory_dt * transition.interpolation_steps / trajectory.interpolation_dt
                    )
                )
                + 1
            )
            trajectory.interpolation_buffer_size = knots * steps + 2
            for optimizer in trajectory.optimizer_configs:
                optimizer.num_iters = 25
        # Only IK screens independent candidates. Padding TrajOpt to the screen
        # batch wastes B times the work and breaks cuRobo's single-plan results.
        motion_config.ik_solver_config.max_batch_size = max_batch_size
        self.motion = MotionPlanner(motion_config)
        kinematics = self.motion.kinematics.config.kinematics_config
        static = torch.zeros_like(kinematics.link_sphere_idx_map, dtype=torch.bool)
        for link in self._static_scene_links:
            static |= kinematics.link_sphere_idx_map == kinematics.link_name_to_idx_map[link]
        self.static_scene_sphere_count = int(static.sum())
        if self.static_scene_sphere_count:
            _skip_static_scene_spheres(self.motion.scene_collision_checker, static)
        self._grasp_selections = {}
        self._ik_validity_cache = {}
        try:
            self.checker = RobotCollisionChecker(
                RobotCollisionCheckerCfg.load_from_config(
                    robot_config=config,
                    scene_model=copy.deepcopy(
                        scene if self._pending_validation_scene is not None else validation_scene
                    ),
                    collision_activation_distance=0.0,
                )
            )
            use_gpu_validation_meshes(self.checker.scene_model)
            spheres = self.checker.kinematics.config.kinematics_config.total_spheres
            if len(self.collision_sphere_links) != spheres:
                raise ValueError("Checker spheres differ from the configured collision spheres")
            self._hand_spheres = torch.as_tensor(
                [link.startswith("R_") for link in self.collision_sphere_links], device="cuda"
            )
            for cost in (self.checker.collision_cost, self.checker.collision_constraint):
                cost.config.update_num_spheres(spheres)
            if tuple(self.motion.joint_names) != LEFT_ARM or tuple(self.checker.kinematics.joint_names) != LEFT_ARM:
                raise ValueError("cuRobo joint order differs from SDK left arm order")
            state = self.motion.compute_kinematics(self._state(feedback.vector(LEFT_ARM)))
            fine_state = self.checker.get_kinematics(self._state(feedback.vector(LEFT_ARM)).position[:, None])
            self.mesh_query_distance = (
                max(float(state.robot_spheres[..., 3].max()), float(fine_state.robot_spheres[..., 3].max()))
                + activation_distance
                + 0.001
            )
            self.screen_ik = None
            if self._screen_batch_size:
                self.screen_ik = self._build_screen_ik(config, scene)
            for model in self._scene_models():
                self._bound_mesh_query(model)
            self._checker_stream = torch.cuda.Stream(priority=-1)
            self._checker_stream.wait_stream(torch.cuda.current_stream())
        except Exception:
            if getattr(self, "screen_ik", None) is not None:
                self.screen_ik.destroy()
            self.motion.destroy()
            raise

    def _build_screen_ik(self, config, scene):
        from curobo._src.geom.collision import SceneCollisionCfg, create_collision_checker
        from curobo._src.geom.types import SceneCfg
        from curobo._src.solver.solver_ik import IKSolver
        from curobo._src.solver.solver_ik_cfg import IKSolverCfg
        from curobo._src.types.device_cfg import DeviceCfg

        screen_robot = screen_robot_config(config, self.hand_links)
        # L-BFGS IK as in MotionPlannerCfg.create, but with a fixed 1 cm
        # activation (not _optimizer_activation_m): it only screens, and every
        # returned state is fine-checked at 0.
        ik_config = IKSolverCfg.create(
            robot=screen_robot,
            optimizer_configs=["ik/lbfgs_ik.yml"],
            self_collision_check=False,
            num_seeds=self._screen_ik_seeds,
            optimizer_collision_activation_distance=0.01,
            max_batch_size=self._screen_batch_size,
            max_goalset=1,
        )
        checker = create_collision_checker(
            SceneCollisionCfg(device_cfg=DeviceCfg(), scene_model=SceneCfg.create(copy.deepcopy(scene)))
        )
        ik_config.exit_early_batch_success_threshold = self._screen_exit_early_fraction
        solver = IKSolver(ik_config, checker)
        if tuple(solver.joint_names) != LEFT_ARM:
            solver.destroy()
            raise ValueError("cuRobo screen joint order differs from SDK left arm order")
        return solver

    def _scene_models(self):
        """Coarse optimizer, fine checker and (if built) screen scene models."""
        screen = getattr(self, "screen_ik", None)
        return (self.motion.scene_collision_checker, self.checker.scene_model) + (
            () if screen is None else (screen.scene_collision_checker,)
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        if getattr(self, "screen_ik", None) is not None:
            self.screen_ik.destroy()
        self.motion.destroy()

    def warmup(self):
        """Compile/capture IK, goalset and trajectory kernels before requests."""
        if self._closed:
            raise RuntimeError("Planner is closed")
        if self._warmed:
            return
        self.motion.warmup(num_warmup_iterations=1)
        start = self.motion.default_joint_state.position.reshape(-1).detach().cpu().numpy()
        if getattr(self, "screen_ik", None) is not None:
            # Capture every screen graph now, as MotionPlanner.warmup does: with
            # exit_early the seed solver solves these easy goals and the
            # optimizer graph would first be captured during a request. A
            # capture after the motion graphs are in use (torch empties its
            # allocator cache at each capture) crashed the next motion IK
            # replay after a planner rebuild.
            exit_early, self.screen_ik.config.exit_early = self.screen_ik.config.exit_early, False
            try:
                self._batch_contact_ik(np.repeat(self._palm_poses(start[None]), self._screen_batch_size, axis=0), start)
            finally:
                self.screen_ik.config.exit_early = exit_early
            self.screen_ik.reset_seed()
        # The fine checks load their kernels at their first call: live on the
        # Orin the first request after a graph start screened in 0.86-1.15 s,
        # every later one in ~0.3 s (3090 replay: first call 0.20 s, then 0.04).
        self.constraint_diagnostics(np.repeat(start[None], 8, axis=0))
        # contact_target_clearance's first call too (3090: 0.042 s, then 0.009):
        # a contact state at the start and a cloud whose box holds the hand.
        states = getattr(self, "_screen_contact_states", None)
        self._screen_contact_states = {0: start}
        try:
            self.contact_target_clearance([0], np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
        finally:
            if states is None:
                del self._screen_contact_states
            else:
                self._screen_contact_states = states
        self._grasp_selections.clear()
        self._ik_validity_cache.clear()
        self._warmed = True

    def update_scene(self, feedback, estimate):
        """Replace coarse/fine meshes without rebuilding solvers or CUDA graphs.

        Serialized calls only. Receipt time comes from the new depth estimate,
        never from completion time. Capacity is fixed by the initial scene.
        """
        import torch
        from curobo._src.geom.types import SceneCfg

        if self._closed or not self.locked_joints_match(feedback):
            raise RuntimeError("Planner closed or locked joints changed; rebuild planner")
        if not isinstance(estimate, SceneEstimate) or estimate.frame_id != "torso_link":
            raise ValueError("A torso-frame SceneEstimate with both resolutions is required")
        if not np.isfinite(estimate.received_at) or estimate.received_at < self.scene_received_at:
            raise ValueError("Scene receipt time must be finite and nondecreasing")
        pending = estimate.validation_scene if isinstance(estimate.validation_scene, LazyScene) else None
        carried_target_cover = (
            None
            if pending is not None
            else _mesh_sphere_cover(
                estimate.validation_scene.get("mesh", {}).get("target"), self._carried_target_max_spheres
            )
        )
        pairs = (
            (self.motion.scene_collision_checker, estimate.scene),
            (self.checker.scene_model, estimate.scene if pending is not None else estimate.validation_scene),
        )
        if getattr(self, "screen_ik", None) is not None:
            pairs += ((self.screen_ik.scene_collision_checker, estimate.scene),)
        configs = []
        for model, scene in pairs:
            if not scene or set(scene) - {"cuboid", "mesh"}:
                raise ValueError("Expected nonempty mesh/cuboid scene")
            for kind, data in (("mesh", model.data.meshes), ("cuboid", model.data.cuboids)):
                if len(scene.get(kind, {})) > (0 if data is None else data.max_n):
                    raise ValueError(f"New scene exceeds {kind} capacity; rebuild planner")
            configs.append(SceneCfg.create(copy.deepcopy(scene)))
        # Scene replacement mutates GPU tensors and may create new Warp mesh
        # objects.  Do not let either operation race a previous solver call or
        # leave Warp's default stream ahead of the CUDA stream used by cuRobo.
        # The latter is important for a warmed planner: construction is allowed
        # to happen once, but a live scene update must become visible before the
        # first post-update IK call.
        self._grasp_selections.clear()
        self._ik_validity_cache.clear()
        # cuRobo caches meshes by name, not by geometry. Drop Warp meshes as
        # well as enabled slots or an updated 'environment' silently stays old.
        self.scene_received_at = -np.inf
        try:
            torch.cuda.synchronize()
            for (model, _), cfg in zip(pairs, configs, strict=True):
                if model.data.meshes is not None:
                    model.data.meshes.clear(clear_warp_cache=True)
                model.load_collision_model(cfg)
                self._bound_mesh_query(model)
            # IK and TrajOpt share the updated scene tensors. Invalidate the
            # graph planner's paths, which were checked against the old scene.
            if self.motion.graph_planner is not None:
                self.motion.graph_planner.reset_buffer()
            torch.cuda.synchronize()
        except Exception:
            self.close()
            raise
        self._checker_stream.wait_stream(torch.cuda.current_stream())
        self._pending_validation_scene = pending
        self._carried_target_cover = carried_target_cover
        self._carried_target_support = (np.array(estimate.support.point), np.array(estimate.support.normal))
        self.scene_received_at = estimate.received_at
        if (
            feedback is not None
            and self._approach_graph_seed in ("escape", "over")
            and self.planning_profile == "latency"
        ):
            self._escape_chain(feedback.vector(LEFT_ARM))  # scene-independent; ready for the first attempt

    def locked_joints_match(self, feedback):
        return bool(
            np.allclose(feedback.vector(RIGHT_ARM), self.right, atol=0.025, rtol=0)
            and np.allclose(feedback.vector(HEAD), self.head, atol=0.025, rtol=0)
            and np.allclose(self.hand_mapping.check(feedback.left_hand), self.hand, atol=0.025, rtol=0)
        )

    def _bound_mesh_query(self, model):
        if model is self.motion.scene_collision_checker and self._motion_mesh_query_margin is not None:
            _set_mesh_query_distance(model, self.mesh_query_distance + self._motion_mesh_query_margin)
        else:
            _ensure_mesh_query_distance(model, self.mesh_query_distance)

    def _state(self, positions):
        import torch
        from curobo.types import JointState

        q = np.asarray(positions, dtype=np.float32)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Seven finite left arm positions required")
        return JointState.from_position(torch.tensor(q[None], device="cuda"), joint_names=list(LEFT_ARM))

    def snapshot_matches(self, feedback, now):
        return bool(
            not self._closed
            and feedback.healthy
            and 0 <= now - feedback.received_at <= 0.1
            and 0 <= now - self.scene_received_at <= self.max_scene_age_s
            and self.locked_joints_match(feedback)
        )

    def constraint_diagnostics(self, positions):
        """Per-state constraint results; self cost is squared, world depth is metres."""
        # Final path validation and an offline replay may share this checker.
        # Its batch buffers must not be resized by two threads simultaneously.
        import torch

        with self._checker_lock:
            if self._closed:
                raise RuntimeError("Planner is closed")
            self._ensure_validation_scene()
            with torch.cuda.stream(self._checker_stream):
                return self._constraint_diagnostics(positions)

    def load_validation_scene(self):
        """Install the pending fine mesh now instead of at the first fine check."""
        with self._checker_lock:
            if self._closed:
                raise RuntimeError("Planner is closed")
            self._ensure_validation_scene()

    def _ensure_validation_scene(self):
        """Install the fine mesh before *any* checker result can be returned."""
        import torch
        from curobo._src.geom.types import SceneCfg

        pending = self._pending_validation_scene
        if pending is None:
            return
        try:
            scene = pending.materialize()
            carried_target_cover = _mesh_sphere_cover(
                scene.get("mesh", {}).get("target"), self._carried_target_max_spheres
            )
            config = SceneCfg.create(copy.deepcopy(scene))
            torch.cuda.synchronize()
            model = self.checker.scene_model
            for kind, data in (("mesh", model.data.meshes), ("cuboid", model.data.cuboids)):
                if len(pending[kind]) > (0 if data is None else data.max_n):
                    raise ValueError(f"Fine scene exceeds {kind} capacity; rebuild planner")
            if model.data.meshes is not None:
                model.data.meshes.clear(clear_warp_cache=True)
            model.load_collision_model(config)
            self._bound_mesh_query(model)
            torch.cuda.synchronize()
            self._checker_stream.wait_stream(torch.cuda.current_stream())
            self._pending_validation_scene = None
            self._carried_target_cover = carried_target_cover
        except Exception:
            self.close()
            raise

    def _scene_cost_once_per_sphere_pose(self, state):
        """Fine scene cost per state and sphere; a sphere at the same pose as in state 0 is queried once.

        Links the left arm does not move (torso, right arm, head: 340 of 795
        fine spheres) sit at bitwise identical poses in every state, so their
        cost equals state 0's. Such spheres get a negative radius, which the
        kernels skip, and state 0's value is copied: the result is identical
        to querying every sphere. Orin replay of 39 live inputs (2026-10-09):
        planning median 0.89 -> 0.85 s, same candidates.
        """
        import torch

        spheres = state.robot_spheres
        if not self._query_shared_spheres_once or len(spheres) < 2:
            return self.checker.collision_constraint.forward(state)
        shared = (spheres == spheres[:1]).all(dim=-1)
        shared[0] = False
        masked = spheres.clone()
        masked[..., 3] = torch.where(shared, -100.0, spheres[..., 3])
        query = copy.copy(state)
        query.robot_spheres = masked
        cost = self.checker.collision_constraint.forward(query)
        shared = shared.reshape(cost.shape)
        return torch.where(shared, cost[:1].expand_as(cost), cost)

    def _constraint_diagnostics(self, positions):
        import torch

        q = np.asarray(positions, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 7 or not len(q) or not np.isfinite(q).all():
            raise ValueError("Expected finite Nx7 positions")
        with torch.no_grad():
            positions = torch.tensor(q[:, None], device="cuda").contiguous()
            if self._checker_batch_size != len(q):
                self.checker.setup_batch_tensors(len(q), 1)
                self._checker_batch_size = len(q)
            state = self.checker.get_kinematics(positions)
            # cuRobo 0.8.0 validate() passes spheres to a cost requiring KinematicsState.
            bound = self.checker.get_bound(positions)
            self_cost = self.checker.get_self_collision(state.robot_spheres)
            scene_cost = self._scene_cost_once_per_sphere_pose(state)
            valid = (
                (bound.reshape(len(q), -1) == 0).all(dim=1)
                & (self_cost.reshape(len(q), -1) == 0).all(dim=1)
                & (scene_cost.reshape(len(q), -1) == 0).all(dim=1)
            )
            behind = torch.zeros_like(valid)
            if self._hand_min_x_m is not None:
                hand_x = state.robot_spheres.reshape(len(q), -1, 4)[:, self._hand_spheres, 0]
                behind = (hand_x < self._hand_min_x_m).any(dim=1)
                valid &= ~behind
        return {
            "valid": valid.cpu().numpy().reshape(-1),
            "hand_behind_torso": behind.cpu().numpy().reshape(-1),
            "bound_cost": bound.cpu().numpy().reshape(len(q), -1),
            "self_collision_cost": self_cost.cpu().numpy().reshape(len(q), -1),
            "world_penetration_m": scene_cost.cpu().numpy().reshape(len(q), -1),
            "spheres": state.robot_spheres.cpu().numpy().reshape(len(q), -1, 4),
        }

    def violations(self, positions) -> str:
        """What the fine check rejects along a path: where, which links, how deep, and the deepest sphere.

        The sphere centre is in the torso frame, so it tells the table (its
        height) from the target or a neighbour (their place) without the scene.
        """
        diagnostics = self.constraint_diagnostics(positions)
        bad = np.flatnonzero(~diagnostics["valid"])
        if not len(bad):
            return "no violation"
        world, count = diagnostics["world_penetration_m"], len(diagnostics["valid"])
        parts = [f"samples {bad[0]}-{bad[-1]} of {count}"]
        if world[bad].max() > 0:
            links = sorted(
                {n for n, v in zip(self.collision_sphere_links, world[bad].max(axis=0), strict=False) if v > 0}
            )
            sample, sphere = np.unravel_index(np.argmax(world), world.shape)
            x, y, z, radius = diagnostics["spheres"][sample, sphere]
            parts.append(
                f"world {world.max() * 1000:.1f} mm at sample {sample}, links={links}, deepest "
                f"{self.collision_sphere_links[sphere]} sphere at torso ({x:.3f}, {y:.3f}, {z:.3f}) r={radius:.3f}"
            )
        if diagnostics["self_collision_cost"][bad].max() > 0:
            parts.append(f"self cost {diagnostics['self_collision_cost'][bad].max():.3g}")
        bound = diagnostics["bound_cost"][bad].max(axis=0)
        if bound.max() > 0:
            parts.append(f"bound_joints={[n for n, v in zip(LEFT_ARM, bound, strict=False) if v > 0]}")
        if diagnostics["hand_behind_torso"][bad].any():
            parts.append("hand behind the torso")
        return "; ".join(parts)

    def valid_positions(self, positions):
        return self.constraint_diagnostics(positions)["valid"]

    def _valid_contact_positions(self, positions):
        """Fine-check IK endpoints once per exact float32 state and scene.

        The independent checker includes the measured fingers even while the
        optimizer's hand links are masked. Locked joints belong to this planner;
        scene replacement and warmup clear the cache. Final paths are always
        checked independently, including endpoints and intermediate samples.
        """
        q = np.asarray(positions, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 7 or not len(q) or not np.isfinite(q).all():
            raise ValueError("Expected finite Nx7 contact positions")
        cache = self.__dict__.setdefault("_ik_validity_cache", {})
        keys = [row.tobytes() for row in q]
        missing = {key: row for key, row in zip(keys, q, strict=True) if key not in cache}
        if missing:
            values = self.valid_positions(np.stack(list(missing.values())))
            cache.update(zip(missing, map(bool, values), strict=True))
        valid = np.array([cache[key] for key in keys], dtype=bool)
        if len(cache) > self._ik_validity_cache_capacity:
            self._ik_validity_cache = {key: cache[key] for key in keys[-self._ik_validity_cache_capacity :]}
        return valid

    def _require_valid_start(self, positions):
        diagnostics = self.constraint_diagnostics(np.asarray(positions)[None])
        if diagnostics["valid"].all():
            return
        bound = diagnostics["bound_cost"][0]
        world = diagnostics["world_penetration_m"][0]
        raise RuntimeError(
            "Start state violates collision or joint constraints: "
            f"bound_joints={[n for n, v in zip(LEFT_ARM, bound, strict=False) if v > 0]}, "
            f"self_squared_cost={diagnostics['self_collision_cost'].max():.8g}, "
            f"world_links={sorted({n for n, v in zip(self.collision_sphere_links, world, strict=False) if v > 0})}, "
            f"world_max_m={world.max():.8g}, "
            f"hand_behind_torso={bool(diagnostics['hand_behind_torso'][0])}"
        )

    def _solve_motion(self, goal_pose, state, *, deadline=None, seed_config=None, iters=25, graph_seed=None):
        """One bounded IK/TrajOpt pass in latency mode; native retries otherwise."""
        if self.planning_profile != "latency":
            return self.motion.plan_pose(goal_pose, state)

        def check_deadline():
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Planning deadline exceeded")

        check_deadline()
        count = self._trajopt_seeds
        if seed_config is None and count == 1:
            seed = self.motion.ik_solver.solve_pose(goal_pose, state, return_seeds=1)
            check_deadline()
            if seed is None or seed.success is None or not bool(seed.success.all()):
                return None
            seed_config = seed.solution
        elif count > 1:
            # Fill the parallel TrajOpt seeds with further IK branches after
            # any caller-ranked seed; TrajOpt still returns only its best.
            import torch

            seed = self.motion.ik_solver.solve_pose(goal_pose, state, return_seeds=count)
            check_deadline()
            configs = [] if seed_config is None else [seed_config.reshape(1, -1, seed_config.shape[-1])]
            if seed is not None and seed.success is not None and bool(seed.success.any()):
                success = seed.success.reshape(-1).bool()
                configs.append(seed.solution.reshape(1, -1, seed.solution.shape[-1])[:, success])
            if not configs:
                return None
            configs = torch.cat(configs, dim=1)
            index = torch.arange(count, device=configs.device) % configs.shape[1]
            seed_config = configs[:, index].contiguous()
        self.graph_seed_trace = trace = dict(mode=graph_seed)
        if graph_seed in ("escape", "over") and count > 1:
            result = self._escape_seeded_solve(goal_pose, state, seed_config, iters, trace, check_deadline)
            if result is not None:
                return result
        result = self.motion.trajopt_solver.solve_pose(
            goal_pose, state, seed_config=seed_config, use_implicit_goal=True, initial_iters=iters, finetune_attempts=0
        )
        check_deadline()
        return result

    # Escape seed (_escape_seeded_solve): palm translation (m) at the start
    # orientation, and the Cartesian steps it is solved in.
    _escape_m = 0.10
    _escape_cartesian_steps = 5

    def _palm_poses(self, positions):
        """Torso-frame R_palm transforms for Nx7 joint positions."""
        import torch
        from curobo.types import JointState
        from scipy.spatial.transform import Rotation

        q = torch.as_tensor(np.ascontiguousarray(positions, dtype=np.float32), device="cuda")
        poses = self.motion.compute_kinematics(JointState.from_position(q, joint_names=list(LEFT_ARM))).tool_poses
        frame = poses.tool_frames.index("R_palm")
        position = poses.position[:, 0, frame].detach().cpu().numpy()
        quaternion = poses.quaternion[:, 0, frame].detach().cpu().numpy()
        transforms = np.repeat(np.eye(4)[None], len(q), axis=0)
        transforms[:, :3, :3] = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
        transforms[:, :3, 3] = position
        return transforms

    def _dls_update(self, q, target):
        """One damped least-squares step of Kx7 states toward Kx4x4 palm targets.

        The Jacobian is a forward difference of one batched FK call; rotation
        is weighted 0.1. At most 0.3 rad per update keeps a chain local near
        singularities. Clipped to the joint position limits.
        """
        limits = self.motion.kinematics.config.kinematics_config.joint_limits.position.detach().cpu().numpy()
        weight = np.array((1.0, 1.0, 1.0, 0.1, 0.1, 0.1))
        k, eps = len(q), 1e-3
        probe = np.concatenate((q[:, None], q[:, None] + eps * np.eye(7)[None]), axis=1)
        poses = self._palm_poses(probe.reshape(-1, 7)).reshape(k, 8, 4, 4)
        error = _palm_twist(target, poses[:, 0]) * weight
        jac = np.stack([_palm_twist(poses[:, j], poses[:, 0]) * weight for j in range(1, 8)], axis=2) / eps
        jt = np.transpose(jac, (0, 2, 1))
        dq = (jt @ np.linalg.solve(jac @ jt + 1e-4 * np.eye(6), error[..., None]))[..., 0]
        dq *= np.minimum(1.0, 0.3 / np.maximum(np.abs(dq).max(axis=1), 1e-9))[:, None]
        return np.clip(q + dq, limits[0], limits[1])

    def _cartesian_paths(self, origins, offsets, steps, check_deadline):
        """Fixed-orientation palm translations by damped least squares.

        Each of the K chains moves its origin's palm by its offset in `steps`
        equal targets, one DLS update each plus two final corrections. The
        Jacobian is a forward difference of one batched FK call per update, so
        no IK solver runs. Returns Kx(steps+1)x7 joint chains (clipped to joint
        limits) and a mask of chains whose final palm is within 5 mm and 3 deg.
        """
        base = self._palm_poses(origins)
        q = np.asarray(origins, dtype=float).copy()
        chains = [q]
        for i in range(1, steps + 3):
            check_deadline()
            target = base.copy()
            target[:, :3, 3] += offsets * min(i, steps) / steps
            q = self._dls_update(q, target)
            if i <= steps:
                chains.append(q)
            else:
                chains[-1] = q
        target = base.copy()
        target[:, :3, 3] += offsets
        error = _palm_twist(target, self._palm_poses(q))
        ok = (np.linalg.norm(error[:, :3], axis=1) < 5e-3) & (np.linalg.norm(error[:, 3:], axis=1) < np.radians(3.0))
        return np.stack(chains, axis=1), ok

    def _escape_seeded_solve(self, goal_pose, state, seed_config, iters, trace, check_deadline):
        """TrajOpt whose first seed escapes before its joint line; None if the escape fails.

        The palm first moves _escape_m up and outward (away from the body)
        at its start orientation, then a joint line leads to the approach IK;
        with "over" it leads to the approach pose raised _over_m and the seed
        walks down from there (see _approach_graph_seed).
        The hand starts beside and below the support surface, and the straight
        joint line sweeps it through the surface edge (all of 73 live
        approaches, 2026-10-10); a search over via paths (removed: +0.29 s
        per approach on the Orin) chose this escape 54 of 73 times.
        Other seeds stay straight lines. Only seeds change: TrajOpt and the
        caller's fine validation decide.
        """
        import torch

        start = state.position.detach().cpu().numpy().reshape(-1)
        end = seed_config.detach().cpu().numpy().reshape(-1, start.size)[0]
        before = time.perf_counter()
        chain = self._escape_chain(start, check_deadline)
        trace["escape_ok"] = chain is not None
        if chain is None:
            return None
        path = np.concatenate((chain, end[None]))
        if self._approach_graph_seed == "over":
            # The approach pose raised _over_m, walked back down into it.
            lift = self._over_m * np.asarray(self.lift_up, dtype=float) / np.linalg.norm(self.lift_up)
            above, ok = self._cartesian_paths(end[None], lift[None], self._over_cartesian_steps, check_deadline)
            trace["over_ok"] = bool(ok[0])
            if ok[0]:
                path = np.concatenate((chain, above[0][::-1]))
        solver = self.motion.trajopt_solver
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
        s = np.linspace(0.0, arc[-1], solver.action_horizon)
        seed = np.stack([np.interp(s, arc, path[:, j]) for j in range(7)], axis=1)
        seed_traj = torch.as_tensor(seed, dtype=seed_config.dtype, device=seed_config.device)[None, None]
        trace["escape_s"] = time.perf_counter() - before
        check_deadline()
        return solver.solve_pose(
            goal_pose,
            state,
            seed_config=seed_config[:, 1:].contiguous(),
            seed_traj=seed_traj.contiguous(),
            use_implicit_goal=True,
            initial_iters=iters,
            finetune_attempts=0,
        )

    def _escape_chain(self, start, check_deadline=lambda: None):
        """The escape joint chain from ``start`` (see _escape_seeded_solve), or None; cached per start.

        It depends on the start state only: every attempt of a request, and
        update_scene before planning, share one computation.
        """
        key = np.asarray(start, dtype=np.float32).tobytes()  # solver states are float32
        cache = self.__dict__.setdefault("_escape_chains", {})
        if key not in cache:
            side = 1.0 if self._palm_poses(np.asarray(start)[None])[0, 1, 3] >= 0 else -1.0
            direction = np.array((0.0, side, 1.0))
            chains, ok = self._cartesian_paths(
                np.asarray(start, dtype=float)[None],
                (self._escape_m * direction / np.linalg.norm(direction))[None],
                self._escape_cartesian_steps,
                check_deadline,
            )
            if len(cache) >= 8:
                cache.clear()
            cache[key] = chains[0] if ok[0] else None
        return cache[key]

    def _approach_seed_index(self, selection, start):
        """Prefer a clear sampled joint path only when the native IK seed is blocked.

        These 12 samples rank the already-computed IK alternatives; they never
        authorize execution or reject a candidate. If none is clear, preserve
        cuRobo's first successful seed. Every executed segment still needs
        TrajOpt and full interpolated fine validation.
        """
        success = selection.success.detach().cpu().numpy().reshape(-1)
        choices = np.flatnonzero(success)
        if not len(choices):
            raise RuntimeError("No successful approach IK seed")
        first = int(choices[0])
        if len(choices) == 1:
            return first
        solutions = selection.solution.detach().cpu().numpy().reshape(-1, 7)
        paths = np.stack([np.linspace(start, end, 12) for end in solutions[choices]])
        clear = self.valid_positions(paths.reshape(-1, 7)).reshape(len(choices), 12).all(axis=1)
        if clear[0] or not clear.any():
            return first
        distance = np.sum((solutions[choices] - start) ** 2, axis=1)
        return int(choices[np.argmin(np.where(clear, distance, np.inf))])

    def _fixed_goalset_transforms(self, transforms):
        """Pad a goalset to the planner's capture size with a duplicate goal.

        cuRobo captures solver buffers from the first goalset shape it sees.
        Reusing one fixed shape avoids a graph rebuild when a retry removes a
        failed candidate or when a singleton segment follows a multi-goal
        selection.  The caller keeps the real candidate count and maps any
        selected padding slot back to the first real goal.
        """
        transforms = np.asarray(transforms, dtype=float)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not len(transforms):
            raise ValueError("Expected a nonempty Kx4x4 transform goalset")
        if len(transforms) > self.max_goalset:
            raise ValueError(f"Goalset has {len(transforms)} entries; capacity is {self.max_goalset}")
        if len(transforms) == self.max_goalset:
            return np.ascontiguousarray(transforms)
        padding = np.repeat(transforms[:1], self.max_goalset - len(transforms), axis=0)
        return np.ascontiguousarray(np.concatenate((transforms, padding), axis=0))

    def _batch_contact_ik(self, transforms, start):
        """Screen IK for up to _screen_batch_size goals, one problem each.

        Returns per-goal seed success and solutions (KxSx7). Unused problems
        repeat the first goal so the captured batch shape never changes.
        """
        import torch
        from curobo.types import JointState

        count, size = len(transforms), self._screen_batch_size
        if not 1 <= count <= size:
            raise ValueError(f"Screen batch has {count} goals; capacity is {size}")
        padded = np.concatenate((transforms, np.repeat(transforms[:1], size - count, axis=0)))
        pose = palm_goal_pose(padded, batched=True)
        q = np.repeat(np.asarray(start, dtype=np.float32)[None], size, axis=0)
        seeds = self._screen_ik_seeds
        result = self.screen_ik.solve_pose(
            pose,
            JointState.from_position(torch.tensor(q, device="cuda"), joint_names=list(LEFT_ARM)),
            return_seeds=seeds,
        )
        if result is None or result.success is None or result.solution is None:
            raise RuntimeError("Screen IK returned no result")
        success = result.success.detach().cpu().numpy().reshape(size, seeds)[:count]
        positions = result.solution.detach().cpu().numpy().reshape(size, seeds, 7)[:count]
        return success, positions

    def _fine_valid_states(self, success, positions):
        """Fine-check IK-successful states (KxS mask, KxSx7) for per-goal feasibility.

        Returns (valid, checked) KxS masks. A goal is feasible iff any of its
        successful states is valid, so the first round checks one state per goal and the second only the remaining states of
        goals whose first state collides: the same feasibility as checking all
        (the scene query is linear in states, 0.25 ms each on the Orin).
        Orin replay of 39 live inputs (2026-10-09): planning median 1.11 ->
        0.90 s, same candidates.
        """
        valid = np.zeros_like(success)
        batch = success & (np.cumsum(success, axis=1) == 1)
        checked = np.zeros_like(success)
        while batch.any():
            valid[batch] = self._valid_contact_positions(positions[batch])
            checked |= batch
            batch = success & ~checked & ~valid.any(axis=1, keepdims=True)
        return valid, checked

    def _batch_screen(self, q, transforms, check_deadline, diagnostics):
        """One batched IK problem per candidate; see screen_grasps."""
        summary = self.contact_screen_report
        summary["method"] = "batch"
        success, positions = [], []
        for offset in range(0, len(transforms), self._screen_batch_size):
            check_deadline()
            indices = np.arange(offset, min(offset + self._screen_batch_size, len(transforms)))
            batch_success, batch_positions = self._batch_contact_ik(transforms[indices], q)
            success.append(batch_success)
            positions.append(batch_positions)
            summary["rounds"] += 1
        success, positions = np.concatenate(success), np.concatenate(positions)
        check_deadline()
        fine, checked = self._fine_valid_states(success, positions)
        reached, feasible = success.any(axis=1), fine.any(axis=1)
        # One fine-valid contact state per feasible candidate (grasp_line_clear).
        self._screen_contact_states = {
            int(index): positions[index][np.argmax(fine[index])].copy() for index in np.flatnonzero(feasible)
        }
        summary["ik_successful_states"] += int(success.sum())
        summary["fine_valid_states"] += int(fine.sum())
        summary["fine_rejected_states"] += int(checked.sum() - fine.sum())
        summary["fine_unchecked_states"] = int(success.sum() - checked.sum())
        summary["coarse_candidates"] = np.flatnonzero(reached).tolist()
        summary["fine_valid_candidates"] = np.flatnonzero(feasible).tolist()
        if diagnostics is not None:
            rows = [
                {
                    "candidates": list(range(offset, min(offset + self._screen_batch_size, len(transforms)))),
                    "success": success[offset : offset + self._screen_batch_size].tolist(),
                    "fine_valid": fine[offset : offset + self._screen_batch_size].tolist(),
                }
                for offset in range(0, len(transforms), self._screen_batch_size)
            ]
            diagnostics.append(
                {
                    "method": "batch",
                    "rounds": rows,
                    "feasible": np.flatnonzero(feasible).tolist(),
                    "seeds": self._screen_ik_seeds,
                    "position_tolerance_m": self.screen_ik.config.position_tolerance,
                    "orientation_tolerance_rad": self.screen_ik.config.orientation_tolerance,
                }
            )
        check_deadline()
        return feasible

    def approach_reachable(self, feedback, camera_from_grasps, base_from_camera, offset_m):
        """Per candidate: does the screen IK reach its approach pose (fine-valid)?

        The light batched IK solves each pose moved offset_m along the palm
        axis; successful states are fine-checked with the full model. Ranking
        only: on 60 live inputs a candidate without such a state planned 1 % of
        the time (6/429), with one about 60 % (2026-10-09).
        """
        grasps = np.asarray(camera_from_grasps, dtype=float)
        if getattr(self, "screen_ik", None) is None or not len(grasps):
            return np.zeros(len(grasps), dtype=bool)
        q = feedback.vector(LEFT_ARM)
        goals = palm_goals_from_camera_grasps(grasps, base_from_camera, self.grasp_to_palm)
        goals[:, :3, 3] += float(offset_m) * goals[:, :3, 2]
        batches = [
            self._batch_contact_ik(goals[offset : offset + self._screen_batch_size], q)
            for offset in range(0, len(goals), self._screen_batch_size)
        ]
        success = np.concatenate([batch[0] for batch in batches])
        positions = np.concatenate([batch[1] for batch in batches])
        return self._fine_valid_states(success, positions)[0].any(axis=1)

    def contact_target_clearance(self, candidates, target_points_base, camera_from_grasps=None, base_from_camera=None):
        """Per screened candidate: the open hand's clearance (m) to the target points at its contact state.

        The hand's fine spheres at the screen's contact state (where the grasp
        segment ends), against the target's points in the base frame: the
        smallest distance from a point to a sphere's surface, negative inside.
        With the candidates' grasps and base_from_camera, the smallest along
        the grasp line too (the hand at _grasp_line_offsets_m, as in
        grasp_line_clear): live 2026-10-14 clr_3 pushed the bottle with the
        pinky halfway along the segment, 12.5 mm clear at contact.
        Candidates without a stored contact state read inf. Ranking only.
        """
        import torch

        candidates = [int(index) for index in candidates]
        clearance = np.full(len(candidates), np.inf)
        states = getattr(self, "_screen_contact_states", {})
        known = [k for k, index in enumerate(candidates) if index in states]
        points = np.asarray(target_points_base, dtype=float).reshape(-1, 3)
        if not known or not len(points):
            return clearance
        q = np.ascontiguousarray(np.stack([states[candidates[k]] for k in known]), dtype=np.float32)
        with self._checker_lock, torch.no_grad():
            if self._closed:
                raise RuntimeError("Planner is closed")
            with torch.cuda.stream(self._checker_stream):
                state = self.checker.get_kinematics(torch.tensor(q[:, None], device="cuda").contiguous())
                axes = (
                    None
                    if camera_from_grasps is None
                    else palm_goals_from_camera_grasps(
                        np.asarray(camera_from_grasps, dtype=float)[known], base_from_camera, self.grasp_to_palm
                    )[:, :3, 2]
                )
                spheres = state.robot_spheres.reshape(len(q), -1, 4)[:, self._hand_sphere_mask()]
                clearance[known] = self._hand_target_clearance(spheres, axes, points)
        return clearance

    def _hand_sphere_mask(self):
        """Bool mask over the checker's spheres: the hand's links (fixed per planner)."""
        mask = getattr(self, "_hand_spheres_checker", None)
        if mask is None:
            import torch

            kinematics = self.checker.kinematics.config.kinematics_config
            mask = torch.zeros_like(kinematics.link_sphere_idx_map, dtype=torch.bool)
            for link in self.hand_links:
                mask |= kinematics.link_sphere_idx_map == kinematics.link_name_to_idx_map[link]
            self._hand_spheres_checker = mask
        return mask

    def _hand_target_clearance(self, spheres, axes, points):
        """Per candidate (rows of spheres: N x hand spheres x 4 at contact): its smallest clearance (m) to points.

        With axes (N x 3 palm axes), along the grasp line too: the hand is rigid
        along it, so the contact spheres are shifted back along the palm axis.
        """
        import torch

        spheres = spheres[:, spheres[0, :, 3] > 0]
        if axes is not None:
            offsets = np.array((0.0, *self._grasp_line_offsets_m))
            shift = torch.tensor(offsets[None, :, None] * axes[:, None, :], dtype=spheres.dtype, device=spheres.device)
            centers = (spheres[:, None, :, :3] + shift[:, :, None, :]).reshape(len(spheres), -1, 3)
            radii = spheres[..., 3].repeat(1, len(offsets))
        else:
            centers, radii = spheres[..., :3], spheres[..., 3]
        cloud = torch.tensor(points, dtype=centers.dtype, device=centers.device)
        # A lower bound from the cloud's box; only spheres it puts within
        # _clearance_exact_m are measured against every point (Orin: all
        # ~25k against ~17k points took 0.15 s). Clearances below that are
        # exact, larger ones read as that bound.
        low, high = cloud.min(dim=0).values, cloud.max(dim=0).values
        gap = ((low - centers).clamp_min(0) + (centers - high).clamp_min(0)).norm(dim=-1) - radii
        near = gap < self._clearance_exact_m
        flat = centers[near]
        if len(flat):
            distance = torch.cat(
                [
                    torch.cdist(flat[start : start + 4096], cloud).min(dim=1).values
                    for start in range(0, len(flat), 4096)
                ]
            )
            gap[near] = distance - radii[near]
        return gap.min(dim=1).values.double().cpu().numpy()

    # contact_target_clearance is exact below this (m); see there.
    _clearance_exact_m = 0.03

    # Palm offsets (m, along the palm axis from contact) at which
    # grasp_line_clear places the hand: the grasp segment's straight line.
    _grasp_line_offsets_m = (-0.09, -0.07, -0.05, -0.03, -0.01)

    def grasp_line_clear(self, candidates, camera_from_grasps, base_from_camera):
        """Per screened candidate: does the open hand clear the scene along its grasp line?

        The grasp segment moves the palm straight along its axis at a fixed
        orientation, so the hand is rigid relative to the palm: its fine
        spheres at the screen's contact state, translated back along the palm
        axis by _grasp_line_offsets_m, are where the hand will be. Only those
        spheres are queried (no IK). Ranking only: a contact pose can be valid
        while this line skims the target (live 2026-10-10: three candidates in
        a row, 0.2-3.4 mm of finger tips in the bottle's edge, 0.8 s each);
        plan_grasp's fine check of the real segment still decides. Candidates
        without a stored contact state read as clear. Orin replay of 88 live
        inputs: grasp-segment failures 6 -> 1, attempts 95 -> 90, slowest
        request 2.76 -> 1.67 s, 38 ms per request for 10 candidates.
        """
        return self.grasp_line_checks(candidates, camera_from_grasps, base_from_camera)[0]

    def grasp_line_checks(self, candidates, camera_from_grasps, base_from_camera, target_points_base=None):
        """(grasp_line_clear, contact_target_clearance along the line) from one kinematics pass.

        Both place the same hand spheres (the contact state's, shifted along
        the palm axis) for the same candidates; computed together they share
        the kinematics, the lock and the stream (Orin: 0.069 + 0.096 s apart).
        Without target points the clearance reads inf.
        """
        import torch

        candidates = [int(index) for index in candidates]
        clear = np.ones(len(candidates), dtype=bool)
        clearance = np.full(len(candidates), np.inf)
        states = getattr(self, "_screen_contact_states", {})
        known = [k for k, index in enumerate(candidates) if index in states]
        points = None if target_points_base is None else np.asarray(target_points_base, dtype=float).reshape(-1, 3)
        if not known:
            return clear, clearance
        goals = palm_goals_from_camera_grasps(
            np.asarray(camera_from_grasps, dtype=float)[known], base_from_camera, self.grasp_to_palm
        )
        q = np.ascontiguousarray(np.stack([states[candidates[k]] for k in known]), dtype=np.float32)
        offsets = np.asarray(self._grasp_line_offsets_m, dtype=np.float32)
        with self._checker_lock, torch.no_grad():
            if self._closed:
                raise RuntimeError("Planner is closed")
            self._ensure_validation_scene()
            with torch.cuda.stream(self._checker_stream):
                rows = len(q) * len(offsets)
                if self._checker_batch_size != rows:
                    self.checker.setup_batch_tensors(rows, 1)
                    self._checker_batch_size = rows
                repeated = torch.tensor(np.repeat(q, len(offsets), axis=0)[:, None], device="cuda").contiguous()
                state = self.checker.get_kinematics(repeated)
                hand = self._hand_sphere_mask()
                shift = torch.tensor(
                    np.repeat(goals[:, :3, 2], len(offsets), axis=0).astype(np.float32)
                    * np.tile(offsets, len(q))[:, None],
                    device="cuda",
                )
                spheres = state.robot_spheres.clone()
                spheres[..., :3] += shift[:, None, None, :]
                spheres[..., 3] = torch.where(hand, spheres[..., 3], -100.0)
                query = copy.copy(state)
                query.robot_spheres = spheres
                cost = self.checker.collision_constraint.forward(query).reshape(len(q), len(offsets), -1)
                clear[known] = (cost == 0).all(dim=2).all(dim=1).cpu().numpy()
                if points is not None and len(points):
                    # Each candidate's first row, unshifted, is its contact state.
                    contact = state.robot_spheres.reshape(len(q), len(offsets), -1, 4)[:, 0][:, hand]
                    clearance[known] = self._hand_target_clearance(contact, goals[:, :3, 2], points)
        return clear, clearance

    def _goal_pose(self, transforms):
        """Use one capture shape and contiguous quaternion storage for all solves."""
        return palm_goal_pose(self._fixed_goalset_transforms(transforms))

    def _input_goal_index(self, index, count):
        if not 0 <= index < self.max_goalset:
            raise RuntimeError("cuRobo selected a goal outside the captured goalset")
        return index if index < count else 0

    @staticmethod
    def _motion_failure(result):
        if result is None:
            return "solver returned no result"
        return "trajectory optimization rejected solution"

    def _retry_approach(self, goal_pose, state, result, *, deadline=None, seed_config=None):
        """Try one clear-path seed after the native approach trajectory fails.

        The goalset here contains copies of the selected approach only. Ranking
        cannot silently change the grasp candidate. All-link collision checking
        remains enabled, and the returned trajectory still needs fine validation.
        """
        import torch

        if self.planning_profile != "latency" or self.approach_seed_policy != "clear_path":
            return result
        if result is not None and result.success is not None and bool(result.success.all()):
            return result
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        selection = self.motion.ik_solver.solve_pose(
            goal_pose, state, return_seeds=self.motion.ik_solver.config.num_seeds
        )
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        if selection is None or selection.success is None or not bool(selection.success.any()):
            return result
        index = self._approach_seed_index(selection, state.position.detach().cpu().numpy().reshape(7))
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        seed = selection.solution[:, index : index + 1].clone()
        if seed_config is not None and torch.equal(seed, seed_config):
            return result
        return self._solve_motion(goal_pose, state, deadline=deadline, seed_config=seed)

    def _rescuable(self, result):
        """A TrajOpt rejection that still reached the goal within limits (see _fine_rescue).

        Collision is not judged here: _grasp_trajectory fine-checks every
        interpolated state of a rescued result exactly as of an accepted one.
        """
        if (
            not getattr(self, "_fine_rescue", False)
            or result is None
            or result.success is None
            or bool(result.success.all())
        ):
            return False
        trajectory = getattr(result, "interpolated_trajectory", None)
        errors = [getattr(result, name, None) for name in ("position_error", "rotation_error")]
        tolerances = [getattr(result, name, None) for name in ("position_tolerance", "orientation_tolerance")]
        if trajectory is None or any(value is None for value in errors + tolerances):
            return False
        for error, tolerance in zip(errors, tolerances, strict=True):
            error = error.detach().cpu().numpy()
            tolerance = tolerance.detach().cpu().numpy() if hasattr(tolerance, "detach") else np.asarray(tolerance)
            if (
                not error.size
                or not tolerance.size
                or not np.isfinite(error).all()
                or not np.isfinite(tolerance).all()
                or error.max() > tolerance.max()
            ):
                return False
        last = getattr(result, "interpolated_last_tstep", None)
        count = None if last is None else int(last.reshape(-1)[0].item()) + 1
        names = list(trajectory.joint_names)
        if not set(LEFT_ARM) <= set(names):
            return False
        columns = [names.index(name) for name in LEFT_ARM]
        limits = self.motion.kinematics.config.kinematics_config.joint_limits
        for field in ("velocity", "acceleration", "jerk"):
            value, bound = getattr(trajectory, field, None), getattr(limits, field, None)
            if value is None or bound is None:
                return False
            value = value.detach().cpu().numpy().reshape(-1, len(names))[:count, columns]
            bound = bound.detach().cpu().numpy().reshape(2, -1)[:, : len(LEFT_ARM)]
            if not np.isfinite(value).all() or np.any(value < bound[0] - 1e-6) or np.any(value > bound[1] + 1e-6):
                return False
        return True

    def _carried_target_world_clear(self, spheres, *, ignore_support=False):
        """Check all carried spheres against the scene except their source and support."""
        from types import SimpleNamespace

        import torch
        from curobo._src.geom.collision import CollisionBuffer

        values = np.asarray(spheres, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != 4 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Finite nonempty NxMx4 carried spheres required")
        constraint = self.checker.collision_constraint
        scene = self.checker.scene_model
        cuboids = self.checker.scene_model.data.cuboids
        support = ignore_support and cuboids is not None and "table" in cuboids.get_names()
        _ensure_mesh_query_distance(scene, float(values[..., 3].max()) + 0.001)
        if support:
            cuboids.set_enabled("table", False)
            torch.cuda.synchronize()
        try:
            with torch.cuda.stream(self._checker_stream):
                query = torch.as_tensor(values[:, None], device="cuda")
                state = SimpleNamespace(robot_spheres=query)
                buffer = CollisionBuffer.from_shape(query.shape, scene.device_cfg)
                collision = scene.get_sphere_collision(
                    state,
                    buffer,
                    constraint._weight,
                    constraint.config.activation_distance,
                    return_loss=constraint.config.use_grad_input,
                )
                collision = collision.detach().cpu().numpy().reshape(len(values), -1)
        finally:
            if support:
                cuboids.set_enabled("table", True)
                torch.cuda.synchronize()
        return (collision == 0).all(axis=1)

    def _carried_target_robot_clear(self, carried, robot, tolerance_m=1e-6):
        """Check the carried cover against non-hand robot spheres on the checker stream."""
        import torch

        carried = np.asarray(carried, dtype=np.float32)
        robot = np.asarray(robot, dtype=np.float32)
        mask = self._carried_target_robot_mask
        if carried.ndim != 3 or carried.shape[2] != 4 or robot.ndim != 3 or robot.shape[2] != 4:
            raise ValueError("Expected NxMx4 carried and robot sphere arrays")
        if len(carried) != len(robot) or mask.shape != (robot.shape[1],):
            raise ValueError("Carried and robot sphere batches must align")
        robot = robot[:, mask]
        if not robot.shape[1]:
            return np.ones(len(carried), dtype=bool)
        with torch.cuda.stream(self._checker_stream):
            payload = torch.as_tensor(carried, device="cuda")
            links = torch.as_tensor(robot, device="cuda")
            payload_centers, link_centers = payload[..., :3], links[..., :3]
            distance_squared = (
                (payload_centers * payload_centers).sum(dim=2)[:, :, None]
                + (link_centers * link_centers).sum(dim=2)[:, None, :]
                - 2 * torch.matmul(payload_centers, link_centers.transpose(1, 2))
            ).clamp_min_(0)
            required = (payload[:, :, None, 3] + links[:, None, :, 3] - tolerance_m).clamp_min_(0)
            clear = ((distance_squared >= required * required) | (links[:, None, :, 3] <= 0)).all(dim=(1, 2))
            return clear.cpu().numpy()

    def _valid_carried_target_positions(self, positions, base_from_palm_at_contact):
        """Fine-check robot and carried target for a lift; call inside _carried_target."""
        cover = self._carried_target_cover
        if cover is None:
            return np.zeros(len(positions), dtype=bool)
        diagnostics = self.constraint_diagnostics(positions)
        palms = self._palm_poses(positions)
        carried = _carried_spheres(cover, base_from_palm_at_contact, palms)
        robot_clear = self._carried_target_robot_clear(carried, diagnostics["spheres"])
        support = self._carried_target_support
        support_clear = np.ones(len(positions), dtype=bool)
        if support is not None:
            support_clear = _carried_support_clear(carried, support)
        world_clear = self._carried_target_world_clear(carried, ignore_support=support is not None)
        return diagnostics["valid"] & robot_clear & support_clear & world_clear

    def _linear_grasp_segment(self, start, goal, check_deadline, *, valid_positions=None, invalid_reason="fine"):
        """The grasp segment along the palm line without TrajOpt, or (None, reason); see _linear_grasp.

        Local IK (seeded at the approach end) gives the contact state. Interior
        waypoints start on the joint line and take _linear_grasp_corrections
        damped least-squares steps toward evenly spaced palm poses on the
        Cartesian line (goal orientation); a cubic spline through them is timed
        by a minimum-jerk profile, stretched until the sampled velocity,
        acceleration and jerk are within _linear_grasp_limit_fraction of the
        solver's limits. Call with the grasp phase's collision mask (hand links
        disabled for the coarse IK); the fine check covers every link. A caller
        may add a stricter batched validator, as the lift does for its payload.
        """
        import torch
        from scipy.interpolate import CubicSpline

        check_deadline()
        state = self._state(start)
        seeds = self.motion.ik_solver.config.num_seeds
        result = self.motion.ik_solver.solve_pose(
            self._goal_pose(goal[None]), state, seed_config=state.position[:, None].repeat(1, seeds, 1)
        )
        check_deadline()
        if result is None or result.success is None or not bool(result.success.any()):
            return None, "ik"
        success = result.success.detach().cpu().numpy().reshape(-1)
        solutions = result.solution.detach().cpu().numpy().reshape(-1, 7)[success]
        start = np.asarray(start, dtype=np.float64)
        end = solutions[np.argmin(np.max(np.abs(solutions - start), axis=1))].astype(np.float64)
        origin = self._palm_poses(start[None])[0, :3, 3]
        nodes = np.linspace(0.0, 1.0, self._linear_grasp_waypoints + 2)
        targets = np.repeat(goal[None], len(nodes) - 2, axis=0)
        targets[:, :3, 3] = origin + nodes[1:-1, None] * (goal[:3, 3] - origin)
        interior = start + nodes[1:-1, None] * (end - start)
        for _ in range(self._linear_grasp_corrections):
            check_deadline()
            interior = self._dls_update(interior, targets)
        path = CubicSpline(nodes, np.concatenate((start[None], interior, end[None])), axis=0)
        limits = self.motion.kinematics.config.kinematics_config.joint_limits
        bounds = [
            self._linear_grasp_limit_fraction
            * np.min(np.abs(getattr(limits, field).detach().cpu().numpy().reshape(2, -1)[:, :7]), axis=0)
            for field in ("velocity", "acceleration", "jerk")
        ]
        dt = self.motion.trajopt_solver.config.interpolation_dt
        # Minimum-jerk peaks on the joint line: velocity 1.875 d/T,
        # acceleration 5.774 d/T^2, jerk 60 d/T^3; the samples decide below.
        delta = np.abs(end - start)
        duration = max(
            np.max(1.875 * delta / bounds[0]),
            np.max(np.sqrt(5.774 * delta / bounds[1])),
            np.max(np.cbrt(60.0 * delta / bounds[2])),
            2 * dt,
        )
        for _ in range(4):
            count = int(np.ceil(duration / dt)) + 1
            times = np.arange(count) * dt
            tau = times / times[-1]
            positions = np.ascontiguousarray(path(10 * tau**3 - 15 * tau**4 + 6 * tau**5))
            ratio = 1.0
            for order, bound in enumerate(bounds, 1):
                derivative = np.diff(positions, n=order, axis=0) / dt**order
                ratio = max(ratio, (np.max(np.abs(derivative) / bound)) ** (1.0 / order))
            if ratio <= 1.0:
                break
            duration = times[-1] * ratio * 1.01
        else:
            return None, "limits"
        palms = self._palm_poses(positions)
        axis = goal[:3, 3] - origin
        along = np.clip((palms[:, :3, 3] - origin) @ axis / max(axis @ axis, 1e-12), 0.0, 1.0)
        deviation = np.linalg.norm(palms[:, :3, 3] - origin - along[:, None] * axis, axis=1).max()
        relative = np.einsum("ji,njk->nik", goal[:3, :3], palms[:, :3, :3])
        rotation = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)).max()
        check_deadline()
        if deviation > self._linear_grasp_max_deviation_m or rotation > self._linear_grasp_max_rotation_rad:
            return None, "line"
        validate = self.valid_positions if valid_positions is None else valid_positions
        if not validate(positions).all():
            return None, invalid_reason
        check_deadline()
        torch.cuda.synchronize()
        return TimedTrajectory(times, positions), None

    def _grasp_trajectory(self, result, phase, *, deadline=None):
        """Keep inclusive endpoints and fine-check every returned segment."""
        from types import SimpleNamespace

        import torch

        rescued = getattr(result, "aimdk_fine_rescue", False)
        if result is None or result.success is None or not (rescued or bool(result.success.all())):
            raise RuntimeError(f"cuRobo grasp planning failed at {phase}: {self._motion_failure(result)}")
        raw_result = SimpleNamespace(
            success=torch.ones_like(result.success) if rescued else result.success,
            get_interpolated_plan=lambda: result.interpolated_trajectory,
        )
        out = TimedTrajectory.from_curobo(
            raw_result, self.motion.trajopt_solver.config.interpolation_dt, locked_positions=self.locked_positions
        )
        last = result.interpolated_last_tstep
        if last is not None:
            n = int(last.reshape(-1)[0].item()) + 1
            if not 2 <= n <= len(out.times):
                raise RuntimeError(f"Incomplete interpolated {phase} trajectory")
            if n < len(out.times):
                out = TimedTrajectory(out.times[:n], out.positions[:n])
        if not self.valid_positions(out.positions).all():
            raise RuntimeError(
                f"Interpolated {phase} segment violates fine collision constraints: {self.violations(out.positions)}"
            )
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        return out

    def _vertical_lift(self, grasp, check_deadline):
        """(lift, lift_method): the palm raised vertical_lift_m along lift_up, or a hold; see vertical_lift_m.

        Call with the grasp phase's collision mask, like _linear_grasp_segment.
        """
        report = self.__dict__.setdefault("vertical_lift_report", {})
        lift, reason = None, "disabled"
        if self.vertical_lift_m > 0:
            up = np.asarray(self.lift_up, dtype=float)
            up = up / np.linalg.norm(up)
            contact = self._palm_poses(grasp.positions[-1][None])[0]
            goal = contact.copy()
            goal[:3, 3] += self.vertical_lift_m * up
            with self._carried_target():
                lift, reason = self._linear_grasp_segment(
                    grasp.positions[-1],
                    goal,
                    check_deadline,
                    valid_positions=lambda positions: self._valid_carried_target_positions(positions, contact),
                    invalid_reason="carried",
                )
        outcome = "used" if lift is not None else f"hold_{reason}"
        report[outcome] = report.get(outcome, 0) + 1
        if lift is None:
            return self._hold_lift(grasp), "hold"
        # Exact shared endpoint with the grasp segment (grasp_request checks it).
        positions = lift.positions.copy()
        positions[0] = grasp.positions[-1]
        return TimedTrajectory(lift.times, positions), "vertical"

    @contextmanager
    def _carried_target(self):
        """Hide the target's old pose while its palm-attached cover is checked exclusively."""
        import torch

        with self._checker_lock:
            self._ensure_validation_scene()
            # The mesh store directly: cuRobo 0.8.0's enable_obstacle tries the
            # cuboids first and their lookup raises ValueError, not the
            # RuntimeError it catches.
            meshes = self.checker.scene_model.data.meshes
            carried = meshes is not None and "target" in meshes.get_names()
            if carried:
                meshes.set_enabled("target", False)
                torch.cuda.synchronize()
            try:
                yield
            finally:
                if carried:
                    meshes.set_enabled("target", True)
                    torch.cuda.synchronize()

    @staticmethod
    def _hold_lift(grasp):
        """A zero-motion lift segment at the grasp pose (lift not planned)."""
        end = grasp.positions[-1]
        return TimedTrajectory(np.array([0.0, 0.1]), np.stack([end, end]), grasp.joint_names)

    def plan_pose(self, feedback, torso_from_palm, now, *, deadline=None):
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("Finite monotonic planning deadline required")
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        if not self.snapshot_matches(feedback, now):
            raise RuntimeError("Feedback or collision scene changed/stale; rebuild snapshot")
        q = feedback.vector(LEFT_ARM)
        self._require_valid_start(q)
        goal = rigid(torso_from_palm)
        pose, state = self._goal_pose(goal[None]), self._state(q)
        result = self._solve_motion(pose, state, deadline=deadline)
        result = self._retry_approach(pose, state, result, deadline=deadline)
        if result is None or result.success is None or not bool(result.success.all()):
            raise RuntimeError(f"cuRobo pose planning failed: {self._motion_failure(result)}")
        trajectory = TimedTrajectory.from_curobo(
            result, self.motion.trajopt_solver.config.interpolation_dt, locked_positions=self.locked_positions
        )
        if not self.valid_positions(trajectory.positions).all():
            raise RuntimeError("Interpolated trajectory violates collision or joint constraints")
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        return trajectory

    def screen_grasps(self, feedback, camera_from_grasps, base_from_camera, now, *, diagnostics=None, deadline=None):
        """Discover fine-valid contact candidates with the batched screen IK.

        Every candidate is its own IK problem over the moving arm spheres
        (screen_ik); successful states are fine-checked with the measured-hand
        model (see _batch_screen). This is a contact prefilter, not a path
        guarantee: approach, contact and fine all-link path checks still run in
        plan_grasp. A false flag means not discovered by these seeds, not
        proven unreachable. The caller orders the discoveries afterward.
        """
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("Finite monotonic planning deadline required")

        def check_deadline():
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Planning deadline exceeded")

        self.contact_screen_report = {
            "rounds": 0,
            "ik_successful_states": 0,
            "fine_valid_states": 0,
            "fine_rejected_states": 0,
            "coarse_candidates": [],
            "fine_valid_candidates": [],
        }
        check_deadline()
        grasps = list(camera_from_grasps)
        if not grasps:
            return np.zeros(0, dtype=bool)
        if getattr(self, "screen_ik", None) is None:
            raise RuntimeError("Planner has no screen IK (_screen_batch_size is 0)")
        if not self.snapshot_matches(feedback, now):
            raise RuntimeError("Feedback or collision scene changed/stale; rebuild snapshot")
        q = feedback.vector(LEFT_ARM)
        self._require_valid_start(q)
        transforms = palm_goals_from_camera_grasps(np.stack(grasps), base_from_camera, self.grasp_to_palm)
        return self._batch_screen(q, transforms, check_deadline, diagnostics)

    def plan_grasp(
        self,
        feedback,
        camera_from_grasp,
        base_from_camera,
        now,
        *,
        approach_offset=-0.05,
        lift_offset=-0.05,
        on_approach=None,
        deadline=None,
    ):
        """Plan three segments to one of up to max_goalset camera-frame grasps.

        Accepts a single 4x4 transform or Kx4x4 alternatives. goalset_index in
        the returned plan identifies the input chosen by cuRobo. The initial
        goalset solve is reused for offset strategies with identical start and
        goals. Contact links are disabled in contact selection (default profile),
        grasp and lift optimization; all links remain active in approach
        selection and fine final checks.
        on_approach, when supplied, receives the fine-validated approach and
        selected local index before the remaining segments are solved. The
        caller must hold the scene snapshot and stop/wait if continuation fails.
        deadline uses time.perf_counter(), checked between GPU calls; an in-flight
        call is not preempted. Expired results are never returned or published.
        """
        import torch
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        if not self.snapshot_matches(feedback, now):
            raise RuntimeError("Feedback or collision scene changed/stale; rebuild snapshot")
        if not np.isfinite([approach_offset, lift_offset]).all():
            raise ValueError("Finite approach and lift offsets required")
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("Finite monotonic planning deadline required")

        def check_deadline():
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Planning deadline exceeded")

        check_deadline()
        grasps = np.asarray(camera_from_grasp, dtype=float)
        if grasps.shape == (4, 4):
            grasps = grasps[None]
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or not 1 <= len(grasps) <= self.max_goalset:
            raise ValueError(f"Expected 1..{self.max_goalset} camera-frame grasps with shape Kx4x4")
        goals = palm_goals_from_camera_grasps(grasps, base_from_camera, self.grasp_to_palm)
        q = feedback.vector(LEFT_ARM)
        self._require_valid_start(q)
        hand = list(self.hand_links)

        def solve(transforms, start, phase, seed_config=None):
            check_deadline()
            goal_pose, state = self._goal_pose(transforms), self._state(start)
            iters = self._approach_iters if phase == "approach" else 25
            # Experimental options are passed only when set.
            options = {} if iters == 25 else dict(iters=iters)
            if phase == "approach" and self._approach_graph_seed is not None:
                options["graph_seed"] = self._approach_graph_seed
            result = self._solve_motion(goal_pose, state, deadline=deadline, seed_config=seed_config, **options)
            rescued = self._rescuable(result)
            if not rescued and phase == "approach" and (abs(approach_offset) > 0.05 or self._short_approach_retry):
                result = self._retry_approach(goal_pose, state, result, deadline=deadline, seed_config=seed_config)
                rescued = self._rescuable(result)
            check_deadline()
            if rescued:
                result.aimdk_fine_rescue = True
                report = self.__dict__.setdefault("fine_rescue_report", {})
                report[phase] = report.get(phase, 0) + 1
            elif result is None or result.success is None or not bool(result.success.all()):
                raise RuntimeError(f"cuRobo grasp planning failed at {phase}: {self._motion_failure(result)}")
            return result

        def adapt(result, phase):
            return self._grasp_trajectory(result, phase, deadline=deadline)

        selection_goals = goals
        if self.planning_profile == "latency":
            # Rank by the first executable pose with all links enabled. Contact
            # IK can favor a pose whose offset approach is unreachable, wasting
            # the request on repeated single-goal failures. Grasp/lift still
            # receive their own solves and full fine validation below.
            selection_goals = goals.copy()
            selection_goals[:, :3, 3] += float(approach_offset) * goals[:, :3, 2]
        key = (q.tobytes(), goals.tobytes(), selection_goals.tobytes())
        index = None
        try:
            check_deadline()
            if key not in self._grasp_selections:
                # Default mode selects contact; latency mode selects an approach
                # and must use the same all-link model as its trajectory solve.
                if self.planning_profile != "latency":
                    self.motion.disable_link_collision(hand)
                # Select a reachable goal without optimizing a fourth, unused
                # trajectory straight to contact. The three executed segments
                # still require trajectory solves and all-link fine checks.
                # Short approaches benefit from clear-path IK alternatives. For
                # longer clearances, native ranking preserves branches needed by
                # the contact continuation (approach-only ranking can regress it).
                rank_seeds = (
                    self.planning_profile == "latency"
                    and self.approach_seed_policy == "clear_path"
                    and abs(approach_offset) <= 0.05
                )
                selection = self.motion.ik_solver.solve_pose(
                    self._goal_pose(selection_goals),
                    self._state(q),
                    return_seeds=self.motion.ik_solver.config.num_seeds if rank_seeds else 1,
                )
                check_deadline()
                if (
                    selection is None
                    or selection.success is None
                    or not bool(selection.success.any())
                    or selection.goalset_index is None
                ):
                    raise RuntimeError("cuRobo grasp planning failed at goalset IK")
                seed_index = self._approach_seed_index(selection, q) if rank_seeds else 0
                check_deadline()
                raw_index = int(selection.goalset_index.reshape(-1)[seed_index].item())
                index = self._input_goal_index(raw_index, len(goals))
                if len(self._grasp_selections) >= self.max_goalset * 2:
                    self._grasp_selections.clear()
                seed = (
                    selection.solution[:, seed_index : seed_index + 1].clone()
                    if self.planning_profile == "latency"
                    else None
                )
                self._grasp_selections[key] = (index, seed)
            index, seed = self._grasp_selections[key]
            goal = goals[index]
            approach_goal = goal.copy()
            approach_goal[:3, 3] += float(approach_offset) * goal[:3, 2]
            # Finish the goalset work before changing the collision-link mask
            # shared by cuRobo/Warp rollouts for the approach phase.
            torch.cuda.synchronize()
            self.motion.enable_link_collision(hand)
            approach = adapt(solve(approach_goal[None], q, "approach", seed_config=seed), "approach")
            if on_approach is not None:
                on_approach(approach, index)
            linear = ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True)
            self.motion.update_tool_pose_criteria({"R_palm": linear})
            self.motion.disable_link_collision(hand)
            check_deadline()
            grasp = None
            if self._linear_grasp and self.planning_profile == "latency":
                grasp, reason = self._linear_grasp_segment(approach.positions[-1], goal, check_deadline)
                report = self.__dict__.setdefault("linear_grasp_report", {})
                outcome = "used" if grasp is not None else f"fallback_{reason}"
                report[outcome] = report.get(outcome, 0) + 1
            if grasp is None:
                grasp = adapt(solve(goal[None], approach.positions[-1], "grasp"), "grasp")
            # lift_offset is not planned (see _plan_lift): the lift goes straight up, or holds.
            lift, lift_method = self._vertical_lift(grasp, check_deadline)
            return GraspPlan(approach, grasp, lift, index, approach_offset_m=approach_offset, lift_method=lift_method)
        except RuntimeError as exc:
            if device_fault(exc):
                raise PlannerFault(f"{type(exc).__name__}: {exc}") from exc
            raise GraspPlanningError(str(exc), index) from exc
        finally:
            self.motion.enable_link_collision(hand)
            self.motion.update_tool_pose_criteria({"R_palm": ToolPoseCriteria()})
