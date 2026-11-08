"""CUDA-only planning with locked-joint snapshots and replaceable scene buffers."""

import copy
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from types import SimpleNamespace

import numpy as np

from .frames import rigid
from .hand import HandMapping, hand_links
from .joints import HEAD, arm_joints, check_side, other_side
from .mesh_loading import use_gpu_validation_meshes
from .robot_model import collision_robot_config, config_directory, planning_model
from .robot_spheres import RobotSphereModel
from .scene_estimation import LazyScene, SceneEstimate
from .trajectory import TimedTrajectory

#: How far (m) any collision sphere of a locked group (other arm, head, the
#: planning hand) may have moved since the snapshot the planner was built on
#: before the snapshot has to be rebuilt. The rebuild is the service's most
#: expensive operation (47.6-49.9 s on the Orin) and no request deadline can
#: preempt it, so `planner_service` names which group crossed this when it
#: happens. Judged in metres, not radians: one joint-angle tolerance moved the
#: other arm's hand ~12 mm but a finger ~1 mm, and the OmniHand thumb comes to
#: rest anywhere within ~0.05 rad of the open preset after every reopening
#: (live 2026-11-06: hand=0.0319 rad rebuilt the planner inside a request, 39 s).
#: 5 mm is a quarter of the optimizer's 0.02 m collision activation distance.
LOCKED_SPHERE_SHIFT_M = 0.005
#: How far (rad) past a model joint bound a measured planning-arm start may be
#: and still plan, from just inside the bound (START_BOUND_INSET_RAD); see
#: LeftArmPlanner._measured_start. Far below the 0.025 rad start tolerance the
#: executor allows between the planned start and the measured arm.
START_BOUND_SLACK_RAD = 0.01
START_BOUND_INSET_RAD = 0.001
#: Palm-fixed link whose reserved spheres carry a held object (and the closed
#: fingers outside the open-hand model). cuRobo rewrites them in place, so a
#: carry never rebuilds the planner. Disabled slots have radius -100.
ATTACH_LINK = "attached_object"
ATTACH_SLOTS = 16
DISABLED_RADIUS = -100.0


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


class PlacementSearchExhausted(RuntimeError):
    """No complete placement path in this snapshot; a fresh observation may help."""


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
    side="left",
    reference_hand_collisions=True,
    torso_merge_radius_m=None,
    torso_merge_protected_spheres=None,
    hand_merge_radius_m=None,
):
    directory = config_directory(check_side(side))
    arm, other = arm_joints(side), arm_joints(other_side(side))
    config = collision_robot_config(
        directory,
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
    locks.update(zip(other + HEAD, feedback.vector(other + HEAD), strict=True))
    locks.update(HandMapping(planning_model(directory), side=side).kinematic_alias(feedback.hand(side)))
    if movable - set(locks) != set(arm):
        raise ValueError(f"Robot must expose exactly the seven {side} arm joints")
    kin["lock_joints"] = {n: float(v) for n, v in locks.items()}
    kin["cspace"]["default_joint_position"] = feedback.vector(arm).tolist()
    reserve_attachment(kin, directory, side)
    return config


def reserve_attachment(kin, directory, side):
    """Add ATTACH_LINK under R_palm with ATTACH_SLOTS disabled spheres.

    The slots are listed explicitly so every model (and collision_sphere_links)
    counts them; the hand and wrist never collide with what the hand holds.
    """
    extra = kin.get("extra_links") or {}
    extra[ATTACH_LINK] = {
        "link_name": ATTACH_LINK,
        "parent_link_name": "R_palm",
        "joint_name": f"{ATTACH_LINK}_joint",
        "joint_type": "FIXED",
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
    }
    kin["extra_links"] = extra
    near = [*hand_links(planning_model(directory)), *(n for n in kin["collision_link_names"] if f"{side}_wrist" in n)]
    kin["collision_link_names"] = [*kin["collision_link_names"], ATTACH_LINK]
    kin["collision_spheres"][ATTACH_LINK] = [
        {"center": [0.0, 0.0, 0.0], "radius": DISABLED_RADIUS} for _ in range(ATTACH_SLOTS)
    ]
    kin["self_collision_ignore"] = {**(kin.get("self_collision_ignore") or {}), ATTACH_LINK: list(dict.fromkeys(near))}
    kin["self_collision_buffer"] = {**(kin.get("self_collision_buffer") or {}), ATTACH_LINK: 0.0}


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


def screen_robot_config(config, excluded_links, side="left"):
    """The ``side`` arm's moving collision links of ``config`` minus ``excluded_links``.

    The contact screen disables the hand links anyway, and the static body
    cannot move toward a candidate, so a batched screen over 100 candidates
    needs only the few arm spheres. The kinematic tree is trimmed to those
    links and the R_palm tool frame; locks outside it are dropped.
    """
    config = copy.deepcopy(config)
    kin = config["robot_cfg"]["kinematics"]
    joints = {child: (name, parent) for name, _, parent, child, _ in urdf_joints(kin["urdf_path"])}
    shoulder = next(child for child, (name, _) in joints.items() if name == arm_joints(side)[0])
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


def scene_capacity(scene, validation_scene):
    """Reserve environment, target and standing-leg slots before CUDA graph capture.

    Warmup has no target mask, so its mesh count is smaller than a pick's.
    A lazy fine scene stays deferred; its standard third mesh is the leg.
    Preserve larger caller-supplied scenes too.
    """
    scenes = [scene]
    if not isinstance(validation_scene, LazyScene):
        scenes.append(validation_scene)
    return {
        "mesh": max(3, *(len(value.get("mesh", {})) for value in scenes)),
        "cuboid": max(1, *(len(value.get("cuboid", {})) for value in scenes)),
    }


class LeftArmPlanner:
    """Plans and checks only the motion of one arm (``side``, the left by default); hardware transport is separate.

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

    #: The planned arm; ``__init__`` sets them from ``side`` (class defaults: the left arm).
    side = "left"
    arm, other_arm = arm_joints("left"), arm_joints("right")

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
    # segment. Its fine check leaves out the target, which the hand carries
    # (live 2026-10-13: with the target in, the open hand sliding up the
    # standing bottle failed it); the table and the rest of the scene stay.
    # When the line cannot be planned (IK, limits, deviation, fine check) the
    # lift is a hold at the grasp pose, which the executor refuses to run
    # until lift. 0 always holds.
    vertical_lift_m = 0.05
    lift_up = (0.0, 0.0, 1.0)
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
        approach_over_height_m=0.18,
        side="left",
    ):
        import curobo.runtime as runtime
        import torch
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if isinstance(scene, SceneEstimate):
            if validation_scene is not None or scene_received_at != scene.received_at:
                raise ValueError("SceneEstimate owns both geometry resolutions and their receipt time")
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
        if (
            isinstance(approach_over_height_m, bool)
            or not isinstance(approach_over_height_m, int | float)
            or not np.isfinite(approach_over_height_m)
            or approach_over_height_m <= 0
        ):
            raise ValueError("Finite positive approach_over_height_m required")
        if planning_profile == "latency" and max_batch_size != 1:
            raise ValueError("The latency profile requires max_batch_size=1 (direct goalsets)")
        configure_runtime(runtime)
        self.side = check_side(side)
        self.arm, self.other_arm = arm_joints(side), arm_joints(other_side(side))
        self.other = feedback.vector(self.other_arm)
        self.head = feedback.vector(HEAD)
        model = planning_model(config_directory(side))
        self.hand_mapping = HandMapping(model, side=side)
        self.hand_links = list(hand_links(model))
        self.grasp_to_palm = grasp_to_palm_transform(gripper)
        self.hand = self.hand_mapping.check(feedback.hand(side)).copy()
        # CPU forward kinematics of the planning model's spheres, to judge how
        # far the locked groups moved (locked_shift_m) without touching the GPU.
        self._lock_spheres = RobotSphereModel(self.hand_mapping, config_directory(side))
        self.scene_received_at = scene_received_at
        self.max_scene_age_s = max_scene_age_s
        self.reference_hand_collisions = reference_hand_collisions
        #: Spheres (palm frame, Nx4) attached to ATTACH_LINK; the last is the object.
        self.attached = np.empty((0, 4))
        self.max_goalset = max_goalset
        self.max_batch_size = max_batch_size
        self.planning_profile = planning_profile
        self.approach_seed_policy = approach_seed_policy
        self._over_m = float(approach_over_height_m)
        self._closed = False
        self._warmed = False
        self._checker_lock = RLock()
        self._checker_batch_size = None
        self._pending_validation_scene = validation_scene if isinstance(validation_scene, LazyScene) else None
        self._scene_capacity = scene_capacity(scene, validation_scene)
        config = robot_config(
            feedback,
            side=side,
            reference_hand_collisions=reference_hand_collisions,
        )
        kin = config["robot_cfg"]["kinematics"]
        self.collision_sphere_links = [
            name for name in kin["collision_link_names"] for _ in kin["collision_spheres"][name]
        ]
        self.locked_positions = config["robot_cfg"]["kinematics"]["lock_joints"].copy()
        activation_distance = self._optimizer_activation_m
        motion_robot = (
            robot_config(
                feedback,
                side=side,
                reference_hand_collisions=reference_hand_collisions,
                torso_merge_radius_m=torso_merge_radius_m,
                torso_merge_protected_spheres=torso_merge_protected_spheres,
                hand_merge_radius_m=self._motion_hand_merge_radius_m,
            )
            if torso_merge_radius_m is not None or self._motion_hand_merge_radius_m is not None
            else copy.deepcopy(config)
        )
        # Preserve the defaults from construction, also after cached reuse.
        self._diagnostic_models = copy.deepcopy({"fine": config, "planning": motion_robot})
        self._diagnostic_scenes = (copy.deepcopy(scene), validation_scene)
        self.planning_sphere_count = sum(
            len(v) for v in motion_robot["robot_cfg"]["kinematics"]["collision_spheres"].values()
        )
        motion_config = MotionPlannerCfg.create(
            robot=motion_robot,
            scene_model=copy.deepcopy(scene),
            collision_cache=dict(self._scene_capacity),
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
        limits = self.motion.kinematics.config.kinematics_config.joint_limits.position.detach().cpu().numpy()
        #: Planning-arm position bounds (lower, upper) the solvers enforce, for _measured_start.
        self._arm_bounds = (limits[0].astype(float), limits[1].astype(float))
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
                    n_meshes=self._scene_capacity["mesh"],
                    n_cuboids=self._scene_capacity["cuboid"],
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
            if tuple(self.motion.joint_names) != self.arm or tuple(self.checker.kinematics.joint_names) != self.arm:
                raise ValueError("cuRobo joint order differs from SDK left arm order")
            state = self.motion.compute_kinematics(self._state(feedback.vector(self.arm)))
            fine_state = self.checker.get_kinematics(self._state(feedback.vector(self.arm)).position[:, None])
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

        screen_robot = screen_robot_config(config, self.hand_links, self.side)
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
            SceneCollisionCfg(
                device_cfg=DeviceCfg(),
                scene_model=SceneCfg.create(copy.deepcopy(scene)),
                cache=dict(self._scene_capacity),
            )
        )
        ik_config.exit_early_batch_success_threshold = self._screen_exit_early_fraction
        solver = IKSolver(ik_config, checker)
        if tuple(solver.joint_names) != self.arm:
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

    def update_scene(self, feedback, estimate, *, carrying=None):
        """Replace coarse/fine meshes without rebuilding solvers or CUDA graphs.

        Serialized calls only. Receipt time comes from the new depth estimate,
        never from completion time. Capacity is reserved before warmup, including
        the target mesh absent from the startup scene.
        """
        import torch
        from curobo._src.geom.types import SceneCfg

        matches = self.locked_joints_match(feedback, carrying=True) if carrying else self.locked_joints_match(feedback)
        if self._closed or not matches:
            raise RuntimeError("Planner closed or locked joints changed; rebuild planner")
        if not isinstance(estimate, SceneEstimate) or estimate.frame_id != "torso_link":
            raise ValueError("A torso-frame SceneEstimate with both resolutions is required")
        if not np.isfinite(estimate.received_at) or estimate.received_at < self.scene_received_at:
            raise ValueError("Scene receipt time must be finite and nondecreasing")
        pending = estimate.validation_scene if isinstance(estimate.validation_scene, LazyScene) else None
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
        self._diagnostic_scenes = (copy.deepcopy(estimate.scene), estimate.validation_scene)
        self.scene_received_at = estimate.received_at
        if (
            feedback is not None
            and self._approach_graph_seed in ("escape", "over")
            and self.planning_profile == "latency"
        ):
            self._escape_chain(feedback.vector(self.arm))  # scene-independent; ready for the first attempt

    def locked_shift_m(self, feedback, *, carrying=None):
        """Per locked group, the largest collision-sphere displacement (m) from
        the built snapshot when only that group takes its measured value.

        While carrying, the hand is the open-hand model plus the attached cover,
        so a closed hand is not a change (0.0). Hand spheres ride on the
        planning arm; a rigid motion keeps distances, so the planning arm's own
        pose does not change the result and its measured value is used.
        """
        carrying = self.carrying if carrying is None else carrying
        other = np.asarray(feedback.vector(self.other_arm), float)
        head = np.asarray(feedback.vector(HEAD), float)
        hand = self.hand if carrying else self.hand_mapping.check(feedback.hand(self.side))
        arm = np.asarray(feedback.vector(self.arm), float)
        # Row 0 is the built snapshot; each later row changes one group.
        others = np.array([self.other, other, self.other, self.other])
        heads = np.array([self.head, self.head, head, self.head])
        hands = np.array([self.hand, self.hand, self.hand, hand])
        spheres, _ = self._lock_spheres.spheres(feedback, np.tile(arm, (4, 1)), other=others, head=heads, hand=hands)
        moved = np.linalg.norm(spheres[1:, :, :3] - spheres[:1, :, :3], axis=2).max(axis=1)
        return dict(zip(("other_arm", "head", "hand"), (float(value) for value in moved), strict=True))

    def locked_joints_match(self, feedback, *, carrying=None):
        """No locked group moved any collision sphere past LOCKED_SPHERE_SHIFT_M."""
        return max(self.locked_shift_m(feedback, carrying=carrying).values()) <= LOCKED_SPHERE_SHIFT_M

    @property
    def carrying(self):
        return len(self.attached) > 0

    def _attachment_parameters(self):
        """Every kinematics copy holding ATTACH_LINK spheres: the motion planner's
        (shared by IK, TrajOpt and the graph planner) and the fine checker's own."""
        found = []
        for model in (self.motion.kinematics, self.checker.kinematics):
            parameters = model.config.kinematics_config
            index = parameters.get_sphere_index_from_link_name(ATTACH_LINK)
            if len(index) != ATTACH_SLOTS:
                raise RuntimeError(f"planner model lacks the {ATTACH_SLOTS} reserved {ATTACH_LINK} spheres")
            found.append((parameters, index))
        return found

    def _write_attachment(self, spheres):
        import torch

        values = np.full((ATTACH_SLOTS, 4), [0.0, 0.0, 0.0, DISABLED_RADIUS])
        values[: len(spheres)] = spheres
        for parameters, index in self._attachment_parameters():
            tensor = torch.as_tensor(values, device=parameters.link_spheres.device, dtype=parameters.link_spheres.dtype)
            parameters.link_spheres[:, index, :] = tensor
        torch.cuda.synchronize()
        self._diagnostic_attachment = np.asarray(spheres).copy()

    def attach_object(self, spheres):
        """Attach palm-frame Nx4 spheres (N <= ATTACH_SLOTS, the object last); no rebuild."""
        spheres = np.asarray(spheres, dtype=float).reshape(-1, 4)
        if not 1 <= len(spheres) <= ATTACH_SLOTS or not np.isfinite(spheres).all() or (spheres[:, 3] <= 0).any():
            raise ValueError(f"1..{ATTACH_SLOTS} finite positive palm-frame spheres required")
        self._write_attachment(spheres)
        self.attached = spheres.copy()

    def detach_object(self):
        self._write_attachment(np.empty((0, 4)))
        self.attached = np.empty((0, 4))

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
        return JointState.from_position(torch.tensor(q[None], device="cuda"), joint_names=list(self.arm))

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
            config = SceneCfg.create(copy.deepcopy(pending.materialize()))
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
            hand_min_x = None
            if self._hand_min_x_m is not None:
                hand_x = state.robot_spheres.reshape(len(q), -1, 4)[:, self._hand_spheres, 0]
                hand_min_x = hand_x.min(dim=1).values.cpu().numpy()
                behind = (hand_x < self._hand_min_x_m).any(dim=1)
                valid &= ~behind
        diagnostics = {
            "valid": valid.cpu().numpy().reshape(-1),
            "hand_behind_torso": behind.cpu().numpy().reshape(-1),
            "bound_cost": bound.cpu().numpy().reshape(len(q), -1),
            "self_collision_cost": self_cost.cpu().numpy().reshape(len(q), -1),
            "world_penetration_m": scene_cost.cpu().numpy().reshape(len(q), -1),
            "spheres": state.robot_spheres.cpu().numpy().reshape(len(q), -1, 4),
        }
        if hand_min_x is not None:
            diagnostics.update(hand_min_x_m=hand_min_x, required_hand_min_x_m=self._hand_min_x_m)
        if (diagnostics["self_collision_cost"] > 0).any():
            config = self.checker.self_collision_cost.config.self_collision_kin_config
            diagnostics["self_collision_pairs"] = config.collision_pairs.cpu().numpy()
            diagnostics["self_collision_padding"] = config.sphere_padding.cpu().numpy()
        return diagnostics

    def violations(self, positions, *, diagnostics=None, times=None) -> str:
        """What the fine check rejects along a path: where, which links, how deep, and the deepest sphere.

        The sphere centre is in the torso frame, so it tells the table (its
        height) from the target or a neighbour (their place) without the scene.
        """
        diagnostics = self.constraint_diagnostics(positions) if diagnostics is None else diagnostics
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
            parts.append(f"bound_joints={[n for n, v in zip(self.arm, bound, strict=False) if v > 0]}")
        if diagnostics["hand_behind_torso"][bad].any():
            parts.append("hand behind the torso")
        from .path_diagnostics import summarize

        detail = summarize(
            positions, times, diagnostics, self.arm, self.collision_sphere_links, getattr(self, "_arm_bounds", None)
        )
        parts.append(f"constraint_details={detail['types']}")
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

    def _measured_start(self, feedback):
        """The measured planning arm as a start state, a hair past a model bound taken back to it.

        The planning model's bounds sit inside the motion controller's (right
        wrist pitch 0.5236 vs 0.558), and a loaded arm settles past its target:
        live 2026-11-06 a grasp planned to the bound measured 0.5272 at contact
        and 0.526 carrying. Planning from that rejected every start, the drop
        recovery's path home included (DROP_RECOVERY_FAILED). A joint at most
        START_BOUND_SLACK_RAD past a bound starts just inside it instead -- well
        within the executor's start tolerance -- and is logged; one further
        out is left as measured for _require_valid_start to reject. The bounds
        themselves are unchanged: no plan is allowed to go there.
        """
        q = np.asarray(feedback.vector(self.arm), dtype=float)
        bounds = getattr(self, "_arm_bounds", None)
        if bounds is None:  # a planner assembled without solvers (offline tools, tests)
            return q
        lower, upper = bounds
        past = np.maximum(lower - q, q - upper)
        if past.max() <= 0.0 or past.max() > START_BOUND_SLACK_RAD:
            return q
        inside = np.clip(q, lower + START_BOUND_INSET_RAD, upper - START_BOUND_INSET_RAD)
        start = np.where(past > 0.0, inside, q)
        moved = {name: round(float(value), 4) for name, value in zip(self.arm, past, strict=True) if value > 0.0}
        print(f"planning start taken back inside the model bounds (rad past): {moved}", file=sys.stderr, flush=True)
        return start

    def _require_valid_start(self, positions):
        diagnostics = self.constraint_diagnostics(np.asarray(positions)[None])
        if diagnostics["valid"].all():
            return
        bound = diagnostics["bound_cost"][0]
        world = diagnostics["world_penetration_m"][0]
        raise RuntimeError(
            "Start state violates collision or joint constraints: "
            f"bound_joints={[n for n, v in zip(self.arm, bound, strict=False) if v > 0]}, "
            f"self_squared_cost={diagnostics['self_collision_cost'].max():.8g}, "
            f"world_links={sorted({n for n, v in zip(self.collision_sphere_links, world, strict=False) if v > 0})}, "
            f"world_max_m={world.max():.8g}, "
            f"hand_behind_torso={bool(diagnostics['hand_behind_torso'][0])}"
        )

    def _solve_motion(self, goal_pose, state, *, deadline=None, seed_config=None, iters=25, graph_seed=None):
        """One bounded IK/TrajOpt pass in latency mode; native retries otherwise."""
        self.graph_seed_trace = trace = dict(
            mode=graph_seed,
            requested_seed=graph_seed or "native",
            actual_seed="native",
            requested_over_height_m=self._over_m if graph_seed == "over" else None,
            actual_over_height_m=None,
        )
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
        if graph_seed in ("escape", "over") and count > 1:
            result = self._escape_seeded_solve(goal_pose, state, seed_config, iters, trace, check_deadline)
            if result is not None:
                return result
            if trace.get("escape_ok"):
                trace.update(
                    actual_seed="native",
                    actual_over_height_m=None,
                    fallback_reason="seeded_optimizer_returned_no_result",
                )
        elif graph_seed in ("escape", "over"):
            trace["fallback_reason"] = "parallel_seeds_required"
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
        poses = self.motion.compute_kinematics(JointState.from_position(q, joint_names=list(self.arm))).tool_poses
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
            trace.update(actual_seed="native", actual_over_height_m=None, fallback_reason="escape_cartesian_ik_failed")
            return None
        trace.update(actual_seed="escape", actual_over_height_m=None)
        path = np.concatenate((chain, end[None]))
        if self._approach_graph_seed == "over":
            trace["requested_over_height_m"] = self._over_m
            # The approach pose raised _over_m, walked back down into it.
            lift = self._over_m * np.asarray(self.lift_up, dtype=float) / np.linalg.norm(self.lift_up)
            above, ok = self._cartesian_paths(end[None], lift[None], self._over_cartesian_steps, check_deadline)
            trace["over_ok"] = bool(ok[0])
            if ok[0]:
                path = np.concatenate((chain, above[0][::-1]))
                trace.update(actual_seed="over", actual_over_height_m=self._over_m)
            else:
                trace["fallback_reason"] = "over_cartesian_ik_failed"
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
            JointState.from_position(torch.tensor(q, device="cuda"), joint_names=list(self.arm)),
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
        q = feedback.vector(self.arm)
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
        if not set(self.arm) <= set(names):
            return False
        columns = [names.index(name) for name in self.arm]
        limits = self.motion.kinematics.config.kinematics_config.joint_limits
        for field in ("velocity", "acceleration", "jerk"):
            value, bound = getattr(trajectory, field, None), getattr(limits, field, None)
            if value is None or bound is None:
                return False
            value = value.detach().cpu().numpy().reshape(-1, len(names))[:count, columns]
            bound = bound.detach().cpu().numpy().reshape(2, -1)[:, : len(self.arm)]
            if not np.isfinite(value).all() or np.any(value < bound[0] - 1e-6) or np.any(value > bound[1] + 1e-6):
                return False
        return True

    def _linear_grasp_segment(self, start, goal, check_deadline):
        """The grasp segment along the palm line without TrajOpt, or (None, reason); see _linear_grasp.

        Local IK (seeded at the approach end) gives the contact state. Interior
        waypoints start on the joint line and take _linear_grasp_corrections
        damped least-squares steps toward evenly spaced palm poses on the
        Cartesian line (goal orientation); a cubic spline through them is timed
        by a minimum-jerk profile, stretched until the sampled velocity,
        acceleration and jerk are within _linear_grasp_limit_fraction of the
        solver's limits. Call with the grasp phase's collision mask (hand links
        disabled for the coarse IK); the fine check covers every link.
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
        if not self.valid_positions(positions).all():
            return None, "fine"
        check_deadline()
        torch.cuda.synchronize()
        return TimedTrajectory(times, positions, self.arm), None

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
            raw_result,
            self.motion.trajopt_solver.config.interpolation_dt,
            locked_positions=self.locked_positions,
            arm=self.arm,
        )
        last = result.interpolated_last_tstep
        if last is not None:
            n = int(last.reshape(-1)[0].item()) + 1
            if not 2 <= n <= len(out.times):
                raise RuntimeError(f"Incomplete interpolated {phase} trajectory")
            if n < len(out.times):
                out = TimedTrajectory(out.times[:n], out.positions[:n], out.joint_names)
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
            goal = self._palm_poses(grasp.positions[-1][None])[0]
            goal[:3, 3] += self.vertical_lift_m * up
            with self._carried_target():
                lift, reason = self._linear_grasp_segment(grasp.positions[-1], goal, check_deadline)
        outcome = "used" if lift is not None else f"hold_{reason}"
        report[outcome] = report.get(outcome, 0) + 1
        if lift is None:
            return self._hold_lift(grasp), "hold"
        # Exact shared endpoint with the grasp segment (grasp_request checks it).
        positions = lift.positions.copy()
        positions[0] = grasp.positions[-1]
        return TimedTrajectory(lift.times, positions, lift.joint_names), "vertical"

    def carry_lift(self, grasp, requested_m, check_deadline):
        previous = self.vertical_lift_m
        try:
            for height in dict.fromkeys([requested_m, min(requested_m, 0.10), min(requested_m, 0.05)]):
                check_deadline()
                self.vertical_lift_m = height
                lift, method = self._vertical_lift(grasp, check_deadline)
                if method == "vertical":
                    return lift, height
        finally:
            self.vertical_lift_m = previous
        raise RuntimeError("no straight carry lift at the requested or fallback heights")

    @contextmanager
    def _carried_target(self):
        """The fine checker without the target mesh, held exclusively until the block ends."""
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

    def plan_pose(self, feedback, torso_from_palm, now, *, deadline=None, seed_config=None):
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("Finite monotonic planning deadline required")
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        if not self.snapshot_matches(feedback, now):
            raise RuntimeError("Feedback or collision scene changed/stale; rebuild snapshot")
        q = self._measured_start(feedback)
        self._require_valid_start(q)
        goal = rigid(torso_from_palm)
        pose, state = self._goal_pose(goal[None]), self._state(q)
        result = self._solve_motion(pose, state, deadline=deadline, seed_config=seed_config)
        result = self._retry_approach(pose, state, result, deadline=deadline, seed_config=seed_config)
        if result is None or result.success is None or not bool(result.success.all()):
            raise RuntimeError(f"cuRobo pose planning failed: {self._motion_failure(result)}")
        trajectory = TimedTrajectory.from_curobo(
            result,
            self.motion.trajopt_solver.config.interpolation_dt,
            locked_positions=self.locked_positions,
            arm=self.arm,
        )
        diagnostics = self.constraint_diagnostics(trajectory.positions)
        if not diagnostics["valid"].all():
            detail = self.violations(trajectory.positions, diagnostics=diagnostics, times=trajectory.times)
            archive = self._save_pose_failure(feedback, q, goal, trajectory, diagnostics, result, seed_config)
            raise RuntimeError(f"Interpolated trajectory violates collision or joint constraints: {detail}; {archive}")
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Planning deadline exceeded")
        return trajectory

    def _save_pose_failure(self, feedback, start, goal, trajectory, diagnostics, result, seed_config):
        """A recording failure must preserve the original planning rejection."""
        from .path_diagnostics import save_failure, summarize

        try:
            planning_scene, validation_scene = self._diagnostic_scenes
            validation_scene = (
                validation_scene.materialize() if isinstance(validation_scene, LazyScene) else validation_scene
            )
            optimized = getattr(result, "js_solution", None)
            summary = summarize(
                trajectory.positions,
                trajectory.times,
                diagnostics,
                self.arm,
                self.collision_sphere_links,
                self._arm_bounds,
            )
            record = dict(
                positions=trajectory.positions,
                times=trajectory.times,
                joint_names=tuple(self.arm),
                diagnostics=diagnostics,
                summary=summary,
                sphere_links=self.collision_sphere_links,
                measured_start=feedback.vector(self.arm),
                planning_start=start,
                feedback_received_at=feedback.received_at,
                target_palm=goal,
                attached_spheres=getattr(self, "_diagnostic_attachment", self.attached),
                planning_scene=planning_scene,
                validation_scene=dict(validation_scene),
                scene_received_at=self.scene_received_at,
                models=self._diagnostic_models,
                bounds=self._arm_bounds,
                optimized_positions=None if optimized is None else optimized.position.detach().cpu().numpy(),
                optimized_joint_names=None if optimized is None else optimized.joint_names,
                seed_config=None if seed_config is None else seed_config.detach().cpu().numpy(),
                planning_profile=self.planning_profile,
                approach_seed_policy=self.approach_seed_policy,
                approach_over_height_m=self._over_m,
                hand_min_x_m=self._hand_min_x_m,
                mesh_query_distance=self.mesh_query_distance,
            )
            path = save_failure(record)
            print(f"pose path rejected: {summary}; archive={path}", file=sys.stderr, flush=True)
            return f"archive={path}"
        except Exception as exc:  # noqa: BLE001 - evidence cannot change the safety decision
            message = f"failure archive unavailable: {type(exc).__name__}: {exc}"
            print(message, file=sys.stderr, flush=True)
            return message

    #: Peak joint speed (rad/s) and check spacing (rad) of a joint-space path home.
    HOME_JOINT_SPEED = 0.5
    HOME_CHECK_STEP = 0.02

    def _checked_joint_path(self, start, goal):
        """Straight joint-space path, every 2 cm-rad sample fine-checked; None if any collides.

        Cosine time scaling (zero end velocities) at the TrajOpt interpolation dt.
        """
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        span = float(np.max(np.abs(goal - start)))
        samples = np.linspace(start, goal, max(2, int(np.ceil(span / self.HOME_CHECK_STEP)) + 1))
        if not np.asarray(self.valid_positions(samples), bool).all():
            return None
        dt = float(self.motion.trajopt_solver.config.interpolation_dt)
        duration = max(dt, span * np.pi / (2 * self.HOME_JOINT_SPEED))
        times = np.arange(0.0, duration + dt / 2, dt)
        if len(times) < 2:
            times = np.array([0.0, dt])
        times[-1] = max(times[-1], duration)
        fraction = (1 - np.cos(np.pi * np.clip(times / times[-1], 0, 1))) / 2
        return TimedTrajectory(times - times[0], start + fraction[:, None] * (goal - start), tuple(self.arm))

    @staticmethod
    def _joined(*paths):
        """Concatenate timed paths that meet end to start, keeping time increasing."""
        times, positions = [paths[0].times], [paths[0].positions]
        for path in paths[1:]:
            if not np.allclose(positions[-1][-1], path.positions[0], atol=1e-6):
                raise ValueError("paths do not meet")
            step = path.times[1] - path.times[0]
            times.append(times[-1][-1] + step + path.times[1:] - path.times[1])
            positions.append(path.positions[1:])
        return TimedTrajectory(np.concatenate(times), np.vstack(positions), paths[0].joint_names)

    def _route(self, start, pieces, home):
        """``pieces`` driven in order from ``start`` to ``home``, re-checked now; None if any part collides.

        Pieces that do not meet are bridged by checked straight joint paths.
        """
        joined, at = [], np.asarray(start, float)
        for piece in [*pieces, None]:
            target = np.asarray(home, float) if piece is None else piece.positions[0]
            if not np.allclose(at, target, atol=1e-6):
                bridge = self._checked_joint_path(at, target)
                if bridge is None:
                    return None
                joined.append(bridge)
            if piece is None:
                break
            if not np.asarray(self.valid_positions(piece.positions), bool).all():
                return None
            joined.append(piece)
            at = piece.positions[-1]
        return self._joined(*joined) if joined else None

    def _path_home(self, start, home, now, feedback, *, deadline, back=None, return_path=None):
        """A checked path to the exact ``home`` joints without a joint-space TrajOpt goal.

        The warmed solvers keep their captured pose-goal shape, so a cspace
        solve cannot run on them. In order: straight; back the way it came
        (``back``: a validated path ending near ``start``, reversed) and then
        straight, or then along ``return_path`` (pick's own moves reversed,
        from the handover to home); ``return_path`` from here; a pose plan to
        home's palm pose, then straight to the exact joints. Every part is
        fine-checked in the current scene.
        """
        direct = self._checked_joint_path(start, home)
        if direct is not None:
            return direct
        routes = []
        if back is not None:
            reverse = TimedTrajectory(back.times[-1] - back.times[::-1], back.positions[::-1], back.joint_names)
            routes.append([reverse])
            if return_path is not None:
                routes.append([reverse, return_path])
        if return_path is not None:
            routes.append([return_path])
        for pieces in routes:
            path = self._route(start, pieces, home)
            if path is not None:
                return path
        goal = self._palm_poses(np.asarray(home, float)[None])[0]
        moved = SimpleNamespace(
            received_at=feedback.received_at,
            healthy=feedback.healthy,
            vector=lambda names: np.asarray(start, float) if list(names) == list(self.arm) else feedback.vector(names),
            hand=feedback.hand,
        )
        reach = self.plan_pose(moved, goal, now, deadline=deadline)
        finish = self._checked_joint_path(reach.positions[-1], home)
        if finish is None:
            raise RuntimeError("no checked path to home")
        return self._joined(reach, finish)

    def plan_place(
        self, feedback, targets, home, now, *, check_deadline, deadline, opening=None, return_path=None, max_rounds=3
    ):
        """Transfer, descent, upward retreat and checked joint-space home for one of ``targets``.

        ``targets`` are (above, release) palm poses in preference order. The
        start is checked once; one goalset IK over the "above" poses picks a
        reachable one (chunks of max_goalset), whose seed then drives a single
        TrajOpt solve. A candidate whose descent, opening, retreat or home
        fails is dropped and the goalset solved again, at most ``max_rounds``
        times. ``opening`` (palm-frame spheres the opening fingers sweep) is
        checked once at the release pose; the open, empty hand then retreats
        and goes home with nothing attached.
        """
        if not self.snapshot_matches(feedback, now):
            raise RuntimeError("Feedback or collision scene changed/stale; rebuild snapshot")
        if not self.carrying:
            raise RuntimeError("placement planning needs the carried object attached")
        q = self._measured_start(feedback)
        self._require_valid_start(q)
        aboves = np.stack([rigid(above) for above, _ in targets])
        releases = np.stack([rigid(release) for _, release in targets])
        remaining, failures, stages = list(range(len(targets))), [], {}
        rounds = 0
        while remaining and rounds < max_rounds:
            check_deadline()
            chunk = remaining[: self.max_goalset]
            selection = self.motion.ik_solver.solve_pose(self._goal_pose(aboves[chunk]), self._state(q), return_seeds=1)
            check_deadline()
            if selection is None or selection.success is None or not bool(selection.success.any()):
                failures.append(f"no reachable opening pose among {len(chunk)} candidates")
                remaining = remaining[len(chunk) :]
                continue
            rounds += 1
            index = chunk[self._input_goal_index(int(selection.goalset_index.reshape(-1)[0].item()), len(chunk))]
            seed = selection.solution[:, 0:1].clone()
            stage = "transfer"
            try:
                transfer = self.plan_pose(feedback, aboves[index], now, deadline=deadline, seed_config=seed)
                stage = "descend"
                descend, reason = self._linear_grasp_segment(transfer.positions[-1], releases[index], check_deadline)
                if descend is None:
                    raise RuntimeError(f"descent: {reason}")
                stage = "opening"
                if opening is not None and len(opening):
                    with self._attachment(opening):
                        if not self.valid_positions(descend.positions[-1:]).all():
                            raise RuntimeError(f"opening hand: {self.violations(descend.positions[-1:])}")
                with self._without_placed_object():
                    stage = "retreat"
                    retreat, reason = self._linear_grasp_segment(descend.positions[-1], aboves[index], check_deadline)
                    if retreat is None:
                        raise RuntimeError(f"retreat: {reason}")
                    stage = "home"
                    # The retreat ends where the transfer ended: the transfer,
                    # checked with the closed hand and object, leads back.
                    home_path = self._path_home(
                        retreat.positions[-1],
                        home,
                        now,
                        feedback,
                        deadline=deadline,
                        back=transfer,
                        return_path=return_path,
                    )
                check_deadline()
                self.place_report = {
                    "candidate": index,
                    "rounds": rounds,
                    "candidates": len(targets),
                    "failed_stages": dict(stages),
                }
                return transfer, descend, retreat, home_path
            except (TimeoutError, PlannerFault):
                raise
            except RuntimeError as exc:
                if device_fault(exc):
                    raise PlannerFault(str(exc)) from exc
                failures.append(f"candidate {index} at {stage}: {exc}")
                stages[stage] = stages.get(stage, 0) + 1
                remaining.remove(index)
        self.place_report = {"rounds": rounds, "candidates": len(targets), "failed_stages": dict(stages)}
        raise PlacementSearchExhausted(f"no complete placement path after {rounds} rounds {stages}: {failures[-3:]}")

    def plan_home(self, feedback, home, *, check_deadline, deadline, return_path=None):
        """Checked joint-space path from the measured arm to home, the hand empty.

        For an object dropped while carried. A placement planner detaches its
        object sphere first, as for its planned post-release paths.
        """
        check_deadline()
        q = self._measured_start(feedback)
        try:
            with self._without_placed_object():
                self._require_valid_start(q)
                now = max(feedback.received_at, self.scene_received_at)
                path = self._path_home(q, home, now, feedback, deadline=deadline, return_path=return_path)
        except RuntimeError as exc:
            if device_fault(exc):
                raise PlannerFault(str(exc)) from exc
            raise
        check_deadline()
        return path

    @contextmanager
    def _attachment(self, spheres):
        """Temporarily attach ``spheres`` instead of the carried cover (empty: nothing)."""
        if not self.carrying:
            yield
            return
        self._write_attachment(np.asarray(spheres, dtype=float).reshape(-1, 4))
        try:
            yield
        finally:
            self._write_attachment(self.attached)

    def _without_placed_object(self):
        """After the release the hand is open and empty: exactly the warmed model."""
        return self._attachment(np.empty((0, 4)))

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
        q = self._measured_start(feedback)
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

        self.approach_seed_traces = []
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
        hand = list(self.hand_links)

        def solve(transforms, start, phase, seed_config=None):
            check_deadline()
            goal_pose, state = self._goal_pose(transforms), self._state(start)
            iters = self._approach_iters if phase == "approach" else 25
            # Experimental options are passed only when set.
            options = {} if iters == 25 else dict(iters=iters)
            if phase == "approach" and self._approach_graph_seed is not None:
                options["graph_seed"] = self._approach_graph_seed
            try:
                result = self._solve_motion(goal_pose, state, deadline=deadline, seed_config=seed_config, **options)
            finally:
                if phase == "approach":
                    self.approach_seed_traces.append(dict(getattr(self, "graph_seed_trace", {})))
            rescued = self._rescuable(result)
            if not rescued and phase == "approach" and (abs(approach_offset) > 0.05 or self._short_approach_retry):
                try:
                    result = self._retry_approach(goal_pose, state, result, deadline=deadline, seed_config=seed_config)
                finally:
                    if phase == "approach":
                        trace = dict(getattr(self, "graph_seed_trace", {}))
                        if trace != self.approach_seed_traces[-1]:
                            self.approach_seed_traces.append(trace)
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
        index = None
        try:
            q = self._measured_start(feedback)
            self._require_valid_start(q)
            key = (q.tobytes(), goals.tobytes(), selection_goals.tobytes())
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
            active_error = sys.exc_info()[1]
            cleanup_errors = []
            try:
                self.motion.enable_link_collision(hand)
            except Exception as exc:  # noqa: BLE001 - preserve the planning fault already propagating
                cleanup_errors.append(exc)
            try:
                self.motion.update_tool_pose_criteria({"R_palm": ToolPoseCriteria()})
            except Exception as exc:  # noqa: BLE001 - make both cleanup steps best effort
                cleanup_errors.append(exc)
            cleanup_fault = next(
                (exc for exc in cleanup_errors if isinstance(exc, RuntimeError) and device_fault(exc)), None
            )
            if cleanup_fault is not None and not isinstance(active_error, PlannerFault):
                raise PlannerFault(f"{type(cleanup_fault).__name__}: {cleanup_fault}") from cleanup_fault
            if cleanup_errors and active_error is None:
                raise cleanup_errors[0]
