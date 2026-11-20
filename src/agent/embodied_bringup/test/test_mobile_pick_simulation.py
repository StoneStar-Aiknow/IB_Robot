"""The production-pipeline pick E2E on the sim platforms (R9, revision 2).

One test per platform (MuJoCo, Gazebo), gated by
``IBROBOT_REQUIRE_SIM_PLATFORMS`` (acceptance sets ``mujoco,gazebo``; unset,
the sim E2E skips with a reason). Maintainer decision 2026-11-02: the pick
E2E is ACCEPTED ON MUJOCO; on Gazebo it still runs when required and its
failure is reported as a known issue ONLY at the approved pick-failure
verdicts (the retry-budget class with the listed preparation codes or the
jaw-at-open wedge evidence; the true-contact transport slip with paired
gripper-frame evidence) — everything else stays a hard failure.

Sequence (design R9 first half, R1/R2/R6/R7): stage + launch, readiness,
staging drive + the /cmd_vel attribution census, mode, planning scene (the
chassis, applied and read back), the synchronized pre-arm GT snapshot in the
pan frame, detection warm-up, the frozen rest height over a continuous
one-second window, the pick through the production client, then — on EVERY
path, before any verdict is raised (review r2 #2) — the recording endpoint,
the reset/runtime-status/clock invariants, the R6 audit of every executed
arm motion (explicit no-motion evidence when the arm never moved), the
judge (successful picks), the scene removal and the cleanup aggregation.
The verdict (fail or xfail) is raised only after the ExitStack has run
every registered cleanup, so a cleanup failure stays a hard failure even
behind a provisional xfail (review r2 #3).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from ibrobot_msgs.action import PickObject
from ibrobot_msgs.msg import RuntimeStatus
from robot_runtime_sim.testing.entity_ground_truth import open_entity_ground_truth
from robot_runtime_sim.testing.sim_ground_truth import RUN_MARKER_ENV, terminate_marked_processes
from sim_models.scene_compiler import get_scene_layout, mobile_manipulation_spec
from test.mobile_grasp_support import (
    JUDGE_GRASP_REGION_M,
    JUDGE_MAX_SAMPLE_GAP_SEC,
    JudgeInput,
    _package_mesh_path,
    _stl_vertices,
    apply_transform,
    chassis_box,
    clock_stall_verdict,
    collision_object_message,
    first_lift_stamp,
    frozen_rest_height,
    judge_hold,
    matrix_from_xyz_rpy,
    motion_audit_window,
    motion_sample_verdict,
    pick_feedback_candidate_rounds,
    point_in_frame,
    quat_in_frame,
    quat_roll_pitch_deg,
    runtime_status_verdict,
    transform_matrix,
    transform_scene_box,
)

PLATFORMS = ("mujoco", "gazebo")
REQUIRED_SIM_PLATFORMS_ENV = "IBROBOT_REQUIRE_SIM_PLATFORMS"
RUN_DIR_ENV = "IBROBOT_E2E_RUN_DIR"
WORLD_NAME = "demo"
SCENE_NAME = "nav_grasp_room"
OBJECT_NAME = "grasp_object"
ROBOT_MODEL_NAME = "lekiwi_robot"
EXPECTED_CMD_VEL_PUBLISHERS = {"behavior_server", "collision_monitor"}
LAUNCH_READY_TIMEOUT = 1200.0  # the ceiling; a crashed launch fails fast via the log scan
NAV_RESULT_TIMEOUT = 300.0
PICK_TIMEOUT_S = 230.0

#: The pick E2E's Gazebo known issues (maintainer decision 2026-11-02: the
#: navigate-then-grasp E2E is ACCEPTED ON MUJOCO; on Gazebo the run still
#: happens and its failure is reported as a known issue — never a silent
#: skip). The two measured modes, from the 2026-11-02 acceptance sequence
#: (6 runs, 0 passes, runs/e2e/gazebo-182552..-184202):
GAZEBO_KNOWN_ISSUES = (
    "Gazebo known issues (2026-11-02 acceptance, 0/6 runs): "
    "(1) retry budget exhausted on marginal candidates across the base stop-pose band "
    "(0.177-0.267 m from the table edge) — including deep-descend candidates whose "
    "moving-jaw CoACD hull wedges the table slab at the fully-open end so the close "
    "goal never moves the jaw (e2e/gazebo-180245, -183150); "
    "(2) a verified-contact close whose grip does not survive the transport swing: the "
    "gripper-effort-1.0 hold slips and the cube is never carried (e2e/gazebo-174538, "
    "-183757, -184202). Root cause of (1)'s wedge: the production FK tabletop filter "
    "checks the raw-STL jaw mesh while Gazebo's physics uses the fatter CoACD hulls."
)

#: The verifier's jaw-at-open wedge evidence (manipulation_service
#: grasp_verification, the 2026-11-02 open-jaw fix): a close goal whose jaw
#: stays at/near the fully-open end holds nothing. The executor propagates
#: the verifier's evidence into the PickObject failure message, so the
#: GRASP_UNCERTAIN budget class is granted ONLY on this specific evidence —
#: a missing/stale-sensor "No usable grasp-verification evidence" or a
#: generic fused-uncertain close stays a hard failure (review r2 #8).
JAW_AT_OPEN_WEDGE_EVIDENCE = "jaw at/near fully open after close goal"

#: The judge rejections that carry PAIRED gripper-frame evidence of a
#: physical slip (review r2 #6/#8): the object observed outside the grasp
#: region or moving within it. Absent or unsynchronized pairs are NOT slip
#: evidence — they are a separate hard failure.
SLIP_REJECTIONS = ("outside the gripper's grasp region", "slipping")


def pick_e2e_gate(platform: str, required: list[str]) -> tuple[str, str]:
    """The pick E2E's platform gate (maintainer decision 2026-11-02).

    A platform outside ``required`` skips (unchanged). MuJoCo and Gazebo
    both RUN when required — MuJoCo is the accepted platform, and Gazebo's
    failure is attributed to the known issues ONLY at the point of an
    approved pick-failure verdict; every other failure stays a hard
    failure."""
    if platform not in required:
        return "skip", f"set {REQUIRED_SIM_PLATFORMS_ENV}={','.join(PLATFORMS)} to enable the sim pick E2E"
    return "run", ""


#: The pick-failure codes whose retry-budget exhaustion the Gazebo known
#: issues cover (review r1 item 9): the marginal-candidate preparation codes
#: measured across the base stop-pose band and the jaw-at-open
#: GRASP_UNCERTAIN of the table-wedge mode (which additionally requires the
#: wedge evidence in the message, review r2 #8).
GAZEBO_KNOWN_BUDGET_CODES = frozenset(
    {
        "IK_ORIENTATION_REJECTED",
        "CONTACT_REALIGN_FAILED",
        "CONTACT_COMPENSATION_FAILED",
        "GRASP_UNCERTAIN",
    }
)


def _gazebo_budget_exhaustion_verdict(pick_result) -> str | None:
    """The approved Gazebo pick-failure class (1): the retry budget
    exhausted with one of the listed preparation codes / the jaw-at-open
    wedge UNCERTAIN (the wedge needs its specific evidence in the message —
    review r2 #8). Returns the xfail reason, or ``None`` when the failure is
    NOT the approved class (the caller hard-fails)."""
    if pick_result is None or pick_result.success:
        return None
    code = str(pick_result.error_code or "")
    message = str(pick_result.message or "")
    if "grasp retry budget exhausted" not in message or code not in GAZEBO_KNOWN_BUDGET_CODES:
        return None
    if code == "GRASP_UNCERTAIN" and JAW_AT_OPEN_WEDGE_EVIDENCE not in message:
        return None
    return f"{GAZEBO_KNOWN_ISSUES} measured: {code} — {message}"


def _verification_is_true_contact(pick_result) -> bool:
    """True contact is the verification ENUM (review r2 #8):
    ``verification_status == VERIFICATION_SUCCESS`` — the uint8 enum, never
    a score; the confidence is recorded separately."""
    return int(pick_result.verification_status) == PickObject.Result.VERIFICATION_SUCCESS


def _gazebo_slip_verdict(pick_result, judge_detail: str) -> str | None:
    """The approved Gazebo pick-failure class (2): a true-contact grasp
    (the enum) whose transport physically slipped — the judge rejected it
    WITH paired gripper-frame evidence showing the object outside the jaws
    (review r2 #6). Absent or unsynchronized evidence is a hard failure, so
    is any other judge rejection. Returns the xfail reason, or ``None``."""
    if pick_result is None or not pick_result.success:
        return None
    if not _verification_is_true_contact(pick_result):
        return None
    if not any(marker in str(judge_detail) for marker in SLIP_REJECTIONS):
        return None
    return (
        f"{GAZEBO_KNOWN_ISSUES} measured: the pick verified "
        f"(status {int(pick_result.verification_status)}, "
        f"confidence {float(pick_result.verification_confidence):.2f}) but the transport "
        f"physically slipped: {judge_detail}"
    )


# ---------------------------------------------------------------------------
# The platform gate: required platforms FAIL when they do not run
# ---------------------------------------------------------------------------


def _required_platforms() -> list[str]:
    return [entry.strip() for entry in os.environ.get(REQUIRED_SIM_PLATFORMS_ENV, "").split(",") if entry.strip()]


# ---------------------------------------------------------------------------
# Staging, launch and teardown
# ---------------------------------------------------------------------------


def _stage(tmp_path: Path, sim_platform: str) -> tuple[Path, Path]:
    from test.pick_sim_staging import stage_pick_sim_config

    return stage_pick_sim_config(
        tmp_path / "staged",
        sim_platform=sim_platform,
        e2e_package_root=Path(__file__).resolve().parents[1],
        execution_store_path=tmp_path / "executions.sqlite3",
    )


def _run_dir(sim_platform: str, tmp_path: Path) -> Path:
    override = os.environ.get(RUN_DIR_ENV, "").strip()
    if not override and _required_platforms():
        pytest.fail(
            f"{RUN_DIR_ENV} must be set for acceptance E2E runs "
            f"(e.g. /tmp/lekiwi-grasp-evidence/runs/e2e/{sim_platform}-<timestamp>) — "
            "the launch log, feedback, metrics and GT evidence must land on a stable path"
        )
    run_dir = (Path(override) if override else tmp_path / "run") / sim_platform
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class _Stack:
    """The launched hybrid stack and its process group."""

    def __init__(self, process: subprocess.Popen, log_path: Path) -> None:
        self.process = process
        self.log_path = log_path

    def alive(self) -> bool:
        return self.process.poll() is None

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="ignore").splitlines()[-lines:])
        except OSError:
            return ""


def _launch_stack(staged_overlay: Path, sim_platform: str, env: dict, log_path: Path) -> _Stack:
    command = [
        "ros2",
        "launch",
        "embodied_bringup",
        "embodied_pipeline.launch.py",
        f"config_path:={staged_overlay}",
        "nav_stage:=hybrid",
        "use_sim:=true",
        f"sim_platform:={sim_platform}",
        "entry_mode:=hermes",
        "authorize_motion:=true",
    ]
    if sim_platform == "gazebo":
        env.setdefault("IBROBOT_GAZEBO_HEADLESS", "1")
    else:
        env.setdefault("IBROBOT_MUJOCO_HEADLESS", "1")
    env.setdefault("RCUTILS_COLORIZED_OUTPUT", "0")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return _Stack(process, log_path)


def _stop_stack(stack: _Stack, env: dict) -> None:
    """Teardown with INDEPENDENT steps (review r2 #3): the process-group
    termination, the run-marker sweep and the ros2 daemon stop each run in
    its own try — a failure in one never skips the others — and every
    failure surfaces together (the caller aggregates them)."""
    errors: list[str] = []
    try:
        if stack.process.poll() is None:
            os.killpg(os.getpgid(stack.process.pid), signal.SIGINT)
            for _ in range(30):
                if stack.process.poll() is not None:
                    break
                time.sleep(1.0)
            if stack.process.poll() is None:
                os.killpg(os.getpgid(stack.process.pid), signal.SIGKILL)
        stack.process.wait(timeout=30)
    except Exception as exc:  # noqa: BLE001 - one failing step must not skip the rest
        errors.append(f"process-group termination: {type(exc).__name__}: {exc}")
    try:
        # ros2 launch's grandchildren (the simulator, nav2 nodes, runtime
        # nodes) escape the launch's process group — sweep every process
        # carrying this run's marker (the same rule the navigation tests
        # use).
        if RUN_MARKER_ENV in env:
            terminate_marked_processes({RUN_MARKER_ENV: env[RUN_MARKER_ENV]})
    except Exception as exc:  # noqa: BLE001
        errors.append(f"run-marker sweep: {type(exc).__name__}: {exc}")
    try:
        subprocess.run(["ros2", "daemon", "stop"], env=env, capture_output=True, timeout=60, check=False)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ros2 daemon stop: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("the stack teardown steps failed: " + "; ".join(errors))


def _join_pick_worker(pick_thread: threading.Thread, client, timeout_s: float = 5.0) -> None:
    """Bounded shutdown of the pick client's worker (review r2 #3): join
    with a bound and VERIFY termination — a watchdog-stranded worker is a
    hard failure, never a leaked thread over a destroyed context."""
    pick_thread.join(timeout=timeout_s)
    if pick_thread.is_alive():
        raise TimeoutError(f"the pick worker thread did not terminate within {timeout_s:.0f} s")
    client.destroy_node()


def _remove_scene_objects(apply_client, scene_client, scene_objects, call, metrics: dict) -> None:
    """The scene cleanup registered right after the scene is applied (review
    r2 #3): remove the objects and VERIFY the removal on every exit path —
    a failure raises and surfaces as a hard failure through the cleanup
    aggregation, including behind a provisional xfail."""
    removal_request = ApplyPlanningScene.Request()
    removal_scene = PlanningScene()
    removal_scene.is_diff = True
    for message in scene_objects:
        removal = CollisionObject()
        removal.id = message.id
        removal.header.frame_id = message.header.frame_id
        removal.operation = CollisionObject.REMOVE
        removal_scene.world.collision_objects.append(removal)
    removal_request.scene = removal_scene
    removal = call(apply_client, removal_request, 30.0)
    scene_query = call(scene_client, GetPlanningScene.Request(), 30.0)
    remaining = [world.id for world in scene_query.scene.world.collision_objects] if scene_query else ["no scene"]
    removal_verified = bool(removal and removal.success) and not any(
        remaining.count(scene_box.id) for scene_box in scene_objects
    )
    metrics["planning_scene_removal"] = {"verified": removal_verified, "remaining": remaining}
    if not removal_verified:
        raise RuntimeError(f"the planning-scene removal was not verified: {remaining}")


# ---------------------------------------------------------------------------
# The probe node and its helpers
# ---------------------------------------------------------------------------


class _ProbeNode(Node):
    """The E2E's ROS face: readiness clients, TF, the sim clock, the arm
    sample buffer of the R6 validity audit, and the runtime-status /
    clock-stream records of the pick-window invariants (timestamped —
    review r2 #4: (sim, lifecycle, latch) status observations plus their
    wall receipts, and (steady-wall, sim) clock receipts)."""

    #: The arm-only MoveIt model's joints (the audit sends exactly these).
    ARM_JOINTS = ("1", "2", "3", "4", "5", "6")

    def __init__(self, platform: str) -> None:
        super().__init__("mobile_pick_e2e")
        from ibrobot_msgs.srv import GetSkillGatewayStatus, MoveToPose, PlanGrasp, SetRuntimeMode

        self.platform = platform
        self.robot_description: str | None = None
        self.wrist_frames: list = []
        self.clock_sim = 0.0
        # the R6 arm-sample audit buffer: (sim stamp, [positions of ARM_JOINTS])
        # from the recording start (the audit window trims to the actual first
        # arm motion and checks endpoint coverage against the markers)
        self.recording = False
        self.arm_samples: list[tuple[float, list[float]]] = []
        # the pick-window invariants: every RuntimeStatus observation
        # (sim, lifecycle, stop latch) with its wall receipt AND the message's
        # own source stamp (the Gateway's admission checks BOTH ages — review
        # r2 finding 9's instrumentation), and every /clock message's
        # (steady-wall, sim) pair
        self.runtime_statuses: list[tuple[float, str, bool]] = []
        self.status_receipts: list[tuple[float, float, float]] = []
        self.clock_pairs: list[tuple[float, float]] = []
        self.status_client = self.create_client(GetSkillGatewayStatus, "/embodied/get_skill_gateway_status")
        self.mode_client = self.create_client(SetRuntimeMode, "/runtime/set_mode")
        self.move_client = self.create_client(MoveToPose, "/motion/move_to_pose")
        self.planner_client = self.create_client(PlanGrasp, "/grasp_planner/plan_grasp")
        self.apply_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        from ibrobot_msgs.srv import GroundingDetect

        self.detect_client = self.create_client(GroundingDetect, "/perception/grasp/grounding_detect")
        self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        description_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, "/robot_description", self._on_description, description_qos)
        self.create_subscription(JointState, "/joint_states", self._on_arm_joints, 50)
        self.create_subscription(RuntimeStatus, "/runtime_status", self._on_runtime_status, 50)
        from rosgraph_msgs.msg import Clock

        self.create_subscription(
            Clock,
            "/clock",
            self._on_clock,
            200,
        )

    def _on_clock(self, message) -> None:
        self.clock_sim = message.clock.sec + message.clock.nanosec / 1e9
        self.clock_pairs.append((time.monotonic(), self.clock_sim))

    def _on_runtime_status(self, message) -> None:
        source_stamp = message.stamp.sec + message.stamp.nanosec / 1e9
        self.runtime_statuses.append((self.clock_sim, str(message.lifecycle), bool(message.stop_latched)))
        self.status_receipts.append((time.monotonic(), self.clock_sim, source_stamp))

    def _on_arm_joints(self, msg: JointState) -> None:
        if not self.recording:
            return
        positions = dict(zip(msg.name, (float(value) for value in msg.position), strict=False))
        if not set(self.ARM_JOINTS) <= positions.keys():
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # a frozen sim clock makes the joint-state broadcaster re-emit the
        # SAME stamp — a duplicate carries no new state and would fail the
        # audit's strictly-increasing rule, so only the first of consecutive
        # identical stamps is kept. A stamp that runs BACKWARDS still fails
        # the audit.
        if self.arm_samples and stamp == self.arm_samples[-1][0]:
            return
        self.arm_samples.append((stamp, [positions[joint] for joint in self.ARM_JOINTS]))

    def _on_description(self, msg: String) -> None:
        # multiple publishers share /robot_description (the runtime RSP's full
        # LeKiwi model and MoveIt's arm-only one, both TRANSIENT_LOCAL). The
        # RUNTIME description is selected deterministically: its ros2_control
        # block carries this platform's sim system plugin — and, on MuJoCo,
        # the mujoco_model parameter.
        if self._is_runtime_description(msg.data):
            self.robot_description = msg.data

    def _is_runtime_description(self, data: str) -> bool:
        plugin = "MujocoSystemInterface" if self.platform == "mujoco" else "GazeboSimSystem"
        return (
            plugin in data
            and "<ros2_control" in data
            and ("mujoco_model" in data if self.platform == "mujoco" else True)
        )

    def gateway_ready(self) -> bool:
        if not self.status_client.wait_for_service(timeout_sec=2.0):
            return False
        from ibrobot_msgs.srv import GetSkillGatewayStatus

        request = GetSkillGatewayStatus.Request()
        request.schema_version = 1
        request.task_id = "pick-sim-e2e"
        future = self.status_client.call_async(request)
        deadline = time.monotonic() + 15.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.2)
        if not future.done():
            return False
        status = future.result()
        capabilities = {capability.name: capability.ready for capability in status.capabilities}
        return bool(status.control_plane_ready and capabilities.get("pick_object"))

    def wait_for(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.3)
        return False

    def action_present(self, action: str) -> bool:
        try:
            from rclpy.action import get_action_names_and_types

            return any(name == action for name, _ in get_action_names_and_types(self))
        except Exception:
            return False


def call_service(probe: _ProbeNode, client, request, timeout: float, stack: _Stack):
    if not client.wait_for_service(timeout_sec=15.0):
        return None
    future = client.call_async(request)
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        if not stack.alive():
            return None
        time.sleep(0.1)
    return future.result() if future.done() else None


def publishers_of(probe: _ProbeNode, topic: str) -> set[str]:
    try:
        return {info.node_name for info in probe.get_publishers_info_by_topic(topic)}
    except Exception:
        return set()


def navigate_to_staging(probe: _ProbeNode, stack: _Stack) -> tuple[int, set[str]]:
    # the staging goal is the layout's single definition (D1), read through
    # the validated task spec — with the test-only Nav2 inflation 0.12 it
    # stops the base ~0.24 m from the cube (the real robot's validated pick
    # band is ~0.18-0.25 m out).
    staging = mobile_manipulation_spec(get_scene_layout(SCENE_NAME))["staging_pose"]
    deadline = time.monotonic() + NAV_RESULT_TIMEOUT
    goal = NavigateToPose.Goal()
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = "map"
    pose_stamped.header.stamp = probe.get_clock().now().to_msg()
    pose_stamped.pose.position.x = float(staging["x"])
    pose_stamped.pose.position.y = float(staging["y"])
    pose_stamped.pose.orientation.w = 1.0
    goal.pose = pose_stamped
    cmd_vel_publishers: set[str] = set()
    handle = None
    for attempt in range(1, 6):
        handle_future = probe.nav_client.send_goal_async(goal)
        while not handle_future.done() and time.monotonic() < deadline:
            if not stack.alive():
                pytest.fail("the launch exited during navigation")
            cmd_vel_publishers |= publishers_of(probe, "/cmd_vel")
            time.sleep(0.1)
        if not handle_future.done():
            print(f"[pick-sim] nav retry {attempt}: goal handle not ready", flush=True)
        elif not handle_future.result().accepted:
            print(f"[pick-sim] nav retry {attempt}: goal rejected", flush=True)
        else:
            handle = handle_future.result()
            break
        time.sleep(15.0)
    if handle is None:
        pytest.fail("staging goal not accepted after retries")
    result_future = handle.get_result_async()
    while not result_future.done() and time.monotonic() < deadline:
        if not stack.alive():
            pytest.fail("the launch exited during navigation")
        cmd_vel_publishers |= publishers_of(probe, "/cmd_vel")
        time.sleep(0.1)
    if not result_future.done():
        handle.cancel_goal_async()
        pytest.fail("navigation did not complete within the window (host contention)")
    response = result_future.result()
    return int(response.status), cmd_vel_publishers


def spec_prompt() -> str:
    return mobile_manipulation_spec(get_scene_layout(SCENE_NAME))["object_prompt"]


def spec_success() -> dict:
    return mobile_manipulation_spec(get_scene_layout(SCENE_NAME))["success"]


def build_scene_objects(probe: _ProbeNode, robot_description: str) -> list[CollisionObject]:
    """The chassis only (review r1 item 4): the base_link collision mesh's
    AABB with the mesh's FULL collision origin applied, cut at the arm
    mount's underside DERIVED from the model, then fully transformed into
    the arm ``base`` frame. The table is NOT added to MoveIt's planning
    scene (maintainer decision 2026-11-02) — the E2E judges the table
    physically through the R7 pre-lift window."""
    chassis_mesh_file = None
    origin_xyz = [0.0, 0.0, 0.0]
    origin_rpy = [0.0, 0.0, 0.0]
    for link in ElementTree.fromstring(robot_description).iter("link"):
        if link.get("name") == "base_link":
            collision = link.find("collision")
            if collision is not None:
                chassis_mesh_file = collision.find("geometry").find("mesh").get("filename")
                origin = collision.find("origin")
                origin_xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
                origin_rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
    if not chassis_mesh_file:
        pytest.fail("no base_link collision mesh in /robot_description")
    for scheme in ("file://", "file:"):
        if chassis_mesh_file.startswith(scheme):
            chassis_mesh_file = chassis_mesh_file[len(scheme) :]
            break
    chassis_vertices = apply_transform(
        np.vstack(_stl_vertices(_package_mesh_path(chassis_mesh_file))),
        matrix_from_xyz_rpy(origin_xyz, origin_rpy),
    )
    # the arm mount's underside, derived from the model's own mount TF —
    # never a hard-coded height
    mount = probe.tf_buffer.lookup_transform("base_link", "base", Time(), timeout=Duration(seconds=5.0))
    box_baselink = chassis_box(chassis_vertices, float(mount.transform.translation.z))
    # fully into the arm base frame: MoveIt transforms the stamped pose, but
    # the box's geometry must already BE in the frame it is stamped in
    baselink_to_base = transform_matrix(
        probe.tf_buffer.lookup_transform("base", "base_link", Time(), timeout=Duration(seconds=5.0))
    )
    return [collision_object_message(transform_scene_box(box_baselink, baselink_to_base, "base"), CollisionObject.ADD)]


def _verify_scene_readback(scene_objects, scene_state, transform_for) -> dict:
    """Review r1 item 5: verify the read-back chassis object (id, frame,
    size, pose) against the applied box BEFORE the pick starts. Returns the
    readback metrics entry; raises AssertionError when the object is missing
    or does not match."""
    readback = (
        next((obj for obj in scene_state.scene.world.collision_objects if obj.id == "chassis"), None)
        if scene_state is not None
        else None
    )
    if readback is None:
        raise AssertionError("the chassis collision object is not in the read-back planning scene")
    applied_message = scene_objects[0]
    applied_frame = str(applied_message.header.frame_id)
    readback_frame = str(readback.header.frame_id)
    expected_centre = np.array(
        [
            applied_message.primitive_poses[0].position.x,
            applied_message.primitive_poses[0].position.y,
            applied_message.primitive_poses[0].position.z,
        ]
    )
    if readback_frame != applied_frame:
        # MoveIt answered in its planning frame: transform the applied
        # centre into that frame before the comparison
        expected_centre = apply_transform(expected_centre.reshape(1, 3), transform_for(readback_frame, applied_frame))[
            0
        ]
    readback_centre = np.array([readback.pose.position.x, readback.pose.position.y, readback.pose.position.z])
    readback_dims = np.array(list(readback.primitives[0].dimensions)) if readback.primitives else None
    applied_dims = np.array(list(applied_message.primitives[0].dimensions))
    pose_ok = bool(np.linalg.norm(readback_centre - expected_centre) <= 1e-3)
    size_ok = readback_dims is not None and bool(np.all(np.abs(readback_dims - applied_dims) <= 1e-3))
    entry = {
        "id": readback.id,
        "frame": readback_frame,
        "size_ok": size_ok,
        "pose_ok": pose_ok,
    }
    if not (pose_ok and size_ok):
        raise AssertionError(
            f"the read-back chassis object does not match the applied box: frame={readback_frame} "
            f"centre={readback_centre.tolist()} expected={expected_centre.tolist()} "
            f"dims={readback_dims.tolist() if readback_dims is not None else None} expected_dims={applied_dims.tolist()}"
        )
    return entry


def _pre_arm_snapshot(probe: _ProbeNode, object_stream, base_stream) -> dict:
    """Review r2 #11: synchronized GT frozen IMMEDIATELY after the navigation
    success and BEFORE any arm motion, transformed into the arm's pan
    (``base``) frame — the staging reach band's evidence. The base and cube
    poses are taken at the nearest common stamps (within the streams' 0.1 s
    pairing bound); the final poses are recorded separately after the pick."""
    layout = get_scene_layout(SCENE_NAME)
    table_entry = next(entry for entry in layout["static_geometry"] if entry["name"] == "table")

    def _synchronized() -> bool:
        obj, base = object_stream.latest(), base_stream.latest()
        return obj is not None and base is not None and abs(obj.stamp_sec - base.stamp_sec) <= JUDGE_MAX_SAMPLE_GAP_SEC

    if not probe.wait_for(_synchronized, 30.0):
        pytest.fail("no synchronized base/cube ground-truth samples for the pre-arm snapshot")
    obj, base = object_stream.latest(), base_stream.latest()
    # the cube in the arm's pan frame: the synchronized base GT pose plus
    # the model's static base_link -> base mount TF
    mount = probe.tf_buffer.lookup_transform("base", "base_link", Time(), timeout=Duration(seconds=5.0))
    cube_in_baselink = np.array(point_in_frame(base.position, base.quat_xyzw, obj.position))
    cube_in_pan = apply_transform(cube_in_baselink.reshape(1, 3), transform_matrix(mount))[0]
    edge_x = float(table_entry["pose"][0]) - float(table_entry["size"][0]) / 2
    return {
        "stamp_s": round(obj.stamp_sec, 2),
        "pair_dt_s": round(abs(obj.stamp_sec - base.stamp_sec), 3),
        "base_map": [round(float(v), 4) for v in base.position],
        "cube_map": [round(float(v), 4) for v in obj.position],
        "table_edge_distance_m": round(edge_x - float(base.position[0]), 4),
        "cube_pan_x_m": round(float(cube_in_pan[0]), 4),
        "cube_pan_y_m": round(float(cube_in_pan[1]), 4),
        "cube_pan_radius_m": round(float(np.hypot(cube_in_pan[0], cube_in_pan[1])), 4),
    }


def judge_from_ground_truth(
    object_stream, gripper_stream, base_stream, window_start_s: float, end_s: float, z_rest: float, hold_point
) -> tuple[bool, str, dict]:
    """The retained D6 judge over the collected GT streams (R7 windows).

    ``window_start_s`` (the PickObject result's sim stamp) starts the hold;
    ``end_s`` (the recording endpoint's sim stamp) bounds it. The pairing
    PRESERVES every observation (review r2 #6): every object sample after
    the result is paired with the nearest gripper sample within the 0.1 s
    bound and passed to the judge — in or out of the grasp region — so the
    judge's region and gap rules do the rejecting. Missing or
    unsynchronized evidence is a SEPARATE hard failure, never an approved
    slip class; the base stream must cover the window and its roll/pitch is
    judged. ``z_rest`` is the FROZEN rest height; ``hold_point`` is the
    model TCP in the gripper link frame (review r2 #7)."""
    layout = get_scene_layout(SCENE_NAME)
    spec = mobile_manipulation_spec(layout)
    table_entry = next(entry for entry in layout["static_geometry"] if entry["name"] == spec["table"])
    object_samples = [
        (sample.stamp_sec, tuple(sample.position), tuple(sample.quat_xyzw)) for sample in object_stream.samples()
    ]
    gripper_samples = [
        (sample.stamp_sec, tuple(sample.position), tuple(sample.quat_xyzw)) for sample in gripper_stream.samples()
    ]
    base_samples = [
        (sample.stamp_sec, tuple(sample.position), tuple(sample.quat_xyzw)) for sample in base_stream.samples()
    ]
    if not object_samples or not gripper_samples or not base_samples:
        return False, "no ground-truth samples collected", {}
    stream_freshness: list[str] = []
    for name, samples in (("object", object_samples), ("gripper", gripper_samples), ("base", base_samples)):
        if samples[-1][0] < end_s - 0.5:
            stream_freshness.append(f"{name}: last sample at {samples[-1][0]:.2f} s")
    if stream_freshness:
        return False, "missing ground-truth coverage over the window (" + ", ".join(stream_freshness) + ")", {}
    if len(base_samples) < 2:
        return False, "missing base ground-truth coverage over the window (fewer than two samples)", {}
    paired: list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]] = []
    unpaired = 0
    for stamp, position, quat in object_samples:
        if stamp < window_start_s:
            continue  # the hold is judged after the result, not in transport
        if stamp > end_s + JUDGE_MAX_SAMPLE_GAP_SEC:
            # ... and never past the RECORDING ENDPOINT: the judge runs minutes
            # after the endpoint (the R6 audit's validity calls) while the
            # streams keep appending — pairing the post-endpoint samples would
            # judge a hold window the design never defined, and a load-induced
            # stream hole there (measured: 0.517 s at sim 145.3, 27 s past the
            # endpoint, r2fix-final-mujoco) would fail runs on evidence outside
            # the hold. The hold is hold_s AFTER THE RESULT, bounded by the
            # endpoint marker (+ the streams' own 0.1 s pairing bound).
            continue
        nearest = min(gripper_samples, key=lambda sample: abs(sample[0] - stamp))
        if abs(nearest[0] - stamp) > JUDGE_MAX_SAMPLE_GAP_SEC:
            unpaired += 1
            continue
        # the object measured in the gripper LINK frame against the MODEL
        # TCP, with the orientation carried in the gripper frame too
        paired.append(
            (
                stamp,
                point_in_frame(nearest[1], nearest[2], position),
                quat_in_frame(nearest[2], quat),
            )
        )
    if not paired:
        return (
            False,
            "unsynchronized ground-truth evidence: no object sample paired with a gripper sample within "
            f"0.1 s after the result ({unpaired} unpaired object samples)",
            {"paired_samples": 0, "unpaired_object_samples": unpaired},
        )
    # the base stillness spans the attempt "from the observe step to THE END"
    # — bounded by the recording endpoint for the same reason as the pairing:
    # the streams keep appending while the judge waits behind the audit
    base_speeds = [
        (
            following[0],
            float(np.linalg.norm(np.array(following[1][:2]) - np.array(previous[1][:2])))
            / max(following[0] - previous[0], 1e-3),
        )
        for previous, following in zip(base_samples, base_samples[1:], strict=False)
        if following[0] <= end_s + JUDGE_MAX_SAMPLE_GAP_SEC
    ]
    base_roll_pitch = [(stamp, quat_roll_pitch_deg(quat)) for stamp, _position, quat in base_samples]
    data = JudgeInput(
        z_rest=float(z_rest),
        hold_seconds=spec["success"]["hold_s"],
        lift_m=spec["success"]["lift_m"],
        max_object_rotation_in_gripper_deg=spec["success"]["max_object_rotation_in_gripper_deg"],
        object_samples=object_samples,
        object_in_gripper_samples=paired,
        hold_point_in_gripper=tuple(float(v) for v in hold_point),
        object_table_region=(
            (
                float(table_entry["pose"][0]) - float(table_entry["size"][0]) / 2,
                float(table_entry["pose"][0]) + float(table_entry["size"][0]) / 2,
            ),
            (
                float(table_entry["pose"][1]) - float(table_entry["size"][1]) / 2,
                float(table_entry["pose"][1]) + float(table_entry["size"][1]) / 2,
            ),
        ),
        base_planar_speeds=base_speeds,
        base_roll_pitch_deg=base_roll_pitch,
    )
    ok, detail = judge_hold(data)
    lift = first_lift_stamp(object_samples, float(z_rest), detect_m=spec["success"]["lift_m"])
    # the base tilt is RECORD ONLY (maintainer decision 2026-11-03): the max
    # |roll|/|pitch| over the HOLD WINDOW goes to metrics.json — the judge
    # applies no tilt bound (the spec judges the base by planar speed alone)
    if paired:
        hold_lo, hold_hi = paired[0][0] - 1e-9, paired[-1][0] + 1e-9
        hold_tilts = [max(abs(r), abs(p)) for stamp, (r, p) in base_roll_pitch if hold_lo <= stamp <= hold_hi]
    else:
        hold_tilts = [0.0]
    info = {
        "hold_start_s": round(paired[0][0], 2) if paired else None,
        "hold_end_s": round(paired[-1][0], 2) if paired else None,
        "hold_window_s": round(paired[-1][0] - paired[0][0], 2) if paired else 0.0,
        "first_lift_s": round(lift, 2) if lift is not None else None,
        "paired_samples": len(paired),
        "unpaired_object_samples": unpaired,
        "object_samples": len(object_samples),
        "base_samples": len(base_samples),
        "max_base_tilt_deg_hold_window": round(max(hold_tilts), 3),
    }
    print(
        f"[pick-sim] judge ok={ok} detail={detail} paired={len(paired)} unpaired={unpaired} "
        f"object={len(object_samples)} base={len(base_samples)}",
        flush=True,
    )
    return ok, detail, info


def audit_arm_samples(
    probe: _ProbeNode, stack: _Stack, window
) -> tuple[bool, str, int, list[tuple[float, list[float]]]]:
    """The R6 passive audit over the supplied window (``motion_audit_window``):
    every sample checked against the applied chassis scene. The primary
    move_group's model is the ARM-ONLY one, so each validity request carries
    exactly the arm joint names/positions sampled at that instant. A failed
    or timed-out validity call leaves the sample UNCHECKED, which fails the
    run like an invalid one; every early return is the same 4-tuple shape
    (review r2 #5)."""
    stamps: list[float] = []
    results: list[bool] = []
    invalid_samples: list[tuple[float, list[float]]] = []
    for stamp, positions in window:
        request = GetStateValidity.Request()
        request.group_name = "arm"
        request.robot_state.joint_state.name = list(_ProbeNode.ARM_JOINTS)
        request.robot_state.joint_state.position = positions
        future = probe.validity_client.call_async(request)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            if not stack.alive():
                return False, "the launch exited during the validity audit", len(stamps), invalid_samples
            time.sleep(0.005)
        if not future.done() or future.result() is None:
            return (
                False,
                f"sample t={stamp:.2f}: the validity call never completed (unchecked sample)",
                len(stamps),
                invalid_samples,
            )
        stamps.append(stamp)
        results.append(bool(future.result().valid))
        if not future.result().valid:
            invalid_samples.append((round(stamp, 3), [round(v, 4) for v in positions]))
    ok, detail = motion_sample_verdict(stamps, results)
    return ok, detail, len(stamps), invalid_samples


# ---------------------------------------------------------------------------
# The E2E
# ---------------------------------------------------------------------------


def _record_verdict_entry(verdict: list[tuple[str, str]], kind: str, message: str) -> None:
    """Record a deferred verdict entry (review r2 #2): the FIRST recorded
    verdict stands, except that a HARD failure overrides a provisional
    xfail — an invariant, audit or cleanup failure can never be hidden
    behind an approved failure code."""
    if not verdict:
        verdict.append((kind, message))
    elif kind == "fail" and verdict[0][0] == "xfail":
        verdict[0] = (kind, message)


def _raise_deferred_verdict(verdict: list[tuple[str, str]], cleanup_errors: list[str]) -> None:
    """Raise the deferred verdict AFTER the cleanup aggregation (review r2
    #2/#3): cleanup failures are hard failures that win even over a
    provisional xfail; only then the recorded verdict itself."""
    if cleanup_errors:
        pytest.fail("the E2E's cleanup steps failed:\n- " + "\n- ".join(cleanup_errors))
    if verdict:
        kind, message = verdict[0]
        if kind == "xfail":
            pytest.xfail(message)
        pytest.fail(message)


@pytest.mark.parametrize("sim_platform", PLATFORMS)
def test_pick_end_to_end(sim_platform: str, tmp_path: Path) -> None:
    decision, gate_reason = pick_e2e_gate(sim_platform, _required_platforms())
    if decision == "skip":
        pytest.skip(gate_reason)

    from ibrobot_msgs.srv import SetRuntimeMode

    run_dir = _run_dir(sim_platform, tmp_path)
    overlay, _catalog = _stage(tmp_path, sim_platform)

    env = dict(os.environ)
    partition = f"pick_sim_{uuid.uuid4().hex[:8]}"
    run_marker = f"pick_e2e_{uuid.uuid4().hex[:12]}"
    env["ROS_LOCALHOST_ONLY"] = "1"
    env.setdefault("ROS_DOMAIN_ID", os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "0"))
    env["IGN_PARTITION"] = env["GZ_PARTITION"] = partition
    env[RUN_MARKER_ENV] = run_marker
    env["RCUTILS_COLORIZED_OUTPUT"] = "0"

    metrics: dict = {"platform": sim_platform}
    # review r1 item 10 + r2 #3: every resource's cleanup is registered
    # RIGHT AFTER the resource is created (an ExitStack; LIFO), a failing
    # cleanup step never skips the rest, and the aggregated failures are
    # hard failures raised AFTER the stack unwinds — before the deferred
    # verdict, so they win even over a provisional xfail.
    cleanup_errors: list[str] = []

    def _register_cleanup(exit_stack: contextlib.ExitStack, action) -> None:
        label = getattr(action, "__name__", str(action))

        def _run_cleanup():
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - one failing step must not skip the rest
                cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")

        exit_stack.callback(_run_cleanup)

    # the deferred verdict (review r2 #2): every post-pick decision point
    # RECORDS its verdict instead of raising, so the invariants, the audit,
    # the judge and the whole cleanup complete first; a HARD failure
    # overrides a provisional xfail, never the other way around.
    verdict: list[tuple[str, str]] = []

    def _record_verdict(kind: str, message: str) -> None:
        _record_verdict_entry(verdict, kind, message)

    with contextlib.ExitStack() as exit_stack:
        # the run's metrics are written when the stack unwinds — on EVERY
        # exit path (failure, xfail, interruption, success)
        _register_cleanup(
            exit_stack,
            lambda: (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8"),
        )
        stack = _launch_stack(overlay, sim_platform, env, run_dir / "launch" / "launch.log")
        _register_cleanup(exit_stack, lambda: _stop_stack(stack, env))

        def _on_sigterm(_signum, _frame):
            # A SIGTERM must not orphan the launched stack: sweep every
            # process carrying this run's marker now, then stop the test
            # (the exit stack still unwinds and sweeps again — idempotent).
            terminate_marked_processes({RUN_MARKER_ENV: run_marker})
            raise KeyboardInterrupt

        previous_term = signal.signal(signal.SIGTERM, _on_sigterm)
        _register_cleanup(exit_stack, lambda: signal.signal(signal.SIGTERM, previous_term))
        rclpy.init()
        probe = _ProbeNode(sim_platform)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(probe)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        def _shutdown_probe():
            probe.recording = False
            executor.shutdown()
            spin_thread.join(timeout=5.0)
            if spin_thread.is_alive():
                raise TimeoutError("the probe's spin thread did not terminate within 5 s")
            probe.destroy_node()
            rclpy.shutdown()

        _register_cleanup(exit_stack, _shutdown_probe)

        def crashed_nodes_now() -> list[str]:
            try:
                return [
                    line.strip()[:160]
                    for line in stack.log_path.read_text(errors="ignore").splitlines()
                    if "process has died" in line
                ]
            except OSError:
                return []

        def fail_on_crash(what: str) -> None:
            """A launched node dying is a fail-fast event — report with the
            launch-log excerpt."""
            deaths = crashed_nodes_now()
            if deaths:
                pytest.fail(f"{what}: a launched node died during startup; deaths:\n" + "\n".join(deaths[:6]))

        # readiness: the Gateway control plane + the pick_object capability
        if not probe.wait_for(probe.gateway_ready, LAUNCH_READY_TIMEOUT):
            fail_on_crash("the Gateway readiness")
            pytest.fail(f"the Gateway never became ready; log tail:\n{stack.log_tail()}")
        # readiness: the production services, polled
        from ibrobot_msgs.srv import ComputeIk

        ik_worker_ik = probe.create_client(ComputeIk, "/ik_worker_0/motion/compute_ik")
        for client, what in (
            (ik_worker_ik, "the IK worker pool (/ik_worker_0/motion/compute_ik)"),
            (probe.planner_client, "the grasp planner"),
            (probe.move_client, "the motion move_to_pose service"),
            (probe.apply_client, "the planning-scene service"),
            (probe.scene_client, "the planning-scene query service"),
            (probe.validity_client, "the state-validity service (the R6 audit)"),
        ):
            ready = False
            deadline = time.monotonic() + LAUNCH_READY_TIMEOUT
            while time.monotonic() < deadline:
                fail_on_crash(what)
                if client.wait_for_service(timeout_sec=5.0):
                    ready = True
                    break
            if not ready:
                fail_on_crash(what)
                pytest.fail(
                    f"{what}: service never appeared within {LAUNCH_READY_TIMEOUT:.0f}s; log tail:\n{stack.log_tail()}"
                )
        if not probe.wait_for(lambda: probe.action_present("/move_action"), LAUNCH_READY_TIMEOUT):
            fail_on_crash("move_group's /move_action")
            pytest.fail("move_group's /move_action never appeared")

        # staging drive + the /cmd_vel attribution census
        nav_status, cmd_vel_publishers = navigate_to_staging(probe, stack)
        metrics["navigate_to_staging"] = {"status": nav_status, "cmd_vel_publishers": sorted(cmd_vel_publishers)}
        if nav_status != 4:
            pytest.fail(f"navigation did not SUCCEED: status {nav_status}")
        unexpected = cmd_vel_publishers - EXPECTED_CMD_VEL_PUBLISHERS
        if unexpected:
            pytest.fail(f"the /cmd_vel attribution audit found unexpected publishers: {sorted(unexpected)}")

        # trajectory mode
        mode_request = SetRuntimeMode.Request()
        mode_request.mode = "trajectory"
        mode = call_service(probe, probe.mode_client, mode_request, 30.0, stack)
        if not (mode and mode.success):
            pytest.fail("SetRuntimeMode('trajectory') was not confirmed")

        # the planning scene: the chassis (base_link), applied — the table
        # is deliberately absent (maintainer decision 2026-11-02)
        if not probe.wait_for(lambda: probe.robot_description is not None, 120.0):
            pytest.fail("no RUNTIME /robot_description received within 120s")
        scene_objects = build_scene_objects(probe, probe.robot_description)
        apply_request = ApplyPlanningScene.Request()
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects = list(scene_objects)
        apply_request.scene = planning_scene
        applied = call_service(probe, probe.apply_client, apply_request, 30.0, stack)
        # review r1 item 5: a scene that failed to apply must NEVER reach the
        # PickObject goal. Assert the apply, read the chassis object back and
        # verify it, and register the removal+verification cleanup right
        # after the apply (review r2 #3 — it runs on every exit path).
        if not (applied and applied.success):
            metrics["planning_scene"] = {"applied": False}
            pytest.fail(f"ApplyPlanningScene was not successful for the chassis scene: applied={applied}")

        def _transform_for(target_frame: str, source_frame: str) -> np.ndarray:
            return transform_matrix(
                probe.tf_buffer.lookup_transform(target_frame, source_frame, Time(), timeout=Duration(seconds=5.0))
            )

        scene_state = call_service(probe, probe.scene_client, GetPlanningScene.Request(), 30.0, stack)
        # MoveIt returns the scene in its planning frame; the dump pins the
        # box poses where the planner actually checks them
        if scene_state is not None:
            metrics["planning_scene_dump"] = [
                {
                    "id": obj.id,
                    "frame": obj.header.frame_id,
                    "pose": [
                        round(obj.pose.position.x, 4),
                        round(obj.pose.position.y, 4),
                        round(obj.pose.position.z, 4),
                    ],
                    "dims": [round(v, 4) for v in obj.primitives[0].dimensions] if obj.primitives else [],
                }
                for obj in scene_state.scene.world.collision_objects
            ]
        try:
            metrics["planning_scene"] = {
                "applied": True,
                "readback": _verify_scene_readback(scene_objects, scene_state, _transform_for),
            }
        except AssertionError as exc:
            metrics["planning_scene"] = {"applied": True, "readback": f"failed: {exc}"}
            pytest.fail(str(exc))

        def _scene_call(client, request, timeout: float):
            return call_service(probe, client, request, timeout, stack)

        _register_cleanup(
            exit_stack,
            lambda: _remove_scene_objects(probe.apply_client, probe.scene_client, scene_objects, _scene_call, metrics),
        )
        map_base = probe.tf_buffer.lookup_transform("map", "base", Time(), timeout=Duration(seconds=5.0))
        metrics["map_base_tf"] = {
            "z": round(map_base.transform.translation.z, 4),
            "x": round(map_base.transform.translation.x, 4),
            "y": round(map_base.transform.translation.y, 4),
        }

        # the grasp region's centre (review r2 #7): the MODEL TCP — the
        # static gripper -> tcp origin from the URDF, read through TF —
        # never an empirical constant
        tcp_in_gripper = probe.tf_buffer.lookup_transform("gripper", "tcp", Time(), timeout=Duration(seconds=5.0))
        hold_point = (
            float(tcp_in_gripper.transform.translation.x),
            float(tcp_in_gripper.transform.translation.y),
            float(tcp_in_gripper.transform.translation.z),
        )
        metrics["grasp_region"] = {
            "tcp_in_gripper": [round(v, 4) for v in hold_point],
            "region_radius_m": JUDGE_GRASP_REGION_M,
            "pre_lift_boundary_m": spec_success()["lift_m"],
        }

        # MuJoCo link streams run through the FK of the model file the
        # provider actually loaded; the MuJoCo BASE pose comes from the
        # simulator's floating-base odometry
        mjcf_path = None
        if sim_platform == "mujoco":
            for param in ElementTree.fromstring(probe.robot_description).findall(".//ros2_control/hardware/param"):
                if param.get("name") == "mujoco_model":
                    mjcf_path = (param.text or "").strip()
            if not mjcf_path:
                pytest.fail("no mujoco_model hardware parameter in /robot_description")
        from robot_runtime_sim.testing.entity_ground_truth import MujocoFloatingBaseGroundTruthStream

        object_stream = open_entity_ground_truth(
            sim_platform, world=WORLD_NAME, scene=SCENE_NAME, entity=OBJECT_NAME, env=env
        )
        gripper_stream = open_entity_ground_truth(
            sim_platform,
            world=WORLD_NAME,
            scene=SCENE_NAME,
            entity=ROBOT_MODEL_NAME,
            link="gripper",
            env=env,
            mjcf_path=mjcf_path,
        )
        if sim_platform == "mujoco":
            base_stream = MujocoFloatingBaseGroundTruthStream(body="base_link", env=env)
        else:
            base_stream = open_entity_ground_truth(
                sim_platform, world=WORLD_NAME, scene=SCENE_NAME, entity=ROBOT_MODEL_NAME, env=env
            )
        streams = [object_stream, gripper_stream, base_stream]
        for stream in streams:
            stream.start()
            _register_cleanup(exit_stack, stream.close)  # right after the resource is created
        if not probe.wait_for(lambda: object_stream.latest() is not None, 30.0):
            pytest.fail("no ground-truth object samples at the observe pose")

        # review r2 #11: the staging reach band's evidence — synchronized GT
        # in the pan frame, frozen BEFORE any arm motion
        metrics["pre_arm_snapshot"] = _pre_arm_snapshot(probe, object_stream, base_stream)

        # detection warm-up (readiness, not a PickObject goal): the pick's
        # first plan_grasp runs the grounding model inside the planner's
        # detection call, and the model load (tens of seconds) would blow
        # the planner's detection budget on a cold model. One real
        # GroundingDetect call from the observe pose loads the model and
        # validates the prompt end-to-end.
        colour_sub = probe.create_subscription(
            Image, "/camera/wrist/image_raw", lambda m: probe.wrist_frames.append(m), 5
        )
        if not probe.wait_for(lambda: bool(probe.wrist_frames), 20.0):
            pytest.fail("no wrist frame for the detection warm-up")
        probe.destroy_subscription(colour_sub)
        colour = probe.wrist_frames[-1]
        from ibrobot_msgs.srv import GroundingDetect

        detect_request = GroundingDetect.Request()
        detect_request.image = colour
        detect_request.text_prompt = spec_prompt()
        detect_request.box_threshold = 0.35
        detect_request.text_threshold = 0.25
        detect_future = probe.detect_client.call_async(detect_request)
        deadline = time.monotonic() + 300.0
        while not detect_future.done() and time.monotonic() < deadline:
            if not stack.alive():
                pytest.fail("the launch exited during the detection warm-up")
            time.sleep(0.1)
        detection = detect_future.result() if detect_future.done() else None
        if detection is None:
            pytest.fail("the detection warm-up call timed out after 300s")
        metrics["detection_warmup"] = {
            "success": bool(detection.success),
            "n_detections": len(detection.detections.detections),
        }
        print(f"[pick-sim] detection warm-up: {metrics['detection_warmup']}", flush=True)
        del detection, colour, detect_request

        # the frozen rest height (review r1 item 7, tightened r2 #7): the
        # median of a CONTINUOUS, DENSE 1 s window of fresh samples, taken
        # BEFORE any arm motion — a sparse or gapped window freezes nothing
        # (the wait keeps polling and fails explicitly)
        if not probe.wait_for(
            lambda: frozen_rest_height([(s.stamp_sec, s.position[2]) for s in object_stream.samples()]) is not None,
            30.0,
        ):
            pytest.fail("no continuous 1 s window of ground-truth object samples to freeze the rest height from")
        z_rest = frozen_rest_height([(sample.stamp_sec, sample.position[2]) for sample in object_stream.samples()])
        if z_rest is None:
            pytest.fail("no ground-truth object samples to freeze the rest height from")
        metrics["z_rest_frozen"] = round(z_rest, 4)

        # the supervised-direct pick through the production client; the R6
        # audit buffer starts HERE (the executor's first arm motion is the
        # observe move) with its INDEPENDENT start marker
        from manipulation_execution.pick.action_client import PickActionClient, build_parser

        client_args = build_parser().parse_args(["--prompt", spec_prompt(), "--timeout-s", str(PICK_TIMEOUT_S)])
        client = PickActionClient(client_args, use_sim_time=True)
        captured = io.StringIO()
        pick_error: list[str] = []
        pick_result_holder: list = []

        def run_pick() -> None:
            try:
                pick_result_holder.append(client.execute())
            except Exception as exc:  # noqa: BLE001 - the failure IS the data
                pick_error.append(f"{type(exc).__name__}: {exc}")

        probe.arm_samples.clear()
        recording_start_sim = probe.clock_sim
        probe.recording = True
        # the pick-window invariants start HERE: every RuntimeStatus
        # observation and every /clock (steady-wall, sim) pair recorded from
        # the PickObject goal to the endpoint is judged
        status_window_mark = len(probe.runtime_statuses)
        clock_window_mark = len(probe.clock_pairs)
        pick_sim_start, pick_wall_start = probe.clock_sim, time.monotonic()
        pick_thread = threading.Thread(target=run_pick, daemon=True)
        # review r2 #3: the worker's shutdown is registered right after its
        # creation — a bounded join with termination verification, and the
        # client node is always destroyed
        _register_cleanup(exit_stack, lambda: _join_pick_worker(pick_thread, client))
        with contextlib.redirect_stdout(captured):
            pick_thread.start()
            pick_deadline = time.monotonic() + PICK_TIMEOUT_S + 90.0
            while pick_thread.is_alive() and time.monotonic() < pick_deadline:
                if not stack.alive():
                    pick_thread.join(timeout=10.0)
                    pytest.fail("the launch exited during the pick")
                time.sleep(0.1)
            if pick_thread.is_alive():
                pick_thread.join(timeout=15.0)
                pytest.fail("the pick client outlived its watchdog window")
        pick_wall_s = time.monotonic() - pick_wall_start
        pick_sim_s = max(probe.clock_sim - pick_sim_start, 0.0)
        # the sim stamp of the PickObject result: the R7 hold is judged from
        # here (the transport swing before it is not part of the hold)
        result_stamp = probe.clock_sim

        feedback_lines = [line for line in captured.getvalue().splitlines() if line.startswith("PICK_FEEDBACK")]
        candidate_rounds = pick_feedback_candidate_rounds(feedback_lines)
        pick_error_text = pick_error[0] if pick_error else None
        pick_result = pick_result_holder[0] if pick_result_holder else None
        Path(run_dir / "pick_feedback.txt").write_text("\n".join(feedback_lines), encoding="utf-8")
        metrics["pick"] = {
            "result": None
            if pick_result is None
            else {
                "success": bool(pick_result.success),
                "error_code": pick_result.error_code,
                "candidate_index": int(pick_result.candidate_index),
                "attempts": int(pick_result.attempts),
                "verification_status": int(pick_result.verification_status),
                "verification_confidence": round(float(pick_result.verification_confidence), 4),
                "message": pick_result.message,
            },
            "error": pick_error_text,
            "candidate_rounds_from_feedback": candidate_rounds if candidate_rounds is not None else "unavailable",
        }
        metrics["run"] = {
            "pick_wall_s": round(pick_wall_s, 2),
            "pick_sim_s": round(pick_sim_s, 2),
            "pick_rtf": round(pick_sim_s / pick_wall_s, 3) if pick_wall_s > 0 else None,
            "load_avg": [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]],
        }
        # the final poses recorded separately (review r2 #11): the base and
        # object world poses AFTER the pick — descriptive only, the staging
        # band's evidence is the pre-arm snapshot above
        base_latest = base_stream.latest()
        object_latest = object_stream.latest()
        final_poses: dict = {}
        if base_latest is not None:
            qx, qy, qz, qw = (float(v) for v in base_latest.quat_xyzw)
            final_poses["base_map"] = [round(float(v), 4) for v in base_latest.position]
            final_poses["base_yaw_rad"] = round(
                float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))), 4
            )
            layout = get_scene_layout(SCENE_NAME)
            table_entry = next(e for e in layout["static_geometry"] if e["name"] == "table")
            edge_x = float(table_entry["pose"][0]) - float(table_entry["size"][0]) / 2
            final_poses["table_edge_distance_from_final_base_m"] = round(edge_x - float(base_latest.position[0]), 4)
        if object_latest is not None:
            final_poses["object_map"] = [round(float(v), 4) for v in object_latest.position]
        metrics["final_poses"] = final_poses

        # the provisional pick verdict (review r2 #2): recorded, NOT raised —
        # the invariants, the audit and the cleanup still complete below
        if pick_result is None or not pick_result.success:
            if sim_platform == "gazebo":
                verdict_reason = _gazebo_budget_exhaustion_verdict(pick_result)
                if verdict_reason is not None:
                    _record_verdict("xfail", verdict_reason)
            if not verdict:
                _record_verdict("fail", f"the pick did not succeed: {metrics['pick']}")
        else:
            # the R7 hold window is "after the result, hold_s of sim time":
            # let the sim RUN through the hold before the judge reads the
            # ground truth
            hold_seconds = spec_success()["hold_s"]
            hold_target = probe.clock_sim + hold_seconds + 0.5
            hold_deadline = time.monotonic() + 60.0
            while probe.clock_sim < hold_target and time.monotonic() < hold_deadline:
                if not stack.alive():
                    pytest.fail("the launch exited during the hold window")
                time.sleep(0.05)

        # the recording endpoint (review r2 #2): recording stops HERE on
        # every path — at the failure/success endpoint (after the hold on a
        # successful pick) — and the independent end markers anchor the
        # audit and the invariant windows
        recording_end_sim = probe.clock_sim
        recording_end_wall = time.monotonic()
        probe.recording = False
        status_end_mark = len(probe.runtime_statuses)
        clock_end_mark = len(probe.clock_pairs)

        # the raw GT streams' positions over the run (the GT-vs-GUI
        # investigation, 2026-11-03): the object/gripper/base samples with
        # their sim stamps — the 2 Hz object-z-next-to-gripper reading
        # around the close comes straight from this dump
        Path(run_dir / "gt_streams.json").write_text(
            json.dumps(
                {
                    name: [
                        [round(sample.stamp_sec, 3), [round(float(v), 4) for v in sample.position]]
                        for sample in stream.samples()
                    ]
                    for name, stream in (("object", object_stream), ("gripper", gripper_stream), ("base", base_stream))
                },
                indent=1,
            ),
            encoding="utf-8",
        )

        # the pick-window continuous invariants, on EVERY path: a world
        # reset, a non-ACTIVE/stale/latched runtime status, or a stalled/
        # dropped/rewound sim clock each void the run's motion evidence —
        # explicit failures, never a silent wait-out
        reset_streams = [index for index, stream in enumerate(streams) if stream.reset_detected()]
        metrics["ground_truth_reset"] = {"streams": reset_streams}
        if reset_streams:
            _record_verdict(
                "fail",
                f"the simulator clock rewound (a world reset) during the run — stream indices {reset_streams}: "
                "the run's ground truth is void",
            )
        status_samples = probe.runtime_statuses[status_window_mark:status_end_mark]
        status_ok, status_detail = runtime_status_verdict(status_samples, end_sim_s=recording_end_sim)
        metrics["runtime_status_window"] = {
            "ok": status_ok,
            "detail": status_detail,
            "samples": len(status_samples),
            # the wall receipts (steady time) and the sim progress at each
            # the wall receipts (steady time), the probe's sim-time view at
            # the receipt and the message's OWN source stamp — the cadence
            # and both-age evidence for the runtime-freshness investigation
            "receipts": [
                [round(wall, 3), round(sim, 3), round(stamp, 3)]
                for wall, sim, stamp in probe.status_receipts[status_window_mark:status_end_mark]
            ],
        }
        if not status_ok:
            _record_verdict("fail", f"the pick window's runtime-status invariant failed: {status_detail}")
        clock_samples = probe.clock_pairs[clock_window_mark:clock_end_mark]
        clock_ok, clock_detail = clock_stall_verdict(clock_samples, end_wall_s=recording_end_wall)
        metrics["clock_stall_window"] = {
            "ok": clock_ok,
            "detail": clock_detail,
            "samples": len(clock_samples),
        }
        if not clock_ok:
            _record_verdict("fail", f"the pick window's sim-clock invariant failed: {clock_detail}")

        # the R6 audit of every attempted execution (review r2 #2): the
        # window comes from the INDEPENDENT recording markers, and a pick
        # that executed NO arm motion needs explicit no-motion evidence
        window, window_reason = motion_audit_window(probe.arm_samples, recording_start_sim, recording_end_sim)
        if window_reason == "":
            audit_ok, audit_detail, audit_count, audit_invalid = audit_arm_samples(probe, stack, window)
        elif window_reason == "no arm motion over the covered window" and not (
            pick_result is not None and pick_result.success
        ):
            # a failed pick that never moved the arm: the covered no-motion
            # buffer IS the evidence; there is nothing to audit
            audit_ok, audit_detail, audit_count, audit_invalid = (
                True,
                "no executed arm motion over the covered recording (the arm stayed at its rest pose)",
                0,
                [],
            )
        else:
            audit_ok, audit_detail, audit_count, audit_invalid = (
                False,
                window_reason
                if window_reason != "no arm motion over the covered window"
                else "a successful pick must leave executed arm motion — the recording shows none",
                0,
                [],
            )
        metrics["arm_sample_audit"] = {
            "ok": audit_ok,
            "samples": audit_count,
            "verdict": audit_detail,
            "invalid_samples": audit_invalid,
        }
        print(f"[pick-sim] arm-sample audit: {audit_count} samples, ok={audit_ok}: {audit_detail}", flush=True)
        if not audit_ok:
            _record_verdict("fail", f"the R6 arm-sample audit rejected the run: {audit_detail}")

        # the judge (successful picks only — R7 windows on the collected
        # ground truth, the frozen rest height, the model TCP hold point)
        if pick_result is not None and pick_result.success and not any(kind == "fail" for kind, _ in verdict):
            judge_ok, judge_detail, judge_info = judge_from_ground_truth(
                object_stream, gripper_stream, base_stream, result_stamp, recording_end_sim, z_rest, hold_point
            )
            metrics["judge"] = {"ok": judge_ok, "detail": judge_detail, **judge_info}
            metrics["ground_truth"] = {
                "object_samples": len(object_stream.samples()),
                "gripper_samples": len(gripper_stream.samples()),
                "base_samples": len(base_stream.samples()),
            }
            if not judge_ok:
                if sim_platform == "gazebo":
                    slip_reason = _gazebo_slip_verdict(pick_result, judge_detail)
                    if slip_reason is not None:
                        _record_verdict("xfail", slip_reason)
                if not any(kind == "xfail" for kind, _ in verdict):
                    _record_verdict("fail", f"the R7 judge rejected the pick: {judge_detail}")

    # the deferred verdict is raised only after the cleanup aggregation
    # (review r2 #2/#3)
    _raise_deferred_verdict(verdict, cleanup_errors)


# ---------------------------------------------------------------------------
# The platform gate and the matchers (unit-checks; no simulator is started)
# ---------------------------------------------------------------------------


def test_pick_e2e_gate_requires_the_platform():
    decision, reason = pick_e2e_gate("gazebo", [])
    assert decision == "skip"
    assert REQUIRED_SIM_PLATFORMS_ENV in reason and "mujoco" in reason and "gazebo" in reason
    assert pick_e2e_gate("mujoco", ["gazebo"])[0] == "skip"


def test_pick_e2e_gate_runs_mujoco_when_required():
    assert pick_e2e_gate("mujoco", ["mujoco"]) == ("run", "")
    assert pick_e2e_gate("mujoco", ["mujoco", "gazebo"]) == ("run", "")


def test_pick_e2e_gate_runs_gazebo_when_required():
    """Review r1 item 9: a required Gazebo leg RUNS (never a skip, never an
    upfront xfail) — its known issues surface only at the approved
    pick-failure verdicts."""
    assert pick_e2e_gate("gazebo", ["gazebo"]) == ("run", "")
    assert pick_e2e_gate("gazebo", ["mujoco", "gazebo"]) == ("run", "")


def _pick_result(
    success: bool,
    error_code: str = "",
    message: str = "",
    *,
    verification_status: int = PickObject.Result.VERIFICATION_NOT_RUN,
    verification_confidence: float = 0.0,
):
    """A wire-realistic PickObject.Result (review r2 #8): the verification
    status is the uint8 enum and the confidence is its own float field."""
    result = PickObject.Result()
    result.success = success
    result.error_code = error_code
    result.message = message
    result.verification_status = verification_status
    result.verification_confidence = verification_confidence
    return result


def test_the_gazebo_budget_exhaustion_class_is_the_only_budget_xfail():
    """Review r1 item 9 + r2 #8: the retry-budget xfail applies ONLY with
    the listed preparation codes, or the jaw-at-open wedge UNCERTAIN with
    its specific evidence — a missing-evidence or generic fused-uncertain
    UNCERTAIN is a hard failure."""
    for code in ("IK_ORIENTATION_REJECTED", "CONTACT_REALIGN_FAILED", "CONTACT_COMPENSATION_FAILED"):
        verdict = _gazebo_budget_exhaustion_verdict(
            _pick_result(False, code, f"grasp retry budget exhausted after 3 attempts; last failure: {code}")
        )
        assert verdict is not None, code
    # the jaw-at-open wedge UNCERTAIN: the verifier's evidence is in the
    # message (the executor propagates it)
    wedge = (
        "grasp retry budget exhausted after 3 attempts; last failure: GRASP_UNCERTAIN: "
        "Grasp verification is uncertain (success=0.00, failure=0.45) "
        f"[{JAW_AT_OPEN_WEDGE_EVIDENCE}; gripper_position: joint=6 value=1.0001]"
    )
    assert _gazebo_budget_exhaustion_verdict(_pick_result(False, "GRASP_UNCERTAIN", wedge)) is not None
    # the missing/stale-sensor UNCERTAIN is NOT the approved class
    missing_evidence = (
        "grasp retry budget exhausted after 3 attempts; last failure: GRASP_UNCERTAIN: "
        "No usable grasp-verification evidence is available [gripper_position: unavailable; "
        "gripper_current_abs_a: unavailable; wrist_visibility: unavailable]"
    )
    assert _gazebo_budget_exhaustion_verdict(_pick_result(False, "GRASP_UNCERTAIN", missing_evidence)) is None
    # a generic fused-uncertain close without the wedge evidence is NOT the class
    generic = (
        "grasp retry budget exhausted after 3 attempts; last failure: GRASP_UNCERTAIN: "
        "Grasp verification is uncertain (success=0.10, failure=0.45)"
    )
    assert _gazebo_budget_exhaustion_verdict(_pick_result(False, "GRASP_UNCERTAIN", generic)) is None
    # an internal/system failure that exhausted the budget is NOT the class
    assert (
        _gazebo_budget_exhaustion_verdict(
            _pick_result(False, "INTERNAL_ERROR", "grasp retry budget exhausted after 3 attempts")
        )
        is None
    )
    # the approved code without the budget-exhaustion verdict is NOT the class
    assert _gazebo_budget_exhaustion_verdict(_pick_result(False, "GRASP_UNCERTAIN", "something else failed")) is None
    # a successful pick never enters the class
    assert _gazebo_budget_exhaustion_verdict(_pick_result(True)) is None


def test_the_gazebo_slip_class_requires_verified_contact_and_paired_evidence():
    """Review r2 #6/#8: the slip xfail applies ONLY to a true-contact grasp
    (the verification ENUM, never a score compared against a threshold)
    whose judge rejection carries paired gripper-frame evidence of the
    object outside the jaws — absent or unsynchronized pairs are a hard
    failure."""
    region = "the object is outside the gripper's grasp region — resting on the wrist or arm"
    slipping = "the object moved 43.1 mm within the gripper frame — slipping"
    verified = _pick_result(
        True, verification_status=PickObject.Result.VERIFICATION_SUCCESS, verification_confidence=0.80
    )
    for detail in (region, slipping):
        verdict = _gazebo_slip_verdict(verified, detail)
        assert verdict is not None and "physically slipped" in verdict, detail
        assert "confidence 0.80" in verdict
    # the enum, not a score: an UNCERTAIN or FAILED verification never enters
    # the class (the old float-status matcher accepted them: 3.0 >= 0.65)
    for status in (PickObject.Result.VERIFICATION_UNCERTAIN, PickObject.Result.VERIFICATION_FAILED):
        assert (
            _gazebo_slip_verdict(_pick_result(True, verification_status=status, verification_confidence=0.90), region)
            is None
        )
    # a hold-continuity rejection is a hard failure even at true contact
    assert (
        _gazebo_slip_verdict(verified, "the trailing hold window is 1.20 s < 3.0 s — a sample gap of 0.264 s split it")
        is None
    )
    # missing/unsynchronized evidence is a hard failure, not a slip
    assert (
        _gazebo_slip_verdict(
            verified, "unsynchronized ground-truth evidence: no object sample paired with a gripper sample"
        )
        is None
    )
    # a failed pick never enters the class
    assert _gazebo_slip_verdict(_pick_result(False), region) is None


# ---------------------------------------------------------------------------
# The cleanup structure (unit-checks; no simulator is started)
# ---------------------------------------------------------------------------


def test_the_deferred_verdict_orders_cleanup_failures_over_a_provisional_xfail():
    """Review r2 #2/#3: the first recorded verdict stands, a hard failure
    overrides a provisional xfail, cleanup failures win over everything, and
    an empty verdict with no cleanup errors passes silently."""
    verdict: list[tuple[str, str]] = []
    _record_verdict_entry(verdict, "xfail", "the approved budget class")
    _record_verdict_entry(verdict, "fail", "the audit rejected the run")
    assert verdict == [("fail", "the audit rejected the run")]  # the hard failure wins
    _record_verdict_entry(verdict, "xfail", "a later xfail changes nothing")
    assert verdict == [("fail", "the audit rejected the run")]

    with pytest.raises(pytest.fail.Exception, match="cleanup steps failed"):
        _raise_deferred_verdict([("xfail", "approved")], ["scene removal: RuntimeError: not verified"])
    with pytest.raises(pytest.fail.Exception, match="the audit rejected the run"):
        _raise_deferred_verdict([("fail", "the audit rejected the run")], [])
    with pytest.raises(pytest.xfail.Exception, match="the approved budget class"):
        _raise_deferred_verdict([("xfail", "the approved budget class")], [])
    _raise_deferred_verdict([], [])  # nothing recorded: a clean pass
    verdict.clear()
    _record_verdict_entry(verdict, "fail", "first")
    _record_verdict_entry(verdict, "fail", "second")
    assert verdict == [("fail", "first")]  # the first hard failure stands


def test_a_failing_cleanup_step_does_not_skip_the_rest():
    """Review r2 #3: every registered cleanup runs even when an earlier one
    fails, and every failure is aggregated."""
    errors: list[str] = []
    ran: list[str] = []

    def _register(exit_stack, action):
        label = getattr(action, "__name__", str(action))

        def _run():
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

        exit_stack.callback(_run)

    def first():
        ran.append("first")
        raise RuntimeError("boom")

    def second():
        ran.append("second")

    with contextlib.ExitStack() as exit_stack:
        _register(exit_stack, first)
        _register(exit_stack, second)
    assert ran == ["second", "first"]  # LIFO, both ran
    assert errors == ["first: RuntimeError: boom"]


def test_the_stack_teardown_runs_every_step_independently(monkeypatch):
    """Review r2 #3: a failing process-group termination must not skip the
    marker sweep or the daemon stop."""
    import sys

    this_module = sys.modules[__name__]
    sweep_calls: list[dict] = []
    daemon_calls: list[list[str]] = []
    monkeypatch.setattr(this_module, "terminate_marked_processes", lambda marker: sweep_calls.append(marker))

    class _FakeSubprocess:
        @staticmethod
        def run(*_args, **_kwargs):
            daemon_calls.append(["ros2", "daemon", "stop"])

    monkeypatch.setattr(this_module, "subprocess", _FakeSubprocess)

    def _raising_killpg(_pgid, _sig):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(os, "killpg", _raising_killpg)
    process = type("FakeProcess", (), {"poll": staticmethod(lambda: None), "wait": staticmethod(lambda **_kw: None)})
    stack = _Stack(process, Path("/nonexistent/launch.log"))  # type: ignore[arg-type]
    marker_env = {RUN_MARKER_ENV: "test-marker"}

    with pytest.raises(RuntimeError, match="process-group termination"):
        _stop_stack(stack, marker_env)
    assert sweep_calls == [marker_env]
    assert daemon_calls == [["ros2", "daemon", "stop"]]


def test_the_pick_worker_join_is_bounded_and_verifies_termination():
    """Review r2 #3: the pick client's worker is joined with a bound and its
    termination verified — a stranded worker is a hard failure and the
    client node is destroyed only after a clean join."""
    destroyed: list[str] = []
    client = type("FakeClient", (), {"destroy_node": staticmethod(lambda: destroyed.append("destroyed"))})()

    def _work(duration: float):
        time.sleep(duration)

    quick = threading.Thread(target=_work, args=(0.01,))
    quick.start()
    _join_pick_worker(quick, client, timeout_s=2.0)
    assert destroyed == ["destroyed"]

    stuck = threading.Thread(target=_work, args=(5.0,), daemon=True)
    stuck.start()
    with pytest.raises(TimeoutError, match="did not terminate"):
        _join_pick_worker(stuck, client, timeout_s=0.05)
    assert destroyed == ["destroyed"]  # not destroyed while the worker lives
    stuck.join(timeout=6.0)


def test_the_scene_readback_verification():
    """Review r1 item 5 + r2 #3: the readback check verifies the applied box
    (id, frame, size, pose) and fails loudly on a missing or mismatched
    object."""
    from test.mobile_grasp_support import SceneBox

    applied = collision_object_message(
        transform_scene_box(
            SceneBox("chassis", (0.02, 0.06, -0.02), (0.22, 0.21, 0.03), frame="base"), np.eye(4), "base"
        ),
        CollisionObject.ADD,
    )
    # MoveIt's GetPlanningScene returns the object with its planning-frame
    # ``pose`` set (the applied primitive pose), as the r1 runs measured
    readback = CollisionObject()
    readback.id = applied.id
    readback.header.frame_id = applied.header.frame_id
    readback.operation = applied.operation
    readback.primitives = [applied.primitives[0]]
    readback.pose = applied.primitive_poses[0]
    scene_state = SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(collision_objects=[readback])))
    entry = _verify_scene_readback([applied], scene_state, lambda *_: np.eye(4))
    assert entry["id"] == "chassis" and entry["pose_ok"] and entry["size_ok"]

    empty_state = SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(collision_objects=[])))
    with pytest.raises(AssertionError, match="not in the read-back"):
        _verify_scene_readback([applied], empty_state, lambda *_: np.eye(4))
