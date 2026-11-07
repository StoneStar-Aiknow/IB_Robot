"""Fixed executor with compiled body and real hand models; only hardware feedback is mocked."""

import copy
import os
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.hand_restore import (
    CommandBoard,
    GatewayState,
    HandRestorer,
    Observation,
    RuntimeState,
    build_hand_plans,
)
from aimdk_grasp.joints import HAND_ALIASES, HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.robot_model import config_directory, planning_model

from ibrobot_msgs.action import ExecuteArmPose
from pose_composer.fixed_backend import FixedBackend
from pose_composer.fixed_library import FixedComposer
from pose_composer.node import PoseComposerNode
from pose_composer.task_poses import TaskPoseRunner
from robot_runtime.task_poses import normalize_task_poses


@pytest.fixture(scope="session", autouse=True)
def configured():
    # Override the legacy module-wide MuJoCo fixture: this suite must run without it.
    return None


@pytest.fixture(scope="module")
def fixed_backend():
    artifact = os.environ.get("IBROBOT_FIXED_LIBRARY")
    if not artifact:
        pytest.skip("set IBROBOT_FIXED_LIBRARY to the offline-compiled action artifact")
    from robot_config.loader import _load_robot_section

    root = Path(__file__).resolve().parents[4]
    _, config = _load_robot_section(root / "src/robot_config/config/robots/aimdk_x2_grasp_310p.yaml")
    return FixedBackend(
        root / "src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml", config["arm_limits"], artifact
    )


class Goal:
    def __init__(self, request, state):
        self.request, self.state, self.terminal = request, state, None

    @property
    def is_cancel_requested(self):
        return self.state.cancelled

    def succeed(self):
        self.terminal = "succeeded"

    def abort(self):
        self.terminal = "aborted"

    def canceled(self):
        self.terminal = "canceled"

    def publish_feedback(self, _feedback):
        pass


def ready(value):
    future = Future()
    future.set_result(value)
    return future


@pytest.mark.parametrize(
    "pose_name,return_kind,policy",
    [
        ("arms_up", "named", {"hand": "off"}),
        ("arms_up", "seed", {"hand": "off", "body": "off"}),
        ("right_forward", "seed", {"hand": "off", "body": "off"}),
        ("left_forward", "seed", {"hand": "off", "body": "off"}),
    ],
)
def test_fixed_raise_return_and_both_hands_without_generation(
    monkeypatch, fixed_backend, pose_name, return_kind, policy
):
    from aimdk_grasp import backend

    from pose_composer import node as node_module

    opened = dict(zip(HAND_ALIASES, [0.129, -1.54, 0.255, 0, 0, 0, 0, 0, 0, 0], strict=True))
    presets = {"open": opened, "close": dict(opened)}

    def robot(config):
        side = config["backend_options"]["arm_side"]
        return SimpleNamespace(hand=HandMapping(planning_model(config_directory(side)), presets, side=side))

    monkeypatch.setattr(backend, "x2_robot", robot)
    plans, missing = build_hand_plans({})
    assert not missing
    events = []
    state = SimpleNamespace(
        now=10.0, cancelled=False, q=dict(zip(LEFT_ARM + RIGHT_ARM, [0.4, 0, 0, -1.2, 0, 0, 0] * 2, strict=True))
    )

    class Ports:
        def __init__(self):
            self.hands = {side: plan.open_pose.copy() for side, plan in plans.items()}
            # Both fingers actually have to move; neither opening may be skipped.
            for values in self.hands.values():
                values[4] = 0.15

        def observe(self):
            body = {**state.q, **dict.fromkeys(HEAD, 0.0)}
            board = CommandBoard(
                LEFT_ARM + RIGHT_ARM, np.array(list(state.q.values())), np.full(14, 40.0), np.full(14, 2.0), 0.0
            )
            return Observation(
                Feedback(state.now, body, self.hands["left"], right_hand=self.hands["right"]),
                0.0,
                board,
                RuntimeState(True, "", "policy_stream", 0.0),
                GatewayState(True, True, True, "mock_task", 0.0),
            )

        def prepare(self, _ctx):
            pass

        def refusal(self):
            return None

        def release(self):
            pass

        def monotonic(self):
            return state.now

        def wait(self, seconds):
            state.now += seconds
            publish()

        def publish(self, left, right):
            self.hands = {"left": left.copy(), "right": right.copy()}

    ports = Ports()
    restorer = HandRestorer(ports, plans)
    server = object.__new__(PoseComposerNode)
    server._selected_backend = fixed_backend
    server._validation_policy = policy
    server._busy, server._joint_lock = Lock(), Lock()
    server._motion_limits, server._trajectory_modes = (0.0, 0.0), {"policy_stream"}
    server._runtime_mode = lambda: "policy_stream"
    server._hand_config = {"enabled": True}
    server._hand_restorer = lambda: restorer
    server.get_logger = lambda: SimpleNamespace(info=lambda _: None)

    def publish():
        server._positions = dict(state.q)
        server._joint_received = dict.fromkeys(state.q, state.now)

    publish()
    monkeypatch.setattr(node_module.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(node_module.time, "sleep", ports.wait)
    server._composer = FixedComposer(fixed_backend.library)
    config = copy.deepcopy(fixed_backend.task_poses) or {
        "schema_version": 1,
        "robot_type": "x2_ultra_omnihand",
        "joint_names": list(LEFT_ARM + RIGHT_ARM),
        "poses": {
            name: {
                "joint_positions": dict(state.q),
                "hardware_approved": True,
                "unloaded_operation_approved": True,
                "timeout_sec": 180.0,
                "hands": {"left": "open", "right": "open"},
            }
            for name in ("arms_rest", "grasp_ready")
        },
    }
    config.pop("digest", None)
    for pose in config["poses"].values():
        pose.update(hardware_approved=True, unloaded_operation_approved=True, timeout_sec=180.0)
    config = normalize_task_poses(config)
    server._task_pose_runner = TaskPoseRunner(config, clock=lambda: state.now, pause=ports.wait, backend=fixed_backend)

    class Trajectory:
        def wait_for_server(self, **_kw):
            return True

        def send_goal_async(self, goal):
            events.append("arm_trajectory")
            state.q = dict(zip(goal.trajectory.joint_names, goal.trajectory.points[-1].positions, strict=True))
            publish()
            terminal = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0, error_string=""))
            return ready(SimpleNamespace(accepted=True, get_result_async=lambda: ready(terminal)))

    server._traj = Trajectory()
    if pose_name not in fixed_backend.library["poses"] or (
        return_kind == "seed" and "arms_down" not in fixed_backend.library["poses"]
    ):
        pytest.skip("set IBROBOT_FIXED_LIBRARY to the rest-bound expanded candidate library")
    up = ExecuteArmPose.Goal(text=pose_name, gateway_task_id="mock_task", timeout_sec=180.0)
    first = Goal(up, state)
    result = server._on_execute(first)
    assert result.success, result.message
    assert first.terminal == "succeeded"
    assert state.q == pytest.approx(fixed_backend.library["poses"][pose_name]["target"])
    for side in ("left", "right"):
        anchor = fixed_backend.library["poses"][pose_name]["description"].get(side, {}).get("task_pose")
        if anchor:
            for joint in state.q:
                if joint.startswith(side + "_"):
                    assert state.q[joint] == config["poses"][anchor]["joint_positions"][joint]
    events.append("raised")
    request = ExecuteArmPose.Goal(
        named_pose="grasp_ready", task_poses_digest=config["digest"], gateway_task_id="mock_task", timeout_sec=180.0
    )
    if return_kind == "seed":
        request = ExecuteArmPose.Goal(text="双臂放下", gateway_task_id="mock_task", timeout_sec=180.0)
    second = Goal(request, state)
    result = server._on_execute(second)
    assert result.success, result.message
    target_name = "arms_rest" if return_kind == "seed" else "grasp_ready"
    assert state.q == pytest.approx(config["poses"][target_name]["joint_positions"])
    assert server._task_pose_runner.report["completed_hands"] == ["left", "right"]
    assert server._task_pose_runner.report["skipped_hands"] == []
    for side, plan in plans.items():
        np.testing.assert_allclose(ports.hands[side], plan.open_pose)
    assert events == ["arm_trajectory", "raised", "arm_trajectory"]
    assert "path_validation=" in result.message
    assert len(result.message) <= 300
    assert "opened_hands=left,right" in result.message
    assert '"stages"' not in result.message
    for part in ("hand", "body"):
        stages = server._path_report["stages"]
        assert stages[f"{part}_path"]["status"] == ("skipped" if policy.get(part) == "off" else "executed")
        assert stages[f"{part}_start"]["status"] == "executed"


def test_unknown_phrase_never_generates(fixed_backend):
    from pose_composer.errors import ComposeError

    composer = FixedComposer(fixed_backend.library)
    with pytest.raises(ComposeError, match="generation is disabled"):
        composer.compose("右手再往左一点", fixed_backend.rest)


def test_fixed_preview_builds_candidate_but_offline_compiler_stays_strict(fixed_backend, monkeypatch):
    composer = FixedComposer(fixed_backend.library)
    start = dict(zip(fixed_backend.ARM_JOINTS, [0.4, 0, 0, -1.2, 0, 0, 0] * 2, strict=True))
    prepared = composer.compose("双手举起来", start)

    def collision(*_, **__):
        raise ValueError("mock midpoint collision")

    monkeypatch.setattr(fixed_backend.body_geometry, "check_path", collision)
    assert fixed_backend.plan(prepared.composition, start).ok
    assert not fixed_backend.plan_safe(start, prepared.composition.frames, [0.0]).ok


@pytest.mark.parametrize(
    "budget,offset,success", [(150.0, 0.0, True), (90.0, 0.0, False), (150.0, 0.09, True), (150.0, 0.11, False)]
)
def test_fixed_seed_respects_gateway_budget_and_arrival_tolerance(monkeypatch, budget, offset, success):
    from pose_composer import node as node_module
    from robot_runtime.pose_backend import Plan, Report, Trajectory

    joints = LEFT_ARM + RIGHT_ARM
    state = SimpleNamespace(now=10.0, cancelled=False)
    server = object.__new__(PoseComposerNode)
    server.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    server._joint_lock = Lock()
    server._positions = dict.fromkeys(joints, 0.0)
    server._motion_limits = (0.0, 0.0)
    target = dict.fromkeys(joints, 0.2)
    moved = []

    def advance(seconds):
        state.now += seconds
        server._joint_received = dict.fromkeys(joints, state.now)

    advance(0.0)

    def plan(_start, _frames, _holds):
        advance(100.0)
        return Plan(Trajectory([0.0, 1.0], [np.zeros(14), np.full(14, 0.2)]), True, [], Report(True))

    def move(_goal, _plan, **_kwargs):
        moved.append(True)
        server._positions = {joint: value + offset for joint, value in target.items()}
        advance(1.0)
        return True, "", "reached"

    server._selected_backend = SimpleNamespace(
        ARM_JOINTS=joints,
        model=SimpleNamespace(ARM_JOINTS=joints),
        safety=SimpleNamespace(check_pose=lambda _: Report(True)),
        plan_safe=plan,
        check_displacement=lambda *_: "",
    )
    server._run_trajectory = move
    monkeypatch.setattr(node_module.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(node_module.time, "sleep", advance)
    goal = Goal(ExecuteArmPose.Goal(timeout_sec=budget, gateway_task_id="mock_task"), state)
    prepared = SimpleNamespace(pose_id="arms_up", composition=SimpleNamespace(frames=[target]))
    result = server._execute_fixed_target(goal, ExecuteArmPose.Result(), prepared)
    assert result.success is success
    assert bool(moved) is (budget >= 101.0)
    if not success:
        assert result.error_code == ("TASK_POSE_TIMEOUT" if budget < 101.0 else "TASK_POSE_NOT_REACHED")


def test_artifact_tampering_and_wrong_model_are_rejected(fixed_backend):
    from pose_composer.fixed_backend import digest

    root = Path(__file__).resolve().parents[4]
    profile = root / "src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml"
    data = copy.deepcopy(fixed_backend.library)
    data["poses"]["arms_up"]["target"][LEFT_ARM[0]] += 0.1
    with pytest.raises(ValueError, match="digest"):
        FixedBackend(profile, fixed_backend.hard_limits, data)
    data["robot_fingerprint"] = "wrong"
    data["digest"] = digest({k: v for k, v in data.items() if k != "digest"})
    with pytest.raises(ValueError, match="model/limits"):
        FixedBackend(profile, fixed_backend.hard_limits, data)


def test_invalid_start_and_excessive_speed_refuse(fixed_backend):
    from robot_runtime.pose_backend import Plan, Report, Trajectory

    target = fixed_backend.library["poses"]["arms_up"]["target"]
    start = dict(target, left_elbow_joint=100.0)
    assert not fixed_backend.plan_safe(start, [target], [0.0]).ok
    row = np.array([target[n] for n in fixed_backend.ARM_JOINTS])
    plan = Plan(Trajectory([0.0, 0.001], [row, row + 0.1]), True, [], Report(True))
    assert not fixed_backend.prepare_execution(plan).ok


def test_body_path_cancel_is_checked_before_work(fixed_backend):
    row = [fixed_backend.rest[n] for n in fixed_backend.ARM_JOINTS]

    def cancel():
        raise RuntimeError("cancel requested")

    with pytest.raises(RuntimeError, match="cancel requested"):
        fixed_backend.body_geometry.check_path(fixed_backend.ARM_JOINTS, [row, row], guard=cancel)


@pytest.mark.parametrize("fault", ["drift", "stale", "nonfinite", "missing_geometry", "body_collision"])
def test_fixed_dispatch_refuses_changed_start_before_sending(fault):
    from robot_runtime.pose_backend import Plan, Report, Trajectory

    joints = LEFT_ARM + RIGHT_ARM
    server = object.__new__(PoseComposerNode)

    def body_check(*_args, **_kwargs):
        if fault == "body_collision":
            raise ValueError("compiled body collision")

    server._selected_backend = SimpleNamespace(
        library={},
        model=SimpleNamespace(ARM_JOINTS=joints),
        ARM_JOINTS=joints,
        body_geometry=SimpleNamespace(check_path=body_check),
        prepare_execution=lambda plan, **_: plan,
        check_pose=lambda _, **_kw: SimpleNamespace(ok=True),
    )
    server._hand_config = {} if fault == "missing_geometry" else {"enabled": True}
    baseline = SimpleNamespace(feedback=SimpleNamespace(positions=dict.fromkeys(HEAD, 0.0)))
    server._hand_restorer = lambda: SimpleNamespace(
        ports=SimpleNamespace(prepare=lambda _: None),
        arm_baseline=lambda _: baseline,
        geometry=SimpleNamespace(check_arm_path=lambda *_a, **_k: None),
        guard_arm_motion=lambda *_: baseline,
    )
    server._joint_lock = Lock()
    current = dict.fromkeys(joints, 0.0)
    server._positions = current
    server._joint_received = dict.fromkeys(joints, time.monotonic())
    if fault == "drift":
        current[joints[0]] = 0.006
    elif fault == "stale":
        server._joint_received[joints[0]] -= 1.0
    elif fault == "nonfinite":
        current[joints[0]] = float("nan")

    def unexpected_dispatch(_goal):
        pytest.fail("changed start must not dispatch any arm trajectory")

    server._traj = SimpleNamespace(wait_for_server=lambda **_: True, send_goal_async=unexpected_dispatch)
    plan = Plan(Trajectory([0.0, 1.0], [np.zeros(14), np.full(14, 0.1)]), True, [], Report(True))
    goal = SimpleNamespace(
        request=ExecuteArmPose.Goal(gateway_task_id="mock_task", timeout_sec=90.0), is_cancel_requested=False
    )
    ok, code, _message = server._run_trajectory(goal, plan)
    expected = {
        "drift": "POSE_START_CHANGED",
        "stale": "TASK_POSE_STALE_STATE",
        "nonfinite": "TASK_POSE_STALE_STATE",
        "missing_geometry": "HAND_CAPABILITY_MISSING",
        "body_collision": "HAND_CAPABILITY_MISSING",
    }
    assert not ok and code == expected[fault]


def test_fixed_runtime_imports_when_mujoco_is_forbidden():
    code = """
import sys
from importlib.abc import MetaPathFinder
class Forbidden(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mujoco' or fullname.startswith('mujoco.'):
            raise RuntimeError('MuJoCo must not be imported on the board')
sys.meta_path.insert(0, Forbidden())
import pose_composer.node, pose_composer.fixed_backend, pose_composer.task_poses
assert 'mujoco' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_fixed_launch_has_no_mjcf_or_language_dependency(monkeypatch, fixed_backend):
    from pose_composer import launch_builder

    monkeypatch.setattr(launch_builder, "_hand_restore_parameters", lambda _: '{"enabled":true}')
    config = {
        "runtime": {"provider": "aimdk_robot", "profile": "x2_ultra_omnihand"},
        "skill_required_control_mode": "moveit_planning",
        "control_modes": {"moveit_planning": {"runtime_mode": "policy_stream"}},
        "pose_composer": {"enabled": True, "mode": "fixed", "fixed_library": os.environ["IBROBOT_FIXED_LIBRARY"]},
    }
    (action,) = launch_builder.generate_pose_composer_nodes(config)
    assert action.node_executable == "pose_composer_node"
    config["pose_composer"]["llm"] = {"model": "must-not-run"}
    with pytest.raises(ValueError, match="does not allow"):
        launch_builder.generate_pose_composer_nodes(config)


def test_convex_checker_rejects_overlap_and_preserves_clearance():
    from itertools import product

    from pose_composer.compiled_geometry import Shape, separated

    cube = {"kind": "mesh", "vertices": list(product([-0.1, 0.1], repeat=3)), "radius": 0.18}
    a = Shape(cube, np.eye(4))
    transform = np.eye(4)
    transform[0, 3] = 0.15
    assert not separated(a, Shape(cube, transform), 0.0)
    transform[0, 3] = 0.25
    assert separated(a, Shape(cube, transform), 0.049)
    assert not separated(a, Shape(cube, transform), 0.051)


@pytest.mark.parametrize("terminal", [True, False])
def test_range_failure_cancels_once_and_unknown_stop_blocks_next_request(monkeypatch, terminal):
    from aimdk_grasp.hand_restore import RestoreError

    from pose_composer import node as node_module
    from robot_runtime.pose_backend import Plan, Report, Trajectory

    clock = SimpleNamespace(now=10.0, cancelled=False)
    monkeypatch.setattr(node_module.time, "monotonic", lambda: clock.now)
    completed = Future()
    cancellations = []
    sent = []

    def pause(seconds):
        clock.now += seconds
        if terminal and cancellations and not completed.done():
            completed.set_result(SimpleNamespace(result=SimpleNamespace(error_code=0, error_string="held")))

    monkeypatch.setattr(node_module.time, "sleep", pause)
    server = object.__new__(PoseComposerNode)
    server._busy, server._joint_lock = Lock(), Lock()
    server._selected_backend = SimpleNamespace(
        library={}, ARM_JOINTS=LEFT_ARM + RIGHT_ARM, model=SimpleNamespace(ARM_JOINTS=LEFT_ARM + RIGHT_ARM)
    )
    server._positions = dict.fromkeys(LEFT_ARM + RIGHT_ARM, 0.0)
    server._joint_received = dict.fromkeys(LEFT_ARM + RIGHT_ARM, clock.now)
    server._hand_config = {"enabled": True}
    baseline = SimpleNamespace(feedback=None)
    server._execution_gate = SimpleNamespace(report={"stages": {}}, monitor=None, check=lambda *_: baseline)

    def guard(*_):
        if sent:
            raise RestoreError("POSE_CERTIFICATE_RANGE", "mock tracking out of range")
        return baseline

    server._hand_restorer = lambda: SimpleNamespace(guard_arm_motion=guard)
    handle = SimpleNamespace(
        accepted=True, get_result_async=lambda: completed, cancel_goal_async=lambda: cancellations.append(clock.now)
    )
    server._traj = SimpleNamespace(
        wait_for_server=lambda **_: True, send_goal_async=lambda _: sent.append(True) or ready(handle)
    )
    plan = Plan(Trajectory([0.0, 1.0], np.zeros((2, 14))), True, [], Report(True))
    goal = Goal(ExecuteArmPose.Goal(timeout_sec=30.0, gateway_task_id="mock"), clock)
    ok, code, _ = server._run_trajectory(goal, plan, deadline=30)
    assert not ok and len(cancellations) == 1
    assert code == ("POSE_CERTIFICATE_RANGE" if terminal else "POSE_STOP_UNKNOWN")
    if not terminal:
        result = server._on_execute(goal)
        assert not result.success and result.error_code == "POSE_STOP_UNKNOWN"
        assert len(sent) == 1
