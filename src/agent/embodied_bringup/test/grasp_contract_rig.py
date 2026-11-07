"""Real Agent/Gateway/PickObject graph, with deterministic peripheral faults.

This module is test-only: no hardware driver, camera or motion publisher is
created. The root conftest reserves a localhost DDS domain before importing it.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from ibrobot_agent.node import IncubationAgentNode
from ibrobot_agent.voice_bridge import _build_node_class
from ibrobot_msgs.msg import SynthesizedAudio
from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech, ValidateSkill
from manipulation_execution.grasp_pipeline.contracts import (
    ArmSegment,
    ContainerEstimate,
    Effect,
    EffectorAction,
    EffectorOutcome,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    HeldObject,
    Hold,
    MotionOutcome,
    MotionState,
    Move,
    Observation,
    PlaceBackends,
    PlacePlan,
    PlaceVerdict,
    Proposals,
    SegmentKind,
)
from manipulation_execution.grasp_pipeline_node import GraspPipelineNode
from manipulation_execution.place_pipeline_node import PlacePipelineNode
from skill_library.skill_executor_node import SkillExecutorNode

from .test_real_grasp_config_startup import _write_graspgen_bundle

_CARRY_MOCKS = {}


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("timed out waiting for mock contract graph")


def parameters(values):
    return [Parameter(name, value=value) for name, value in values.items()]


class MockGrasp:
    def __init__(self, config):
        self.scenario = config.get("scenario", "success")
        self.captures = 0
        self.queries = []
        self.moves = []
        self.closes = 0
        self.releases = 0
        self.container_queries = []
        self.verifications = 0
        self.stops = []
        self.position = np.zeros(3)
        self.gripped = False
        self.started = threading.Event()

    def capture(self, *, deadline, cancelled):
        self.captures += 1
        time.sleep(0.02)
        # A new pick starts at home; container capture preserves the carry pose.
        if not self.gripped:
            self.position = np.zeros(3)
        if self.scenario in {
            "OBSERVATION_UNAVAILABLE",
            "RPC_FAILED",
            "RPC_TIMEOUT",
            "NOT_READY",
            "SERVICE_UNAVAILABLE",
            "TASK_TIMEOUT",
        }:
            raise GraspFailure(self.scenario, "mock sensor failure " * 80, retryable=True)
        if self.scenario == "INTERNAL_ERROR":
            raise RuntimeError("injected unexpected sensor failure")
        return Observation(
            np.zeros((4, 4, 3), np.uint8),
            np.ones((4, 4), np.float32),
            np.eye(3),
            "camera",
            self.captures,
            "base",
            np.eye(4),
        )

    def propose(self, observation, query, *, deadline, cancelled):
        self.queries.append(query)
        if self.scenario in {"INVALID_TARGET", "TARGET_NOT_VISIBLE"}:
            raise GraspFailure(
                self.scenario, "injected perception failure", retryable=self.scenario != "INVALID_TARGET"
            )
        return Proposals(np.ones((4, 4), bool), np.zeros((3, 3)), np.stack([np.eye(4)]), [0.9])

    def plan(self, observation, proposals, *, deadline, cancelled):
        time.sleep(0.02)
        if self.scenario in {
            "NO_EXECUTABLE_CANDIDATE",
            "NO_COMPLETE_PATH",
            "OUT_OF_REACH",
            "PLAN_TIMEOUT",
            "PLAN_REJECTED",
            "PLANNER_FAULT",
            "REOBSERVE_REQUIRED",
            "EXECUTION_BUDGET_INSUFFICIENT",
        }:
            code = (
                "REOBSERVE_REQUIRED"
                if self.scenario == "REOBSERVE_REQUIRED"
                else "SKILL_TIMEOUT"
                if self.scenario == "EXECUTION_BUDGET_INSUFFICIENT"
                else "NO_EXECUTABLE_CANDIDATE"
            )
            raise GraspFailure(code, "injected planner failure", reason_code=self.scenario, retryable=True)

        def move(name, start, end):
            return Move(ArmSegment(name, SegmentKind.TARGET, ("j1", "j2", "j3"), np.array([start, end], dtype=float)))

        return GraspPlan(
            (
                Effect(EffectorAction.PRESHAPE),
                move("approach", [0, 0, 0], [0.2, 0, 0]),
                move("grasp", [0.2, 0, 0], [0.3, 0, 0]),
                Effect(EffectorAction.CLOSE),
                move("lift", [0.3, 0, 0], [0.3, 0, 0.2]),
                Hold(0.01, "hold_lifted"),
            ),
            0,
        )

    def execute(self, segment, *, carrying, halted, cancelled):
        self.started.set()
        self.moves.append(segment.name)
        if self.scenario in {"cancel", "deadline"}:
            wait_until(cancelled, timeout=15.0)
            return MotionOutcome(MotionState.CANCELLED, "mock cancel acknowledged")
        if self.scenario in {"MOTION_FAILED", "stop_unknown"} or (
            self.scenario == "recovery_failed" and segment.name.startswith("retreat_")
        ):
            return MotionOutcome(MotionState.FAILED, "mock tracking failure")
        if halted():
            return MotionOutcome(MotionState.HALTED, "mock watch halt", self.position.copy())
        self.position = segment.end.copy()
        return MotionOutcome(MotionState.SUCCEEDED)

    def hold(self, seconds, *, carrying, cancelled):
        return MotionOutcome(MotionState.CANCELLED if cancelled() else MotionState.SUCCEEDED)

    def measured(self, joint_names):
        return self.position.copy()

    def stop(self, reason):
        self.stops.append(reason)
        if self.scenario == "stop_unknown":
            raise RuntimeError("stop acknowledgement lost")

    def act(self, action, *, halted, cancelled):
        if cancelled():
            return EffectorOutcome(False, "cancelled")
        if self.scenario == "EFFECTOR_FAILED":
            return EffectorOutcome(False, "injected effector failure")
        if action == EffectorAction.CLOSE:
            self.closes += 1
            held = (
                None
                if self.scenario == "GRASP_UNCERTAIN"
                else self.scenario not in {"EMPTY_GRASP", "recovery_failed"}
                and not (self.scenario == "retry_success" and self.closes == 1)
            )
            self.gripped = held
            return EffectorOutcome(True, held=held, confidence=0.9)
        self.gripped = False
        if action == EffectorAction.RELEASE:
            self.releases += 1
        return EffectorOutcome(True)

    def finish(self):
        pass

    def suspend(self):
        pass

    def released(self):
        return not self.gripped

    def retained(self, query, observation, proposals, plan):
        return HeldObject(
            uuid.uuid4().hex,
            query,
            "left",
            ("j1", "j2", "j3"),
            tuple(self.position),
            (0.0, 0.0, 0.0),
            0.04,
            (0.0, 0.0, 0.0),
            "{}",
            lifted_at_ns=self.now_ns(),
            frame_id="base",
        )

    def holding(self):
        return "object slipped" if self.scenario == "GRIP_LOST" else "" if self.gripped else None

    def arm(self, *args):
        pass

    def moved(self):
        return "target moved" if self.scenario == "TARGET_MOVED" else None

    def failed(self):
        return "camera lost" if self.scenario == "WATCH_FAILED" else None

    def disarm(self):
        return {}


def make_mock_backends(*, node, grasp_execution, description):
    mock = MockGrasp(grasp_execution)
    mock.now_ns = lambda: node.get_clock().now().nanoseconds
    node.contract_mock = mock
    key = grasp_execution.get("carry_mock_id")
    if key:
        _CARRY_MOCKS[key] = mock
    return GraspBackends(mock, mock, mock, mock, mock, mock, held_object=mock.retained if key else None)


def make_mock_place_backends(*, placement_execution, **_kwargs):
    mock = _CARRY_MOCKS[placement_execution["carry_mock_id"]]
    container = ContainerEstimate(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([0.0, 0.0, 1.0]), 0)

    def propose(observation, query, held, **_kwargs):
        mock.container_queries.append(query)
        if placement_execution.get("scenario") == "container_missing":
            raise GraspFailure("CONTAINER_NOT_FOUND", f"no {query} in view")
        return container

    def move(name, start, end):
        return Move(ArmSegment(name, SegmentKind.TARGET, ("j1", "j2", "j3"), np.array([start, end], dtype=float)))

    def plan(observation, container, held, **_kwargs):
        return PlacePlan(
            (
                move("transfer", held.joint_positions, [0.5, 0.2, 0.3]),
                move("descend", [0.5, 0.2, 0.3], [0.5, 0.2, 0.2]),
            ),
            (Effect(EffectorAction.RELEASE),),
            (move("retreat", [0.5, 0.2, 0.2], [0.5, 0.2, 0.3]), move("home", [0.5, 0.2, 0.3], [0, 0, 0])),
            container,
        )

    def restore(held):
        if not mock.gripped or not np.allclose(mock.position, held.joint_positions):
            raise GraspFailure("HELD_OBJECT_MISMATCH", "mock handover pose or grip changed")

    def verify(*_args, **_kwargs):
        mock.verifications += 1
        return PlaceVerdict(not mock.gripped and mock.releases == 1)

    def execute(segment, **kwargs):
        if placement_execution.get("scenario") == "transfer_failed" and segment.name == "transfer":
            mock.moves.append(segment.name)
            return MotionOutcome(MotionState.FAILED, "mock placement tracking failure; stop acknowledged")
        return mock.execute(segment, **kwargs)

    return PlaceBackends(
        mock,
        SimpleNamespace(propose=propose),
        SimpleNamespace(plan=plan),
        SimpleNamespace(verify=verify),
        SimpleNamespace(execute=execute, hold=mock.hold, measured=mock.measured, stop=mock.stop, finish=mock.finish),
        SimpleNamespace(holding=mock.holding, act=mock.act, released=lambda: not mock.gripped, finish=mock.finish),
        restore,
    )


class ContractRig:
    def __init__(
        self,
        root: Path,
        scenario="success",
        *,
        tts_failure=False,
        tts_block=False,
        steps=1,
        pick_place=False,
        place_scenario="success",
        held_lease_s=None,
    ):
        assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID") == os.environ.get("ROS_DOMAIN_ID")
        assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
        self.requests = []
        self.events = []
        self.responses = []
        self.speech = []
        self.played = []
        self.tts_failure = tts_failure
        self.speech_gate = threading.Event()
        if not tts_block:
            self.speech_gate.set()
        self.nodes = []
        self.pick_place = pick_place
        self.place_scenario = place_scenario
        self.held_lease_s = held_lease_s
        self.carry_mock_id = uuid.uuid4().hex
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self.executor = MultiThreadedExecutor(num_threads=10)
        self.spin_stop = threading.Event()
        self.thread = None
        self.server = None
        self.server_thread = None
        try:
            self._start(root, scenario, steps)
        except BaseException:
            self.close()
            raise

    def _start(self, root, scenario, steps):
        rig = self

        class ModelHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                rig.requests.append(payload)
                outcome = {
                    "kind": "workflow",
                    "user_message": "准备抓取苹果。",
                    "summary": "抓取苹果",
                    "steps": [
                        {"schema_version": 1, "skill_name": "pick_object", "target_name": "an apple"}
                        for _ in range(steps)
                    ],
                }
                if rig.pick_place:
                    outcome["steps"].append(
                        {
                            "schema_version": 1,
                            "skill_name": "place_in_container",
                            "target_name": "an apple",
                            "container_name": "the blue bowl on the left",
                        }
                    )
                body = json.dumps(
                    {"choices": [{"message": {"content": json.dumps(outcome, ensure_ascii=False), "tool_calls": []}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        # Reuse the strict-valid, metadata-only model fixture. Catalog identity
        # validation is real even though no model is loaded for inference.
        _write_graspgen_bundle(root / "models")
        config = {
            "enabled": True,
            "backend": f"{__name__}:make_mock_backends",
            "scenario": scenario,
            "model_bundle_path": str(root / "models" / "graspgen"),
            "model_deployment": "torch_cpu",
            "pipeline": {"max_attempts": 3, "max_regrasps": 1, "max_moved_replans": 1, "execute_reserve_s": 0.05},
        }
        if self.pick_place:
            config["carry_mock_id"] = self.carry_mock_id
        if self.held_lease_s is not None:
            config["held_lease_s"] = self.held_lease_s
        placement = {"enabled": True}
        place_action = f"/contract_{uuid.uuid4().hex}/execute_place"
        if self.pick_place:
            placement.update(
                {
                    "backend": f"{__name__}:make_mock_place_backends",
                    "carry_mock_id": self.carry_mock_id,
                    "scenario": self.place_scenario,
                    "pipeline": {"max_attempts": 1, "execute_reserve_s": 0.05},
                }
            )
        pick_action = f"/contract_{uuid.uuid4().hex}/execute_pick"
        self.pick = None
        if scenario != "server_unavailable":
            self.pick = GraspPipelineNode(
                parameter_overrides=parameters({"grasp_execution_json": json.dumps(config), "action_name": pick_action})
            )
            self.nodes.append(self.pick)
            self.mock = self.pick.contract_mock
        else:
            self.mock = MockGrasp(config)
        self.place = None
        if self.pick_place:
            self.place = PlacePipelineNode(
                parameter_overrides=parameters(
                    {
                        "placement_execution_json": json.dumps(placement),
                        "action_name": place_action,
                    }
                )
            )
            self.nodes.append(self.place)
        self.io = Node("contract_mock_io")
        self.nodes.append(self.io)
        group = ReentrantCallbackGroup()

        def validate(request, response):
            response.allowed = True
            response.actual_registry_epoch = request.dispatch_binding.expected_registry_epoch
            response.actual_registry_generation = request.dispatch_binding.expected_registry_generation
            response.actual_registry_digest = request.dispatch_binding.expected_registry_digest
            return response

        self.io.create_service(ValidateSkill, "/embodied/validate_skill", validate, callback_group=group)

        def synthesize(request, response):
            self.speech.append(request.text)
            self.speech_gate.wait(timeout=5.0)
            response.success = not self.tts_failure
            if response.success:
                response.audio_segments = [SynthesizedAudio(audio_data=[0, 1], audio_format="wav")]
            return response

        def playback(request, response):
            self.played.append(Path(request.file_path).read_bytes())
            response.success = True
            return response

        self.io.create_service(SynthesizeSpeech, "/voice_tts/synthesize", synthesize, callback_group=group)
        self.io.create_service(PlayAudioFile, "/voice_tts/play", playback, callback_group=group)
        catalog_root = Path(__file__).resolve().parents[2] / "skill_catalog"
        self.gateway = SkillExecutorNode(
            parameter_overrides=parameters(
                {
                    "robot_name": "aimdk_x2_grasp",
                    "active_control_mode": "moveit_planning",
                    "skill_required_control_mode": "moveit_planning",
                    "motion_authorized": True,
                    "named_poses_json": '{"home": {}}',
                    "default_skill_timeout_sec": 15.0,
                    "task_budget_sec": 180.0,
                    "rpc_timeout_sec": 0.5,
                    "robot_state_freshness_sec": 1.0,
                    "scene_freshness_sec": 1.0,
                    "model_idle_timeout_sec": 15.0,
                    "gripper_settle_sec": 0.1,
                    "grasp_execution_json": json.dumps(config),
                    # Pick-only fixtures must also register the placement
                    # descriptor advertised by the shared X2 profile.
                    "placement_execution_json": json.dumps(placement),
                    "pick_action_name": pick_action,
                    "place_action_name": place_action,
                    "skill_catalog_profile": "aimdk_x2_grasp",
                    "skill_catalog_source_mode": "development",
                    "skill_catalog_source_root": str(catalog_root),
                    "execution_store_path": str(root / "executions.sqlite3"),
                    "execution_caller_scopes_json": '["contract_test"]',
                }
            )
        )
        self.nodes.append(self.gateway)
        self.agent = IncubationAgentNode(
            parameter_overrides=parameters(
                {
                    "ledger_path": str(root / "requests.sqlite3"),
                    "conversation_path": str(root / "conversation.sqlite3"),
                    "deployment_lock_path": str(root / "agent.lock"),
                    "execution_enabled": True,
                    "robot_scope": "aimdk_x2_grasp",
                    "channel_id": "text",
                    "principal_id": "operator",
                    "allowed_skills_json": json.dumps(
                        ["pick_object", "place_in_container"] if self.pick_place else ["pick_object"]
                    ),
                    "simulation_mode": True,
                    "rpc_timeout_sec": 2.0,
                    "task_budget_sec": 1.0 if scenario == "deadline" else 15.0,
                    "gateway_caller_scope": "contract_test",
                    "gateway_caller_token_env": "",
                    "planner_config_json": json.dumps(
                        {
                            "mode": "vlm",
                            "provider": "openai_compatible",
                            "base_url": f"http://127.0.0.1:{self.server.server_port}/v1",
                            "api_key_env": "",
                            "model": "deterministic-contract-mock",
                            "api_protocol": "chat_completions",
                            "timeout_sec": 3.0,
                        }
                    ),
                }
            )
        )
        self.nodes.append(self.agent)
        self.voice = _build_node_class()(
            parameter_overrides=parameters(
                {
                    "robot_scope": "aimdk_x2_grasp",
                    "channel_id": "text",
                    "principal_id": "operator",
                    "feedback_only": True,
                    "tts_enabled": True,
                    "mute_asr_during_tts": False,
                    "tts_timeout_sec": 1.0,
                }
            )
        )
        self.nodes.append(self.voice)
        self.io.create_subscription(String, "/agent/event", lambda msg: self.events.append(json.loads(msg.data)), 128)
        self.io.create_subscription(
            String, "/agent/response", lambda msg: self.responses.append(json.loads(msg.data)), 128
        )
        self.input = self.io.create_publisher(String, "/agent/request", 10)
        self.control = self.io.create_publisher(String, "/agent/control", 10)
        for node in self.nodes:
            self.executor.add_node(node)

        def spin():
            while not self.spin_stop.is_set():
                self.executor.spin_once(timeout_sec=0.1)

        self.thread = threading.Thread(target=spin, daemon=True)
        self.thread.start()
        # DDS discovery shares CPU with other colcon package tests. Allow a
        # separate startup budget without extending any execution deadline.
        wait_until(lambda: self.agent._gateway_ready, timeout=30.0)
        if self.pick is not None:
            wait_until(self.gateway._pick_client.server_is_ready)
        if self.place is not None:
            wait_until(self.gateway._place_client.server_is_ready)
        wait_until(lambda: self.input.get_subscription_count() > 0)

    def submit(self, request_id=None):
        from datetime import datetime, timezone

        request_id = request_id or f"mock-{uuid.uuid4().hex}"
        envelope = {
            "schema_version": 1,
            "request_id": request_id,
            "session_id": request_id,
            "channel_id": "text",
            "principal_id": "operator",
            "robot_scope": "aimdk_x2_grasp",
            "text": "帮我抓取苹果并放进左边的蓝色碗里" if self.pick_place else "帮我抓取苹果",
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        self.input.publish(String(data=json.dumps(envelope, ensure_ascii=False)))
        return request_id, envelope

    def terminal(self, request_id):
        return wait_until(
            lambda: next(
                (
                    event
                    for event in reversed(self.events)
                    if event.get("request_key", {}).get("request_id") == request_id
                    and event.get("state")
                    in {"SUCCEEDED", "FAILED", "UNKNOWN", "CANCELLED", "CANCELLED_BEFORE_EXECUTION"}
                ),
                None,
            ),
            20,
        )

    def query(self, request_id):
        return self.agent._bridge.get_execution(request_id=request_id, execution_id="", timeout_sec=2.0)

    def close(self):
        self.speech_gate.set()
        if self.thread:
            # Leave the native wait set before shutdown destroys its guards.
            self.spin_stop.set()
            self.executor.wake()
            self.thread.join(timeout=5.0)
            assert not self.thread.is_alive(), "contract executor did not stop"
        assert self.executor.shutdown(timeout_sec=5.0), "contract callbacks did not finish"
        # Humble's executor shutdown does not join its ThreadPoolExecutor.
        # Drain submitted handlers before destroying nodes or their context.
        self.executor._executor.shutdown(wait=True)
        for node in reversed(self.nodes):
            if isinstance(node, IncubationAgentNode):
                bridge_executor = node._bridge._executor
                node.close()
                if bridge_executor is not None:
                    bridge_executor._executor.shutdown(wait=True)
            node.destroy_node()
        # Humble's executor can retain its last wait-set generator after spin
        # exits. Release it while the context still exists, before another rig
        # registers clients and services with the same public names.
        self.executor = None
        gc.collect()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        _CARRY_MOCKS.pop(self.carry_mock_id, None)
        # The rig contains node/callback cycles. Collect them before another
        # test initializes the default ROS context and reuses native handles.
        self.nodes.clear()
        self.pick = self.place = self.io = self.gateway = self.agent = self.voice = None
        gc.collect()
