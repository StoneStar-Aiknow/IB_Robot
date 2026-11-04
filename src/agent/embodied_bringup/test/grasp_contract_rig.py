"""Real Agent/Gateway/PickObject graph, with deterministic peripheral faults.

This module is test-only: no hardware driver, camera or motion publisher is
created. The root conftest reserves a localhost DDS domain before importing it.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
    Effect,
    EffectorAction,
    EffectorOutcome,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    Hold,
    MotionOutcome,
    MotionState,
    Move,
    Observation,
    Proposals,
    SegmentKind,
)
from manipulation_execution.grasp_pipeline_node import GraspPipelineNode
from skill_library.skill_executor_node import SkillExecutorNode

from .test_real_grasp_config_startup import _write_graspgen_bundle


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
        self.stops = []
        self.position = np.zeros(3)
        self.gripped = False
        self.started = threading.Event()

    def capture(self, *, deadline, cancelled):
        self.captures += 1
        time.sleep(0.02)
        # Each independent simulated scene starts with the arm at home.
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
        return EffectorOutcome(True)

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
    node.contract_mock = mock
    return GraspBackends(mock, mock, mock, mock, mock, mock)


class ContractRig:
    def __init__(self, root: Path, scenario="success", *, tts_failure=False, tts_block=False, steps=1):
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
        rclpy.init()
        self.executor = MultiThreadedExecutor(num_threads=10)
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
                    "pick_action_name": pick_action,
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
                    "allowed_skills_json": '["pick_object"]',
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
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        # DDS discovery shares CPU with other colcon package tests. Allow a
        # separate startup budget without extending any execution deadline.
        wait_until(lambda: self.agent._gateway_ready, timeout=30.0)
        if self.pick is not None:
            wait_until(self.gateway._pick_client.server_is_ready)
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
            "text": "帮我抓取苹果",
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
            self.executor.shutdown(timeout_sec=5.0)
            self.thread.join(timeout=5.0)
        for node in reversed(self.nodes):
            if isinstance(node, IncubationAgentNode):
                node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
