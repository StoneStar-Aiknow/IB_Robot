"""Real-ROS fake Gateway regression for online planning without local compile.

The fixture serves the public Gateway services the CLI calls and deliberately no
motion action: a ``PrepareAgentPlan`` mismatch must fail between presentation and
confirmation, and neither ``ExecuteAgentPlan`` nor ``ExecuteProgram`` may receive
a goal. The bound configuration is the repository G1 preset, which has no local
interface description, model directory or skill source.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from ibrobot_msgs.action import ExecuteAgentPlan, ExecuteProgram
from ibrobot_msgs.srv import (
    GetSkillGatewayStatus,
    GetSkillSnapshot,
    PlanAgentCommand,
    PrepareAgentPlan,
)
from robot_config.loader import load_robot_config_dict
from robot_skill_cli import cli
from robot_skill_cli.catalog import compile_local_snapshot

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "test" / "fixtures" / "so101_single_arm_offline.yaml"

STATUS_SERVICE = "/embodied/get_skill_gateway_status"
SNAPSHOT_SERVICE = "/embodied/get_skill_snapshot"
PLAN_SERVICE = "/embodied/plan_agent_command"
PREPARE_PLAN_SERVICE = "/embodied/prepare_agent_plan"
EXECUTE_PLAN_ACTION = "/embodied/execute_agent_plan"
EXECUTE_PROGRAM_ACTION = "/embodied/execute_program"

ROBOT_NAME = "unitree_g1_arm_presets"
REGISTRY_EPOCH = "epoch-online-plan"
REGISTRY_GENERATION = 1
STEP = {
    "schema_version": 1,
    "skill_name": "move_relative_ee",
    "motion_direction": "forward",
    "motion_distance": 0.03,
}


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def _gateway_snapshot():
    return compile_local_snapshot(load_robot_config_dict(CONFIG_PATH), CONFIG_PATH)


class _FakePlanGateway:
    def __init__(self, snapshot, *, suffix):
        self.snapshot = snapshot
        self.node = rclpy.create_node(f"online_plan_gateway_{suffix}")
        self.plan_requests = []
        self.prepare_requests = []
        self.execute_plan_goals = 0
        self.execute_program_goals = 0
        self.node.create_service(GetSkillGatewayStatus, STATUS_SERVICE, self._status)
        self.node.create_service(GetSkillSnapshot, SNAPSHOT_SERVICE, self._snapshot)
        self.node.create_service(PlanAgentCommand, PLAN_SERVICE, self._plan)
        self.node.create_service(PrepareAgentPlan, PREPARE_PLAN_SERVICE, self._prepare)
        self.execute_plan_server = ActionServer(
            self.node,
            ExecuteAgentPlan,
            EXECUTE_PLAN_ACTION,
            self._execute_plan,
        )
        self.execute_program_server = ActionServer(
            self.node,
            ExecuteProgram,
            EXECUTE_PROGRAM_ACTION,
            self._execute_program,
        )

    def _status(self, _request, response):
        response.schema_version = 1
        response.robot_name = ROBOT_NAME
        response.motion_authorized = False
        response.active_control_mode = ""
        response.busy = False
        response.default_skill_timeout_sec = 30.0
        response.task_budget_sec = 180.0
        response.rpc_timeout_sec = 5.0
        response.config_digest = self.snapshot.capability_digest
        response.registry_epoch = REGISTRY_EPOCH
        response.registry_generation = REGISTRY_GENERATION
        response.registry_digest = self.snapshot.registry_digest
        response.capability_digest = self.snapshot.capability_digest
        response.provenance_digest = self.snapshot.provenance_digest
        response.control_plane_ready = True
        response.control_plane_state = "READY"
        return response

    def _snapshot(self, _request, response):
        response.success = True
        response.registry_epoch = REGISTRY_EPOCH
        response.generation = REGISTRY_GENERATION
        response.registry_digest = self.snapshot.registry_digest
        response.capability_digest = self.snapshot.capability_digest
        response.provenance_digest = self.snapshot.provenance_digest
        response.snapshot_json = self.snapshot.snapshot_json
        return response

    def _plan(self, request, response):
        self.plan_requests.append(request)
        response.success = True
        response.plan = self._plan_message(request)
        return response

    def _prepare(self, request, response):
        self.prepare_requests.append(request)
        response.success = True
        response.allowed = True
        # Deliberately differ from the requested distance so the controller
        # fails its exact-plan check before confirmation or execution.
        plan = self._plan_message(request)
        for step in plan.workflow_steps:
            if step.skill_name == STEP["skill_name"]:
                step.motion_distance = 0.04
        response.plan = plan
        return response

    def _plan_message(self, request):
        from ibrobot_msgs.msg import AgentPlan

        plan = AgentPlan()
        plan.schema_version = 1
        plan.plan_id = "plan-online-plan"
        plan.plan_token = "token-online-plan"
        plan.plan_kind = 2
        plan.raw_command = request.raw_command
        plan.workflow_steps = list(request.workflow_steps)
        plan.plan_digest = "digest-online-plan"
        plan.registry_epoch = REGISTRY_EPOCH
        plan.registry_generation = REGISTRY_GENERATION
        plan.registry_digest = self.snapshot.registry_digest
        plan.execution_mode = request.execution_mode
        return plan

    def _execute_plan(self, goal_handle):
        self.execute_plan_goals += 1
        return ExecuteAgentPlan.Result()

    def _execute_program(self, goal_handle):
        self.execute_program_goals += 1
        return ExecuteProgram.Result()

    def close(self):
        self.execute_plan_server.destroy()
        self.execute_program_server.destroy()
        self.node.destroy_node()


def _spin(gateway):
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(gateway.node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread


def test_online_plan_prepare_and_query_do_not_reach_a_motion_action(monkeypatch, capsys):
    _assert_isolated_ros_domain()
    rclpy.init()
    snapshot = _gateway_snapshot()
    gateway = _FakePlanGateway(snapshot, suffix=f"{os.getpid()}_{uuid.uuid4().hex[:8]}")
    executor, _thread = _spin(gateway)
    try:
        assert (
            cli.main(
                [
                    "--config-name",
                    ROBOT_NAME,
                    "plan-workflow",
                    "--text",
                    "向前移动",
                    "--request-id",
                    "request-online-plan",
                    "--workflow-json",
                    json.dumps([STEP]),
                ]
            )
            == 0
        )
        planned = json.loads(capsys.readouterr().out)
        assert planned["data"]["plan"]["plan_id"] == "plan-online-plan"
        assert len(gateway.plan_requests) == 1
        assert gateway.plan_requests[0].request_id == "request-online-plan"
        assert gateway.plan_requests[0].workflow_steps[0].skill_name == STEP["skill_name"]

        assert cli.main(["--config-name", ROBOT_NAME, "status"]) == 0
        assert cli.main(["--config-name", ROBOT_NAME, "list-skills"]) == 0
        capsys.readouterr()
        assert gateway.execute_plan_goals == 0
        assert gateway.execute_program_goals == 0

        assert (
            cli.main(
                [
                    "--config-name",
                    ROBOT_NAME,
                    "run-workflow",
                    "--text",
                    "向前移动",
                    "--request-id",
                    "request-online-run",
                    "--workflow-json",
                    json.dumps([STEP]),
                ]
            )
            == 13
        )
        ran = json.loads(capsys.readouterr().out)
        assert ran["error"]["code"] == "SKILL_SNAPSHOT_DIGEST_MISMATCH"
        assert len(gateway.prepare_requests) == 1
        assert gateway.prepare_requests[0].request_id == "request-online-run"
        assert gateway.prepare_requests[0].execution_mode == "immediate_after_presentation"
        assert gateway.execute_plan_goals == 0
        assert gateway.execute_program_goals == 0
    finally:
        gateway.close()
        executor.shutdown()
        rclpy.shutdown()
