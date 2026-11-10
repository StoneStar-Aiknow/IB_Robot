"""Real-ROS mock Gateway: a registry restart re-keys cached previews.

This fixture is deliberately minimal: it provides only the services and the
Program action the client touches on the preview/submit path, and a mocked
GetExecution ledger. The plan pipeline (snapshot, plan, confirm) is replaced by
a small controller stub so the test does not rebuild the Gateway. No robot is
connected and no motion is commanded.

Covered contract:
* a preview cached against registry identity A is rejected after a restart that
  publishes identity B on the same endpoints, before presentation or a goal;
* no ExecuteProgram goal is sent for the rejected preview;
* the same endpoints with the new identity accept a freshly prepared preview;
* an unknown execution outcome keeps the target locked and is never replayed.

The client binds one deployment and starts its transport on the first
operation, so the fixture never starts a second bridge behind its back.
"""

import contextlib
import copy
import os
import threading
import time
import uuid

import pytest

from robot_client.client import RobotClient
from robot_client.errors import ClientError
from robot_client.transport import gateway_transport


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and 1 <= int(allocated) <= 232
    assert os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


class MockGateway:
    """One restartable Gateway process publishing status, program and ledger."""

    def __init__(self) -> None:
        import rclpy
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.context import Context

        from ibrobot_msgs.action import ExecuteProgram
        from ibrobot_msgs.srv import GetExecution, GetSkillGatewayStatus

        self._rclpy = rclpy
        self._ActionServer = ActionServer
        self._CancelResponse = CancelResponse
        self._GoalResponse = GoalResponse
        self._ExecuteProgram = ExecuteProgram
        self._GetExecution = GetExecution
        self._GetSkillGatewayStatus = GetSkillGatewayStatus
        self.context = Context()
        rclpy.init(context=self.context)
        suffix = f"registry_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.names = {
            "status": f"/{suffix}/gateway_status",
            "program": f"/{suffix}/execute_program",
            "execution": f"/{suffix}/get_execution",
        }
        self.identity = {"epoch": "epoch-a", "generation": 1, "digest": "digest-a"}
        self.goals = []
        self.ledger = {"found": False, "state": "", "execution_id": ""}
        self.node = None
        self.action_server = None
        self.executor = None
        self.thread = None
        self._start()

    def _start(self) -> None:
        suffix = self.names["status"].rsplit("/", 2)[-2]
        self.node = self._rclpy.create_node(
            f"mock_gateway_{suffix}_{uuid.uuid4().hex[:6]}",
            context=self.context,
            use_global_arguments=False,
        )
        self.node.create_service(self._GetSkillGatewayStatus, self.names["status"], self._status)
        self.node.create_service(self._GetExecution, self.names["execution"], self._execution)

        def goal_callback(_goal_request):
            return self._GoalResponse.ACCEPT

        def cancel_callback(_goal_handle):
            return self._CancelResponse.ACCEPT

        def execute(goal_handle):
            self.goals.append(copy.deepcopy(goal_handle.request))
            result = self._ExecuteProgram.Result()
            result.success = True
            result.state = "SUCCEEDED"
            result.completed_step_count = len(goal_handle.request.workflow_steps)
            result.execution_id = "execution-1"
            goal_handle.succeed()
            return result

        self.action_server = self._ActionServer(
            self.node,
            self._ExecuteProgram,
            self.names["program"],
            execute_callback=execute,
            goal_callback=goal_callback,
            cancel_callback=cancel_callback,
        )
        self.executor = self._rclpy.executors.MultiThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, name="mock-gateway", daemon=True)
        self.thread.start()

    def _status(self, _request, response):
        response.schema_version = 1
        response.robot_name = "g1"
        response.motion_authorized = True
        response.control_plane_ready = True
        response.control_plane_state = "READY"
        response.registry_epoch = self.identity["epoch"]
        response.registry_generation = self.identity["generation"]
        response.registry_digest = self.identity["digest"]
        response.capability_digest = self.identity["digest"]
        response.config_digest = self.identity["digest"]
        response.provenance_digest = ""
        response.source_release_digest = ""
        response.primitive_contract_digest = ""
        return response

    def _execution(self, _request, response):
        response.found = self.ledger["found"]
        response.state = self.ledger["state"]
        response.execution_id = self.ledger["execution_id"]
        return response

    def restart(self, *, epoch: str, generation: int = 1, digest: str = "digest-b") -> None:
        self.identity = {"epoch": epoch, "generation": generation, "digest": digest}
        self.stop(keep_context=True)
        self._start()

    def stop(self, *, keep_context: bool = False) -> None:
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=2.0)
            self.executor = None
        if self.action_server is not None:
            self.action_server.destroy()
            self.action_server = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if not keep_context:
            with contextlib.suppress(Exception):
                self._rclpy.shutdown(context=self.context)


class StubController:
    """Plan pipeline stand-in: the mock Gateway's identity is read for real."""

    def __init__(self, bridge, **_kwargs):
        self.bridge = bridge
        self.plans = 0

    def discover(self):
        pass

    def prepare_workflow(self, _raw, steps, *, request_id):
        status = self.bridge.get_status(timeout_sec=10.0)
        self.plans += 1
        return {
            "steps": copy.deepcopy(steps),
            "registry_identity": {
                "registry_epoch": status["registry_epoch"],
                "registry_generation": status["registry_generation"],
                "registry_digest": status["registry_digest"],
            },
            "plan_id": f"plan-{self.plans}",
            "proposed_task_budget_sec": 30.0,
        }

    def confirm_plan(self):
        return {"confirmed_task_budget_sec": 30.0, "stopped_before_execution": False}


@pytest.fixture
def gateway_rig():
    _assert_isolated_ros_domain()
    from robot_client.transports.ros_bridge import RosBridge

    gateway = MockGateway()
    transport = gateway_transport(
        {
            "skill_gateway_status_service": gateway.names["status"],
            "execute_program_action": gateway.names["program"],
            "get_execution_service": gateway.names["execution"],
        },
        robot_id="g1",
        robot_name="g1",
    )
    client = RobotClient(
        transport,
        transport_factory=lambda **kwargs: RosBridge(**kwargs),
    )
    yield gateway, client
    client.close()
    gateway.stop()


def _wait(predicate, *, timeout_sec=15.0, interval=0.1):
    deadline = time.monotonic() + timeout_sec
    last = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:  # noqa: BLE001 - discovery converges asynchronously
            last = exc
        time.sleep(interval)
    raise AssertionError(f"condition not reached: {last!r}")


@pytest.fixture
def stub_planner(monkeypatch):
    from robot_client import workflow

    monkeypatch.setattr(workflow, "InteractiveController", StubController)


def _prepare(client):
    return client.prepare_workflow(
        {
            "workflow_steps": [{"skill_name": "wave_hand", "schema_version": 1}],
            "raw_command": "wave",
            "request_id": "request-registry-1",
        }
    )


def test_registry_restart_rejects_cached_preview_and_accepts_a_new_one(gateway_rig, stub_planner):
    gateway, client = gateway_rig
    _wait(lambda: client.get_status()["registry_epoch"] == "epoch-a")
    preview = _prepare(client)
    assert preview["gateway_identity"]["registry_epoch"] == "epoch-a"
    assert gateway.goals == []

    # Same endpoints, new Gateway process identity.
    gateway.restart(epoch="epoch-b")
    _wait(lambda: client.get_status()["registry_epoch"] == "epoch-b")

    presented = []
    with pytest.raises(ClientError) as rejected:
        client.submit_prepared(
            preview["preview_id"],
            presenter=presented.append,
            persist_reference=lambda _value: None,
        )
    assert rejected.value.code == "STALE_PREVIEW"
    assert presented == []
    assert gateway.goals == []

    # The same endpoints under the new identity accept a fresh preview.
    fresh = _prepare(client)
    assert fresh["gateway_identity"]["registry_epoch"] == "epoch-b"
    reference = client.submit_prepared(
        fresh["preview_id"],
        presenter=presented.append,
        persist_reference=lambda _value: None,
    )
    _wait(lambda: len(gateway.goals) == 1)
    assert presented and presented[0]["preview_id"] == fresh["preview_id"]
    assert reference["target_key"] == client.session["target_key"]
    assert gateway.goals[0].expected_registry_epoch == "epoch-b"
    assert gateway.goals[0].expected_registry_generation == 1
    assert gateway.goals[0].expected_registry_digest == "digest-b"
    assert gateway.goals[0].request_id == "request-registry-1"


def test_unknown_execution_keeps_the_target_locked_without_replay(gateway_rig, stub_planner):
    gateway, client = gateway_rig
    _wait(lambda: client.get_status()["registry_epoch"] == "epoch-a")
    preview = _prepare(client)
    reference = client.submit_prepared(
        preview["preview_id"],
        presenter=lambda _value: None,
        persist_reference=lambda _value: None,
    )
    _wait(lambda: len(gateway.goals) == 1)

    observed = client.get_execution(reference)
    assert observed["observation_unknown"] is True
    assert client._unresolved, "an unknown outcome must keep the target locked"
    with pytest.raises(ClientError) as blocked:
        _prepare(client)
    assert blocked.value.code == "EXECUTION_UNRESOLVED"
    assert len(gateway.goals) == 1

    # Ledger reconciliation releases the lock; nothing is re-sent.
    gateway.ledger.update({"found": True, "state": "SUCCEEDED", "execution_id": "execution-1"})
    reconciled = client.get_execution(reference)
    assert reconciled["observation_unknown"] is False
    assert client._unresolved == {}
    assert len(gateway.goals) == 1
