"""Direct behaviour tests for RobotClient.catalog_view's read-only retry loop.

The client reads status and the skill snapshot as two separate RPCs. A registry
restart or reload can land between them, so exactly three "not ready/not visible"
codes retry once after re-reading status; every other failure is final. These
tests use a fake bridge and the existing catalog document fixture, never ROS or
a robot, and assert that a catalog read never reaches prepare/confirm/execute.
"""

from __future__ import annotations

import pytest
from test_lazy_bridge_start import ROBOT_NAME, SKILL_NAMES, _catalog_documents

from embodied_common.catalog_digest import to_canonical_json
from robot_client.client import RobotClient
from robot_client.errors import ClientError
from robot_client.transport import gateway_transport

PAYLOAD, DIGESTS = _catalog_documents()
SNAPSHOT_JSON = to_canonical_json(PAYLOAD)

ALLOWED_RETRY_CODES = (
    "SKILL_SNAPSHOT_NOT_FOUND",
    "SKILL_REGISTRY_EPOCH_MISMATCH",
    "SKILL_REGISTRY_NOT_READY",
)
OTHER_ERROR_CODES = (
    "PERMISSION_DENIED",
    "UNKNOWN_ERROR",
    "SKILL_SNAPSHOT_DIGEST_MISMATCH",
)


def _status(epoch: str, generation: int) -> dict:
    return {
        "robot_name": ROBOT_NAME,
        "registry_epoch": epoch,
        "registry_generation": generation,
        "registry_digest": DIGESTS["registry_digest"],
        "capability_digest": DIGESTS["capability_digest"],
        "control_plane_ready": True,
        "capabilities": [],
    }


def _error(code: str) -> dict:
    return {"success": False, "error_code": code, "message": f"{code} injected for the test"}


def _success(epoch: str, generation: int) -> dict:
    return {
        "success": True,
        "registry_epoch": epoch,
        "generation": generation,
        "registry_digest": DIGESTS["registry_digest"],
        "capability_digest": DIGESTS["capability_digest"],
        "provenance_digest": DIGESTS["provenance_digest"],
        "snapshot_json": SNAPSHOT_JSON,
    }


class FakeBridge:
    """Started by RobotClient; exposes only read methods plus action traps."""

    def __init__(self, *, statuses, snapshots):
        self._statuses = list(statuses)
        self._snapshots = list(snapshots)
        self.calls: list[str] = []
        self.status_calls: list[dict] = []
        self.snapshot_calls: list[dict] = []
        self.action_calls: list[str] = []
        self.started = 0
        self.closed = 0

    def start(self):
        self.calls.append("start")
        self.started += 1
        return True

    def close(self):
        self.closed += 1

    def get_status(self, **kwargs):
        assert self.started == 1, "get_status must run on a started bridge"
        self.calls.append("status")
        self.status_calls.append(kwargs)
        if len(self._statuses) > 1:
            return self._statuses.pop(0)
        return self._statuses[0]

    def get_skill_snapshot(self, **kwargs):
        assert self.started == 1, "get_skill_snapshot must run on a started bridge"
        self.calls.append("snapshot")
        self.snapshot_calls.append(dict(kwargs))
        return self._snapshots.pop(0)

    def _action(self, name: str):
        self.action_calls.append(name)
        raise AssertionError(f"catalog_view must not call {name}")

    def plan_agent_command(self, *args, **kwargs):
        self._action("plan_agent_command")

    def validate_agent_plan(self, *args, **kwargs):
        self._action("validate_agent_plan")

    def confirm_agent_plan(self, *args, **kwargs):
        self._action("confirm_agent_plan")

    def send_agent_plan_goal(self, *args, **kwargs):
        self._action("send_agent_plan_goal")

    def send_program_goal(self, *args, **kwargs):
        self._action("send_program_goal")

    def get_execution(self, *args, **kwargs):
        self._action("get_execution")

    def cancel_execution(self, *args, **kwargs):
        self._action("cancel_execution")

    def cancel_agent_plan(self, *args, **kwargs):
        self._action("cancel_agent_plan")

    def get_agent_plan_result(self, *args, **kwargs):
        self._action("get_agent_plan_result")


def _client(bridge: FakeBridge) -> RobotClient:
    transport = gateway_transport({}, robot_id="g1", robot_name=ROBOT_NAME)
    return RobotClient(transport, transport_factory=lambda **_kwargs: bridge)


def _assert_catalog_only(bridge: FakeBridge) -> None:
    assert bridge.started == 1
    assert bridge.calls and bridge.calls[0] == "start"
    assert bridge.action_calls == []
    assert bridge.closed == 0


@pytest.mark.parametrize("code", ALLOWED_RETRY_CODES)
def test_catalog_view_retries_an_allowed_error_once_with_the_second_status(code):
    bridge = FakeBridge(
        statuses=[_status("epoch-1", 1), _status("epoch-2", 2)],
        snapshots=[_error(code), _success("epoch-2", 2)],
    )

    view = _client(bridge).catalog_view()

    assert view["robot_name"] == ROBOT_NAME
    assert [skill["name"] for skill in view["skills"]] == list(SKILL_NAMES)
    assert [call["registry_epoch"] for call in bridge.snapshot_calls] == ["epoch-1", "epoch-2"]
    assert [call["generation"] for call in bridge.snapshot_calls] == [1, 2]
    assert len(bridge.status_calls) == 2
    _assert_catalog_only(bridge)


@pytest.mark.parametrize("code", ALLOWED_RETRY_CODES)
def test_catalog_view_does_not_retry_a_second_identical_allowed_error(code):
    bridge = FakeBridge(
        statuses=[_status("epoch-1", 1)],
        snapshots=[_error(code), _error(code)],
    )

    with pytest.raises(ClientError) as raised:
        _client(bridge).catalog_view()

    assert raised.value.code == code
    assert len(bridge.status_calls) == 2
    assert len(bridge.snapshot_calls) == 2
    _assert_catalog_only(bridge)


@pytest.mark.parametrize("code", OTHER_ERROR_CODES)
def test_catalog_view_raises_other_snapshot_errors_without_a_retry(code):
    bridge = FakeBridge(
        statuses=[_status("epoch-1", 1), _status("epoch-2", 2)],
        snapshots=[_error(code)],
    )

    with pytest.raises(ClientError) as raised:
        _client(bridge).catalog_view()

    assert raised.value.code == code
    assert len(bridge.status_calls) == 1
    assert len(bridge.snapshot_calls) == 1
    _assert_catalog_only(bridge)


def test_catalog_view_does_not_retry_a_malformed_success_payload():
    malformed = _success("epoch-1", 1)
    malformed["snapshot_json"] = "{}"
    bridge = FakeBridge(
        statuses=[_status("epoch-1", 1), _status("epoch-2", 2)],
        snapshots=[malformed],
    )

    with pytest.raises(ClientError) as raised:
        _client(bridge).catalog_view()

    assert raised.value.code == "SKILL_SNAPSHOT_DIGEST_MISMATCH"
    assert len(bridge.status_calls) == 1
    assert len(bridge.snapshot_calls) == 1
    _assert_catalog_only(bridge)
