"""Behaviour of the pose_composer delegated executor, without a ROS graph."""

from types import SimpleNamespace

import pytest
from test_runtime_named_motion_executor import _Client, _Future, _GoalHandle, _MotionHandle, _node

from skill_library import skill_executor_node as executor_module
from skill_library.skill_executor_node import PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, SkillExecutorNode

TEMPLATE = {"executor": "pose_composer", "required_args": ["target_name"], "timeout_sec": 5.0}


@pytest.fixture(autouse=True)
def _rclpy_running(monkeypatch):
    monkeypatch.setattr(executor_module.rclpy, "ok", lambda: True)


def _pose_node(client):
    node = _node(None)
    node._pose_composer_client = client
    node._pose_composer_action = "/pose_composer/execute"
    return node


def _handle(text="双手向前伸直", **kw):
    handle = _GoalHandle(skill_name="free_pose", **kw)
    handle.request.target_name = text
    return handle


def _pose_result(*, success=True, error_code="", message="好的；moved in 4.1s"):
    return SimpleNamespace(result=SimpleNamespace(success=success, error_code=error_code, message=message))


def _run(node, handle, timeout=5.0):
    return node._execute_pose_composer_skill(handle, TEMPLATE, effective_timeout_sec=timeout)


def test_the_description_is_sent_and_success_reported():
    client = _Client(_Future(_MotionHandle(_Future(_pose_result()))))
    node = _pose_node(client)
    handle = _handle()
    result = _run(node, handle)
    assert client.goals[0].text == "双手向前伸直"
    assert result.success and handle.terminal == "succeeded"
    assert result.executed_primitives == ["pose_composer:free_pose"]
    assert node.confirmed == node.registered


def test_an_empty_description_is_refused_before_dispatch():
    client = _Client(_Future(None))
    result = _run(_pose_node(client), _handle(text="  "))
    assert result.error_code == "POSE_EMPTY_TEXT"
    assert client.goals == []


def test_a_disabled_executor_fails_closed():
    result = _run(_pose_node(None), _handle())
    assert result.error_code == "POSE_COMPOSER_UNAVAILABLE"


def test_a_composer_failure_keeps_its_code():
    failed = _pose_result(success=False, error_code="POSE_NO_SAFE_PATH", message="no collision-free path")
    result = _run(_pose_node(_Client(_Future(_MotionHandle(_Future(failed))))), _handle())
    assert (result.success, result.error_code, result.message) == (False, "POSE_NO_SAFE_PATH", "no collision-free path")


def test_a_cancel_is_forwarded_and_ends_the_skill_cancelled():
    stopped = _pose_result(success=False, error_code="POSE_CANCELLED", message="stopped; arms held where they were")
    pose = _MotionHandle(_Future(stopped, done_after_polls=3), cancel_accepted=True)
    node = _pose_node(_Client(_Future(pose)))
    node._audit_cancel_propagated = lambda: node.audits.append("cancel_propagated")
    handle = _handle(cancel_requested=True)
    result = _run(node, handle)
    assert pose.cancel_calls == 1
    assert handle.terminal == "canceled"
    assert result.error_code == "SKILL_CANCELLED" and "held" in result.message


def test_the_deadline_cancels_and_holds_the_admission_until_the_end():
    pose = _MotionHandle(_Future(None, done_after_polls=None))
    node = _pose_node(_Client(_Future(pose)))
    result = _run(node, _handle(), timeout=0.0)
    assert result.error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
    assert pose.cancel_calls == 1
    assert node.confirmed == []


@pytest.mark.parametrize(("enabled", "in_catalog"), [(True, False), (False, True)])
def test_runtime_and_catalog_must_agree_on_the_executor(enabled, in_catalog):
    node = object.__new__(SkillExecutorNode)
    node._runtime_named_motion_enabled = False
    node._pose_composer_enabled = enabled
    templates = {"free_pose": dict(TEMPLATE)} if in_catalog else {}
    snapshot = SimpleNamespace(templates=templates, enabled_skill_names=tuple(templates))
    with pytest.raises(ValueError, match="pose_composer configuration mismatch"):
        node._validate_hri_runtime_catalog_consistency(snapshot)
