"""Every pick with a configured entry pose must prepare before dispatch."""

from types import SimpleNamespace

import pytest

from skill_library.skill_executor_node import SkillExecutorNode


class _PreparationEntered(Exception):
    pass


def _gateway():
    node = object.__new__(SkillExecutorNode)
    node._task_poses = {
        "skill_entry_poses": {"pick_object": "grasp_ready"},
        "poses": {"grasp_ready": {"timeout_sec": 30.0}},
    }
    node._execute_pick_skill = lambda goal, template, effective_timeout_sec=None: "picked"

    def prepare(*_):
        raise _PreparationEntered

    node._ensure_skill_capabilities = prepare
    return node


def _goal():
    return SimpleNamespace(
        request=SimpleNamespace(skill_name="pick_object", timeout_sec=0.0), is_cancel_requested=False
    )


def test_a_pick_first_moves_to_its_entry_pose():
    node = _gateway()
    with pytest.raises(_PreparationEntered):
        node._execute_prepared_pick_skill(_goal(), {})


def test_a_robot_without_an_entry_pose_can_dispatch_its_pick():
    node = _gateway()
    node._task_poses = {}
    assert node._execute_prepared_pick_skill(_goal(), {}) == "picked"
