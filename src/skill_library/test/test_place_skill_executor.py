from types import SimpleNamespace

from embodied_common.dispatch_binding import new_binding
from ibrobot_msgs.action import PlaceObject, SkillCommand
from skill_library.skill_executor_node import SkillExecutorNode


def _result(release, verification, error_code=""):
    return SimpleNamespace(
        release_status=release,
        verification_status=verification,
        error_code=error_code,
    )


def test_place_public_errors_preserve_release_state():
    assert (
        SkillExecutorNode._place_public_error(
            _result(PlaceObject.Result.RELEASE_RELEASED, PlaceObject.Result.VERIFICATION_FAILED)
        )
        == "PLACE_RELEASED_VERIFICATION_FAILED"
    )
    assert (
        SkillExecutorNode._place_public_error(
            _result(PlaceObject.Result.RELEASE_RELEASED, PlaceObject.Result.VERIFICATION_UNCERTAIN)
        )
        == "PLACE_RELEASED_VERIFICATION_UNCERTAIN"
    )
    assert (
        SkillExecutorNode._place_public_error(
            _result(PlaceObject.Result.RELEASE_UNKNOWN, PlaceObject.Result.VERIFICATION_NOT_RUN)
        )
        == "PLACE_RELEASE_STATE_UNKNOWN"
    )


def test_not_released_has_distinct_public_error():
    result = _result(
        PlaceObject.Result.RELEASE_NOT_RELEASED,
        PlaceObject.Result.VERIFICATION_NOT_RUN,
        "PRIMITIVE_FAILED",
    )
    assert SkillExecutorNode._place_public_error(result) == "PLACE_NOT_RELEASED"


def test_unknown_primitive_stop_state_is_preserved_for_hermes():
    result = _result(
        PlaceObject.Result.RELEASE_NOT_RELEASED,
        PlaceObject.Result.VERIFICATION_NOT_RUN,
        "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
    )
    assert SkillExecutorNode._place_public_error(result) == "CANCEL_CLEANUP_TIMEOUT"


def test_place_delegation_forwards_runtime_target_and_container_queries():
    sent_goals = []

    class PlaceClient:
        @staticmethod
        def wait_for_server(**_kwargs):
            return True

        @staticmethod
        def send_goal_async(goal, **_kwargs):
            sent_goals.append(goal)
            raise RuntimeError("stop after inspecting the delegated goal")

    executor = SimpleNamespace(
        name="placement_pipeline",
        contract_version="1",
        endpoint_kind="ros_action",
        endpoint_name="/manipulation/execute_place",
        configuration_digest="a" * 64,
        model_deployment_name="",
        model_fingerprint="",
        model_bundle_digest="",
    )
    node = object.__new__(SkillExecutorNode)
    node._place_client = PlaceClient()
    node._place_action_name = "/manipulation/execute_place"
    node._rpc_timeout = 0.01
    node._active_runtime_bundle = SimpleNamespace(
        snapshot=SimpleNamespace(delegated_executors={"placement_pipeline": executor})
    )
    node._active_skill_admission = object()
    node._register_delegated_dispatch = lambda *_args: b"cleanup"
    node._confirm_delegated_terminal = lambda *_args: None
    node._abort_skill = lambda _result, _handle, _phases, code, message: SimpleNamespace(
        success=False,
        error_code=code,
        message=message,
    )
    node.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    goal = SkillCommand.Goal()
    goal.dispatch_binding = new_binding(task_id="place-1")
    goal.skill_name = "place_in_container"
    goal.target_name = "marker"
    goal.container_name = "black bowl"
    goal.timeout_sec = 10.0

    result = node._execute_place_skill(SimpleNamespace(request=goal), {"timeout_sec": 10.0})

    assert result.success is False
    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert len(sent_goals) == 1
    assert sent_goals[0].target_query == "marker"
    assert sent_goals[0].container_query == "black bowl"
