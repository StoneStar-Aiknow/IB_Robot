from types import SimpleNamespace

from manipulation_execution import pick_action_client


class _Future:
    def __init__(self, *, done: bool, result=None) -> None:
        self._done = done
        self._result = result
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._result

    def cancel(self) -> None:
        self.cancelled = True
        self._done = True


def _successful_result():
    return SimpleNamespace(
        success=True,
        error_code="",
        candidate_index=7,
        attempts=1,
        verification_status=1,
        verification_confidence=1.0,
        released_after_success=True,
        debug_output_dir="",
        pipeline_timings_json="{}",
        message="ok",
    )


def test_missing_goal_response_recovers_result_by_original_uuid(monkeypatch):
    send_future = _Future(done=False)
    result_future = _Future(done=True, result=SimpleNamespace(result=_successful_result()))
    observed = {}

    class FakeActionClient:
        def wait_for_server(self, timeout_sec):
            observed["ready_timeout"] = timeout_sec
            return True

        def send_goal_async(self, goal, feedback_callback, goal_uuid):
            observed["goal_uuid"] = goal_uuid
            feedback_callback(
                SimpleNamespace(feedback=SimpleNamespace(phase="planning", progress=0.2, attempt=0, detail="planning"))
            )
            return send_future

    class RecoveredGoalHandle:
        def get_result_async(self):
            return result_future

    monkeypatch.setattr(pick_action_client.rclpy, "spin_until_future_complete", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pick_action_client,
        "ClientGoalHandle",
        lambda action_client, goal_uuid, response: (
            observed.update(action_client=action_client, recovered_uuid=goal_uuid, accepted=response.accepted)
            or RecoveredGoalHandle()
        ),
    )

    node = pick_action_client.PickActionClient.__new__(pick_action_client.PickActionClient)
    node._action_name = "/manipulation/execute_pick"
    node._client = FakeActionClient()
    result = pick_action_client.PickActionClient.execute(
        node,
        task_id="",
        target_query="marker",
        timeout_sec=200.0,
        mode="execute",
        release_after_success=True,
        release_drop_height_m=0.015,
        ready_timeout_sec=30.0,
        goal_response_timeout_sec=10.0,
    )

    assert result.success is True
    assert send_future.cancelled is True
    assert observed["recovered_uuid"] == observed["goal_uuid"]
    assert observed["accepted"] is True


def test_main_sends_exactly_one_goal(monkeypatch):
    instances = []

    class FakePickActionClient:
        def __init__(self, action_name):
            self.action_name = action_name
            self.calls = []
            self.destroyed = False
            instances.append(self)

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return _successful_result()

        def destroy_node(self):
            self.destroyed = True

    monkeypatch.setattr(pick_action_client, "PickActionClient", FakePickActionClient)
    monkeypatch.setattr(pick_action_client.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(pick_action_client.rclpy, "shutdown", lambda: None)

    pick_action_client.main(
        [
            "--prompt",
            "marker",
            "--task-id",
            "single-test",
        ]
    )

    assert len(instances) == 1
    assert [call["task_id"] for call in instances[0].calls] == ["single-test"]
    assert instances[0].destroyed is True


def test_parser_does_not_expose_repeat_options():
    option_strings = {
        option for action in pick_action_client.build_parser()._actions for option in action.option_strings
    }

    assert "--repeat" not in option_strings
    assert "--repeat-delay-s" not in option_strings
