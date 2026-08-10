"""Scheduled reset barriers for the pick-banana simulation task."""

from __future__ import annotations

from types import SimpleNamespace

from sim_models.tasks.pick_banana import PickBananaTask


def _task() -> PickBananaTask:
    task = object.__new__(PickBananaTask)
    task._scheduled_dispatch = True
    task._disp_stop = object()
    task._disp_start = object()
    task._disp_reset = object()
    task._node = SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=lambda _message: None))
    return task


def test_scheduled_stop_failure_aborts_before_world_reset():
    task = _task()
    calls: list[str] = []
    task._call_service_sync = lambda _client, _request, label, **_kwargs: calls.append(label) or False
    task._publish_rest_pose = lambda: (_ for _ in ()).throw(
        AssertionError("rest pose and world reset must not run after failed Close")
    )

    result = PickBananaTask._clean_world_reset(task, randomize=False, resume=True, settle_s=0)

    assert result == (False, "scheduled dispatcher stop/safe-stop/Close failed; world reset aborted")
    assert calls == ["dispatcher/stop"]


def test_scheduled_open_failure_is_reported_after_successful_world_reset():
    task = _task()
    calls: list[str] = []

    def call_service(_client, _request, label, **_kwargs):
        calls.append(label)
        return label == "dispatcher/stop"

    task._call_service_sync = call_service
    task._publish_rest_pose = lambda: None
    task.reset = lambda: (True, "reset")

    result = PickBananaTask._clean_world_reset(task, randomize=False, resume=True, settle_s=0)

    assert result == (False, "world reset succeeded but scheduled session Open failed")
    assert calls == ["dispatcher/stop", "dispatcher/start"]
