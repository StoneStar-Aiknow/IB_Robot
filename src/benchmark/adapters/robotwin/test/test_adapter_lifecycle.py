"""Resource lifecycle tests for the RoboTwin adapter."""

from __future__ import annotations

from types import SimpleNamespace

from benchmark_robotwin.adapter import RobotwinAdapter


class _FakeViewer:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeEnvironment:
    def __init__(self) -> None:
        self.viewer = _FakeViewer()
        self.close_calls = 0

    def close_env(self) -> None:
        self.close_calls += 1


def test_clear_active_environment_releases_env_and_observation_caches() -> None:
    adapter = RobotwinAdapter()
    env = _FakeEnvironment()
    adapter._env = env
    adapter._last_raw_observation = {"large": object()}
    adapter._last_head_frame = object()

    adapter._clear_active_environment(close=True)

    assert adapter._env is None
    assert adapter._last_raw_observation is None
    assert adapter._last_head_frame is None
    assert env.close_calls == 1
    assert env.viewer.close_calls == 1
    assert adapter._task_environment_close_count == 1


def test_clear_without_environment_still_releases_observation_caches() -> None:
    adapter = RobotwinAdapter()
    adapter._last_raw_observation = {"large": object()}
    adapter._last_head_frame = object()

    adapter._clear_active_environment(close=True)

    assert adapter._last_raw_observation is None
    assert adapter._last_head_frame is None
    assert adapter._task_environment_close_count == 0


def test_close_is_idempotent_and_counts_adapter_close_not_env_creation() -> None:
    adapter = RobotwinAdapter()
    env = _FakeEnvironment()
    adapter._env = env
    adapter._env_generation = 4

    adapter.close()
    adapter.close()

    assert adapter.close_count == 1
    assert env.close_calls == 1
    assert env.viewer.close_calls == 1
    assert adapter._task_environment_close_count == 1


def test_reset_metadata_reports_environment_reuse_and_close_count(monkeypatch) -> None:
    adapter = RobotwinAdapter()
    adapter._config = object()
    adapter._task_name = "fake_task"
    adapter._task_config_args = {}
    adapter._plan_request = SimpleNamespace(
        suite="robotwin",
        episodes_per_task=2,
        embodiment="fake-robot",
        task_config="fake-config",
    )
    adapter._plan = SimpleNamespace(tasks=(SimpleNamespace(prompt="do the task"),))

    created = []

    class _ResetEnvironment(_FakeEnvironment):
        step_lim = 10

        def setup_demo(self, **_kwargs) -> None:
            pass

        def get_obs(self):
            return {"raw": object()}

    def create_environment() -> None:
        adapter._env = _ResetEnvironment()
        adapter._env_generation += 1
        created.append(adapter._env)

    monkeypatch.setattr(adapter, "_create_task_environment", create_environment)
    monkeypatch.setattr(
        "benchmark_robotwin.adapter.convert_observation",
        lambda _raw, arm_dim: {"observation.images.cam_high": object()},
    )
    request = SimpleNamespace(task_id=0, seed=100, init_state_id=0)

    first = adapter.reset(request)
    request.init_state_id = 1
    second = adapter.reset(request)

    assert len(created) == 1
    assert first.metadata["task_environment_generation"] == 1
    assert first.metadata["task_environment_reused"] is False
    assert first.metadata["task_environment_close_count"] == 0
    assert second.metadata["task_environment_generation"] == 1
    assert second.metadata["task_environment_reused"] is True
    assert second.metadata["task_environment_close_count"] == 0

    adapter.on_task_finalized("robotwin", 0)
    assert created[0].close_calls == 1
    assert adapter._task_environment_close_count == 1
