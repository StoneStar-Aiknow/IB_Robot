"""RoboDojo eval-client loop (robodojo_client/deploy.py) against the in-process adapter and core.

No Isaac: a fake TASK_ENV stands in for RoboDojo's eval env (tick counter, step limit,
native success flag, IK), and the model client calls the policy model directly instead of
going over the websocket. The motion request comes from another thread, as the ROS
services do in the runtime node.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from robodojo_robot.adapter import RoboDojoAdapter

from robot_runtime_sim.direct_control.engine import Command, MotionConfig, SimulationCore

PACKAGE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = PACKAGE_DIR / "robodojo_client" / "artifacts.py"
PROFILE = yaml.safe_load((PACKAGE_DIR / "profiles" / "arx_x5_dual.yaml").read_text())


def make_runtime(motion: MotionConfig, events: list | None = None, native: list | None = None):
    """The runtime node's core and RoboDojo adapter, without ROS or a websocket."""
    adapter = RoboDojoAdapter(PROFILE, {"robodojo_root": "/unused"})
    core = SimulationCore(
        adapter.descriptor,
        motion,
        on_event=(lambda kind, info: events.append((kind, info))) if events is not None else None,
        on_platform_result=native.append if native is not None else None,
    )
    return core, adapter.bind(core)


@pytest.fixture
def deploy(tmp_path, monkeypatch):
    """Import deploy.py the way RoboDojo does: as a module of a wrapper package next to artifacts.py."""
    np = pytest.importorskip("numpy")
    assert np is not None
    wrapper = tmp_path / "vrt_wrapper"
    wrapper.mkdir()
    (wrapper / "__init__.py").write_text("")
    (wrapper / "deploy.py").symlink_to(PACKAGE_DIR / "robodojo_client" / "deploy.py")
    (wrapper / "artifacts.py").symlink_to(ARTIFACTS)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("IBROBOT_RUNTIME_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ROBODOJO_RUN_ID", "unit")
    module = importlib.import_module("vrt_wrapper.deploy")
    yield module
    sys.modules.pop("vrt_wrapper.deploy", None)
    sys.modules.pop("vrt_wrapper.artifacts", None)
    sys.modules.pop("vrt_wrapper", None)


class FakeRobotManager:
    def get_robot_by_arm_name(self, name):
        return name

    def solve_ik(self, target_pose, env_idx, robot):
        return {"status": "Success", "joint_value": [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]]}


class FakeTaskEnv:
    task_name = "unit_task"
    step_lim = 40

    def __init__(self, succeed_on_close=True):
        self.take_action_cnt = [0]
        self.success = [True]
        self.env_seeds = [3]
        self.eval_seed = 3
        self.robot_manager = FakeRobotManager()
        self.joints = {"left": [0.0] * 6, "right": [0.0] * 6}
        self.grip = {"left": 1.0, "right": 1.0}
        self.succeed_on_close = succeed_on_close
        self.done = False

    def get_obs(self):
        state = {}
        for side in ("left", "right"):
            state[f"{side}_arm_joint_state"] = list(self.joints[side])
            state[f"{side}_ee_joint_state"] = [self.grip[side]]
            state[f"{side}_ee_pose"] = [0.1 * sum(self.joints[side]), 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]
        return {"state": state, "instruction": "pick it up"}

    def take_action(self, action):
        self.take_action_cnt[0] += 1
        for side in ("left", "right"):
            self.joints[side] = [float(v) for v in action[f"{side}_arm_joint_state"]]
            self.grip[side] = float(action[f"{side}_ee_joint_state"][0])
        if self.succeed_on_close and self.grip["right"] < 0.01:
            self.done = True  # RoboDojo's reward says the task is done

    def is_episode_end(self):
        if self.done:
            return True
        if self.take_action_cnt[0] >= self.step_lim:
            self.success[0] = False
            return True
        return False


class DirectClient:
    def __init__(self, model):
        self.model = model

    def call(self, func_name=None, obs=None):
        method = getattr(self.model, func_name)
        return method() if obs is None else method(obs)


def _drive(core, commands):
    results = []

    def worker():
        deadline = time.monotonic() + 5.0
        while core.snapshot()[0] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        for cmd in commands:
            results.append(core.submit(cmd, timeout_s=5.0))
        core.finish()

    thread = threading.Thread(target=worker)
    thread.start()
    return thread, results


def test_motion_requests_become_ticks_and_the_native_verdict_is_recorded(deploy, tmp_path):
    events, native = [], []
    core, model = make_runtime(
        MotionConfig(min_motion_ticks=3, settle_ticks=1, gripper_ticks=2, idle_wait_s=0.01, position_tolerance_m=1.0),
        events,
        native,
    )
    env = FakeTaskEnv()
    thread, results = _drive(
        core,
        [
            Command(kind="pose", group="left_arm", pose=[0.12, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
            Command(kind="gripper", group="right_arm", opening=0.0),
        ],
    )
    artifact = deploy.eval_one_episode(env, DirectClient(model))
    thread.join(timeout=5.0)

    # RoboDojo judged success while the gripper was still settling: the episode ends at once and the
    # interrupted command says so (the robot no longer exists for this episode).
    assert [r["outcome"] for r in results] == ["SUCCEEDED", "EXECUTION_FAILED"]
    assert "session ended" in results[1]["message"]
    assert env.joints["left"] == pytest.approx([0.2] * 6)  # cuRobo's answer, executed tick by tick
    assert artifact["success"] is True and artifact["failure_reason"] is None
    assert [d["kind"] for d in artifact["decisions"]] == ["pose", "gripper"]
    assert len(artifact["ticks"]) == env.take_action_cnt[0]
    assert artifact["idle_polls"] >= 1  # waiting for the first command did not step physics
    public = [(kind, info) for kind, info in events if kind != "state"]
    assert [kind for kind, _ in public] == ["session_start", "session_end"]
    assert public[0][1]["instruction"] == "pick it up"
    # The verdict, seed and layout reach the platform-result listener only, never the public events.
    for _kind, info in events:
        assert not {"success", "native_success", "seed", "layout", "camera_calibration"} & set(info)
    (verdict,) = native
    assert verdict["native_success"] is True and verdict["seed"] == 3
    written = json.loads(Path(verdict["artifact"]).read_text())
    assert written["seed"] == 3


def test_finish_holds_until_robodojo_times_out(deploy):
    native = []
    core, model = make_runtime(MotionConfig(idle_wait_s=0.01), native=native)
    env = FakeTaskEnv(succeed_on_close=False)
    thread, _ = _drive(core, [])
    artifact = deploy.eval_one_episode(env, DirectClient(model))
    thread.join(timeout=5.0)
    assert artifact["success"] is False and artifact["failure_reason"] == "timeout"
    assert len(artifact["ticks"]) == FakeTaskEnv.step_lim
    assert native[0]["native_success"] is False and native[0]["failure_reason"] == "timeout"


class FailingClient(DirectClient):
    """The runtime refuses or drops a call: ``fail(func_name, call_index)`` decides which."""

    def __init__(self, model, fail):
        super().__init__(model)
        self.fail = fail
        self.calls = []

    def call(self, func_name=None, obs=None):
        self.calls.append(func_name)
        if self.fail(func_name, len(self.calls)):
            raise ConnectionError(f"synthetic failure on {func_name}")
        return super().call(func_name, obs)


@pytest.mark.parametrize("failing", ["update_obs", "get_action"])
def test_a_refused_or_lost_runtime_call_steps_no_more_physics(deploy, failing):
    native = []
    core, model = make_runtime(MotionConfig(idle_wait_s=0.01), native=native)
    env = FakeTaskEnv(succeed_on_close=False)
    env.unstable_envs = set()
    client = FailingClient(model, lambda name, _n: name == failing)
    with pytest.raises(deploy.EpisodeInterrupted):
        deploy.eval_one_episode(env, client)
    assert env.take_action_cnt[0] == 0  # no drain with a guessed hold
    assert env.unstable_envs == {0}  # RoboDojo will not score or keep this episode
    assert "episode_interrupted" in client.calls and "episode_end" not in client.calls
    (verdict,) = native
    assert verdict["native_success"] is None and verdict["interrupted"] is True
    written = json.loads(Path(verdict["artifact"]).read_text())
    assert written["failure_reason"] == "interrupted" and written["success"] is None
    assert core.state in ("FAILED", "UNKNOWN") and core.halted


def test_a_stale_generation_mid_episode_interrupts_without_stepping(deploy):
    core, model = make_runtime(MotionConfig(idle_wait_s=0.01))
    env = FakeTaskEnv(succeed_on_close=False)
    env.unstable_envs = set()
    calls = {"n": 0}

    class Resetting(DirectClient):
        def call(self, func_name=None, obs=None):
            if func_name == "get_action":
                calls["n"] += 1
                if calls["n"] == 3:
                    core.reset()  # another client took the runtime: this client's episode is stale
            return super().call(func_name, obs)

    thread, _results = _drive(core, [Command(kind="joint", group="left_arm", joints=[2.0] * 6)])
    with pytest.raises(deploy.EpisodeInterrupted):
        deploy.eval_one_episode(env, Resetting(model))
    ticks = env.take_action_cnt[0]
    thread.join(timeout=5.0)
    assert ticks <= 2 and env.take_action_cnt[0] == ticks


def test_the_episode_end_carries_the_platforms_final_tick(deploy):
    events = []
    core, model = make_runtime(MotionConfig(idle_wait_s=0.01), events)
    env = FakeTaskEnv(succeed_on_close=False)
    thread, _ = _drive(core, [])
    deploy.eval_one_episode(env, DirectClient(model))
    thread.join(timeout=5.0)
    (end,) = [info for kind, info in events if kind == "session_end"]
    assert end["ticks"] == end["platform_ticks"] == FakeTaskEnv.step_lim
    assert end["sim_time_s"] == pytest.approx(FakeTaskEnv.step_lim * 0.04)


def test_the_reconnect_budget_is_bounded_after_reset(deploy):
    class Config:
        max_connect_seconds = 900.0
        max_connect_attempts = 180
        connect_retry_delay_s = 5.0

    client = SimpleNamespace(_client=SimpleNamespace(config=Config()))
    assert deploy._bound_reconnect(client, 30.0)
    assert client._client.config.max_connect_seconds == 30.0 and client._client.config.max_connect_attempts == 7
    assert not deploy._bound_reconnect(SimpleNamespace(), 30.0)
