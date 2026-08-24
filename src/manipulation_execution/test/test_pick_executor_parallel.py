import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, _message: str) -> None:
        pass


class _ReadyClient(str):
    def service_is_ready(self) -> bool:
        return True


class _MutableReadyClient:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    def service_is_ready(self) -> bool:
        return self.ready


def test_parallel_candidate_preparation_preserves_ranked_order_and_worker_affinity():
    harness = SimpleNamespace(
        _ik_worker_clients=["ik_0", "ik_1", "ik_2"],
        _fk_worker_clients=["fk_0", "fk_1", "fk_2"],
        _ik_worker_count=3,
        get_logger=lambda: _Logger(),
    )
    harness._verify_ik_worker_pool = lambda *_args: None
    calls: list[tuple[int, str, str]] = []

    def prepare(candidate, _scene, _goal, _deadline, *, initial_seed, ik_client, fk_client):
        del initial_seed
        time.sleep(0.002 * (3 - candidate.index % 3))
        calls.append((candidate.index, ik_client, fk_client))
        return candidate.index

    harness._prepare_candidate = prepare
    ranked = [SimpleNamespace(index=index) for index in range(8)]

    prepared, error = PickExecutorNode._prepare_ranked_candidates(
        cast(Any, harness),
        cast(Any, ranked),
        cast(Any, None),
        SimpleNamespace(),
        None,
        time.monotonic() + 10.0,
    )

    assert error is None
    assert prepared == list(range(8))
    assert sorted(calls) == [(index, f"ik_{index % 3}", f"fk_{index % 3}") for index in range(8)]


def test_kinematics_joint_state_filters_aggregate_hardware_feedback():
    harness = SimpleNamespace(_arm_joint_names=["1", "2", "3", "4", "5"])
    state = JointState()
    state.name = ["2", "4", "1", "3", "5", "6", "7", "8", "9"]
    state.position = [-0.2, 0.4, 0.1, 0.3, 0.5, 0.6, 1696.0, 2868.0, 2309.0]
    state.velocity = [0.0] * 9
    state.effort = [float("nan")] * 9

    normalized = PickExecutorNode._kinematics_joint_state(cast(Any, harness), state)

    assert normalized.name == ["1", "2", "3", "4", "5"]
    assert list(normalized.position) == [0.1, -0.2, 0.3, 0.4, 0.5]
    assert list(normalized.velocity) == []
    assert list(normalized.effort) == []


@pytest.mark.parametrize(
    ("names", "positions", "message"),
    [
        (["1", "2"], [0.1], "names but"),
        (["1", "1", "2", "3", "4", "5"], [0.1] * 6, "duplicate"),
        (["1", "2", "3", "4"], [0.1] * 4, "missing configured arm joints"),
        (["1", "2", "3", "4", "5"], [0.1, 0.2, float("nan"), 0.4, 0.5], "must be finite"),
    ],
)
def test_kinematics_joint_state_rejects_invalid_input(names, positions, message):
    harness = SimpleNamespace(_arm_joint_names=["1", "2", "3", "4", "5"])
    state = JointState(name=names, position=positions)

    with pytest.raises(PickFlowError, match=message) as exc_info:
        PickExecutorNode._kinematics_joint_state(cast(Any, harness), state)

    assert exc_info.value.code == "INVALID_JOINT_STATE"


def test_worker_pool_verification_checks_every_configured_worker():
    logger = _Logger()
    primary_ik = _ReadyClient("primary_ik")
    primary_fk = _ReadyClient("primary_fk")
    harness = SimpleNamespace(
        _config={
            "ik": {
                "check_orientation": False,
                "verification_position_tolerance_m": 0.001,
                "verification_orientation_tolerance_deg": 1.0,
            }
        },
        _ik_client=primary_ik,
        _fk_client=primary_fk,
        _ik_worker_clients=[_ReadyClient("ik_0"), _ReadyClient("ik_1")],
        _fk_worker_clients=[_ReadyClient("fk_0"), _ReadyClient("fk_1")],
        _ik_worker_count=2,
        _ik_worker_verification=None,
        _ik_worker_verification_lock=threading.Lock(),
        _ik_worker_service_state=None,
        _ik_worker_service_generation=0,
        get_logger=lambda: logger,
    )
    calls: list[str | None] = []
    fk_calls: list[tuple[JointState, str]] = []
    probe_poses: list[Pose] = []

    def joint_state(joint5: float) -> JointState:
        state = JointState()
        state.name = ["1", "2", "3", "4", "5"]
        state.position = [0.0, 0.0, 0.0, 0.0, joint5]
        return state

    seed = joint_state(0.0)

    def compute_fk(current_seed, _goal, _deadline, *, client=None):
        fk_calls.append((current_seed, client))
        pose = Pose()
        pose.position.x = 0.42
        pose.position.y = -0.08
        pose.position.z = 0.31
        pose.orientation.x = 0.1
        pose.orientation.y = 0.2
        pose.orientation.z = 0.3
        pose.orientation.w = 0.9
        return pose

    def solve(pose, _goal, _deadline, _seed, *, client=None):
        calls.append(client)
        probe_poses.append(pose)
        return joint_state(0.1 if client == "ik_1" else 0.0)

    harness._compute_fk = compute_fk
    harness._solve_ik = solve

    PickExecutorNode._verify_ik_worker_pool(
        cast(Any, harness),
        seed,
        None,
        time.monotonic() + 5.0,
    )

    assert fk_calls == [
        (seed, primary_fk),
        (seed, primary_fk),
        (seed, "fk_0"),
        (joint_state(0.1), "fk_1"),
    ]
    assert calls == [None, "ik_0", "ik_1"]
    assert all((pose.position.x, pose.position.y, pose.position.z) == (0.42, -0.08, 0.31) for pose in probe_poses)
    assert all(
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w) == (0.0, 0.0, 0.0, 1.0)
        for pose in probe_poses
    )
    assert any("max_joint_delta=0.100000000000" in message for message in logger.messages)


def _successful_verification_harness():
    logger = _Logger()
    harness = SimpleNamespace(
        _config={
            "ik": {
                "check_orientation": False,
                "verification_position_tolerance_m": 0.001,
                "verification_orientation_tolerance_deg": 1.0,
            }
        },
        _ik_client=_MutableReadyClient(),
        _fk_client=_MutableReadyClient(),
        _ik_worker_clients=[_MutableReadyClient(), _MutableReadyClient()],
        _fk_worker_clients=[_MutableReadyClient(), _MutableReadyClient()],
        _ik_worker_count=2,
        _ik_worker_verification=None,
        _ik_worker_verification_lock=threading.Lock(),
        _ik_worker_service_state=None,
        _ik_worker_service_generation=0,
        get_logger=lambda: logger,
    )
    calls = {"fk": 0, "ik": 0}

    def joint_state(joint5: float = 0.0) -> JointState:
        return JointState(name=["1", "2", "3", "4", "5"], position=[0.0, 0.0, 0.0, 0.0, joint5])

    def compute_fk(_seed, _goal, _deadline, *, client=None):
        assert client is harness._fk_client or client in harness._fk_worker_clients
        calls["fk"] += 1
        pose = Pose()
        pose.orientation.w = 1.0
        return pose

    def solve(_pose, _goal, _deadline, _seed, *, client=None):
        assert client is None or client in harness._ik_worker_clients
        calls["ik"] += 1
        return joint_state()

    harness._compute_fk = compute_fk
    harness._solve_ik = solve
    return harness, joint_state(), calls, logger


def _verify(harness, seed: JointState) -> None:
    PickExecutorNode._verify_ik_worker_pool(
        cast(Any, harness),
        seed,
        None,
        time.monotonic() + 5.0,
    )


def test_worker_pool_verification_rejects_fk_position_mismatch():
    harness, seed, _calls, _logger = _successful_verification_harness()

    def compute_fk(_seed, _goal, _deadline, *, client=None):
        pose = Pose()
        pose.position.x = 0.002 if client is harness._fk_worker_clients[1] else 0.0
        pose.orientation.w = 1.0
        return pose

    harness._compute_fk = compute_fk

    with pytest.raises(PickFlowError, match="IK worker 1 FK") as exc_info:
        _verify(harness, seed)

    assert exc_info.value.code == "IK_WORKER_MISMATCH"


def test_worker_pool_verification_rejects_fk_orientation_mismatch():
    harness, seed, _calls, _logger = _successful_verification_harness()
    harness._config["ik"]["check_orientation"] = True

    def compute_fk(_seed, _goal, _deadline, *, client=None):
        pose = Pose()
        if client is harness._fk_worker_clients[1]:
            pose.orientation.z = 0.017452406
            pose.orientation.w = 0.999847695
        else:
            pose.orientation.w = 1.0
        return pose

    harness._compute_fk = compute_fk

    with pytest.raises(PickFlowError, match="orientation_error") as exc_info:
        _verify(harness, seed)

    assert exc_info.value.code == "IK_WORKER_MISMATCH"


def test_worker_pool_verification_cache_ignores_seed_motion():
    harness, seed, calls, logger = _successful_verification_harness()
    _verify(harness, seed)

    jittered_seed = JointState(name=list(seed.name), position=[0.0, 0.0, 0.0, 0.0, 0.25])
    _verify(harness, jittered_seed)

    assert calls == {"fk": 4, "ik": 3}
    assert any("cached=true" in message for message in logger.messages)


def test_worker_pool_verification_revalidates_after_service_recovery():
    harness, seed, calls, _logger = _successful_verification_harness()
    _verify(harness, seed)
    harness._ik_worker_clients[0].ready = False

    with pytest.raises(PickFlowError, match="worker_0_ik") as exc_info:
        _verify(harness, seed)

    assert exc_info.value.code == "IK_WORKER_UNAVAILABLE"
    assert harness._ik_worker_verification is None
    assert calls == {"fk": 4, "ik": 3}

    harness._ik_worker_clients[0].ready = True
    _verify(harness, seed)

    assert calls == {"fk": 8, "ik": 6}
    assert harness._ik_worker_service_generation == 2


def test_worker_pool_verification_revalidates_after_timer_observes_restart():
    harness, seed, calls, _logger = _successful_verification_harness()
    _verify(harness, seed)
    harness._fk_worker_clients[1].ready = False
    PickExecutorNode._refresh_ik_worker_service_generation(cast(Any, harness))
    harness._fk_worker_clients[1].ready = True
    PickExecutorNode._refresh_ik_worker_service_generation(cast(Any, harness))

    _verify(harness, seed)

    assert calls == {"fk": 8, "ik": 6}
    assert harness._ik_worker_service_generation == 2
