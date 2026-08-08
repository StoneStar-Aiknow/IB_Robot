import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


class _Logger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass


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


def test_worker_pool_verification_checks_every_configured_worker():
    logger = _Logger()
    harness = SimpleNamespace(
        _config={"ik": {"check_orientation": False}},
        _ik_worker_clients=["ik_0", "ik_1"],
        _ik_worker_count=2,
        _fk_client="fk_primary",
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

    with pytest.raises(PickFlowError, match="worker 1") as exc_info:
        PickExecutorNode._verify_ik_worker_pool(
            cast(Any, harness),
            seed,
            None,
            time.monotonic() + 5.0,
        )

    assert exc_info.value.code == "IK_WORKER_MISMATCH"
    assert fk_calls == [(seed, "fk_primary")]
    assert calls == [None, "ik_0", "ik_1"]
    assert all((pose.position.x, pose.position.y, pose.position.z) == (0.42, -0.08, 0.31) for pose in probe_poses)
    assert all(
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w) == (0.0, 0.0, 0.0, 1.0)
        for pose in probe_poses
    )
