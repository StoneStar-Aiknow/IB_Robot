import time
from types import SimpleNamespace
from typing import Any, cast

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, _message: str) -> None:
        self.messages.append(_message)

    def warning(self, _message: str) -> None:
        self.messages.append(_message)


class _ServiceClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []
        self.removed = []

    def call_async(self, request):
        future = SimpleNamespace(client=self, request=request)
        self.calls.append(future)
        return future

    def remove_pending_request(self, future) -> None:
        self.removed.append(future)


class _KinematicsHarness:
    _solve_ik = PickExecutorNode._solve_ik
    _compute_fk = PickExecutorNode._compute_fk
    _kinematics_worker_index = PickExecutorNode._kinematics_worker_index
    _kinematics_unhealthy_snapshot = PickExecutorNode._kinematics_unhealthy_snapshot
    _mark_kinematics_worker_unhealthy = PickExecutorNode._mark_kinematics_worker_unhealthy
    _mark_kinematics_worker_healthy = PickExecutorNode._mark_kinematics_worker_healthy
    _kinematics_client_candidates = PickExecutorNode._kinematics_client_candidates
    _discard_pending_service_request = staticmethod(PickExecutorNode._discard_pending_service_request)

    def __init__(self) -> None:
        self._config = {"ik": {"group_name": "arm", "timeout_sec": 0.2, "avoid_collisions": False}}
        self._ee_frame = "gripper"
        self._base_frame = "base"
        self._rpc_timeout = 1.0
        self._ik_client = _ServiceClient("primary_ik")
        self._fk_client = _ServiceClient("primary_fk")
        self._ik_worker_clients = [_ServiceClient("worker_0_ik"), _ServiceClient("worker_1_ik")]
        self._fk_worker_clients = [_ServiceClient("worker_0_fk"), _ServiceClient("worker_1_fk")]
        self._kinematics_unhealthy_workers = set()
        self._ik_worker_verification = ("cached", 0.0)
        self._logger = _Logger()

    def get_logger(self):
        return self._logger


def test_parallel_candidate_preparation_balances_shared_queue_and_preserves_ranked_order():
    harness = SimpleNamespace(
        _ik_worker_clients=["ik_0", "ik_1", "ik_2"],
        _fk_worker_clients=["fk_0", "fk_1", "fk_2"],
        _ik_worker_count=3,
        get_logger=lambda: _Logger(),
    )
    harness._verify_ik_worker_pool = lambda *_args: None
    calls: list[tuple[int, str, str, bool]] = []

    def prepare(
        candidate,
        _scene,
        _goal,
        _deadline,
        *,
        initial_seed,
        ik_client,
        fk_client,
        allow_failover,
    ):
        del initial_seed
        if candidate.index == 0:
            time.sleep(0.05)
        calls.append((candidate.index, ik_client, fk_client, allow_failover))
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
    assert sorted(index for index, _, _, _ in calls) == list(range(8))
    assert all(ik_client.removeprefix("ik_") == fk_client.removeprefix("fk_") for _, ik_client, fk_client, _ in calls)
    assert all(allow_failover is False for _, _, _, allow_failover in calls)
    assert any(index % 3 != int(ik_client.removeprefix("ik_")) for index, ik_client, _, _ in calls)


def test_grasp_ik_fk_preserves_worker_failover_policy_for_all_subcalls():
    calls = []
    joint_state = JointState(name=["1"], position=[0.2])
    fk_pose = Pose()
    fk_pose.orientation.w = 1.0
    harness = SimpleNamespace(
        _solve_ik=lambda *_args, **kwargs: (calls.append(("ik", kwargs["allow_failover"])), joint_state)[1],
        _apply_joint5_retry_if_needed=lambda *_args, **kwargs: (
            calls.append(("joint5_retry", kwargs["allow_failover"])),
            (joint_state, None),
        )[1],
        _validate_joint5=lambda _joint_state: None,
        _compute_fk=lambda *_args, **kwargs: (calls.append(("fk", kwargs["allow_failover"])), fk_pose)[1],
        _pose_components=lambda pose: (
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
        ),
        _grasp_orientation_errors=lambda *_args: None,
    )

    payload = PickExecutorNode._solve_grasp_ik_fk(
        cast(Any, harness),
        Pose(),
        None,
        time.monotonic() + 5.0,
        joint_state,
        allow_failover=False,
    )

    assert payload.joint_state is joint_state
    assert calls == [("ik", False), ("joint5_retry", False), ("fk", False)]


def test_worker_verification_uses_current_fk_pose_checks_every_worker_and_caches():
    logger = _Logger()
    harness = SimpleNamespace(
        _ik_client="primary_ik",
        _fk_client="primary_fk",
        _ik_worker_clients=["ik_0", "ik_1", "ik_2"],
        _fk_worker_clients=["fk_0", "fk_1", "fk_2"],
        _ik_worker_count=3,
        _ee_frame="gripper",
        _base_frame="base",
        _config={
            "ik": {
                "group_name": "arm",
                "avoid_collisions": False,
                "check_orientation": False,
            }
        },
        get_logger=lambda: logger,
    )
    seed = JointState(name=["1", "2"], position=[0.1, 0.2])
    fk_calls = []
    solve_calls = []
    timed_out_clients = set()

    def compute_fk(joint_seed, goal_handle, deadline, *, client=None, allow_failover=True):
        fk_calls.append((joint_seed, goal_handle, deadline, client, allow_failover))
        pose = Pose()
        pose.position.x = 0.101
        pose.position.y = -0.163
        pose.position.z = 0.214
        pose.orientation.x = 0.3
        pose.orientation.w = 0.95
        return pose

    def solve_ik(pose, goal_handle, deadline, joint_seed, *, client=None, allow_failover=True):
        solve_calls.append((pose, goal_handle, deadline, joint_seed, client, allow_failover))
        if client == "ik_1" and client not in timed_out_clients:
            timed_out_clients.add(client)
            raise PickFlowError("RPC_TIMEOUT", "transient worker timeout")
        return JointState(name=["1", "2"], position=[0.25, -0.5])

    harness._compute_fk = compute_fk
    harness._solve_ik = solve_ik
    harness._mark_kinematics_worker_healthy = lambda _client: None
    harness._kinematics_unhealthy_snapshot = lambda: set()

    for _ in range(2):
        PickExecutorNode._verify_ik_worker_pool(cast(Any, harness), seed, "goal", 123.0)

    assert len(fk_calls) == 1
    assert fk_calls[0][3:] == ("primary_fk", False)
    assert [call[4] for call in solve_calls] == ["primary_ik", "ik_0", "ik_1", "ik_1", "ik_2"]
    assert all(call[5] is False for call in solve_calls)
    assert all(call[0].position.x == 0.101 for call in solve_calls)
    assert all(call[0].orientation.x == 0.0 and call[0].orientation.w == 1.0 for call in solve_calls)
    assert all(call[3] is seed for call in solve_calls)
    assert any("cached=false" in message for message in logger.messages)
    assert any("cached=true" in message for message in logger.messages)


def test_worker_verification_skips_quarantined_worker_on_next_pick():
    logger = _Logger()
    harness = SimpleNamespace(
        _ik_client="primary_ik",
        _fk_client="primary_fk",
        _ik_worker_clients=["ik_0", "ik_1", "ik_2"],
        _fk_worker_clients=["fk_0", "fk_1", "fk_2"],
        _ik_worker_count=3,
        _ee_frame="gripper",
        _base_frame="base",
        _config={"ik": {"group_name": "arm", "avoid_collisions": False, "check_orientation": False}},
        get_logger=lambda: logger,
        _kinematics_unhealthy_snapshot=lambda: {1},
        _mark_kinematics_worker_healthy=lambda _client: None,
    )
    pose = Pose()
    pose.orientation.w = 1.0
    harness._compute_fk = lambda *_args, **_kwargs: pose
    solve_clients = []

    def solve_ik(*_args, client=None, **_kwargs):
        solve_clients.append(client)
        return JointState(name=["1"], position=[0.1])

    harness._solve_ik = solve_ik

    PickExecutorNode._verify_ik_worker_pool(
        cast(Any, harness),
        JointState(name=["1"], position=[0.1]),
        "goal",
        123.0,
    )

    assert solve_clients == ["primary_ik", "ik_0", "ik_2"]
    assert any("workers=2" in message for message in logger.messages)


def test_runtime_ik_uses_isolated_workers_removes_timeout_and_fails_over():
    harness = _KinematicsHarness()
    seed = JointState(name=["1"], position=[0.1])
    response = SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        solution=SimpleNamespace(joint_state=JointState(name=["1"], position=[0.2])),
    )

    def wait_future(future, *_args):
        if future.client is harness._ik_worker_clients[0]:
            raise PickFlowError("RPC_TIMEOUT", "IK timed out")
        return response

    harness._wait_future = wait_future

    result = harness._solve_ik(Pose(), "goal", time.monotonic() + 5.0, seed)

    assert list(result.position) == [0.2]
    assert len(harness._ik_client.calls) == 0
    assert len(harness._ik_worker_clients[0].calls) == 1
    assert len(harness._ik_worker_clients[0].removed) == 1
    assert len(harness._ik_worker_clients[1].calls) == 1
    assert harness._kinematics_unhealthy_workers == {0}
    assert harness._ik_worker_verification is None
    assert any("quarantined IK/FK worker 0" in message for message in harness._logger.messages)

    harness._solve_ik(Pose(), "goal", time.monotonic() + 5.0, seed)
    assert len(harness._ik_worker_clients[0].calls) == 1
    assert len(harness._ik_worker_clients[1].calls) == 2


def test_runtime_fk_skips_worker_quarantined_by_ik_timeout():
    harness = _KinematicsHarness()
    harness._kinematics_unhealthy_workers.add(0)
    response_pose = Pose()
    response_pose.position.z = 0.11
    response = SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        pose_stamped=[SimpleNamespace(pose=response_pose)],
    )
    harness._wait_future = lambda *_args: response

    result = harness._compute_fk(JointState(name=["1"], position=[0.2]), "goal", time.monotonic() + 5.0)

    assert result.position.z == 0.11
    assert len(harness._fk_client.calls) == 0
    assert len(harness._fk_worker_clients[0].calls) == 0
    assert len(harness._fk_worker_clients[1].calls) == 1
