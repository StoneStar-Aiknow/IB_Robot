from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.collision_spheres import merge_sphere_cover
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.planner import LeftArmPlanner, robot_config
from aimdk_grasp.planner_service import GraspPlanningService
from aimdk_grasp.robot_model import planning_model
from test_hand import PRESETS
from test_planner_contracts import GRIPPER


def test_merged_torso_contains_every_original_solid_sphere():
    feedback = Feedback(
        0.0, dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), HandMapping(planning_model(), PRESETS).preset("open")
    )
    fine = robot_config(feedback)["robot_cfg"]["kinematics"]
    coarse = robot_config(feedback, torso_merge_radius_m=0.06)["robot_cfg"]["kinematics"]
    original = fine["collision_spheres"]["torso_link"]
    merged = coarse["collision_spheres"]["torso_link"]
    assert len(merged) < len(original) * 0.6
    centers = np.array([s["center"] for s in merged])
    radii = np.array([s["radius"] for s in merged])
    for sphere in original:
        assert np.any(np.linalg.norm(centers - sphere["center"], axis=1) + sphere["radius"] <= radii + 1e-12)
    assert coarse["self_collision_ignore"] == fine["self_collision_ignore"]
    for link in fine["collision_spheres"]:
        if link != "torso_link":
            assert coarse["collision_spheres"][link] == fine["collision_spheres"][link]


def test_merge_handles_coincident_and_contained_spheres():
    original = [
        {"center": [0, 0, 0], "radius": 0.02},
        {"center": [0, 0, 0], "radius": 0.03},
        {"center": [0.01, 0, 0], "radius": 0.005},
    ]
    assert merge_sphere_cover(original)[0]["radius"] == pytest.approx(0.03)
    with pytest.raises(ValueError):
        merge_sphere_cover(original, -1.0)


def test_protected_merge_never_inflates_into_work_envelope():
    balls = [{"center": [x, 0, 0], "radius": 0.02} for x in (0.0, 0.03, 1.0, 1.03)]
    protected = np.array([[0.015, 0.05, 0, 0.02]])
    result = merge_sphere_cover(balls, 0.06, protected_spheres=protected)
    assert len(result) == 3
    assert result[:2] == balls[:2]
    merged = result[2]
    assert np.linalg.norm(np.array(merged["center"]) - protected[0, :3]) > merged["radius"] + protected[0, 3] + 0.01


def test_service_reuses_warmed_planner_updates_scene_and_rebuilds_locked_model():
    events = []
    instances = []

    class Planner:
        def __init__(self, feedback, estimate, received_at, **options):
            self.locked = feedback["locked"]
            self.max_goalset = options["max_goalset"]
            self.received_at = received_at
            instances.append(self)
            events.append("init")

        def locked_joints_match(self, feedback):
            return feedback["locked"] == self.locked

        def warmup(self):
            events.append("warmup")

        def update_scene(self, feedback, estimate):
            events.append("update")
            self.received_at = estimate.received_at

        def screen_grasps(self, feedback, grasps, matrix, now):
            assert now == 7.0
            events.append("screen")
            return np.ones(len(grasps), dtype=bool)

        def plan_grasp(self, *args, **kwargs):
            events.append("plan")
            return SimpleNamespace(goalset_index=1)

        def close(self):
            events.append("close")

    feedback = {"locked": 0}
    estimate = SimpleNamespace(received_at=5.0)
    with GraspPlanningService(gripper=GRIPPER, clock=lambda: 7.0, planner_factory=Planner) as service:
        service.prepare(feedback, estimate)
        service.prepare(feedback, estimate)
        _, report = service.plan(feedback, estimate, np.stack([np.eye(4)] * 2), [0.8, 0.9], np.eye(4))
        assert report["candidate_index"] == 0
        fresh = SimpleNamespace(received_at=6.0)
        service.prepare(feedback, fresh)
        assert len(instances) == 1 and instances[0].received_at == 6.0
        service.prepare({"locked": 1}, fresh)
    assert report["selection_mode"] == "direct_goalset" and "screened_candidates" not in report
    assert events == ["init", "warmup", "plan", "update", "close", "init", "warmup", "close"]


def test_a_planner_build_holds_the_lock_the_other_gpu_users_share():
    """cuRobo captures CUDA graphs while it builds; no other GPU work of the process may run meanwhile."""
    gpu, building, release = RLock(), Event(), Event()

    class Planner:
        def __init__(self, *args, **kwargs):
            pass

        def warmup(self):
            building.set()
            release.wait(5.0)

        def close(self):
            pass

    with (
        GraspPlanningService(gripper=GRIPPER, planner_factory=Planner, gpu_lock=gpu) as service,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        built = pool.submit(service.prepare, None, SimpleNamespace(received_at=0.0))
        assert building.wait(5.0)
        assert not pool.submit(gpu.acquire, timeout=0.2).result()  # another thread waits
        release.set()
        built.result(timeout=5.0)
        assert gpu.acquire(timeout=1.0)
        gpu.release()


@pytest.mark.parametrize("error", [RuntimeError("capture failed"), KeyboardInterrupt()])
def test_service_cleans_up_failed_startup(error):
    events = []

    class Broken:
        def __init__(self, *args, **kwargs):
            pass

        def warmup(self):
            raise error

        def close(self):
            events.append("close")

    with GraspPlanningService(gripper=GRIPPER, planner_factory=Broken) as service, pytest.raises(type(error)):
        service.prepare(None, SimpleNamespace(received_at=0.0))
    assert events == ["close"]


def test_expired_service_request_cannot_initialize_a_planner(monkeypatch):
    monkeypatch.setattr("aimdk_grasp.planner_service.time.perf_counter", lambda: 5.0)

    def unexpected(*args, **kwargs):
        pytest.fail("Expired request started GPU work")

    with (
        GraspPlanningService(gripper=GRIPPER, planning_profile="latency", planner_factory=unexpected) as service,
        pytest.raises(TimeoutError),
    ):
        service.plan(None, SimpleNamespace(received_at=0.0), np.eye(4)[None], [1.0], np.eye(4), deadline=4.0)
    with pytest.raises(ValueError, match="max_batch_size=1"):
        GraspPlanningService(gripper=GRIPPER, planning_profile="latency", max_batch_size=10)


def test_snapshot_solve_keeps_admission_time_but_wall_deadline_remains(monkeypatch):
    from aimdk_grasp.planner import GraspPlan
    from aimdk_grasp.trajectory import TimedTrajectory

    feedback = Feedback(
        1.0, dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), HandMapping(planning_model(), PRESETS).preset("open")
    )
    events = []

    class Planner:
        def __init__(self, feedback, estimate, at, **options):
            self.max_goalset = 100
            self.estimate = estimate

        def warmup(self):
            pass

        def close(self):
            pass

        def locked_joints_match(self, value):
            return True

        def update_scene(self, feedback, estimate):
            self.estimate = estimate

        def screen_grasps(self, feedback, grasps, matrix, now):
            events.append(now)
            return np.ones(len(grasps), dtype=bool)

        def plan_grasp(self, feedback, grasps, matrix, now, **options):
            events.append(now)
            trajectory = TimedTrajectory(np.array([0.0, 0.1]), np.zeros((2, 7)))
            return GraspPlan(trajectory, trajectory, trajectory)

    with GraspPlanningService(gripper=GRIPPER, clock=lambda: 20.0, planner_factory=Planner) as service:
        plan, report = service.plan(
            feedback,
            SimpleNamespace(received_at=1.0),
            np.eye(4)[None],
            [1.0],
            np.eye(4),
            snapshot_now=1.0,
            screen_check=True,
        )
        assert events == [1.0, 1.0]
        assert report["planning_clock"] == "admitted_snapshot" and not report["execution_authorized"]
        with pytest.raises(TimeoutError):
            service.plan(
                feedback,
                SimpleNamespace(received_at=1.0),
                np.eye(4)[None],
                [1.0],
                np.eye(4),
                snapshot_now=1.0,
                deadline=0.0,
            )


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "cached",
        "ik_failure",
        "trajopt_failure",
        "expired_before",
        "expired_after_ik",
        "expired_after_trajopt",
    ],
)
def test_latency_motion_checks_deadline_between_gpu_calls_and_never_retries(monkeypatch, outcome):
    clock = [0.0]
    monkeypatch.setattr("aimdk_grasp.planner.time.perf_counter", lambda: clock[0])
    events = []
    goal, state = object(), object()
    seed = object()

    def ik(actual_goal, actual_state, **options):
        assert (actual_goal, actual_state) == (goal, state)
        events.append("ik")
        if outcome == "expired_after_ik":
            clock[0] = 2.0
        return SimpleNamespace(success=np.array([outcome != "ik_failure"]), solution=seed)

    def trajopt(actual_goal, actual_state, **options):
        assert (actual_goal, actual_state) == (goal, state)
        events.append("trajopt")
        assert options["seed_config"] is seed
        assert options["finetune_attempts"] == 0
        if outcome == "expired_after_trajopt":
            clock[0] = 2.0
        return SimpleNamespace(success=np.array([outcome != "trajopt_failure"]))

    planner = object.__new__(LeftArmPlanner)
    planner._trajopt_seeds = 1  # the single-seed path is under test
    planner.planning_profile = "latency"
    planner.motion = SimpleNamespace(
        ik_solver=SimpleNamespace(solve_pose=ik), trajopt_solver=SimpleNamespace(solve_pose=trajopt)
    )
    if outcome == "expired_before":
        clock[0] = 2.0
    if outcome.startswith("expired"):
        with pytest.raises(TimeoutError):
            planner._solve_motion(goal, state, deadline=1.0)
        assert (
            events
            == {"expired_before": [], "expired_after_ik": ["ik"], "expired_after_trajopt": ["ik", "trajopt"]}[outcome]
        )
    else:
        result = planner._solve_motion(goal, state, seed_config=seed if outcome == "cached" else None, deadline=1.0)
        if outcome == "ik_failure":
            assert result is None and events == ["ik"]
        else:
            assert result is not None
            assert bool(result.success.all()) == (outcome != "trajopt_failure")
            assert events == (["trajopt"] if outcome == "cached" else ["ik", "trajopt"])


@pytest.mark.parametrize(
    "clear, expected",
    [
        ([True, True, True], 0),  # Keep a clear native choice, even with shorter alternatives.
        ([False, True, True], 2),  # Prefer the nearest clear successful IK branch.
        ([False, True, False], 1),
        ([False, False, False], 0),  # A blocked straight path does not prove no path exists.
    ],
)
def test_approach_seed_ranking_preserves_fallback_and_ignores_failed_ik(clear, expected):
    torch = pytest.importorskip("torch")
    planner = object.__new__(LeftArmPlanner)
    start = np.zeros(7)
    ends = np.zeros((4, 7))
    ends[:, 0] = [0.8, 0.6, 0.4, 0.1]
    selection = SimpleNamespace(solution=torch.tensor(ends[None]), success=torch.tensor([[True, True, True, False]]))

    def valid(positions):
        paths = positions.reshape(3, 12, 7)
        np.testing.assert_array_equal(paths[:, 0], np.zeros((3, 7)))
        np.testing.assert_array_equal(paths[:, -1], ends[:3])
        flags = np.ones((3, 12), dtype=bool)
        flags[:, 5] = clear  # Test a collision in the path interior, not just endpoints.
        return flags.reshape(-1)

    planner.valid_positions = valid
    assert planner._approach_seed_index(selection, start) == expected


def test_approach_seed_ranking_handles_one_or_no_successful_solutions():
    torch = pytest.importorskip("torch")
    planner = object.__new__(LeftArmPlanner)
    planner.valid_positions = lambda positions: pytest.fail("No alternative to rank")
    selection = SimpleNamespace(solution=torch.zeros((1, 3, 7)), success=torch.tensor([[False, False, True]]))
    assert planner._approach_seed_index(selection, np.zeros(7)) == 2
    selection.success[:] = False
    with pytest.raises(RuntimeError, match="No successful"):
        planner._approach_seed_index(selection, np.zeros(7))
    with pytest.raises(ValueError, match="approach seed policy"):
        GraspPlanningService(gripper=GRIPPER, approach_seed_policy="unknown")


@pytest.mark.parametrize("mode", ["ranked", "native", "long_approach", "expired_during_ranking"])
def test_grasp_selection_keeps_candidate_seed_pair_and_checks_ranking_deadline(monkeypatch, mode):
    import sys
    import time

    from aimdk_grasp.trajectory import TimedTrajectory

    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "curobo.types", SimpleNamespace(GoalToolPose=SimpleNamespace))
    monkeypatch.setattr(LeftArmPlanner, "_linear_grasp", False)  # covers the TrajOpt grasp
    monkeypatch.setattr(LeftArmPlanner, "vertical_lift_m", 0.0)  # lift: hold (see test_vertical_lift)
    monkeypatch.setattr(LeftArmPlanner, "_approach_graph_seed", None)  # its motion stub takes no seed mode

    class Criteria:
        @staticmethod
        def linear_motion(**options):
            return "linear"

    monkeypatch.setitem(sys.modules, "curobo._src.cost.tool_pose_criteria", SimpleNamespace(ToolPoseCriteria=Criteria))
    as_tensor = torch.as_tensor
    monkeypatch.setattr(torch, "as_tensor", lambda *args, device=None, **kw: as_tensor(*args, **kw))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr("aimdk_grasp.planner.palm_goals_from_camera_grasps", lambda g, m, p: np.asarray(g))
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    planner = object.__new__(LeftArmPlanner)
    planner.grasp_to_palm = np.eye(4)
    planner.planning_profile = "latency"
    planner.approach_seed_policy = "native" if mode == "native" else "clear_path"
    planner.max_goalset = 2
    planner._grasp_selections = {}
    planner.locked_positions = {}
    planner.hand_links = ["finger"]
    planner.snapshot_matches = lambda feedback, now: True
    planner._require_valid_start = lambda positions: None
    planner._state = lambda positions: positions
    checks = []
    ik_calls = []
    motion_seeds = []
    collisions = []
    expected_rank = mode not in ("native", "long_approach")
    ends = torch.zeros((1, 3, 7))
    ends[0, :, 0] = torch.tensor([0.8, 0.2, 0.1])

    def ik(goal, state, *, return_seeds):
        assert "disable" not in collisions  # Approach IK must keep contact links enabled.
        assert goal.position.shape == (1, 1, 1, 2, 3)
        ik_calls.append(return_seeds)
        if expected_rank:
            return SimpleNamespace(
                success=torch.tensor([[True, True, False]]),
                solution=ends.clone(),
                goalset_index=torch.tensor([[[1], [0], [1]]]),
            )
        return SimpleNamespace(
            success=torch.tensor([[True]]), solution=ends[:, :1].clone(), goalset_index=torch.tensor([[[1]]])
        )

    def valid(positions):
        checks.append(len(positions))
        flags = np.ones(len(positions), dtype=bool)
        if len(positions) == 24:
            flags[5] = False
            if mode == "expired_during_ranking":
                clock[0] = 2.0
        return flags

    def motion(goal, state, *, deadline, seed_config):
        # Singleton segments retain the same capture size as goal selection.
        assert goal.position.shape == (1, 1, 1, 2, 3)
        assert torch.equal(goal.position[..., 0, :], goal.position[..., 1, :])
        motion_seeds.append(seed_config)
        return SimpleNamespace(success=np.array([True]), interpolated_trajectory=None, interpolated_last_tstep=None)

    planner.valid_positions = valid
    planner._solve_motion = motion
    planner.motion = SimpleNamespace(
        ik_solver=SimpleNamespace(solve_pose=ik, config=SimpleNamespace(num_seeds=3)),
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=0.02)),
        disable_link_collision=lambda links: collisions.append("disable"),
        enable_link_collision=lambda links: collisions.append("enable"),
        update_tool_pose_criteria=lambda criteria: None,
    )
    monkeypatch.setattr(
        TimedTrajectory, "from_curobo", lambda *args, **kw: TimedTrajectory(np.array([0.0, 0.02]), np.zeros((2, 7)))
    )
    grasps = np.repeat(np.eye(4)[None], 2, axis=0)
    grasps[1, 0, 3] = 0.1
    feedback = SimpleNamespace(vector=lambda joints: np.zeros(7))
    options = dict(approach_offset=-0.15 if mode == "long_approach" else -0.05, deadline=1.0)
    if mode == "expired_during_ranking":
        published = []
        with pytest.raises(TimeoutError):
            planner.plan_grasp(
                feedback, grasps, np.eye(4), 0.0, **options, on_approach=lambda *args: published.append(args)
            )
        assert not motion_seeds and not published and not planner._grasp_selections
    else:
        for _ in range(2):
            plan = planner.plan_grasp(feedback, grasps, np.eye(4), 0.0, **options)
            assert plan.goalset_index == (0 if expected_rank else 1)
            # With vertical_lift_m 0 the lift holds the grasp end.
            assert plan.lift_method == "hold"
            assert np.array_equal(plan.lift.positions, np.repeat(plan.grasp.positions[-1:], 2, axis=0))
        assert ik_calls == [3 if expected_rank else 1]  # Repeated strategy reuses its selection.
        segments = 2
        assert len(motion_seeds) == segments * 2
        for index in (0, segments):
            assert motion_seeds[index][0, 0, 0] == pytest.approx(0.2 if expected_rank else 0.8)
        assert all(seed is None for index, seed in enumerate(motion_seeds) if index % segments)
        assert checks.count(24) == int(expected_rank)
        # Every solved segment still receives final validation.
        assert checks.count(2) == 4
    assert collisions[-1] == "enable"


def test_service_session_prevents_scene_updates_during_borrowed_snapshot():
    update_started = Event()
    entered = Event()

    class Planner:
        def __init__(self, *args, **kwargs):
            self.updates = 0

        def warmup(self):
            pass

        def locked_joints_match(self, feedback):
            return True

        def update_scene(self, feedback, estimate):
            self.updates += 1

        def close(self):
            pass

    with GraspPlanningService(gripper=GRIPPER, planner_factory=Planner) as service, ThreadPoolExecutor(1) as pool:
        first, second = SimpleNamespace(received_at=1.0), SimpleNamespace(received_at=2.0)

        def update():
            update_started.set()
            with service.session(None, second) as planner:
                entered.set()
                return planner

        with service.session(None, first) as borrowed:
            future = pool.submit(update)
            assert update_started.wait(timeout=1.0)
            assert not entered.wait(timeout=0.05)
            assert borrowed.updates == 0
        assert future.result(timeout=1.0) is borrowed
        assert borrowed.updates == 1


def test_service_discards_failed_scene_update_before_retry():
    instances = []

    class Planner:
        def __init__(self, *args, **kwargs):
            self.closed = 0
            instances.append(self)

        def warmup(self):
            pass

        def locked_joints_match(self, feedback):
            return True

        def update_scene(self, feedback, estimate):
            raise RuntimeError("mesh load failed")

        def close(self):
            self.closed += 1

    with GraspPlanningService(gripper=GRIPPER, planner_factory=Planner) as service:
        service.prepare(None, SimpleNamespace(received_at=1.0))
        fresh = SimpleNamespace(received_at=2.0)
        with pytest.raises(RuntimeError, match="mesh load failed"):
            service.prepare(None, fresh)
        assert instances[0].closed == 1
        service.prepare(None, fresh)
        assert len(instances) == 2
    assert [p.closed for p in instances] == [1, 1]


def test_unreachable_approach_candidates_are_planned_last_not_dropped():
    planned = []

    class Planner:
        max_goalset = 5
        planning_profile = "latency"

        def __init__(self, *args, **options):
            pass

        def locked_joints_match(self, feedback):
            return True

        def warmup(self):
            pass

        def update_scene(self, feedback, estimate):
            pass

        def screen_grasps(self, feedback, grasps, matrix, now, **options):
            return np.array([True, True, False, True])

        def approach_reachable(self, feedback, grasps, matrix, offset):
            assert offset == pytest.approx(-0.15) and len(grasps) == 3  # screened candidates only
            return np.array([False, True, True])  # candidate 0 cannot reach its approach pose

        def plan_grasp(self, feedback, grasps, matrix, now, **options):
            planned.append(grasps[:, 0, 3].tolist())
            if 0.0 not in grasps[:, 0, 3]:
                raise RuntimeError("reachable group fails")
            return SimpleNamespace(goalset_index=0)

        def close(self):
            pass

    grasps = np.stack([np.eye(4)] * 4)
    grasps[:, 0, 3] = np.arange(4)
    with GraspPlanningService(
        gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
    ) as service:
        service.prepare({}, SimpleNamespace(received_at=0.0))
        _, report = service.plan(
            {}, SimpleNamespace(received_at=0.0), grasps, [0.99, 0.98, 0.97, 0.96], np.eye(4), screen_check=True
        )
    # Reachable candidates first (by confidence) at every approach distance;
    # the unreachable one is tried after them, never dropped.
    assert planned[0] == [1.0, 3.0] and planned[-1] == [0.0]
    assert all(group == [1.0, 3.0] for group in planned[:-1])
    assert report["candidate_index"] == 0 and report["approach_reachable"]["unreachable"] == [0]


def test_device_fault_stops_the_candidate_loop_and_the_service_rebuilds_the_planner():
    from aimdk_grasp.planner import PlannerFault, device_fault

    assert device_fault(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not device_fault(
        RuntimeError("cuRobo grasp planning failed at approach: trajectory optimization rejected solution")
    )
    built, attempts = [], []

    class Planner:
        _closed = False

        def __init__(self, *args, **kwargs):
            built.append(self)
            self.fine_rescue_report = {"approach": 3}  # left over from an earlier request

        def warmup(self):
            pass

        def locked_joints_match(self, feedback):
            return True

        def update_scene(self, feedback, estimate):
            pass

        def plan_grasp(self, *args, **options):
            attempts.append(self.fine_rescue_report.copy())
            raise PlannerFault("RuntimeError: CUDA error: an illegal memory access was encountered")

        def close(self):
            self._closed = True

    grasps = np.repeat(np.eye(4)[None], 3, axis=0)
    with GraspPlanningService(gripper=GRIPPER, planner_factory=Planner, clock=lambda: 0.0) as service:
        with pytest.raises(PlannerFault) as error:
            service.plan(None, SimpleNamespace(received_at=0.0), grasps, [3.0, 2.0, 1.0], np.eye(4))
        assert attempts == [{}]  # counters reset per request; no further candidate tried
        assert built[0]._closed and "CUDA error" in error.value.planning_report["planner_fault"]
        # A planner that closed itself (failed fine-scene load) is replaced as well.
        with pytest.raises(PlannerFault):
            service.plan(None, SimpleNamespace(received_at=0.0), grasps, [3.0, 2.0, 1.0], np.eye(4))
        built[1]._closed = False
        service._planner, service._estimate = built[1], None
        built[1]._closed = True
        service.prepare(None, SimpleNamespace(received_at=0.0))
        assert len(built) == 3

        def cancelled():
            raise TimeoutError("Planning deadline exceeded")

        # A fault near the deadline (or with a cancel pending) is still the fault:
        # the checkpoint after the attempt must not replace it, or the faulted
        # planner would be kept for the next request.
        with pytest.raises(PlannerFault):
            service.plan(
                None, SimpleNamespace(received_at=0.0), grasps, [3.0, 2.0, 1.0], np.eye(4), checkpoint=cancelled
            )
        assert built[2]._closed


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("start", "start-state validation failed"),
        ("planning", "criteria cleanup failed"),
        ("timeout", "link cleanup failed"),
    ],
)
def test_cuda_fault_from_start_or_cleanup_discards_the_planner(monkeypatch, failure, message):
    import sys

    from aimdk_grasp.planner import PlannerFault

    pytest.importorskip("torch")

    class Criteria:
        pass

    monkeypatch.setitem(sys.modules, "curobo._src.cost.tool_pose_criteria", SimpleNamespace(ToolPoseCriteria=Criteria))
    clock = [0.0]
    monkeypatch.setattr("aimdk_grasp.planner.time.perf_counter", lambda: clock[0])
    built, attempts = [], []

    class Planner(LeftArmPlanner):
        def __init__(self, *args, **options):
            built.append(self)
            self._closed = False
            self.max_goalset = options["max_goalset"]
            self.grasp_to_palm = np.eye(4)
            self.hand_links = ["finger"]
            self.planning_profile = options["planning_profile"]
            self._grasp_selections = {}
            self._goal_pose = lambda transforms: transforms
            self._state = lambda positions: positions

            def enable_links(_links):
                if failure == "planning":
                    raise RuntimeError("collision-link reset failed")
                raise RuntimeError("CUDA error: link cleanup failed")

            def reset_criteria(_value):
                if failure == "planning":
                    raise RuntimeError("CUDA error: criteria cleanup failed")

            def select_goal(*args, **options):
                raise RuntimeError("candidate is unreachable")

            self.motion = SimpleNamespace(
                enable_link_collision=enable_links,
                disable_link_collision=lambda links: None,
                update_tool_pose_criteria=reset_criteria,
                ik_solver=SimpleNamespace(solve_pose=select_goal),
            )

        def warmup(self):
            pass

        def locked_joints_match(self, feedback):
            return True

        def update_scene(self, feedback, estimate):
            pass

        def snapshot_matches(self, feedback, now):
            return True

        def _require_valid_start(self, positions):
            attempts.append(positions.copy())
            if failure == "start":
                raise RuntimeError("CUDA error: start-state validation failed")
            if failure == "timeout":
                clock[0] = 2.0

        def close(self):
            self._closed = True

    feedback = SimpleNamespace(vector=lambda joints: np.zeros(7))
    estimate = SimpleNamespace(received_at=0.0, support=None)
    grasps = np.repeat(np.eye(4)[None], 3, axis=0)
    options = {"deadline": 1.0} if failure == "timeout" else {}
    with GraspPlanningService(gripper=GRIPPER, planner_factory=Planner, clock=lambda: 0.0) as service:
        with pytest.raises(PlannerFault, match=message):
            service.plan(feedback, estimate, grasps, [3.0, 2.0, 1.0], np.eye(4), **options)
        assert len(attempts) == 1
        assert built[0]._closed and service._planner is None


def test_candidates_whose_grasp_line_is_blocked_follow_clear_ones_and_precede_unreachable_ones(monkeypatch):
    from aimdk_grasp import planner_service

    monkeypatch.setattr(planner_service, "GRASP_LINE_CANDIDATES", 3)
    planned, checked = [], []

    class Planner:
        max_goalset = 1
        planning_profile = "latency"

        def __init__(self, *args, **options):
            pass

        def locked_joints_match(self, feedback):
            return True

        def warmup(self):
            pass

        def update_scene(self, feedback, estimate):
            pass

        def screen_grasps(self, feedback, grasps, matrix, now, **options):
            return np.ones(len(grasps), dtype=bool)

        def approach_reachable(self, feedback, grasps, matrix, offset):
            return np.array([True, False, True, True, True])  # candidate 1 cannot reach its approach

        def grasp_line_clear(self, candidates, grasps, matrix):
            checked.append((list(candidates), grasps[:, 0, 3].tolist()))
            return np.array([False, True, True])  # the most confident reachable candidate skims the target

        def plan_grasp(self, feedback, grasps, matrix, now, **options):
            planned.append(int(grasps[0, 0, 3]))
            raise RuntimeError("fails")

        def close(self):
            pass

    grasps = np.stack([np.eye(4)] * 5)
    grasps[:, 0, 3] = np.arange(5)
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(
            {}, SimpleNamespace(received_at=0.0), grasps, [0.9, 0.8, 0.7, 0.6, 0.5], np.eye(4), screen_check=True
        )
    # Only the first three in planning order are checked (the unreachable one is last).
    assert checked == [([0, 2, 3], [0.0, 2.0, 3.0])]
    report = error.value.planning_report
    assert report["grasp_line"]["blocked"] == [0] and report["grasp_line"]["checked"] == [0, 2, 3]
    # Tiers in order (each tier runs all its strategies first): clear/unchecked, blocked, unreachable.
    assert list(dict.fromkeys(planned)) == [2, 3, 4, 0, 1]


def test_candidates_whose_hand_ends_close_to_the_target_follow_the_others_of_their_tier(monkeypatch):
    from aimdk_grasp import planner_service

    monkeypatch.setattr(planner_service, "GRASP_LINE_CANDIDATES", 4)
    # Every candidate's closing sweep holds the target.
    monkeypatch.setattr(
        "aimdk_grasp.planner_service.enclosed_points", lambda grasps, points, boxes, indices: np.full(len(grasps), 1000)
    )
    planned, asked = [], []

    class Planner:
        max_goalset = 1
        planning_profile = "latency"

        def __init__(self, *args, **options):
            pass

        def locked_joints_match(self, feedback):
            return True

        def warmup(self):
            pass

        def update_scene(self, feedback, estimate):
            pass

        def screen_grasps(self, feedback, grasps, matrix, now, **options):
            return np.ones(len(grasps), dtype=bool)

        def approach_reachable(self, feedback, grasps, matrix, offset):
            return np.array([True, True, True, True, False])  # candidate 4 cannot reach its approach

        def grasp_line_clear(self, candidates, grasps, matrix):
            return np.array([True, True, False, True])  # candidate 2 skims the scene

        def contact_target_clearance(self, candidates, points, *line):
            asked.append((list(candidates), points.copy()))
            # The most confident ends 5 mm from the target (live: the palm pushed it); 3 has no contact state.
            return np.array([0.005, 0.02, 0.001, np.inf])

        def plan_grasp(self, feedback, grasps, matrix, now, **options):
            planned.append(int(grasps[0, 0, 3]))
            raise RuntimeError("fails")

        def close(self):
            pass

    grasps = np.stack([np.eye(4)] * 5)
    grasps[:, 0, 3] = np.arange(5)
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(
            {},
            SimpleNamespace(received_at=0.0),
            grasps,
            [0.99, 0.98, 0.97, 0.96, 0.95],
            matrix,
            screen_check=True,
            target_points_camera=np.zeros((3, 3)),
        )
    # The target points reach the planner in the base frame.
    ((candidates, points),) = asked
    assert candidates == [0, 1, 2, 3] and np.allclose(points, [1.0, 2.0, 3.0])
    report = error.value.planning_report["target_clearance"]
    assert report["tight"] == [0, 2] and set(report["clearance_m"]) == {0, 1, 2}
    # Tiers: clear and roomy, clear but tight, blocked (tight or not), unreachable. Nothing is dropped.
    assert list(dict.fromkeys(planned)) == [1, 3, 0, 2, 4]
    # Reachable candidates with a clear line exist: no second reachability check.
    assert "fallback" not in error.value.planning_report["approach_reachable"]

    planned.clear()
    monkeypatch.setattr(planner_service, "TARGET_CLEARANCE_M", None)
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate"),
    ):
        service.plan(
            {},
            SimpleNamespace(received_at=0.0),
            grasps,
            [0.99, 0.98, 0.97, 0.96, 0.95],
            matrix,
            screen_check=True,
            target_points_camera=np.zeros((3, 3)),
        )
    assert list(dict.fromkeys(planned)) == [0, 1, 3, 2, 4]

    # Low confidence goes after the tight ones of its tier, alone (3 is clear and roomy but 0.5).
    planned.clear()
    monkeypatch.setattr(planner_service, "TARGET_CLEARANCE_M", 0.01)
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(
            {},
            SimpleNamespace(received_at=0.0),
            grasps,
            [0.99, 0.98, 0.97, 0.5, 0.95],
            matrix,
            screen_check=True,
            target_points_camera=np.zeros((3, 3)),
        )
    assert list(dict.fromkeys(planned)) == [1, 0, 3, 2, 4]
    assert error.value.planning_report["low_confidence"] == {"min": 0.7, "later": [3]}

    # A planner with the merged pass (grasp_line_checks) gives the same order
    # and clearances from one call, with the target points in the base frame.
    class Merged(Planner):
        def grasp_line_checks(self, candidates, grasps, matrix, points):
            asked.append(("merged", list(candidates), points.copy()))
            return self.grasp_line_clear(candidates, grasps, matrix), np.array([0.005, 0.02, 0.001, np.inf])

        def contact_target_clearance(self, *values):
            pytest.fail("the merged pass replaces it")

    planned.clear()
    asked.clear()
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Merged, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(
            {},
            SimpleNamespace(received_at=0.0),
            grasps,
            [0.99, 0.98, 0.97, 0.5, 0.95],
            matrix,
            screen_check=True,
            target_points_camera=np.zeros((3, 3)),
        )
    assert list(dict.fromkeys(planned)) == [1, 0, 3, 2, 4]
    ((kind, candidates, points),) = asked
    assert kind == "merged" and candidates == [0, 1, 2, 3] and np.allclose(points, [1.0, 2.0, 3.0])
    report = error.value.planning_report["target_clearance"]
    assert report["tight"] == [0, 2] and report["with_grasp_line"]

    # A palm facing the robot's left (grasp x -> base +y) goes after the tight
    # ones, before the low-confidence ones (1 is clear, roomy and 0.98).
    planned.clear()
    left = grasps.copy()
    left[1, :3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(
            {},
            SimpleNamespace(received_at=0.0),
            left,
            [0.99, 0.98, 0.97, 0.5, 0.95],
            matrix,
            screen_check=True,
            target_points_camera=np.zeros((3, 3)),
        )
    assert list(dict.fromkeys(planned)) == [0, 1, 3, 2, 4]
    assert error.value.planning_report["palm_left"]["later"] == [1]


def test_when_every_reachable_line_is_blocked_unreachable_candidates_are_checked_at_a_shorter_approach(monkeypatch):
    # Live 2026-10-14 clr_1: four blocked candidates failed 13 attempts (8.9 s on
    # the Orin) before the unreachable ones; one of those planned at -0.10 m.
    from aimdk_grasp import planner_service

    monkeypatch.setattr(planner_service, "GRASP_LINE_CANDIDATES", 5)
    monkeypatch.setattr(planner_service, "TARGET_CLEARANCE_M", None)
    monkeypatch.setattr(
        "aimdk_grasp.planner_service.enclosed_points", lambda grasps, points, boxes, indices: np.full(len(grasps), 1000)
    )
    planned, offsets = [], []

    class Planner:
        max_goalset = 1
        planning_profile = "latency"

        def __init__(self, *args, **options):
            pass

        def locked_joints_match(self, feedback):
            return True

        def warmup(self):
            pass

        def update_scene(self, feedback, estimate):
            pass

        def screen_grasps(self, feedback, grasps, matrix, now, **options):
            return np.ones(len(grasps), dtype=bool)

        def approach_reachable(self, feedback, grasps, matrix, offset):
            offsets.append((offset, [int(g[0, 3]) for g in grasps]))
            if offset == planner_service.APPROACH_REACH_OFFSET_M:
                return np.array([True, True, False, False, False])
            return np.array([False, True, False])  # of 2, 3, 4: only 3 reaches the shorter approach

        def grasp_line_clear(self, candidates, grasps, matrix):
            return np.array([g[0, 3] >= 2 for g in grasps])  # 0 and 1 skim the scene

        def plan_grasp(self, feedback, grasps, matrix, now, **options):
            planned.append(int(grasps[0, 0, 3]))
            raise RuntimeError("fails")

        def close(self):
            pass

    grasps = np.stack([np.eye(4)] * 5)
    grasps[:, 0, 3] = np.arange(5)
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(
            {},
            SimpleNamespace(received_at=0.0),
            grasps,
            [0.9, 0.8, 0.7, 0.6, 0.5],
            np.eye(4),
            screen_check=True,
            target_points_camera=np.zeros((3, 3)),
        )
    assert offsets == [
        (planner_service.APPROACH_REACH_OFFSET_M, [0, 1, 2, 3, 4]),
        (planner_service.APPROACH_REACH_FALLBACK_M, [2, 3, 4]),
    ]
    assert error.value.planning_report["approach_reachable"]["fallback"]["reachable"] == [3]
    # 3 now counts as reachable with a clear line; the blocked ones next, then the unreachable. Nothing dropped.
    assert list(dict.fromkeys(planned)) == [3, 0, 1, 2, 4]


def test_side_first_request_never_plans_a_candidate_from_above():
    from test_grasp_session import _grasps_with_approach

    planned, screened = [], []

    class Planner:
        max_goalset = 1
        planning_profile = "latency"

        def __init__(self, *args, **options):
            pass

        def locked_joints_match(self, feedback):
            return True

        def warmup(self):
            pass

        def screen_grasps(self, feedback, grasps, matrix, now, **options):
            screened.append(len(grasps))
            return np.ones(len(grasps), dtype=bool)

        def plan_grasp(self, feedback, grasps, matrix, now, **options):
            planned.append(round(float(grasps[0, 2, 2]), 2))
            raise RuntimeError("fails")

        def close(self):
            pass

    # side, oblique from above (cos -0.54, live 2026-10-13), straight down
    grasps = _grasps_with_approach([1.0, 0.0, 0.0], [0.84, 0.0, -0.54], [0.0, 0.0, -1.0])
    support = SimpleNamespace(normal=[0.0, 0.0, 1.0], target_heights_m={"p95": 0.19})
    with (
        GraspPlanningService(
            gripper=GRIPPER, clock=lambda: 0.0, planner_factory=Planner, planning_profile="latency"
        ) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(
            {},
            SimpleNamespace(received_at=0.0, support=support),
            grasps,
            [0.5, 0.9, 0.8],
            np.eye(4),
            screen_check=True,
        )
    report = error.value.planning_report
    assert report["approach_priority"]["excluded"] == [1, 2]
    assert report["screened_candidates"] == [0]
    assert set(planned) == {0.0}


def test_service_failed_request_retains_attempts_and_timing():
    class Planner:
        def __init__(self, *args, **kwargs):
            self.max_goalset = 1

        def warmup(self):
            pass

        def plan_grasp(self, *args, **kwargs):
            raise RuntimeError("fine collision")

        def close(self):
            pass

    with (
        GraspPlanningService(gripper=GRIPPER, planner_factory=Planner, clock=lambda: 0.0) as service,
        pytest.raises(RuntimeError, match="No candidate") as error,
    ):
        service.plan(None, SimpleNamespace(received_at=0.0), np.eye(4)[None], [1.0], np.eye(4))
    report = error.value.planning_report
    assert report["planning_attempts"] and "winning_candidate" not in report
    assert all(not row["success"] for row in report["planning_attempts"])
    assert set(report["timings_s"]) == {"prepare", "screen", "plan"}


def test_service_retains_screen_timeout_and_passes_request_deadline():
    import time

    deadline = time.perf_counter() + 60.0

    class Planner:
        def __init__(self, *args, **kwargs):
            pass

        def warmup(self):
            pass

        def screen_grasps(self, *args, **options):
            assert options["deadline"] == deadline
            raise TimeoutError("Planning deadline exceeded")

        def close(self):
            pass

    with (
        GraspPlanningService(gripper=GRIPPER, planner_factory=Planner, clock=lambda: 0.0) as service,
        pytest.raises(TimeoutError) as error,
    ):
        service.plan(
            None,
            SimpleNamespace(received_at=0.0),
            np.eye(4)[None],
            [1.0],
            np.eye(4),
            screen_check=True,
            deadline=deadline,
        )
    report = error.value.planning_report
    assert report["timings_s"]["screen"] >= 0.0 and report["timings_s"]["plan"] == 0.0
    assert "winning_candidate" not in report
