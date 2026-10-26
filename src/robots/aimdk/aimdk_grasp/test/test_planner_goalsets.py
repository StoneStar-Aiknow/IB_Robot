from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.planner import LeftArmPlanner, _skip_static_scene_spheres
from test_planner_contracts import GRIPPER


def test_goalset_padding_preserves_goal_identity_across_retries():
    planner = object.__new__(LeftArmPlanner)
    planner.max_goalset = 4
    goals = np.repeat(np.eye(4)[None], 4, axis=0)
    goals[:, 0, 3] = [1.0, 2.0, 3.0, 4.0]
    for remaining in (goals, goals[[0, 2, 3]], goals[[3]]):
        padded = planner._fixed_goalset_transforms(remaining)
        assert padded.shape == (4, 4, 4) and padded.flags.c_contiguous
        for slot in range(4):
            local = planner._input_goal_index(slot, len(remaining))
            np.testing.assert_array_equal(padded[slot], remaining[local])
    for index in (-1, 4):
        with pytest.raises(RuntimeError, match="outside"):
            planner._input_goal_index(index, 1)
    for invalid in (goals[:0], np.repeat(goals, 2, axis=0)):
        with pytest.raises(ValueError):
            planner._fixed_goalset_transforms(invalid)


def test_static_scene_spheres_are_skipped_only_by_that_scene_model():
    torch = pytest.importorskip("torch")
    calls = []

    class Scene:
        def query(self, state, *args, **kwargs):
            calls.append((state, args, kwargs))
            return state.robot_spheres[..., 3].clone()

        get_sphere_distance = get_sphere_collision = query
        get_swept_sphere_distance = get_swept_sphere_collision = query

    scene, other = Scene(), Scene()
    _skip_static_scene_spheres(scene, torch.tensor([False, True, False, True]))
    spheres = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    state = SimpleNamespace(robot_spheres=spheres, joint_state="q")
    for name in (
        "get_sphere_distance",
        "get_sphere_collision",
        "get_swept_sphere_distance",
        "get_swept_sphere_collision",
    ):
        radii = getattr(scene, name)(state, "buffer", weight=2.0)
        np.testing.assert_array_equal(radii[:, [1, 3]].numpy(), -100.0)
        torch.testing.assert_close(radii[:, [0, 2]], spheres[:, [0, 2], 3])
        passed, args, kwargs = calls[-1]
        assert passed is not state and passed.joint_state == "q" and args == ("buffer",) and kwargs == dict(weight=2.0)
    assert state.robot_spheres is spheres and spheres[0, 1, 3] == 7.0
    torch.testing.assert_close(other.get_sphere_distance(state), spheres[..., 3])


@pytest.fixture
def screen_planner(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("aimdk_grasp.planner.palm_goals_from_camera_grasps", lambda g, matrix, mount: np.asarray(g))
    planner = object.__new__(LeftArmPlanner)
    planner.max_goalset = 100
    planner.snapshot_matches = lambda feedback, now: True
    planner._require_valid_start = lambda positions: None
    planner.valid_positions = lambda positions: np.ones(len(positions), dtype=bool)
    planner._state = lambda positions: positions
    planner._goal_pose = planner._fixed_goalset_transforms
    planner.hand_links = ["finger"]
    planner.grasp_to_palm = np.eye(4)
    stub = SimpleNamespace(planner=planner, calls=[], collisions=[], clock=0.0, discover=None, expire_at=None)
    monkeypatch.setattr("aimdk_grasp.planner.time.perf_counter", lambda: stub.clock)

    def solve(goals, state, **options):
        stub.calls.append(goals[:, 0, 3].astype(int).tolist())
        if stub.expire_at == len(stub.calls):
            stub.clock = 2.0
        return SimpleNamespace(
            success=torch.tensor([[len(stub.calls) == stub.discover]]),
            goalset_index=torch.tensor([[[99]]]),
            solution=torch.zeros((1, 1, 7)),
        )

    planner.motion = SimpleNamespace(
        disable_link_collision=lambda _: stub.collisions.append(False),
        enable_link_collision=lambda _: stub.collisions.append(True),
        ik_solver=SimpleNamespace(
            solve_pose=solve, config=SimpleNamespace(num_seeds=1, position_tolerance=0.005, orientation_tolerance=0.05)
        ),
    )
    stub.feedback = SimpleNamespace(vector=lambda _: np.zeros(7))
    return stub


def test_contact_cache_owns_exact_states_deduplicates_and_is_bounded():
    planner = object.__new__(LeftArmPlanner)
    planner._ik_validity_cache_capacity = 2
    checked = []

    def valid(positions):
        checked.append(positions.copy())
        return positions[:, 0] > 0.0

    planner.valid_positions = valid
    q = np.array([[0.0] * 7, [1.0] * 7, [0.0] * 7])
    result = planner._valid_contact_positions(q)
    assert result.tolist() == [False, True, False] and len(checked[0]) == 2
    result[:] = True
    assert planner._valid_contact_positions(q).tolist() == [False, True, False]
    assert len(checked) == 1
    q[:, 0] += 0.000001
    assert planner._valid_contact_positions(q).all()
    assert len(checked) == 2 and len(planner._ik_validity_cache) <= 2
    for invalid in (np.empty((0, 7)), np.zeros(7), np.full((1, 7), np.nan)):
        with pytest.raises(ValueError):
            planner._valid_contact_positions(invalid)


@pytest.mark.parametrize("mode", ["failed", "success", "native", "expired", "ranking_expired"])
def test_approach_retry_is_bounded_and_keeps_selected_pose(monkeypatch, mode):
    torch = pytest.importorskip("torch")
    clock = [0.0]
    monkeypatch.setattr("aimdk_grasp.planner.time.perf_counter", lambda: clock[0])
    planner = object.__new__(LeftArmPlanner)
    planner._trajopt_seeds = 1  # the single-seed path is under test
    planner.planning_profile = "latency"
    planner.approach_seed_policy = "native" if mode == "native" else "clear_path"
    goal = object()
    state = SimpleNamespace(position=torch.zeros((1, 7)))
    calls = []
    selection = SimpleNamespace(success=torch.tensor([[True, True]]), solution=torch.tensor([[[1.0] * 7, [2.0] * 7]]))

    def ik(actual_goal, actual_state, **options):
        assert actual_goal is goal and actual_state is state
        calls.append("ik")
        return selection

    def rank(actual_selection, start):
        assert actual_selection is selection
        calls.append("rank")
        if mode == "ranking_expired":
            clock[0] = 2.0
        return 1

    def motion(actual_goal, actual_state, **options):
        assert actual_goal is goal and actual_state is state
        calls.append("trajopt")
        assert torch.equal(options["seed_config"], selection.solution[:, 1:2])
        return SimpleNamespace(success=torch.tensor([[True]]))

    planner.motion = SimpleNamespace(
        ik_solver=SimpleNamespace(solve_pose=ik, config=SimpleNamespace(num_seeds=2)),
        trajopt_solver=SimpleNamespace(solve_pose=motion),
    )
    monkeypatch.setattr(planner, "_approach_seed_index", rank)
    failed = SimpleNamespace(success=torch.tensor([[mode == "success"]]))
    if mode == "expired":
        clock[0] = 2.0
    if mode in ("expired", "ranking_expired"):
        with pytest.raises(TimeoutError):
            planner._retry_approach(goal, state, failed, deadline=1.0)
        assert calls == ([] if mode == "expired" else ["ik", "rank"])
    else:
        result = planner._retry_approach(goal, state, failed, deadline=1.0)
        assert calls == (["ik", "rank", "trajopt"] if mode == "failed" else [])
        assert (result is failed) == (mode != "failed")


def test_batch_screen_solves_each_candidate_in_fixed_shape_chunks(screen_planner):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("screen goals are CUDA tensors")
    stub = screen_planner
    planner = stub.planner
    planner._screen_batch_size, planner._screen_ik_seeds = 4, 2
    calls, checked = [], []

    def solve(pose, state, return_seeds):
        x = pose.position[:, 0, 0, 0, 0].cpu()
        calls.append((x.tolist(), tuple(state.position.shape), return_seeds))
        # Seed 0 reaches even candidates; seed 1 reaches multiples of three.
        success = torch.stack([x % 2 == 0, x % 3 == 0], dim=1)
        solution = x[:, None, None].repeat(1, 2, 7) + torch.tensor([0.0, 0.5])[None, :, None]
        return SimpleNamespace(success=success.reshape(-1, 1), solution=solution.reshape(-1, 1, 7))

    def valid(q):
        checked.extend(q[:, 0].tolist())
        return q[:, 0] != 4.0  # candidate 4 reaches only colliding states

    planner.valid_positions = valid
    planner.screen_ik = SimpleNamespace(
        solve_pose=solve, config=SimpleNamespace(position_tolerance=0.005, orientation_tolerance=0.05)
    )
    goals = np.repeat(np.eye(4)[None], 6, axis=0)
    goals[:, 0, 3] = np.arange(6)
    diagnostics = []
    flags = planner.screen_grasps(stub.feedback, goals, np.eye(4), 0.0, diagnostics=diagnostics, deadline=1.0)
    assert np.flatnonzero(flags).tolist() == [0, 2, 3]
    assert [c[0] for c in calls] == [[0, 1, 2, 3], [4, 5, 4, 4]]  # padding repeats the first goal
    assert all(c[1:] == ((4, 7), 2) for c in calls)
    assert sorted(checked) == [0.0, 2.0, 3.5, 4.0]  # one state per goal; candidate 0's second is not needed
    assert stub.collisions == []  # the screen model has no hand links to disable
    assert planner.contact_screen_report == dict(
        rounds=2,
        ik_successful_states=5,
        fine_valid_states=3,
        fine_rejected_states=1,
        fine_unchecked_states=1,
        coarse_candidates=[0, 2, 3, 4],
        fine_valid_candidates=[0, 2, 3],
        method="batch",
    )
    assert diagnostics[0]["method"] == "batch" and len(diagnostics[0]["rounds"]) == 2


def test_batch_screen_fine_checks_further_states_only_for_goals_still_infeasible(screen_planner):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("screen goals are CUDA tensors")
    stub = screen_planner
    planner = stub.planner
    planner._screen_batch_size, planner._screen_ik_seeds = 4, 2
    checks = []

    def solve(pose, state, return_seeds):
        x = pose.position[:, 0, 0, 0, 0].cpu()
        success = torch.stack([x % 2 == 0, x % 3 == 0], dim=1)  # goals 0 and 6 reach both seeds
        solution = x[:, None, None].repeat(1, 2, 7) + torch.tensor([0.0, 0.5])[None, :, None]
        return SimpleNamespace(success=success.reshape(-1, 1), solution=solution.reshape(-1, 1, 7))

    def valid(q):
        checks.append(sorted(q[:, 0].tolist()))
        return ~np.isin(q[:, 0], [0.0, 4.0])  # goal 0's first state and goal 4 collide

    planner.valid_positions = valid
    planner.screen_ik = SimpleNamespace(solve_pose=solve)
    goals = np.repeat(np.eye(4)[None], 7, axis=0)
    goals[:, 0, 3] = np.arange(7)
    flags = planner.screen_grasps(stub.feedback, goals, np.eye(4), 0.0, deadline=1.0)
    assert np.flatnonzero(flags).tolist() == [0, 2, 3, 6]
    assert checks == [[0.0, 2.0, 3.5, 4.0, 6.0], [0.5]]  # 6.5 is never needed
    report = planner.contact_screen_report
    assert (report["fine_valid_states"], report["fine_rejected_states"], report["fine_unchecked_states"]) == (4, 2, 1)


def test_batch_screen_honors_the_deadline_between_chunks(screen_planner):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("screen goals are CUDA tensors")
    stub = screen_planner
    planner = stub.planner
    planner._screen_batch_size, planner._screen_ik_seeds = 2, 1

    def solve(pose, state, return_seeds):
        stub.clock = 2.0
        return SimpleNamespace(success=torch.zeros((2, 1), dtype=torch.bool), solution=torch.zeros((2, 1, 7)))

    planner.screen_ik = SimpleNamespace(solve_pose=solve)
    with pytest.raises(TimeoutError):
        planner.screen_grasps(stub.feedback, np.repeat(np.eye(4)[None], 5, axis=0), np.eye(4), 0.0, deadline=1.0)


def test_screen_robot_config_keeps_only_moving_non_hand_arm_links():
    from aimdk_grasp.hand import hand_links
    from aimdk_grasp.planner import screen_robot_config
    from aimdk_grasp.robot_model import collision_robot_config, planning_model

    config = collision_robot_config()
    kin = config["robot_cfg"]["kinematics"]
    hand = set(hand_links(planning_model()))
    screen = screen_robot_config(config, hand)["robot_cfg"]["kinematics"]
    links = screen["collision_link_names"]
    assert links and not set(links) & hand and "torso_link" not in links
    assert set(links) < set(kin["collision_link_names"])
    assert not any(name.startswith(("head", "right")) for name in screen["lock_joints"])
    assert kin["collision_link_names"] != links  # the source config is not modified


def test_batched_palm_goals_match_the_per_pose_conversion_and_checks():
    from aimdk_grasp.frames import rigid
    from aimdk_grasp.planner import grasp_to_palm_transform, palm_goals_from_camera_grasps
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    grasps = np.repeat(np.eye(4)[None], 50, axis=0)
    grasps[:, :3, :3] = Rotation.random(50, random_state=1).as_matrix()
    grasps[:, :3, 3] = rng.uniform(-1, 1, (50, 3))
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 1.0]).as_matrix()
    base[:3, 3] = [0.1, 0.2, 0.3]
    mount = grasp_to_palm_transform(GRIPPER)
    expected = np.stack([rigid(base) @ rigid(g) @ mount for g in grasps])
    np.testing.assert_allclose(palm_goals_from_camera_grasps(grasps, base, mount), expected, atol=1e-12)
    for bad in (grasps[:, :3], np.where(np.arange(4) == 3, np.nan, grasps), grasps * [[1, 1, 1, 1]] * 2):
        with pytest.raises(ValueError):
            palm_goals_from_camera_grasps(bad, base, mount)
    with pytest.raises(ValueError, match="proper"):
        palm_goals_from_camera_grasps(grasps, base, np.diag([2.0, 1.0, 1.0, 1.0]))


def test_fine_scene_cost_queries_spheres_shared_with_the_first_state_once():
    torch = pytest.importorskip("torch")
    from aimdk_grasp.planner import LeftArmPlanner

    planner = object.__new__(LeftArmPlanner)
    queried = []

    def forward(state):
        spheres = state.robot_spheres
        queried.append(spheres[..., 3].clone())
        # Cost: sphere x coordinate where the radius is positive (a skipped sphere reads 0).
        return torch.where(spheres[..., 3] > 0, spheres[..., 0], torch.zeros(()))

    planner.checker = SimpleNamespace(collision_constraint=SimpleNamespace(forward=forward))
    spheres = torch.ones((3, 1, 4, 4))
    spheres[..., 0] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    spheres[1, 0, 2, 0] = 9.0  # state 1 moves sphere 2
    spheres[2, 0, 3, 1] = 2.0  # state 2 moves sphere 3
    state = SimpleNamespace(robot_spheres=spheres)
    cost = planner._scene_cost_once_per_sphere_pose(state)
    expected = forward(SimpleNamespace(robot_spheres=spheres))
    torch.testing.assert_close(cost, expected)
    assert (queried[0] > 0).tolist() == [[[True] * 4], [[False, False, True, False]], [[False, False, False, True]]]
    assert state.robot_spheres is spheres and (spheres[..., 3] == 1.0).all()


@pytest.fixture
def linear_grasp_planner(monkeypatch):
    torch = pytest.importorskip("torch")
    from aimdk_grasp.planner import LeftArmPlanner

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    planner = object.__new__(LeftArmPlanner)

    def palm_poses(positions):
        q = np.asarray(positions, dtype=float)
        poses = np.repeat(np.eye(4)[None], len(q), axis=0)
        poses[:, :3, 3] = (
            0.3
            * np.stack(
                [
                    np.sin(q[:, 0]) * np.cos(q[:, 1]),
                    np.sin(q[:, 1]) * np.cos(q[:, 2]),
                    np.sin(q[:, 2]) * np.cos(q[:, 0]),
                ],
                axis=1,
            )
            + 0.05 * q[:, 3:6]
        )  # curved
        return poses

    stub = SimpleNamespace(
        planner=planner, palm=palm_poses, valid=lambda q: np.ones(len(q), dtype=bool), solutions=None, start=np.zeros(7)
    )
    stub.end = np.array([1.0, -0.8, 0.6, 0.1, 0.0, 0.0, 0.5])
    near, far = stub.end, stub.end + 1.0
    stub.solutions = torch.tensor(np.stack([far, near])[None], dtype=torch.float32)

    def solve(goal, state, *, seed_config):
        assert seed_config.shape == (1, 2, 7)
        return SimpleNamespace(success=torch.tensor([[True, True]]), solution=stub.solutions)

    limits = SimpleNamespace(
        **{
            name: torch.tensor([[-value] * 7, [value] * 7])
            for name, value in (("position", 3.0), ("velocity", 1.0), ("acceleration", 2.0), ("jerk", 10.0))
        }
    )
    planner.motion = SimpleNamespace(
        ik_solver=SimpleNamespace(solve_pose=solve, config=SimpleNamespace(num_seeds=2)),
        kinematics=SimpleNamespace(config=SimpleNamespace(kinematics_config=SimpleNamespace(joint_limits=limits))),
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=0.02)),
    )
    planner._goal_pose = lambda transforms: transforms
    planner._state = lambda q: SimpleNamespace(position=torch.tensor(np.asarray(q, dtype=np.float32)[None]))
    planner._palm_poses = palm_poses
    planner.valid_positions = lambda q: stub.valid(q)
    stub.goal = palm_poses(stub.end.astype(np.float32)[None])[0]
    return stub


def test_linear_grasp_follows_the_palm_line_within_limits(linear_grasp_planner):
    stub = linear_grasp_planner
    trajectory, reason = stub.planner._linear_grasp_segment(stub.start, stub.goal, lambda: None)
    assert reason is None
    np.testing.assert_allclose(trajectory.positions[0], stub.start)
    np.testing.assert_allclose(trajectory.positions[-1], stub.end, atol=1e-6)  # the nearest IK state
    np.testing.assert_allclose(np.diff(trajectory.times), 0.02)
    for order, bound in ((1, 1.0), (2, 2.0), (3, 10.0)):
        assert np.abs(np.diff(trajectory.positions, n=order, axis=0) / 0.02**order).max() <= 0.9 * bound + 1e-9
    palms = stub.palm(trajectory.positions)[:, :3, 3]
    line = stub.goal[:3, 3] - palms[0]
    along = np.clip((palms - palms[0]) @ line / (line @ line), 0.0, 1.0)
    assert np.linalg.norm(palms - palms[0] - along[:, None] * line, axis=1).max() <= 0.005
    # The plain joint line would leave the palm line by centimetres.
    joint_line = stub.palm(np.linspace(stub.start, stub.end, 50))[:, :3, 3]
    assert (
        np.linalg.norm(
            joint_line - palms[0] - np.clip((joint_line - palms[0]) @ line / (line @ line), 0.0, 1.0)[:, None] * line,
            axis=1,
        ).max()
        > 0.01
    )


@pytest.mark.parametrize("failure", ["ik", "fine", "line"])
def test_linear_grasp_reports_why_trajopt_must_plan_the_grasp(linear_grasp_planner, failure):
    torch = pytest.importorskip("torch")
    stub = linear_grasp_planner
    goal = stub.goal
    if failure == "ik":
        stub.planner.motion.ik_solver.solve_pose = lambda *a, **k: SimpleNamespace(
            success=torch.tensor([[False, False]]), solution=stub.solutions
        )
    elif failure == "fine":
        stub.valid = lambda q: np.arange(len(q)) != len(q) // 2
    else:
        stub.planner._linear_grasp_corrections = 0  # waypoints stay on the curved joint line
    assert stub.planner._linear_grasp_segment(stub.start, goal, lambda: None) == (None, failure)


def test_escape_seed_replaces_only_the_first_trajopt_seed_and_is_cached_per_start(monkeypatch):
    torch = pytest.importorskip("torch")
    from aimdk_grasp.planner import LeftArmPlanner

    planner = object.__new__(LeftArmPlanner)
    planner._approach_graph_seed = "escape"
    paths, solves = [], []

    def cartesian(origins, offsets, steps, check):
        paths.append(offsets.copy())
        return np.linspace(origins[0], origins[0] + 0.2, steps + 1)[None], np.array([True])

    planner._cartesian_paths = cartesian
    planner._palm_poses = lambda q: np.stack([np.eye(4)] * len(q))
    planner.motion = SimpleNamespace(
        trajopt_solver=SimpleNamespace(
            action_horizon=8, solve_pose=lambda goal, state, **options: solves.append(options) or "result"
        )
    )
    ends = torch.stack([torch.full((7,), 1.0), torch.full((7,), 2.0)])[None]
    state = SimpleNamespace(position=torch.zeros((1, 7)))
    # update_scene precomputes from float64 feedback; the solver state is float32.
    assert planner._escape_chain(np.full(7, 1e-9)) is not None
    state.position = torch.tensor(np.full((1, 7), 1e-9), dtype=torch.float32)
    for _ in range(2):
        assert planner._escape_seeded_solve("goal", state, ends, 25, {}, lambda: None) == "result"
    assert len(paths) == 1  # one escape per start state
    offset = paths[0][0]
    np.testing.assert_allclose(offset, [0.0, 0.1 / np.sqrt(2), 0.1 / np.sqrt(2)])  # up and outward (+y side)
    seed_traj = solves[0]["seed_traj"]
    assert seed_traj.shape == (1, 1, 8, 7)
    torch.testing.assert_close(seed_traj[0, 0, 0], torch.zeros(7), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(seed_traj[0, 0, -1], torch.full((7,), 1.0))  # ends at the first IK seed
    torch.testing.assert_close(solves[0]["seed_config"], ends[:, 1:])  # the other seed stays a line


@pytest.mark.parametrize("raised", [True, False])
def test_the_over_seed_comes_down_into_the_approach_pose_from_above(raised):
    torch = pytest.importorskip("torch")
    from aimdk_grasp.planner import LeftArmPlanner

    planner = object.__new__(LeftArmPlanner)
    planner._approach_graph_seed, planner._over_m, planner.lift_up = "over", 0.12, (0.0, 0.0, 2.0)
    calls, solves = [], []

    def cartesian(origins, offsets, steps, check):
        calls.append((origins[0].copy(), offsets[0].copy()))
        # Joint chains stand in for palm paths: the chain moves every joint by the offset's norm.
        chain = np.linspace(origins[0], origins[0] + np.linalg.norm(offsets[0]), steps + 1)[None]
        return chain, np.array([raised or len(calls) == 1])

    planner._cartesian_paths = cartesian
    planner._palm_poses = lambda q: np.stack([np.eye(4)] * len(q))
    planner.motion = SimpleNamespace(
        trajopt_solver=SimpleNamespace(
            action_horizon=64, solve_pose=lambda goal, state, **options: solves.append(options) or "result"
        )
    )
    ends = torch.stack([torch.full((7,), 1.0), torch.full((7,), 2.0)])[None]
    state = SimpleNamespace(position=torch.zeros((1, 7)))
    trace = {}
    assert planner._escape_seeded_solve("goal", state, ends, 25, trace, lambda: None) == "result"
    escape, over = calls
    np.testing.assert_allclose(over[0], np.ones(7))  # from the approach IK seed ...
    np.testing.assert_allclose(over[1], [0.0, 0.0, 0.12])  # ... straight up along lift_up, _over_m
    seed = solves[0]["seed_traj"][0, 0].numpy()
    torch.testing.assert_close(solves[0]["seed_traj"][0, 0, -1], torch.full((7,), 1.0))  # ends at the approach
    assert trace["over_ok"] is raised
    # Raised, the path reaches the top of that chain (1.12) and comes back down to 1.0; otherwise the
    # joint line runs from the escape's end (0.1) straight into the approach and never passes it.
    assert bool(seed.max() > 1.1) is raised
    torch.testing.assert_close(solves[0]["seed_config"], ends[:, 1:])  # the other seed stays a line


def test_grasp_line_clear_queries_only_hand_spheres_shifted_back_along_the_palm_axis(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("checker tensors are CUDA tensors")
    from contextlib import nullcontext

    from aimdk_grasp.planner import LeftArmPlanner

    monkeypatch.setattr("aimdk_grasp.planner.palm_goals_from_camera_grasps", lambda g, matrix, mount: np.asarray(g))
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    planner = object.__new__(LeftArmPlanner)
    planner._grasp_line_offsets_m = (-0.10, -0.05)
    planner._closed, planner._checker_batch_size = False, None
    planner._checker_lock, planner._checker_stream = nullcontext(), None
    planner._ensure_validation_scene = lambda: None
    planner.hand_links = ["hand"]
    planner.grasp_to_palm = np.eye(4)
    batches, queried = [], []

    def kinematics(positions):  # sphere 0 on the arm, sphere 1 on the hand, both at x = joint 0
        spheres = torch.zeros((len(positions), 1, 2, 4), device="cuda")
        spheres[..., 0] = positions[:, :, :1]
        spheres[..., 3] = 0.01
        return SimpleNamespace(robot_spheres=spheres)

    def forward(state):  # an obstacle occupies z in [-0.06, -0.04]
        queried.append(state.robot_spheres.clone())
        spheres = state.robot_spheres
        hit = (spheres[..., 3] > 0) & (spheres[..., 2] < -0.04) & (spheres[..., 2] > -0.06)
        return hit.float()

    planner.checker = SimpleNamespace(
        setup_batch_tensors=lambda rows, horizon: batches.append(rows),
        get_kinematics=kinematics,
        kinematics=SimpleNamespace(
            config=SimpleNamespace(
                kinematics_config=SimpleNamespace(
                    link_sphere_idx_map=torch.tensor([0, 1], device="cuda"), link_name_to_idx_map={"arm": 0, "hand": 1}
                )
            )
        ),
        collision_constraint=SimpleNamespace(forward=forward),
    )
    goals = np.stack([np.eye(4)] * 3)
    goals[1, :3, :3] = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )  # palm axis along +x: the line stays at z = 0, off the obstacle
    planner._screen_contact_states = {4: np.full(7, 0.4, dtype=np.float32), 7: np.full(7, 0.7, dtype=np.float32)}
    clear = planner.grasp_line_clear([4, 7, 9], goals, np.eye(4))
    assert clear.tolist() == [False, True, True]  # 9 has no stored contact state: not judged
    assert batches == [4]
    spheres = queried[0].cpu().numpy()
    assert (spheres[:, 0, 0, 3] < 0).all() and (spheres[:, 0, 1, 3] > 0).all()  # only the hand is queried
    np.testing.assert_allclose(spheres[:2, 0, 1, :3], [[0.4, 0.0, -0.10], [0.4, 0.0, -0.05]], atol=1e-6)
    np.testing.assert_allclose(spheres[2:, 0, 1, :3], [[0.6, 0.0, 0.0], [0.65, 0.0, 0.0]], atol=1e-6)


def test_contact_target_clearance_is_the_hand_spheres_gap_to_the_target_points(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("checker tensors are CUDA tensors")
    from contextlib import nullcontext

    from aimdk_grasp.planner import LeftArmPlanner

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    planner = object.__new__(LeftArmPlanner)
    planner._closed, planner._checker_lock, planner._checker_stream = False, nullcontext(), None
    planner.hand_links = ["hand"]
    planner.grasp_to_palm = np.eye(4)

    def kinematics(positions):  # sphere 0 on the arm at the origin, sphere 1 on the hand at x = joint 0
        spheres = torch.zeros((len(positions), 1, 2, 4), device="cuda")
        spheres[:, 0, 1, 0] = positions[:, 0, 0]
        spheres[..., 3] = 0.01
        return SimpleNamespace(robot_spheres=spheres)

    planner.checker = SimpleNamespace(
        get_kinematics=kinematics,
        kinematics=SimpleNamespace(
            config=SimpleNamespace(
                kinematics_config=SimpleNamespace(
                    link_sphere_idx_map=torch.tensor([0, 1], device="cuda"), link_name_to_idx_map={"arm": 0, "hand": 1}
                )
            )
        ),
    )
    planner._screen_contact_states = {4: np.full(7, 0.4, dtype=np.float32), 7: np.full(7, 0.7, dtype=np.float32)}
    points = np.array([[0.425, 0.0, 0.0], [0.705, 0.0, 0.0], [0.0, 0.0, 0.0]])  # the last one only the arm touches
    clearance = planner.contact_target_clearance([4, 7, 9], points)
    # 25 mm from the centre of a 10 mm sphere; 5 mm inside the other; 9 has no contact state.
    np.testing.assert_allclose(clearance, [0.015, -0.005, np.inf], atol=1e-6)

    # Along the grasp line the hand passes 15 mm from a point 52 mm away at contact
    # (live clr_3: the pinky pushed the bottle halfway along the segment).
    import aimdk_grasp.planner as planner_module

    def goals(grasps, matrix, mount):  # palm axis = base x
        return np.stack([np.array([[0.0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])] * len(grasps))

    monkeypatch.setattr(planner_module, "palm_goals_from_camera_grasps", goals)
    beside = np.array([[0.35, 0.015, 0.0]])
    np.testing.assert_allclose(planner.contact_target_clearance([4], beside), [np.hypot(0.05, 0.015) - 0.01], atol=1e-6)
    np.testing.assert_allclose(
        planner.contact_target_clearance([4, 9], beside, np.stack([np.eye(4)] * 2), np.eye(4)),
        [0.005, np.inf],
        atol=1e-6,
    )
