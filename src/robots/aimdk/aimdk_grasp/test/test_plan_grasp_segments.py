import sys
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.joints import LEFT_ARM
from aimdk_grasp.planner import GraspPlanningError, LeftArmPlanner
from aimdk_grasp.trajectory import TimedTrajectory


@pytest.fixture(autouse=True)
def trajopt_grasp(monkeypatch):
    """These tests cover the TrajOpt grasp segment, kept behind _linear_grasp."""
    monkeypatch.setattr(LeftArmPlanner, "_linear_grasp", False)
    monkeypatch.setattr(LeftArmPlanner, "vertical_lift_m", 0.0)


@pytest.fixture
def branch_planner(monkeypatch):
    torch = pytest.importorskip("torch")

    class Criteria:
        @staticmethod
        def linear_motion(**options):
            return "linear"

    monkeypatch.setitem(sys.modules, "curobo._src.cost.tool_pose_criteria", SimpleNamespace(ToolPoseCriteria=Criteria))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    clock = [0.0]
    monkeypatch.setattr("aimdk_grasp.planner.time.perf_counter", lambda: clock[0])
    planner = object.__new__(LeftArmPlanner)
    planner.planning_profile = "latency"
    planner.locked_positions = {}
    planner.hand_links = ["finger"]
    planner.grasp_to_palm = np.eye(4)
    planner._goal_pose = lambda transforms: transforms.copy()
    planner._state = lambda positions: SimpleNamespace(
        position=torch.tensor(np.array(positions)[None], dtype=torch.float32)
    )
    planner.valid_positions = lambda positions: np.ones(len(positions), dtype=bool)
    events = []
    link_state = [True]
    criteria = [None]
    contacts = np.stack([np.ones(7) * value for value in (1.0, 1.02, 2.0, 3.0)])

    def collision(enabled):
        link_state[0] = enabled
        events.append(("collision", enabled))

    def ik(goal, state, *, return_seeds=1, seed_config=None):
        events.append(("ik", float(goal[0, 2, 3])))
        if seed_config is None:
            assert not link_state[0]
            return SimpleNamespace(
                success=torch.ones((1, 4), dtype=torch.bool), solution=torch.tensor(contacts[None], dtype=torch.float32)
            )
        assert not link_state[0]
        assert seed_config.shape == (1, 4, 7)
        # Preserve the contact branch as we move away from the object.
        return SimpleNamespace(success=torch.tensor([[True]]), solution=state.position[:, None] + 0.01)

    def solve(goal_pose, state, *, seed_config=None, deadline=None):
        start = state.position.numpy().reshape(7)
        end = seed_config.numpy().reshape(7) if seed_config is not None else start + 0.1
        events.append(("motion", link_state[0], criteria[0], start.copy(), end.copy()))
        trajectory = SimpleNamespace(joint_names=list(LEFT_ARM), position=torch.tensor(np.stack([start, end])[None]))
        return SimpleNamespace(
            success=torch.tensor([[True]]),
            interpolated_trajectory=trajectory,
            interpolated_last_tstep=torch.tensor([1]),
        )

    planner._solve_motion = solve
    planner.motion = SimpleNamespace(
        ik_solver=SimpleNamespace(solve_pose=ik, config=SimpleNamespace(num_seeds=4)),
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=0.02)),
        enable_link_collision=lambda _: collision(True),
        disable_link_collision=lambda _: collision(False),
        update_tool_pose_criteria=lambda value: criteria.__setitem__(0, value["R_palm"]),
    )
    return SimpleNamespace(
        planner=planner, clock=clock, events=events, links=link_state, criteria=criteria, torch=torch, contacts=contacts
    )


@pytest.mark.parametrize("published,policy", [(False, "clear_path"), (True, "clear_path"), (False, "native")])
@pytest.mark.parametrize("failed_phase", ["approach", "grasp"])
def test_failed_segment_is_reported_and_a_published_approach_is_never_replaced(
    branch_planner, monkeypatch, published, policy, failed_phase
):
    planner = branch_planner.planner
    planner.planning_profile = "latency"
    planner.approach_seed_policy = policy
    planner.max_goalset = 1
    planner._grasp_selections = {}
    planner.snapshot_matches = lambda *args: True
    planner._require_valid_start = lambda q: None
    planner._approach_seed_index = lambda *args: 0
    monkeypatch.setattr("aimdk_grasp.planner.palm_goals_from_camera_grasps", lambda g, m, p: np.asarray(g))
    planner.motion.ik_solver.solve_pose = lambda *args, **kwargs: SimpleNamespace(
        success=branch_planner.torch.tensor([[True]]),
        goalset_index=branch_planner.torch.tensor([0]),
        solution=branch_planner.torch.ones((1, 1, 7)),
    )
    segment = TimedTrajectory(np.array([0.0, 0.02]), np.zeros((2, 7)))
    publications = []

    def solve(goal, state, **options):
        if (goal[0, 2, 3] == 0.0) == (failed_phase == "grasp"):
            raise RuntimeError("cannot continue")
        return SimpleNamespace(success=np.array([True]))

    planner._solve_motion = solve
    planner._grasp_trajectory = lambda *args, **kwargs: segment
    options = {"on_approach": lambda *args: publications.append(args)} if published else {}
    feedback = SimpleNamespace(vector=lambda _: np.zeros(7))
    with pytest.raises(GraspPlanningError, match="cannot continue") as error:
        planner.plan_grasp(feedback, np.eye(4), np.eye(4), 0.0, **options)
    assert error.value.goalset_index == 0
    assert len(publications) == int(published and failed_phase == "grasp")
    assert branch_planner.links[0]


@pytest.mark.parametrize("fail_load", [False, True])
def test_scene_replacement_invalidates_request_caches_before_loading(branch_planner, monkeypatch, fail_load):
    from aimdk_grasp.scene_estimation import SceneEstimate, SupportPlane

    planner = branch_planner.planner
    monkeypatch.setitem(
        sys.modules, "curobo._src.geom.types", SimpleNamespace(SceneCfg=SimpleNamespace(create=lambda scene: scene))
    )
    monkeypatch.setattr("aimdk_grasp.planner._ensure_mesh_query_distance", lambda *args: None)
    planner._closed = False
    planner.locked_joints_match = lambda _: True
    planner.scene_received_at = 0.0
    planner._grasp_selections = {"old": "seed"}
    planner._ik_validity_cache = {np.zeros(7, dtype=np.float32).tobytes(): True}
    planner.mesh_query_distance = 0.1
    planner._checker_stream = SimpleNamespace(wait_stream=lambda _: None)
    monkeypatch.setattr(branch_planner.torch.cuda, "current_stream", lambda: None)
    closed = []
    planner.close = lambda: closed.append(True)

    def load(scene):
        assert not planner._grasp_selections
        assert not planner._ik_validity_cache
        if fail_load:
            raise RuntimeError("scene load failed")

    model = SimpleNamespace(
        data=SimpleNamespace(meshes=None, cuboids=SimpleNamespace(max_n=1)), load_collision_model=load
    )
    planner.motion.scene_collision_checker = model
    planner.motion.graph_planner = None
    planner.checker = SimpleNamespace(scene_model=model)
    scene = {"cuboid": {"table": {}}}
    estimate = SceneEstimate(
        scene,
        0.01,
        1,
        "camera",
        False,
        SupportPlane(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.empty((0, 3))),
        {},
        scene,
    )
    if fail_load:
        with pytest.raises(RuntimeError, match="scene load failed"):
            planner.update_scene(None, estimate)
        assert closed and planner.scene_received_at == -np.inf
    else:
        planner.update_scene(None, estimate)
        assert not closed and planner.scene_received_at == 0.01
    assert not planner._grasp_selections


@pytest.mark.parametrize("outcome", ["vertical", "hold_line", "disabled"])
def test_vertical_lift(monkeypatch, outcome):
    """The lift raises the grasp-end palm along lift_up, or holds (vertical_lift_m)."""
    planner = LeftArmPlanner.__new__(LeftArmPlanner)
    planner.vertical_lift_m = 0.0 if outcome == "disabled" else 0.05
    planner.lift_up = (0.0, 0.0, 2.0)
    grasp = TimedTrajectory(np.array([0.0, 0.1]), np.zeros((2, 7)) + [[0.0], [0.2]])
    planner._palm_poses = lambda q: np.stack([np.eye(4)] * len(q))
    carried = []
    planner._carried_target = lambda: __import__("contextlib").nullcontext(carried.append(True))
    goals = []

    def segment(start, goal, check_deadline):
        goals.append((start, goal))
        if outcome == "hold_line":
            return None, "line"
        return TimedTrajectory(np.array([0.0, 0.5, 1.0]), np.stack([start + 1e-9, start + 0.1, start + 0.2])), None

    planner._linear_grasp_segment = segment
    lift, method = planner._vertical_lift(grasp, lambda: None)
    if outcome == "disabled":
        assert not goals and not carried
    else:
        assert np.array_equal(goals[0][0], grasp.positions[-1]) and carried == [True]
        assert np.allclose(goals[0][1][:3, 3], [0.0, 0.0, 0.05]) and np.allclose(goals[0][1][:3, :3], np.eye(3))
    if outcome == "vertical":
        assert method == "vertical" and np.array_equal(lift.positions[0], grasp.positions[-1])
        assert planner.vertical_lift_report == {"used": 1}
    else:
        assert method == "hold"
        assert np.array_equal(lift.positions, np.repeat(grasp.positions[-1:], 2, axis=0))
        assert planner.vertical_lift_report == {f"hold_{'line' if outcome == 'hold_line' else 'disabled'}": 1}


def test_carried_target_leaves_the_target_out_of_the_fine_check_only_inside(monkeypatch):
    torch = pytest.importorskip("torch")
    from threading import RLock

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    enabled = {"target": True, "environment": True}
    meshes = SimpleNamespace(
        get_names=lambda: list(enabled), set_enabled=lambda name, value: enabled.__setitem__(name, value)
    )
    planner = LeftArmPlanner.__new__(LeftArmPlanner)
    planner._checker_lock = RLock()
    planner._ensure_validation_scene = lambda: None
    planner.checker = SimpleNamespace(scene_model=SimpleNamespace(data=SimpleNamespace(meshes=meshes)))
    with planner._carried_target():
        assert enabled == {"target": False, "environment": True}
    assert enabled == {"target": True, "environment": True}
    with pytest.raises(RuntimeError), planner._carried_target():
        raise RuntimeError("segment failed")
    assert enabled["target"]
