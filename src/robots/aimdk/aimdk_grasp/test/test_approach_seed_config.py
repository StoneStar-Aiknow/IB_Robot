"""Seed height is a validated robot setting and reaches every planner rebuild."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from aimdk_grasp.backend import _settings
from aimdk_grasp.grasp_request import GraspRequestConfig
from aimdk_grasp.planner import LeftArmPlanner
from aimdk_grasp.planner_service import GraspPlanningService
from test_planner_contracts import GRIPPER


@pytest.mark.parametrize("height", [0.12, 0.18])
def test_robot_height_setting_reaches_initial_and_rebuilt_planner(height):
    settings = _settings(GraspRequestConfig, {"approach_over_height_m": height}, ("revalidation",), "planning")
    received = []

    class Planner:
        def __init__(self, feedback, estimate, received_at, **options):
            received.append(options["approach_over_height_m"])
            self.locked = feedback

        def warmup(self):
            pass

        def locked_joints_match(self, feedback):
            return feedback == self.locked

        def close(self):
            pass

    with GraspPlanningService(
        gripper=GRIPPER, approach_over_height_m=settings.approach_over_height_m, planner_factory=Planner
    ) as service:
        estimate = SimpleNamespace(received_at=0.0)
        service.prepare("first", estimate)
        service.prepare("changed", estimate)
    assert received == [height, height]


@pytest.mark.parametrize("height", [0, -0.12, float("nan"), float("inf"), True, "0.12", None])
def test_bad_seed_height_fails_before_gpu_work(height):
    with pytest.raises(ValueError, match="approach_over_height_m"):
        GraspRequestConfig(approach_over_height_m=height)
    with pytest.raises(ValueError, match="approach_over_height_m"):
        GraspPlanningService(gripper=GRIPPER, approach_over_height_m=height)


def test_both_x2_profiles_select_eighteen_cm_and_match_request_defaults():
    workspace = Path(__file__).resolve().parents[5]
    for name in ("aimdk_x2_grasp", "aimdk_x2_voice_mirror_grasp"):
        path = workspace / "src/robot_config/config/robots" / f"{name}.yaml"
        config = yaml.safe_load(path.read_text())
        options = config["robot"]["grasp_execution"]["backend_options"]["planning"]
        assert options["approach_over_height_m"] == GraspRequestConfig().approach_over_height_m == 0.18
        assert options["approach_order_m"] == [0.10, 0.15, 0.07, 0.05]


def test_failed_escape_records_native_fallback_without_constructing_an_over_seed():
    torch = pytest.importorskip("torch")
    planner = LeftArmPlanner.__new__(LeftArmPlanner)
    planner._escape_chain = lambda *args: None
    planner._cartesian_paths = lambda *args: pytest.fail("Over IK ran after failed escape")
    trace = {}
    result = planner._escape_seeded_solve(
        None,
        SimpleNamespace(position=torch.zeros((1, 7))),
        torch.ones((1, 2, 7)),
        25,
        trace,
        lambda: None,
    )
    assert result is None
    assert not trace["escape_ok"]
    assert trace["actual_seed"] == "native"
    assert trace["actual_over_height_m"] is None
    assert trace["fallback_reason"] == "escape_cartesian_ik_failed"


def test_attempt_report_keeps_approach_seed_instead_of_the_later_grasp_seed():
    from aimdk_grasp.grasp_session import select_plan

    trace = dict(requested_seed="over", actual_seed="escape", requested_over_height_m=0.12)
    planner = SimpleNamespace(max_goalset=1)

    def plan(*args, **kwargs):
        planner.approach_seed_traces = [trace]
        planner.graph_seed_trace = dict(actual_seed="native")  # overwritten by the subsequent grasp solve
        return SimpleNamespace(goalset_index=0)

    planner.plan_grasp = plan
    report = {}
    select_plan(
        planner,
        None,
        SimpleNamespace(grasps=np.eye(4)[None], confidence=np.ones(1)),
        np.eye(4),
        None,
        report,
        strategies=[(-0.10, -0.06)],
    )
    assert report["winning_candidate"]["approach_seeds"] == [trace]
    assert report["planning_attempts"][0]["approach_seeds"] == [trace]
