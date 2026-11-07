"""Recheck an archived fine rejection, including attachment, on a CUDA host."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("path recheck needs CUDA", allow_module_level=True)

from aimdk_grasp.joints import LEFT_ARM  # noqa: E402
from aimdk_grasp.path_diagnostics import recheck, summarize  # noqa: E402
from aimdk_grasp.trajectory import TimedTrajectory  # noqa: E402
from test_planner_cuda import planner as planner  # noqa: E402


@pytest.mark.parametrize("attached", [False, True])
def test_saved_candidate_reproduces_fine_results_with_saved_assets(planner, tmp_path, monkeypatch, attached):  # noqa: F811
    value, feedback, _ = planner
    monkeypatch.setenv("IBROBOT_X2_PATH_FAILURES_DIR", str(tmp_path))
    if attached:
        value.attach_object(np.array([[0.0, 0.0, 0.02, 0.05]]))
    try:
        start = feedback.vector(LEFT_ARM)
        across = np.array([0.25, 0.06, -0.69, -0.95, 0.63, -0.07, 0.0])
        lower, upper = value._arm_bounds
        candidates = np.random.default_rng(7).uniform(lower, upper, size=(64, 7))
        costs = value.constraint_diagnostics(candidates)["self_collision_cost"].reshape(64, -1).max(axis=1)
        assert costs.max() > 0
        self_collision = candidates[np.argmax(costs)]
        bound, behind = start.copy(), start.copy()
        bound[5], behind[0] = upper[5] + 0.01, 1.2
        path = TimedTrajectory(np.arange(5) * 0.1, np.stack([start, across, self_collision, bound, behind]))
        diagnostics = value.constraint_diagnostics(path.positions)
        assert not diagnostics["valid"].all()
        assert diagnostics["bound_cost"].max() > 0 and diagnostics["hand_behind_torso"].any()
        saved = value._save_pose_failure(feedback, start, np.eye(4), path, diagnostics, SimpleNamespace(), None)
        assert saved.startswith("archive="), saved
        archive = pickle.loads(next(tmp_path.glob("pose_failure_*.pkl")).read_bytes())
        # Recheck uses archived URDF bytes rather than the original model path.
        original_urdf = archive["models"]["fine"]["robot_cfg"]["kinematics"]["urdf_path"]
        archive["models"]["fine"]["robot_cfg"]["kinematics"]["urdf_path"] = "/missing/original.urdf"
        archive["assets"]["/missing/original.urdf"] = archive["assets"].pop(original_urdf)
        actual = recheck(archive)
        for key in ("valid", "hand_behind_torso", "bound_cost", "self_collision_cost", "world_penetration_m"):
            np.testing.assert_allclose(actual[key], diagnostics[key], atol=1e-6, rtol=1e-5, err_msg=key)
    finally:
        if attached:
            value.detach_object()


def test_self_collision_diagnostics_name_the_pairs_enabled_by_the_real_checker(planner):  # noqa: F811
    value, _, _ = planner
    lower, upper = value._arm_bounds
    positions = np.random.default_rng(7).uniform(lower, upper, size=(64, 7))
    diagnostics = value.constraint_diagnostics(positions)
    assert diagnostics["self_collision_cost"].max() > 0
    summary = summarize(positions, None, diagnostics, LEFT_ARM, value.collision_sphere_links, value._arm_bounds)
    row = summary["types"]["self_collision"]
    a, b = row["pair_spheres"]
    assert [a, b] in diagnostics["self_collision_pairs"].tolist()
    assert row["pair_links"] == [value.collision_sphere_links[a], value.collision_sphere_links[b]]
    assert row["overlap_m"] > 0 and row["squared_penetration_m2"] > 0
