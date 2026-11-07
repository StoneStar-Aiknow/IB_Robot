"""A rejected interpolated path remains inspectable without the live scene or GPU."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.joints import LEFT_ARM
from aimdk_grasp.path_diagnostics import digest, summarize
from aimdk_grasp.planner import LeftArmPlanner
from test_planner_contracts import result


def evidence():
    q = np.zeros((4, 7))
    q[2, 5], q[3, 5] = 0.55, 0.60
    spheres = np.zeros((4, 2, 4))
    spheres[:, :, 3] = 0.1
    spheres[:, 1, 0] = [1.0, 1.0, 0.1, 0.05]
    return q, {
        "valid": np.array([True, False, False, False]),
        "bound_cost": np.maximum(q - 0.5236, 0),
        "self_collision_cost": np.array([[0], [0], [0.01], [0.02]]),
        "world_penetration_m": np.array([[0, 0], [0.001, 0], [0.002, 0], [0.003, 0]]),
        "hand_behind_torso": np.array([False, False, True, False]),
        "hand_min_x_m": np.array([0.1, 0.0, -0.02, 0.1]),
        "required_hand_min_x_m": -0.01,
        "spheres": spheres,
        "self_collision_pairs": np.array([[0, 1]]),
        "self_collision_padding": np.array([0.01, 0.01]),
    }


def test_classification_reports_first_and_worst_per_constraint_with_real_bounds_and_pairs():
    q, diagnostics = evidence()
    report = summarize(
        q, np.arange(4) * 0.02, diagnostics, LEFT_ARM, ["hand", "torso"], (np.full(7, -1), np.full(7, 0.5236))
    )
    assert report["first_invalid_sample"] == 1
    types = report["types"]
    assert set(types) == {"joint_bounds", "world_collision", "self_collision", "hand_behind_torso"}
    bound = types["joint_bounds"]
    assert (bound["first_sample"], bound["worst_sample"], bound["joint"]) == (2, 3, LEFT_ARM[5])
    assert bound["actual_rad"] == 0.6 and bound["upper_rad"] == 0.5236
    assert bound["excess_rad"] == pytest.approx(0.0764)
    assert types["world_collision"]["first_time_s"] == 0.02
    assert types["world_collision"]["worst_sample"] == 3
    assert types["self_collision"]["pair_links"] == ["hand", "torso"]
    assert types["self_collision"]["overlap_m"] == pytest.approx(0.17)
    assert types["hand_behind_torso"]["worst_sample"] == 2


def test_pose_rejection_archives_exact_path_goal_attachment_models_and_scene(tmp_path, monkeypatch):
    q, diagnostics = evidence()
    value = object.__new__(LeftArmPlanner)
    value.arm = LEFT_ARM
    value.snapshot_matches = lambda *a: True
    value._measured_start = lambda feedback: q[0]
    value._require_valid_start = lambda start: None
    value._goal_pose = value._state = lambda arg: arg
    solved = result(LEFT_ARM, q)
    value._solve_motion = lambda *a, **kw: solved
    value._retry_approach = lambda *a, **kw: solved
    value.constraint_diagnostics = lambda positions: diagnostics
    value.motion = SimpleNamespace(trajopt_solver=SimpleNamespace(config=SimpleNamespace(interpolation_dt=0.02)))
    value._arm_bounds = np.full(7, -1), np.full(7, 0.5236)
    value.locked_positions = {"locked": 0.2}
    value.collision_sphere_links = ["hand", "torso"]
    model = tmp_path / "model.urdf"
    model.write_text("<robot name='recorded'/>")
    value._diagnostic_models = {"fine": {"urdf_path": str(model)}, "planning": {"default_joint_position": [0.1] * 7}}
    value._diagnostic_scenes = ({"cuboid": {"table": {"dims": [1, 1, 0.1]}}}, {"mesh": {"bin": {"vertices": q}}})
    value.attached = np.array([[0, 0, 0, 0.05]])
    value.scene_received_at = 7.0
    value.planning_profile, value.approach_seed_policy = "latency", "clear_path"
    value._hand_min_x_m, value.mesh_query_distance = -0.01, 0.2
    feedback = SimpleNamespace(vector=lambda names: q[0] + 0.001, received_at=7.01)
    monkeypatch.setenv("IBROBOT_X2_PATH_FAILURES_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="world.*archive="):
        value.plan_pose(feedback, np.eye(4), 7.02)
    (archive,) = tmp_path.glob("pose_failure_*.pkl")
    saved = pickle.loads(archive.read_bytes())
    np.testing.assert_array_equal(saved["positions"], q)
    np.testing.assert_allclose(saved["times"], np.arange(4) * 0.02)
    np.testing.assert_array_equal(saved["target_palm"], np.eye(4))
    np.testing.assert_array_equal(saved["attached_spheres"], value.attached)
    assert saved["models"] == value._diagnostic_models
    assert saved["assets"][str(model)] == model.read_bytes()
    assert saved["identities"]["planning_scene"] == digest(saved["planning_scene"])
    assert saved["summary"]["first_invalid_sample"] == 1
    np.testing.assert_allclose(saved["measured_start"], q[0] + 0.001)
    np.testing.assert_array_equal(saved["planning_start"], q[0])
    # A recording failure still rejects the path, with the original violation.
    monkeypatch.setattr(
        "aimdk_grasp.path_diagnostics.save_failure", lambda *a: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(RuntimeError, match="world.*failure archive unavailable: OSError: disk full"):
        value.plan_pose(feedback, np.eye(4), 7.02)
