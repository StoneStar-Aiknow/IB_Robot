import numpy as np
import pytest

from manipulation_service.grasp_planner_node import (
    _fit_execution_table_plane,
    _sample_execution_table_points,
    _stamp_from_ns,
    _stamp_to_ns,
)
from manipulation_service.graspgen_wrapper import GraspDiagnostic


def test_stamp_nanoseconds_round_trip():
    stamp_ns = 1_784_201_190_669_752_215

    assert _stamp_to_ns(_stamp_from_ns(stamp_ns)) == stamp_ns


def test_execution_table_sampling_matches_debug_ply_reader_stride():
    points = np.arange(31 * 3, dtype=np.float32).reshape(31, 3)

    sampled = _sample_execution_table_points(points, max_points=10)

    np.testing.assert_array_equal(sampled, points[::3])


def test_execution_table_plane_uses_completed_scene_and_points_toward_object():
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.2, 0.2, 20))
    scene = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]).astype(np.float32)
    object_points = np.array(
        [
            [-0.01, -0.01, 0.05],
            [0.01, -0.01, 0.05],
            [-0.01, 0.01, 0.05],
            [0.01, 0.01, 0.05],
        ],
        dtype=np.float32,
    )
    diagnostic = GraspDiagnostic(
        scene_pc_after_completion=scene,
        object_pc_after_completion=object_points,
    )

    fit = _fit_execution_table_plane(diagnostic)

    assert fit.plane is not None
    assert fit.plane.normal[2] == pytest.approx(1.0, abs=1e-12)
    assert fit.plane.d == pytest.approx(0.0, abs=1e-12)
    assert fit.plane.inlier_ratio == pytest.approx(1.0)
