from types import SimpleNamespace

import numpy as np

from manipulation_execution.pick_executor_models import PlannerSceneGeometry
from manipulation_execution.pick_executor_node import PickExecutorNode


def _point(x: float, y: float, z: float):
    return SimpleNamespace(x=x, y=y, z=z)


def test_execution_table_plane_takes_priority_over_planning_plane():
    response = SimpleNamespace(
        execution_table_plane_found=True,
        execution_table_plane_normal=_point(0.0, 0.0, 1.0),
        execution_table_plane_offset=-0.12,
        execution_table_plane_inlier_ratio=0.8,
        table_plane_found=True,
        table_plane_normal=_point(1.0, 0.0, 0.0),
        table_plane_offset=-0.5,
        table_plane_inlier_ratio=0.2,
    )

    normal, offset, inlier_ratio = PickExecutorNode._table_geometry_from_response(response)

    assert normal == (0.0, 0.0, 1.0)
    assert offset == -0.12
    assert inlier_ratio == 0.8


def test_planning_table_plane_is_used_as_fallback():
    response = SimpleNamespace(
        execution_table_plane_found=False,
        execution_table_plane_normal=_point(0.0, 0.0, 0.0),
        execution_table_plane_offset=0.0,
        execution_table_plane_inlier_ratio=0.0,
        table_plane_found=True,
        table_plane_normal=_point(0.0, 1.0, 0.0),
        table_plane_offset=-0.3,
        table_plane_inlier_ratio=0.6,
    )

    normal, offset, inlier_ratio = PickExecutorNode._table_geometry_from_response(response)

    assert normal == (0.0, 1.0, 0.0)
    assert offset == -0.3
    assert inlier_ratio == 0.6


def test_scene_table_plane_is_oriented_toward_base_up():
    base_to_camera = np.eye(4, dtype=np.float64)
    base_to_camera[:3, :3] = np.diag((1.0, -1.0, -1.0))
    scene = PlannerSceneGeometry(
        table_normal_camera=(0.0, 0.0, 1.0),
        table_offset_camera=-0.2,
        table_inlier_ratio=0.9,
    )

    geometry = PickExecutorNode._scene_geometry_base(base_to_camera, scene)

    assert geometry.table_plane is not None
    assert geometry.table_plane.normal == (0.0, 0.0, 1.0)
    assert geometry.table_plane.offset == 0.2
    assert geometry.table_plane.inlier_ratio == 0.9
