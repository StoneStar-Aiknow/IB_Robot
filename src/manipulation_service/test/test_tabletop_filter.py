import sys
from types import ModuleType

import numpy as np

from manipulation_service import graspgen_wrapper
from manipulation_service.graspgen_wrapper import (
    _VALID_TABLETOP_FILTER_MODES,
    DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
    GraspGenWrapper,
    TablePlane,
)


def test_tabletop_filter_modes_include_diagnostic():
    assert "diagnostic" in _VALID_TABLETOP_FILTER_MODES


def test_source_gripper_tabletop_sweep_defaults_disabled():
    assert DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP is False


def test_tabletop_pregrasp_sweep_uses_source_mesh_and_pr200_axis():
    wrapper = object.__new__(GraspGenWrapper)
    wrapper._collision_vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    grasp = np.eye(4, dtype=np.float64)
    grasp[:3, 2] = [0.0, 0.0, 1.0]
    grasp[:3, 3] = [0.0, 0.0, 0.02]

    grasps, confidences, clearances = wrapper._filter_tabletop_clearance(
        np.array([grasp], dtype=np.float64),
        np.array([0.9], dtype=np.float64),
        table=TablePlane(normal=np.array([0.0, 0.0, 1.0], dtype=np.float64), d=0.0, inlier_ratio=1.0),
        clearance=0.001,
        pregrasp_distance=0.05,
        pregrasp_steps=1,
    )

    assert len(grasps) == 0
    assert confidences.tolist() == []
    assert clearances.tolist() == [-0.030000000000000002]


def test_tabletop_filter_uses_min_clearance_across_final_and_pregrasp_path():
    wrapper = object.__new__(GraspGenWrapper)
    wrapper._collision_vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    high_grasp = np.eye(4, dtype=np.float64)
    high_grasp[:3, 2] = [0.0, 0.0, -1.0]
    high_grasp[:3, 3] = [0.0, 0.0, 0.02]

    low_grasp = np.eye(4, dtype=np.float64)
    low_grasp[:3, 2] = [0.0, 0.0, -1.0]
    low_grasp[:3, 3] = [0.0, 0.0, 0.0005]

    grasps, confidences, clearances = wrapper._filter_tabletop_clearance(
        np.array([high_grasp, low_grasp], dtype=np.float64),
        np.array([0.9, 0.8], dtype=np.float64),
        table=TablePlane(normal=np.array([0.0, 0.0, 1.0], dtype=np.float64), d=0.0, inlier_ratio=1.0),
        clearance=0.001,
        pregrasp_distance=0.05,
        pregrasp_steps=1,
    )

    assert len(grasps) == 1
    assert confidences.tolist() == [0.9]
    assert clearances.tolist() == [0.02, 0.0005]


def test_disabled_source_gripper_sweep_preserves_table_and_object_top(monkeypatch):
    scene_xy = np.linspace(-0.1, 0.1, 20)
    scene_x, scene_y = np.meshgrid(scene_xy, scene_xy)
    scene_pc = np.column_stack([scene_x.ravel(), scene_y.ravel(), np.full(scene_x.size, 0.5)]).astype(np.float32)

    object_xy = np.linspace(-0.015, 0.015, 5)
    object_x, object_y = np.meshgrid(object_xy, object_xy)
    object_pc = np.column_stack([object_x.ravel(), object_y.ravel(), np.linspace(0.52, 0.54, object_x.size)]).astype(
        np.float32
    )

    point_cloud_utils = ModuleType("grasp_gen.utils.point_cloud_utils")
    point_cloud_utils.__dict__.update(
        {
            "depth_and_segmentation_to_point_clouds": lambda **_kwargs: (scene_pc, object_pc, None, None),
            "filter_colliding_grasps": lambda **_kwargs: None,
            "point_cloud_outlier_removal": lambda points: (points, None),
        }
    )
    grasp_server = ModuleType("grasp_gen.grasp_server")
    grasp_server.__dict__["GraspGenSampler"] = object
    monkeypatch.setitem(sys.modules, "grasp_gen", ModuleType("grasp_gen"))
    monkeypatch.setitem(sys.modules, "grasp_gen.grasp_server", grasp_server)
    monkeypatch.setitem(sys.modules, "grasp_gen.utils", ModuleType("grasp_gen.utils"))
    monkeypatch.setitem(sys.modules, "grasp_gen.utils.point_cloud_utils", point_cloud_utils)

    wrapper = object.__new__(GraspGenWrapper)
    wrapper._inference_point_count = 2048
    grasp = graspgen_wrapper.torch.eye(4, dtype=graspgen_wrapper.torch.float32).reshape(1, 4, 4)
    grasp[0, :3, 3] = graspgen_wrapper.torch.tensor([0.0, 0.0, 0.53])
    confidence = graspgen_wrapper.torch.tensor([0.9], dtype=graspgen_wrapper.torch.float32)
    wrapper._run_batched_inference = lambda *_args, **_kwargs: (grasp, confidence)

    def fail_if_swept(*_args, **_kwargs):
        raise AssertionError("source-gripper tabletop sweep should be skipped")

    wrapper._filter_tabletop_clearance = fail_if_swept

    candidates, diagnostic = wrapper.plan_grasps(
        depth_image=np.full((8, 8), 500, dtype=np.uint16),
        segmentation_mask=np.ones((8, 8), dtype=np.int32),
        camera_intrinsics=np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]),
        enable_collision_filter=False,
        enable_tabletop_filter=True,
        enable_source_gripper_tabletop_sweep=False,
        require_tabletop_filter=True,
        tabletop_filter_mode="diagnostic",
    )

    assert len(candidates) == 1
    assert diagnostic.source_gripper_tabletop_sweep_enabled is False
    assert diagnostic.tabletop_plane_found is True
    assert diagnostic.tabletop_object_top_xyz is not None
    assert np.isclose(diagnostic.tabletop_object_top_xyz[2], np.max(object_pc[:, 2]))
    assert diagnostic.tabletop_filter_before == 1
    assert diagnostic.tabletop_filter_after == 1
