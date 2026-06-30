import numpy as np

from manipulation_service.graspgen_wrapper import GraspGenWrapper, TablePlane


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
