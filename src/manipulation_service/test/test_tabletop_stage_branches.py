"""Committed tabletop-stage branch regression tests (issue #176, D6.5).

Runs ``GraspGenWrapper.plan_grasps`` against a stub backend (no CUDA, no
model) on deterministic synthetic scenes so the D2 extraction of the
tabletop stage into ``table_plane.run_tabletop_filter_stage`` stays guarded:
strict filtering, adaptive retry success and failure, the soft fallback, and
diagnostic mode.
"""

import numpy as np

from manipulation_service.graspgen_wrapper import GraspGenWrapper


def _build_scene(h=48, w=64, object_height=0.03):
    """A deterministic table scene: table at 0.5 m with a noisy plane."""
    rng = np.random.default_rng(42)
    fx = fy = 400.0
    cx, cy = w / 2.0, h / 2.0
    depth = (0.5 + rng.normal(0.0, 0.0005, (h, w))).astype(np.float32)
    depth[18:30, 24:40] = 0.5 + object_height
    mask = np.zeros((h, w), dtype=np.int32)
    mask[18:30, 24:40] = 1
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return depth, mask, intrinsics


def _make_wrapper():
    wrapper = object.__new__(GraspGenWrapper)
    wrapper.inference_backend = "ascend_local"
    wrapper._inference_point_count = 2048
    wrapper._gripper_name = "robotiq_test"
    wrapper._collision_gripper_name = "robotiq_test"
    wrapper._collision_vertices = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.01], [0.0, 0.01, 0.01], [-0.01, -0.005, 0.0]],
        dtype=np.float64,
    )
    wrapper._collision_mesh = "stub-mesh"
    wrapper._gripper_info = "stub-info"
    return wrapper


def _stub_inference(wrapper, poses, confidences, monkeypatch):
    import torch

    def _run(self, object_pc, **_kwargs):
        return torch.from_numpy(poses.copy()), torch.from_numpy(confidences.copy())

    monkeypatch.setattr(GraspGenWrapper, "_run_batched_inference", _run)


_POSES = None


def _poses(count=6):
    poses = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    for i in range(count):
        poses[i, 2, 3] = 0.5005 + 0.0011 * i
    return poses


_CONFIDENCES = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65], dtype=np.float32)


def _plan(wrapper, monkeypatch, *, depth_image=None, **kwargs):
    _stub_inference(wrapper, _poses(), _CONFIDENCES, monkeypatch)
    base = dict(
        depth_image=depth_image if depth_image is not None else _build_scene()[0],
        segmentation_mask=_build_scene()[1],
        camera_intrinsics=_build_scene()[2],
        depth_scale=1.0,
        grasp_threshold=0.5,
        num_grasps=500,
        topk_num_grasps=300,
        enable_collision_filter=False,
        enable_tabletop_filter=True,
        require_tabletop_filter=True,
    )
    base.update(kwargs)
    return wrapper.plan_grasps(**base)


def test_strict_mode_keeps_clearances_above_threshold(monkeypatch):
    depth, mask, K = _build_scene(object_height=0.03)
    wrapper = _make_wrapper()

    def _sweep(grasps, confidences, *, table, clearance, pregrasp_distance, pregrasp_steps):
        clearances = np.array([0.005, 0.004, 0.0035, 0.0045, 0.0038, 0.0032])
        return grasps, confidences, clearances

    wrapper._filter_tabletop_clearance = _sweep
    candidates, diag = _plan(
        wrapper,
        monkeypatch,
        depth_image=depth,
        tabletop_filter_mode="strict",
        enable_source_gripper_tabletop_sweep=True,
    )
    assert diag.tabletop_filter_mode == "strict"
    assert diag.tabletop_plane_found is True
    assert diag.tabletop_filter_before == 6
    assert diag.tabletop_filter_after == 6
    assert len(candidates) == 6
    assert diag.failure_stage == ""


def test_strict_mode_zero_result_reports_tabletop_failure(monkeypatch):
    depth, mask, K = _build_scene(object_height=0.03)
    wrapper = _make_wrapper()

    def _sweep(*_args, **_kwargs):
        grasps = np.zeros((0, 4, 4), dtype=np.float64)
        return (
            grasps,
            np.zeros((0,), dtype=np.float64),
            np.array([0.005, 0.004, 0.003, 0.002, 0.001, 0.0]),
        )

    wrapper._filter_tabletop_clearance = _sweep
    candidates, diag = _plan(
        wrapper,
        monkeypatch,
        depth_image=depth,
        tabletop_filter_mode="strict",
        enable_source_gripper_tabletop_sweep=True,
    )
    assert candidates == []
    assert diag.failure_stage == "tabletop_filter"
    assert diag.tabletop_filter_after == 0


def test_adaptive_retry_success_accepts_lower_clearance(monkeypatch):
    depth, mask, K = _build_scene(object_height=0.03)
    wrapper = _make_wrapper()

    def _sweep(grasps, confidences, *, table, clearance, pregrasp_distance, pregrasp_steps):
        clearances = np.array([0.005, 0.001, 0.004, 0.0025, 0.0015, 0.003])
        if clearance < 0.002:
            keep = np.array([i in (0, 1) for i in range(len(grasps))])
        else:
            keep = np.zeros(len(grasps), dtype=bool)
        return grasps[keep], confidences[keep], clearances

    wrapper._filter_tabletop_clearance = _sweep
    candidates, diag = _plan(
        wrapper,
        monkeypatch,
        depth_image=depth,
        tabletop_filter_mode="adaptive",
        enable_source_gripper_tabletop_sweep=True,
        adaptive_tabletop_retry_clearances=(0.0005,),
        adaptive_tabletop_clearance_min=0.0004,
    )
    assert diag.tabletop_auto_tuned is True
    assert diag.tabletop_auto_tune_attempts >= 1
    assert len(candidates) == 2
    assert diag.tabletop_low_profile is True


def test_adaptive_retry_failure_reports_tabletop_failure(monkeypatch):
    depth, mask, K = _build_scene(object_height=0.03)
    wrapper = _make_wrapper()

    def _sweep(grasps, confidences, *, table, clearance, pregrasp_distance, pregrasp_steps):
        clearances = np.array([0.005, 0.001, 0.004, 0.0025, 0.0015, 0.003])
        keep = np.zeros(len(grasps), dtype=bool)
        return grasps[keep], confidences[keep], clearances

    wrapper._filter_tabletop_clearance = _sweep
    candidates, diag = _plan(
        wrapper,
        monkeypatch,
        depth_image=depth,
        tabletop_filter_mode="adaptive",
        enable_source_gripper_tabletop_sweep=True,
        adaptive_tabletop_retry_clearances=(0.0005,),
        adaptive_tabletop_clearance_min=0.0004,
    )
    assert candidates == []
    assert diag.tabletop_auto_tuned is False
    assert diag.tabletop_auto_tune_attempts >= 1
    assert diag.failure_stage == "tabletop_filter"


def test_soft_mode_falls_back_above_hard_floor(monkeypatch):
    depth, mask, K = _build_scene(object_height=0.03)
    wrapper = _make_wrapper()

    def _sweep(grasps, confidences, *, table, clearance, pregrasp_distance, pregrasp_steps):
        clearances = np.array([0.005, 0.001, 0.004, 0.0025, 0.0015, 0.003])
        keep = np.zeros(len(grasps), dtype=bool)
        return grasps[keep], confidences[keep], clearances

    wrapper._filter_tabletop_clearance = _sweep
    candidates, diag = _plan(
        wrapper,
        monkeypatch,
        depth_image=depth,
        tabletop_filter_mode="soft",
        enable_source_gripper_tabletop_sweep=True,
    )
    assert diag.tabletop_relaxed is True
    # All clearances sit above the hard floor (-0.002): every grasp is
    # re-admitted ordered by (clearance desc, confidence desc).
    assert len(candidates) == 6
    assert diag.tabletop_filter_after == 6


def test_diagnostic_mode_keeps_all_grasps_with_stats(monkeypatch):
    depth, mask, K = _build_scene(object_height=0.03)
    wrapper = _make_wrapper()

    def _sweep(grasps, confidences, *, table, clearance, pregrasp_distance, pregrasp_steps):
        clearances = np.array([0.005, 0.001, 0.004, 0.0025, 0.0015, 0.003])
        keep = np.array([i in (0, 1) for i in range(len(grasps))])
        return grasps[keep], confidences[keep], clearances

    wrapper._filter_tabletop_clearance = _sweep
    candidates, diag = _plan(
        wrapper,
        monkeypatch,
        depth_image=depth,
        tabletop_filter_mode="diagnostic",
        enable_source_gripper_tabletop_sweep=True,
    )
    assert diag.tabletop_filter_after == 6
    assert diag.tabletop_filter_before == 6
    assert diag.tabletop_relaxed is True
    assert diag.tabletop_plane_normal is not None
    assert diag.tabletop_object_top_xyz is not None
