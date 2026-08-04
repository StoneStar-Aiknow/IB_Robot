from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation_service import graspgen_wrapper
from manipulation_service.graspgen_wrapper import (
    AscendLocalBackend,
    GraspGenWrapper,
    Remote310PInferenceClient,
    _depth_and_segmentation_to_point_clouds,
    _point_cloud_outlier_removal,
)


def test_remote_result_validation_accepts_matching_finite_arrays():
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    confidence = np.array([0.9, 0.8], dtype=np.float32)

    Remote310PInferenceClient._validate_result(poses, confidence)


@pytest.mark.parametrize(
    ("poses", "confidence"),
    [
        (np.zeros((2, 3, 4), dtype=np.float32), np.zeros(2, dtype=np.float32)),
        (np.zeros((2, 4, 4), dtype=np.float32), np.zeros(1, dtype=np.float32)),
        (np.full((1, 4, 4), np.nan, dtype=np.float32), np.zeros(1, dtype=np.float32)),
    ],
)
def test_remote_result_validation_rejects_invalid_arrays(poses, confidence):
    with pytest.raises(RuntimeError):
        Remote310PInferenceClient._validate_result(poses, confidence)


def test_remote_backend_delegates_one_request_without_local_sampler():
    class FakeClient:
        def sample(self, object_pc, **kwargs):
            self.object_pc = object_pc
            self.kwargs = kwargs
            return (
                np.eye(4, dtype=np.float32)[None, ...],
                np.array([0.75], dtype=np.float32),
                0.85,
            )

    wrapper = object.__new__(GraspGenWrapper)
    wrapper.inference_backend = "remote_310p"
    wrapper._remote_client = FakeClient()
    points = np.zeros((2048, 3), dtype=np.float32)

    poses, confidence = wrapper._run_batched_inference(
        object,
        points,
        grasp_threshold=0.2,
        num_grasps=5000,
        topk_num_grasps=1000,
        min_grasps=80,
        max_tries=4,
    )

    assert isinstance(poses, torch.Tensor)
    assert isinstance(confidence, torch.Tensor)
    assert poses.shape == (1, 4, 4)
    assert confidence.tolist() == [0.75]
    assert wrapper._remote_client.object_pc is points
    assert wrapper._remote_client.kwargs == {
        "grasp_threshold": 0.2,
        "num_grasps": 5000,
        "topk_num_grasps": 1000,
    }


def test_ascend_local_backend_delegates_one_request_without_local_sampler():
    class FakeClient:
        def sample(self, object_pc, **kwargs):
            self.object_pc = object_pc
            self.kwargs = kwargs
            return (
                np.eye(4, dtype=np.float32)[None, ...],
                np.array([0.75], dtype=np.float32),
                0.92,
            )

    wrapper = object.__new__(GraspGenWrapper)
    wrapper.inference_backend = "ascend_local"
    wrapper._ascend_local_client = FakeClient()
    points = np.zeros((2048, 3), dtype=np.float32)

    poses, confidence = wrapper._run_batched_inference(
        object,
        points,
        grasp_threshold=0.2,
        num_grasps=5000,
        topk_num_grasps=1000,
        min_grasps=80,
        max_tries=4,
    )

    assert isinstance(poses, torch.Tensor)
    assert isinstance(confidence, torch.Tensor)
    assert poses.shape == (1, 4, 4)
    assert confidence.tolist() == [0.75]
    assert wrapper._ascend_local_client.object_pc is points
    assert wrapper._ascend_local_client.kwargs == {
        "grasp_threshold": 0.2,
        "num_grasps": 5000,
        "topk_num_grasps": 1000,
    }


def test_ascend_local_backend_rejects_empty_manifest_path():
    with pytest.raises(ValueError, match="manifest_path"):
        AscendLocalBackend("")


def test_ascend_local_wrapper_warmup_uses_deterministic_synthetic_cloud():
    class FakeClient:
        def warmup(self, object_pc):
            self.object_pc = object_pc

    wrapper = object.__new__(GraspGenWrapper)
    wrapper.inference_backend = "ascend_local"
    wrapper._inference_point_count = 32
    wrapper._ascend_local_client = FakeClient()

    assert wrapper.warmup() is True
    assert wrapper._ascend_local_client.object_pc.shape == (32, 3)
    assert wrapper._ascend_local_client.object_pc.dtype == np.float32
    assert np.isfinite(wrapper._ascend_local_client.object_pc).all()


def test_non_ascend_wrapper_skips_startup_warmup():
    wrapper = object.__new__(GraspGenWrapper)
    wrapper.inference_backend = "remote_310p"

    assert wrapper.warmup() is False


def test_ascend_local_backend_rejects_negative_device_id():
    with pytest.raises(ValueError, match="device_id"):
        AscendLocalBackend("manifest.json", device_id=-1)


def test_ascend_local_warmup_preserves_request_random_stream():
    client = object.__new__(AscendLocalBackend)
    random_generator = np.random.default_rng(7)
    client._backend = SimpleNamespace(_random=random_generator)
    client._ensure_loaded = lambda: None
    client.sample = lambda *_args, **_kwargs: random_generator.standard_normal(16)

    client.warmup(np.zeros((20, 3), dtype=np.float32))

    expected = np.random.default_rng(7).standard_normal()
    assert random_generator.standard_normal() == expected


def test_ascend_local_wrapper_initializes_without_graspgen_source(tmp_path, monkeypatch):
    config_path = tmp_path / "graspgen_test.yml"
    config_path.write_text("data:\n  gripper_name: robotiq_2f_140\n  num_points: 2048\n", encoding="utf-8")

    class FakeAscendBackend:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(graspgen_wrapper, "AscendLocalBackend", FakeAscendBackend)
    wrapper = GraspGenWrapper(
        gripper_config=str(config_path),
        inference_backend="ascend_local",
        ascend_local_manifest_path="/tmp/bundle",
    )

    assert wrapper.gripper_name == "robotiq_2f_140"
    assert wrapper._inference_point_count == 2048
    assert wrapper._collision_mesh is None
    assert wrapper._collision_vertices is None


def test_local_depth_projection_separates_object_from_scene():
    depth = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    mask = np.array([[0, 1], [0, 1]], dtype=np.int32)

    scene, object_pc, scene_colors, object_colors = _depth_and_segmentation_to_point_clouds(
        depth_image=depth,
        segmentation_mask=mask,
        fx=2.0,
        fy=2.0,
        cx=0.0,
        cy=0.0,
        remove_object_from_scene=True,
    )

    assert scene_colors is None
    assert object_colors is None
    assert np.allclose(scene, [[0.0, 0.0, 1.0], [0.0, 1.5, 3.0]])
    assert np.allclose(object_pc, [[1.0, 0.0, 2.0]])


def test_local_outlier_filter_handles_cloud_smaller_than_neighbor_count():
    points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    kept, removed = _point_cloud_outlier_removal(points, threshold=2.0, neighbor_count=20, chunk_size=2)

    assert kept.shape == (2, 3)
    assert removed.shape == (1, 3)
