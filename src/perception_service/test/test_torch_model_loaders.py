from types import SimpleNamespace

import numpy as np

from perception_service.torch_model_loaders import _GraspGenModule, _GroundedSAM2Module, _SAM2Module, _SigLIP2Module


def test_sam2_module_preserves_empty_source_geometry() -> None:
    module = _SAM2Module(SimpleNamespace(generate=lambda _image: []))

    outputs = module({"observation.image": np.zeros((12, 20, 3), dtype=np.uint8)})

    assert outputs["masks"].shape == (0, 12, 20)
    assert outputs["masks"].dtype == np.uint8
    assert outputs["boxes"].shape == (0, 4)
    assert outputs["boxes"].dtype == np.float32
    assert outputs["scores"].dtype == np.float32
    assert outputs["stability_scores"].dtype == np.float32


def test_grounded_sam2_module_forwards_thresholds_and_preserves_empty_geometry() -> None:
    calls = []

    def detect(image, prompt, box_threshold, text_threshold):
        calls.append((image, prompt, box_threshold, text_threshold))
        return []

    module = _GroundedSAM2Module(SimpleNamespace(detect_and_segment=detect))
    image = np.zeros((12, 20, 3), dtype=np.uint8)

    outputs = module(
        {
            "observation.image": image,
            "text_prompt": np.frombuffer(b"red cup", dtype=np.uint8),
            "box_threshold": np.asarray([0.4], dtype=np.float32),
            "text_threshold": np.asarray([0.2], dtype=np.float32),
        }
    )

    assert calls[0][0].shape == image.shape
    assert calls[0][0].flags.c_contiguous
    assert calls[0][1] == "red cup"
    assert calls[0][2] == np.float32(0.4)
    assert calls[0][3] == np.float32(0.2)
    assert outputs["masks"].shape == (0, 12, 20)
    assert outputs["label_indices"].dtype == np.int32


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.device = "cpu"

    def __len__(self):
        return len(self.value)


class _FakeFeature:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _dtype):
        return self.value.astype(np.float32)


class _FakeTorch:
    float32 = np.float32

    @staticmethod
    def empty(shape, dtype, device):
        assert device == "cpu"
        return _FakeFeature(np.empty(shape, dtype=dtype))


class _FakeSigLIP2:
    config = SimpleNamespace(text_config=SimpleNamespace(projection_size=4, hidden_size=4))

    @staticmethod
    def get_image_features(pixel_values):
        return _FakeFeature(np.ones((len(pixel_values), 4)))

    @staticmethod
    def get_text_features(input_ids, attention_mask):
        assert len(input_ids) == len(attention_mask)
        return _FakeFeature(np.ones((len(input_ids), 4)) * 2)


def test_siglip2_module_supports_zero_length_unused_batches() -> None:
    module = _SigLIP2Module(_FakeSigLIP2(), _FakeTorch())

    image_outputs = module(
        {
            "masked_images": _FakeTensor(np.zeros((2, 3, 384, 384), dtype=np.float32)),
            "text_tokens": _FakeTensor(np.empty((0, 64), dtype=np.int64)),
            "text_attention_mask": _FakeTensor(np.empty((0, 64), dtype=np.int64)),
        }
    )
    text_outputs = module(
        {
            "masked_images": _FakeTensor(np.empty((0, 3, 384, 384), dtype=np.float32)),
            "text_tokens": _FakeTensor(np.zeros((3, 64), dtype=np.int64)),
            "text_attention_mask": _FakeTensor(np.ones((3, 64), dtype=np.int64)),
        }
    )

    assert image_outputs["image_embeddings"].shape == (2, 4)
    assert image_outputs["text_embeddings"].shape == (0, 4)
    assert text_outputs["image_embeddings"].shape == (0, 4)
    assert text_outputs["text_embeddings"].shape == (3, 4)


class _FakeGraspTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def to(self, dtype):
        return self.value.astype(dtype)

    def __len__(self):
        return len(self.value)


class _FakeGraspTorch:
    float32 = np.float32

    @staticmethod
    def empty(shape, dtype):
        return _FakeGraspTensor(np.empty(shape, dtype=dtype))


def test_graspgen_module_runs_once_and_preserves_empty_output_geometry(monkeypatch) -> None:
    calls = []

    def run_inference(points, sampler, **options):
        calls.append((points, sampler, options))
        return _FakeGraspTensor(np.empty((0,))), _FakeGraspTensor(np.empty((0,)))

    monkeypatch.setattr("grasp_gen.grasp_server.GraspGenSampler.run_inference", run_inference)
    module = _GraspGenModule(
        "sampler",
        SimpleNamespace(kappa=2.0, grasp_batch_size=1000),
        _FakeGraspTorch(),
    )

    outputs = module({"observation.object_points": _FakeGraspTensor(np.ones((32, 3), dtype=np.float32))})

    assert calls[0][0].shape == (32, 3)
    assert calls[0][1] == "sampler"
    assert calls[0][2]["topk_num_grasps"] == 1000
    assert calls[0][2]["min_grasps"] == 1
    assert calls[0][2]["max_tries"] == 1
    assert calls[0][2]["remove_outliers"] is False
    assert outputs["grasp.poses"].shape == (0, 4, 4)
    assert outputs["grasp.confidence"].shape == (0,)
