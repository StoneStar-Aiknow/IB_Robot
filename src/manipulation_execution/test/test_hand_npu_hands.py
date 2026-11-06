"""The hand perception code without ROS or an NPU: label helpers, the classifier head, where modules load from.

The inference_service backend needs ACL, so it is compared with ais_bench on the board, not here."""

import json
import sys
import types

import numpy as np
import pytest

from manipulation_execution.hand_perception import npu_hands


def _write_head(path, seed=3):
    rng = np.random.default_rng(seed)
    head = {
        "bn_mul": rng.normal(1.0, 0.1, 128),
        "bn_add": rng.normal(0.0, 0.1, 128),
        "W": rng.normal(0.0, 1.0, (128, len(npu_hands.LABELS))),
        "b": rng.normal(0.0, 0.1, len(npu_hands.LABELS)),
    }
    np.savez(path, **head)
    return head


class _FakeSession:
    """ais_bench's InferSession: det sees no hand, the embedder returns a fixed vector."""

    embedding = np.arange(128, dtype=np.float32) / 128.0

    def __init__(self, device, path):
        self.path = str(path)
        self.fed = []
        self.freed = False

    def get_inputs(self):
        return [types.SimpleNamespace(name=n) for n in ("hand", "handedness", "world_hand")]

    def infer(self, feeds):
        self.fed.append(feeds)
        if self.path.endswith(npu_hands.EMB_OM):
            return [self.embedding]
        return [np.zeros((1, 2016, 18), np.float32), np.full((1, 2016, 1), -100.0, np.float32)]

    def free_resource(self):
        self.freed = True


@pytest.fixture
def fake_ais_bench(monkeypatch):
    interface = types.ModuleType("ais_bench.infer.interface")
    interface.InferSession = _FakeSession
    monkeypatch.setitem(sys.modules, "ais_bench", types.ModuleType("ais_bench"))
    monkeypatch.setitem(sys.modules, "ais_bench.infer", types.ModuleType("ais_bench.infer"))
    monkeypatch.setitem(sys.modules, "ais_bench.infer.interface", interface)


@pytest.fixture
def model_dir(tmp_path):
    for name in (npu_hands.DET_OM, npu_hands.LM_OM, npu_hands.EMB_OM):
        (tmp_path / name).write_bytes(b"om")
    _write_head(tmp_path / npu_hands.HEAD)
    return tmp_path


def test_as_right_hand_leaves_a_right_hand_alone():
    lm = np.random.default_rng(0).normal(size=(21, 3))
    assert npu_hands.as_right_hand(lm, 0.9) is lm


def test_as_right_hand_mirrors_a_left_hand_in_x_about_its_mean():
    lm = np.random.default_rng(1).normal(size=(21, 3))
    before = lm.copy()
    out = npu_hands.as_right_hand(lm, 0.1)
    np.testing.assert_array_equal(lm, before)
    np.testing.assert_allclose(out[:, 0], 2.0 * before[:, 0].mean() - before[:, 0])
    np.testing.assert_array_equal(out[:, 1:], before[:, 1:])
    np.testing.assert_allclose(npu_hands.as_right_hand(out, 0.1), before)


@pytest.mark.parametrize(
    ("label", "score", "rule", "expected"),
    [
        ("Victory", 0.4, "thumb_index_middle", ("gun", 1.0)),
        ("None", 0.2, "six", ("six", 1.0)),
        ("ILoveYou", 0.8, "other", ("rock", 0.8)),
        ("Victory", 0.8, "other", ("Victory", 0.8)),
    ],
)
def test_relabel(label, score, rule, expected):
    assert npu_hands.relabel(label, score, rule) == expected


def test_classify_is_the_head_over_the_embedding(fake_ais_bench, model_dir):
    hands = npu_hands.NpuHands(model_dir, backend="ais_bench")
    head = np.load(model_dir / npu_hands.HEAD)
    emb = _FakeSession.embedding.astype(np.float64)
    logits = np.maximum(emb * head["bn_mul"] + head["bn_add"], 0.0) @ head["W"] + head["b"]
    expected = np.exp(logits) / np.exp(logits).sum()

    label, score = hands._classify(np.zeros((21, 3)))

    assert label == npu_hands.LABELS[int(np.argmax(expected))]
    assert score == pytest.approx(float(expected.max()))
    hand, handed, world = hands.npu.emb.fed[-1]
    assert hand.shape == (1, 21, 3) and hand.dtype == np.float32
    assert handed.shape == (1, 1) and world.shape == (1, 21, 3)


def test_head_with_the_wrong_shape_is_refused(fake_ais_bench, model_dir):
    np.savez(model_dir / npu_hands.HEAD, bn_mul=np.ones(128), bn_add=np.ones(128), W=np.ones((128, 7)), b=np.ones(7))
    with pytest.raises(ValueError, match="expected"):
        npu_hands.NpuHands(model_dir, backend="ais_bench")


def test_no_hand_detected_is_an_empty_list(fake_ais_bench, model_dir):
    hands = npu_hands.NpuHands(model_dir, backend="ais_bench")
    img = np.zeros((240, 320, 3), np.uint8)
    assert hands.infer(img, (10.0, 10.0, 200.0, 230.0)) == []
    hands.reset()
    hands.close()
    assert hands.npu.emb.freed


def test_missing_model_files_are_named(fake_ais_bench, model_dir):
    (model_dir / npu_hands.LM_OM).unlink()
    with pytest.raises(FileNotFoundError, match=npu_hands.LM_OM):
        npu_hands.NpuHands(model_dir, backend="ais_bench")


def test_no_backend_given_is_ais_bench(fake_ais_bench, model_dir):
    hands = npu_hands.NpuHands(model_dir)
    assert npu_hands.DEFAULT_BACKEND == "ais_bench"
    assert hands.backend_name == "ais_bench"
    assert isinstance(hands.npu, npu_hands.AisBenchBackend)


def test_unknown_backend_is_refused(model_dir):
    with pytest.raises(ValueError, match="onnx"):
        npu_hands.NpuHands(model_dir, backend="onnx")


def test_model_paths_flat_directory_and_bundle(tmp_path):
    flat = npu_hands.model_paths(tmp_path)
    assert flat[npu_hands.DET_OM] == tmp_path / npu_hands.DET_OM
    artifacts = tmp_path / npu_hands.BUNDLE_ARTIFACTS
    artifacts.mkdir(parents=True)
    assert npu_hands.model_paths(tmp_path)[npu_hands.HEAD] == artifacts / npu_hands.HEAD


def test_relative_model_dir_is_under_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    expected = tmp_path / npu_hands.DEFAULT_MODEL_DIR / npu_hands.DET_OM
    assert npu_hands.model_paths(npu_hands.DEFAULT_MODEL_DIR)[npu_hands.DET_OM] == expected
    monkeypatch.delenv("WORKSPACE")
    with pytest.raises(ValueError, match="WORKSPACE"):
        npu_hands.model_paths(npu_hands.DEFAULT_MODEL_DIR)


@pytest.mark.parametrize(
    ("filename", "public"),
    [
        (npu_hands.PIPELINE_PY, "HandPipeline"),
        (npu_hands.KEYPOINTS_PY, "finger_curls"),
        (npu_hands.HAND_CORE_PY, "HandStream"),
    ],
)
def test_empty_extern_dir_loads_the_packaged_module(filename, public):
    mod = npu_hands.load_extern("", filename)
    assert hasattr(mod, public)
    assert "x2-dexhand-run" not in mod.__file__
    assert mod.__file__.endswith(f"x2/_vendor/{filename}")
    assert npu_hands.load_extern("", filename) is mod


def test_packaged_data_files_are_beside_hand_core():
    hand_core = npu_hands.load_extern("", npu_hands.HAND_CORE_PY)
    assert hand_core.CALIB_MEDIAPIPE_PATH.is_file()
    assert hand_core.ROBOT_PATH.is_file()
    assert hand_core.CALIB_PATH.is_file()
    calib = json.loads(npu_hands.extern_path("", npu_hands.CALIB_MEDIAPIPE_JSON).read_text())
    keypoints = npu_hands.load_extern("", npu_hands.KEYPOINTS_PY)
    for side in ("left", "right"):
        assert set(keypoints.CURL_FINGERS) <= set(calib["endpoints"][side])


def test_extern_dir_overrides_the_package_and_can_go_back(tmp_path):
    (tmp_path / npu_hands.KEYPOINTS_PY).write_text("WHO = 'disk'\n")
    disk = npu_hands.load_extern(tmp_path, npu_hands.KEYPOINTS_PY)
    assert disk.WHO == "disk"
    assert npu_hands.load_extern(tmp_path, npu_hands.KEYPOINTS_PY) is disk
    packaged = npu_hands.load_extern("", npu_hands.KEYPOINTS_PY)
    assert not hasattr(packaged, "WHO")


def test_extern_dir_without_the_file_is_an_error_not_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match=npu_hands.KEYPOINTS_PY):
        npu_hands.load_extern(tmp_path, npu_hands.KEYPOINTS_PY)
