"""The gesture om wrapper and its post-processing (hand_perception/gesture_om.py), on synthetic tensors."""

import sys
import types

import numpy as np
import pytest

from manipulation_execution.hand_perception import gesture_om as go


def tensor(*dets):
    """[1,38,2100] zeros with (anchor, class id, score, xyxy) set."""
    y = np.zeros(go.OUTPUT_SHAPE, np.float32)
    for anchor, cls, score, box in dets:
        y[0, :4, anchor] = box
        y[0, 4 + cls, anchor] = score
    return y


def test_post_fast_keeps_scores_over_conf_best_first():
    y = tensor((10, 2, 0.4, (1, 2, 3, 4)), (20, 7, 0.9, (5, 6, 7, 8)), (30, 8, 0.2, (0, 0, 1, 1)))
    d = go.post_fast(y, conf=0.25)
    assert d.shape == (2, 6)
    np.testing.assert_allclose(d[0], [5, 6, 7, 8, 0.9, 7], rtol=1e-6)
    np.testing.assert_allclose(d[1], [1, 2, 3, 4, 0.4, 2], rtol=1e-6)


def test_post_fast_one_anchor_two_classes_gives_two_rows():
    y = tensor((5, 8, 0.6, (1, 1, 2, 2)))
    y[0, 4 + 9, 5] = 0.7
    d = go.post_fast(y)
    assert [int(c) for c in d[:, 5]] == [9, 8]


def test_post_fast_nothing_over_conf_is_empty():
    assert go.post_fast(tensor((1, 2, 0.1, (0, 0, 1, 1)))).shape == (0, 6)


@pytest.mark.parametrize(
    ("cls", "name"), [(2, "holy"), (7, "xsign"), (8, "heart"), (9, "heart")]
)
def test_kept_classes_and_the_two_hearts_merge(cls, name):
    d = go.post_fast(tensor((3, cls, 0.8, (10, 20, 30, 40))))
    got, score, box = go.best_gesture(d, 0.25)
    assert got == name and score == pytest.approx(0.8)
    np.testing.assert_allclose(box, [10, 20, 30, 40])


def test_other_classes_are_none_with_score_zero():
    d = go.post_fast(tensor((3, 33, 0.95, (0, 0, 1, 1)), (4, 0, 0.9, (0, 0, 1, 1))))
    assert go.best_gesture(d, 0.25) == ("none", 0.0, None)
    assert go.best_gesture(np.zeros((0, 6), np.float32), 0.25) == ("none", 0.0, None)


def test_best_of_the_three_wins_over_a_higher_other_class():
    d = go.post_fast(tensor((1, 33, 0.99, (0, 0, 1, 1)), (2, 7, 0.5, (0, 0, 1, 1)), (3, 9, 0.6, (0, 0, 2, 2))))
    name, score, _ = go.best_gesture(d, 0.25)
    assert name == "heart" and score == pytest.approx(0.6)


def test_box_back_to_source_pixels():
    # a 640 px square at (100, 50) was resized to 320: model px x2, then offset
    assert go.to_source([10, 20, 30, 40], 100, 50, 640) == [120.0, 90.0, 160.0, 130.0]


def test_preprocess_is_rgb_over_255_nchw_float32():
    sq = np.zeros((640, 640, 3), np.uint8)
    sq[..., 0] = 255  # blue in BGR
    x = go.preprocess(sq)
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert x[0, 2].min() == pytest.approx(1.0) and x[0, :2].max() == 0.0


class _Session:
    out = None
    raises = None

    def __init__(self, device, path):
        self.device, self.path, self.freed, self.fed = device, path, False, []

    def infer(self, feeds, mode=None):
        self.fed.append((feeds, mode))
        if self.raises:
            raise self.raises
        return [self.out]

    def free_resource(self):
        self.freed = True


@pytest.fixture
def fake_ais_bench(monkeypatch):
    interface = types.ModuleType("ais_bench.infer.interface")
    interface.InferSession = _Session
    monkeypatch.setitem(sys.modules, "ais_bench", types.ModuleType("ais_bench"))
    monkeypatch.setitem(sys.modules, "ais_bench.infer", types.ModuleType("ais_bench.infer"))
    monkeypatch.setitem(sys.modules, "ais_bench.infer.interface", interface)
    monkeypatch.setattr(_Session, "out", tensor((1, 2, 0.9, (0, 0, 1, 1))))
    monkeypatch.setattr(_Session, "raises", None)


def test_om_session_on_ais_bench_static(fake_ais_bench, tmp_path):
    path = tmp_path / "g.om"
    path.write_bytes(b"om")
    om = go.GestureOm(path, device=1)
    assert om.session.device == 1 and om.session.path == str(path)
    x = np.zeros((1, 3, 320, 320), np.float32)
    assert om.infer(x).shape == go.OUTPUT_SHAPE
    assert om.session.fed[0][1] == "static"
    session = om.session
    om.close()
    om.close()
    assert session.freed and om.session is None


def test_relative_model_path_is_under_the_workspace(fake_ais_bench, tmp_path, monkeypatch):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "g.om").write_bytes(b"om")
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    assert go.GestureOm("m/g.om").path == tmp_path / "m" / "g.om"
    monkeypatch.delenv("WORKSPACE")
    with pytest.raises(ValueError):
        go.GestureOm("m/g.om")


def test_e2e_om_and_missing_file_are_refused(fake_ais_bench, tmp_path):
    e2e = tmp_path / "yolov10n_gestures_320_e2e.om"
    e2e.write_bytes(b"om")
    with pytest.raises(ValueError, match="e2e"):
        go.GestureOm(e2e)
    with pytest.raises(FileNotFoundError):
        go.GestureOm(tmp_path / "nope.om")


@pytest.mark.parametrize(
    "out", [np.zeros((1, 38, 8400), np.float32), np.full(go.OUTPUT_SHAPE, np.nan, np.float32)]
)
def test_wrong_shape_or_nan_output_raises(fake_ais_bench, tmp_path, monkeypatch, out):
    path = tmp_path / "g.om"
    path.write_bytes(b"om")
    monkeypatch.setattr(_Session, "out", out)
    with pytest.raises(go.BadOutput):
        go.GestureOm(path).infer(np.zeros((1, 3, 320, 320), np.float32))
