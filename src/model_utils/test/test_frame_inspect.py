import importlib.util
import warnings
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "model_utils" / "frame_inspect.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("model_utils.frame_inspect", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


frame_inspect_module = _load_module()


def _make_sample(idx: int, *, include_current: bool = False) -> dict:
    sample = {
        "observation.images.top": torch.zeros(3, 4, 5),
        "observation.images.wrist": torch.ones(3, 4, 5),
        "observation.state": torch.tensor([1.0, 2.0, 3.0]),
        "action": torch.tensor([0.5 + idx, -0.25 + idx]),
        "episode_index": torch.tensor(0),
        "frame_index": torch.tensor(idx),
        "index": torch.tensor(idx),
        "task": "pick banana",
    }
    if include_current:
        sample["observation.current"] = torch.tensor([4.0, 5.0, 6.0])
    return sample


def test_normalize_dataset_sample_supports_dict_and_tuple():
    sample = _make_sample(0)

    assert frame_inspect_module._normalize_dataset_sample(sample) is sample
    assert frame_inspect_module._normalize_dataset_sample((sample, 0)) is sample


def test_extract_inference_inputs_falls_back_from_current_to_state():
    sample = _make_sample(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observation, label, metadata = frame_inspect_module.extract_inference_inputs(
            sample=sample,
            input_keys=["observation.current"],
            device=torch.device("cpu"),
        )

    assert observation["observation.current"].shape == (1, 3)
    assert torch.equal(observation["observation.current"].squeeze(0), sample["observation.state"])
    assert label.shape == (2,)
    assert metadata["global_index"] == 0
    assert any("falling back to 'observation.state'" in str(w.message) for w in caught)


def test_extract_inference_inputs_falls_back_from_state_to_current():
    sample = _make_sample(0, include_current=True)
    del sample["observation.state"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observation, _label, _metadata = frame_inspect_module.extract_inference_inputs(
            sample=sample,
            input_keys=["observation.state"],
            device=torch.device("cpu"),
        )

    assert observation["observation.state"].shape == (1, 3)
    assert torch.equal(observation["observation.state"].squeeze(0), sample["observation.current"])
    assert any("falling back to 'observation.current'" in str(w.message) for w in caught)


def test_run_inspection_single_frame_supports_tuple_dataset_samples(tmp_path, monkeypatch):
    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 3}],
                    "stats": {"action": {"mean": torch.tensor([0.0, 0.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 3

        def __getitem__(self, idx):
            return _make_sample(idx), idx

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.top": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            return torch.tensor([[0.0, 0.25]])

    monkeypatch.setattr(frame_inspect_module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(
        frame_inspect_module,
        "load_policy_and_processors",
        lambda **kwargs: (FakePolicy(), lambda batch: batch, lambda action: action),
    )

    result = frame_inspect_module.run_inspection(
        policy_path=Path("pretrained_model"),
        dataset_repo_id="dataset_id",
        dataset_root=Path("dataset"),
        output_dir=tmp_path,
        global_index=2,
    )

    assert result["global_index"] == 2
    assert (tmp_path / "top_frame.png").exists()
    assert (tmp_path / "wrist_frame.png").exists()
    assert (tmp_path / "comparison.json").exists()


def test_run_inspection_range_writes_summary_exports(tmp_path, monkeypatch):
    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 4}],
                    "stats": {"action": {"mean": torch.tensor([0.0, 0.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 4

        def __getitem__(self, idx):
            return _make_sample(idx), idx

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.top": object(),
                    "observation.images.wrist": object(),
                    "observation.current": object(),
                },
            },
        )()

        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def select_action(self, observation):
            return torch.tensor([[0.0, 0.25]])

    def fake_write_video_clip(path, frames, fps):
        path.write_bytes(f"{len(frames)}@{fps}".encode())

    policy = FakePolicy()
    monkeypatch.setattr(frame_inspect_module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(
        frame_inspect_module,
        "load_policy_and_processors",
        lambda **kwargs: (policy, lambda batch: batch, lambda action: action),
    )
    monkeypatch.setattr(frame_inspect_module, "_write_video_clip", fake_write_video_clip)

    result = frame_inspect_module.run_inspection(
        policy_path=Path("pretrained_model"),
        dataset_repo_id="dataset_id",
        dataset_root=Path("dataset"),
        output_dir=tmp_path,
        episode_index=0,
        frame_index="1:3",
    )

    assert result["range"] == {"start": 1, "end": 3}
    assert result["total_frames"] == 2
    assert set(result["exported_paths"]) == {"top_clip", "wrist_clip", "comparison_csv", "comparison_json"}
    assert (tmp_path / "top_clip.mp4").exists()
    assert (tmp_path / "wrist_clip.mp4").exists()
    assert (tmp_path / "comparison.csv").exists()
    assert policy.reset_calls == 2
