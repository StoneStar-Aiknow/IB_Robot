import argparse
import csv
import json
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from PIL import Image

try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import IMAGENET_STATS
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    _LEROBOT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on environment
    PreTrainedConfig = None
    LeRobotDatasetMetadata = None
    IMAGENET_STATS = {}
    LeRobotDataset = None
    make_policy = None
    make_pre_post_processors = None
    ACTION = "action"
    _LEROBOT_IMPORT_ERROR = exc


def _require_lerobot_symbol(symbol: Any, name: str) -> Any:
    if symbol is None:
        raise ImportError(
            "lerobot is required for frame inspection. Please initialize a compatible IB_Robot environment "
            "before running frame_inspect."
        ) from _LEROBOT_IMPORT_ERROR
    return symbol


def parse_frame_index_arg(frame_index: int | str | None) -> tuple[str, int, int | None]:
    if frame_index is None:
        return "none", -1, None
    if isinstance(frame_index, int):
        return "single", frame_index, None
    if ":" not in frame_index:
        return "single", int(frame_index), None

    start_str, end_str = frame_index.split(":", maxsplit=1)
    start = int(start_str)
    end = int(end_str)
    if start >= end:
        raise ValueError("Range frame_index must satisfy start < end.")
    return "range", start, end


def resolve_sample_index(
    *,
    dataset: LeRobotDataset,
    global_index: int | None = None,
    episode_index: int | None = None,
    frame_index: int | None = None,
) -> int:
    has_global = global_index is not None
    has_episode_frame = episode_index is not None or frame_index is not None

    if has_global == has_episode_frame:
        raise ValueError("Provide either global_index or episode_index with frame_index.")

    if has_global:
        if global_index < 0 or global_index >= dataset.num_frames:
            raise IndexError(
                f"global_index {global_index} is out of bounds for dataset with {dataset.num_frames} frames."
            )
        return global_index

    if episode_index is None or frame_index is None:
        raise ValueError("episode_index and frame_index must be provided together.")

    if episode_index < 0 or episode_index >= len(dataset.meta.episodes):
        raise IndexError(
            f"episode_index {episode_index} is out of bounds for dataset with {len(dataset.meta.episodes)} episodes."
        )

    episode = dataset.meta.episodes[episode_index]
    from_index = int(episode["dataset_from_index"])
    to_index = int(episode["dataset_to_index"])
    episode_length = to_index - from_index
    if frame_index < 0 or frame_index >= episode_length:
        raise IndexError(
            f"frame_index {frame_index} is out of bounds for episode {episode_index} with {episode_length} frames."
        )
    return from_index + frame_index


def resolve_sample_range(
    *,
    dataset: LeRobotDataset,
    episode_index: int,
    frame_index: str,
) -> tuple[int, int]:
    mode, start, end = parse_frame_index_arg(frame_index)
    if mode != "range" or end is None:
        raise ValueError("Range mode expects frame_index in start:end format.")

    if episode_index < 0 or episode_index >= len(dataset.meta.episodes):
        raise IndexError(
            f"episode_index {episode_index} is out of bounds for dataset with {len(dataset.meta.episodes)} episodes."
        )

    episode = dataset.meta.episodes[episode_index]
    from_index = int(episode["dataset_from_index"])
    to_index = int(episode["dataset_to_index"])
    episode_length = to_index - from_index
    if start < 0 or end > episode_length:
        raise IndexError(
            f"frame_index range {start}:{end} is out of bounds for episode {episode_index} with {episode_length} frames."
        )
    return from_index + start, from_index + end


def _normalize_dataset_sample(sample_entry: Any) -> dict[str, Any]:
    if isinstance(sample_entry, dict):
        return sample_entry
    if isinstance(sample_entry, tuple) and sample_entry and isinstance(sample_entry[0], dict):
        return sample_entry[0]
    raise TypeError(
        "Unsupported dataset sample type. Expected dict or tuple whose first element is a dict, "
        f"got {type(sample_entry).__name__}."
    )


def _resolve_observation_value(
    *,
    sample: dict[str, Any],
    key: str,
) -> Any:
    if key in sample:
        return sample[key]

    fallback_keys = {
        "observation.current": "observation.state",
        "observation.state": "observation.current",
    }
    fallback_key = fallback_keys.get(key)
    if fallback_key is not None and fallback_key in sample:
        warnings.warn(
            f"Observation key '{key}' missing in dataset sample; falling back to '{fallback_key}' for offline inspection.",
            stacklevel=2,
        )
        return sample[fallback_key]

    raise KeyError(f"Missing required observation key: {key}")


def extract_inference_inputs(
    *,
    sample: dict[str, Any],
    input_keys: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, int]]:
    observation: dict[str, Any] = {}
    for key in input_keys:
        observation[key] = _resolve_observation_value(sample=sample, key=key).unsqueeze(0).to(device)

    if "task" in sample:
        observation["task"] = sample["task"]

    if ACTION not in sample:
        raise KeyError(f"Missing required label key: {ACTION}")

    metadata = {
        "episode_index": int(sample["episode_index"].item()),
        "frame_index": int(sample["frame_index"].item()),
        "global_index": int(sample["index"].item()),
    }
    return observation, sample[ACTION], metadata


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    image_hwc = image.permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
    return Image.fromarray(image_hwc)


def _image_name_from_key(key: str) -> str:
    return key.rsplit(".", maxsplit=1)[-1]


def _image_keys_from_sample(sample: dict[str, Any]) -> list[str]:
    return sorted(key for key in sample if key.startswith("observation.images."))


def save_artifacts(
    *,
    output_dir: Path,
    sample: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_key in _image_keys_from_sample(sample):
        image_name = _image_name_from_key(image_key)
        tensor_image_to_pil(sample[image_key]).save(output_dir / f"{image_name}_frame.png")

    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    summary_lines = ["action\tprediction\tlabel\tdelta"]
    for action_name in comparison["prediction"]:
        summary_lines.append(
            f"{action_name}\t{comparison['prediction'][action_name]}\t"
            f"{comparison['label'][action_name]}\t{comparison['delta'][action_name]}"
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _to_video_tensor(image: torch.Tensor) -> torch.Tensor:
    return image.detach().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).to(torch.uint8)


def _write_video_clip(path: Path, frames: list[torch.Tensor], fps: int) -> None:
    import torchvision

    video = torch.stack([_to_video_tensor(frame) for frame in frames], dim=0)
    torchvision.io.write_video(str(path), video, fps=fps, video_codec="libx264rgb")


def _flatten_row(row: dict[str, Any], action_names: list[str]) -> dict[str, Any]:
    flat_row = {
        "global_index": row["global_index"],
        "episode_index": row["episode_index"],
        "frame_index": row["frame_index"],
    }
    for prefix in ("prediction", "label", "delta"):
        for action_name in action_names:
            flat_row[f"{prefix}.{action_name}"] = row[prefix][action_name]
    return flat_row


def save_range_artifacts(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    fps: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, str] = {}
    if samples:
        for image_key in _image_keys_from_sample(samples[0]):
            image_name = _image_name_from_key(image_key)
            clip_path = output_dir / f"{image_name}_clip.mp4"
            _write_video_clip(clip_path, [sample[image_key] for sample in samples], fps)
            exported_paths[f"{image_name}_clip"] = str(clip_path)

    comparison_csv = output_dir / "comparison.csv"
    comparison_json = output_dir / "comparison.json"

    action_names = list(comparison_rows[0]["prediction"]) if comparison_rows else []
    flat_rows = [_flatten_row(row, action_names) for row in comparison_rows]
    fieldnames = ["global_index", "episode_index", "frame_index"] + [
        f"{prefix}.{action_name}" for prefix in ("prediction", "label", "delta") for action_name in action_names
    ]
    with comparison_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    exported_paths["comparison_csv"] = str(comparison_csv)
    exported_paths["comparison_json"] = str(comparison_json)

    summary = {
        "policy_path": comparison_rows[0].get("policy_path") if comparison_rows else None,
        "dataset_repo_id": comparison_rows[0].get("dataset_repo_id") if comparison_rows else None,
        "dataset_root": comparison_rows[0].get("dataset_root") if comparison_rows else None,
        "stats_dataset_repo_id": comparison_rows[0].get("stats_dataset_repo_id") if comparison_rows else None,
        "stats_dataset_root": comparison_rows[0].get("stats_dataset_root") if comparison_rows else None,
        "episode_index": comparison_rows[0].get("episode_index") if comparison_rows else None,
        "start": comparison_rows[0].get("frame_index") if comparison_rows else None,
        "end": (
            (comparison_rows[-1].get("frame_index") + 1) if comparison_rows[-1].get("frame_index") is not None else None
        )
        if comparison_rows
        else None,
        "total_frames": len(comparison_rows),
        "action_names": action_names,
        "exported_paths": exported_paths,
    }
    comparison_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return exported_paths


def load_policy_and_processors(
    *,
    policy_path: Path,
    dataset_meta: LeRobotDatasetMetadata,
    stats_meta: LeRobotDatasetMetadata,
    device: str | None = None,
):
    pretrained_config_cls = _require_lerobot_symbol(PreTrainedConfig, "PreTrainedConfig")
    make_policy_fn = _require_lerobot_symbol(make_policy, "make_policy")
    make_processors_fn = _require_lerobot_symbol(make_pre_post_processors, "make_pre_post_processors")

    policy_cfg = pretrained_config_cls.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    if device is not None:
        policy_cfg.device = device

    policy = make_policy_fn(cfg=policy_cfg, ds_meta=dataset_meta)
    preprocessor, postprocessor = make_processors_fn(
        policy_cfg=policy.config,
        pretrained_path=str(policy_path),
        dataset_stats=stats_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return policy, preprocessor, postprocessor


def apply_training_dataset_factory_stats(
    *,
    stats_meta: LeRobotDatasetMetadata,
    use_imagenet_stats: bool,
) -> LeRobotDatasetMetadata:
    if not use_imagenet_stats:
        return stats_meta

    stats_meta = deepcopy(stats_meta)
    for camera_key in getattr(stats_meta, "camera_keys", []):
        if camera_key not in stats_meta.stats:
            continue
        for stats_type, stats in IMAGENET_STATS.items():
            stats_meta.stats[camera_key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    return stats_meta


def _as_float_list(tensor: torch.Tensor) -> list[float]:
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _build_named_action_map(values: list[float], action_names: list[str]) -> dict[str, float]:
    if len(values) != len(action_names):
        raise ValueError(f"Action vector length {len(values)} does not match action names length {len(action_names)}.")
    return {name: value for name, value in zip(action_names, values, strict=True)}


def _predict_single_sample(
    *,
    sample: dict[str, Any],
    policy,
    preprocessor,
    postprocessor,
    device_obj: torch.device,
    action_names: list[str],
) -> tuple[dict[str, Any], dict[str, float], dict[str, float], dict[str, float]]:
    input_keys = list(policy.config.input_features)
    observation, label, metadata = extract_inference_inputs(sample=sample, input_keys=input_keys, device=device_obj)

    autocast_context = (
        torch.autocast(device_type=device_obj.type)
        if device_obj.type == "cuda" and policy.config.use_amp
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        processed_observation = preprocessor(observation)
        prediction = postprocessor(policy.select_action(processed_observation))

    prediction_values = _as_float_list(prediction)
    label_values = _as_float_list(label)
    prediction_map = _build_named_action_map(prediction_values, action_names)
    label_map = _build_named_action_map(label_values, action_names)
    delta_map = {name: float(prediction_map[name] - label_map[name]) for name in action_names}
    return metadata, prediction_map, label_map, delta_map


def run_inspection(
    *,
    policy_path: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    output_dir: Path,
    global_index: int | None = None,
    episode_index: int | None = None,
    frame_index: int | str | None = None,
    stats_dataset_repo_id: str | None = None,
    stats_dataset_root: Path | None = None,
    device: str | None = None,
    use_imagenet_stats: bool = True,
    reset_policy: bool = True,
) -> dict[str, Any]:
    if (stats_dataset_repo_id is None) != (stats_dataset_root is None):
        raise ValueError("stats_dataset_repo_id and stats_dataset_root must be provided together.")

    dataset_cls = _require_lerobot_symbol(LeRobotDataset, "LeRobotDataset")

    dataset = dataset_cls(repo_id=dataset_repo_id, root=dataset_root)
    if stats_dataset_repo_id is None:
        raw_stats_meta = dataset.meta
    else:
        dataset_metadata_cls = _require_lerobot_symbol(LeRobotDatasetMetadata, "LeRobotDatasetMetadata")
        raw_stats_meta = dataset_metadata_cls(repo_id=stats_dataset_repo_id, root=stats_dataset_root)
    stats_meta = apply_training_dataset_factory_stats(
        stats_meta=raw_stats_meta,
        use_imagenet_stats=use_imagenet_stats,
    )
    policy, preprocessor, postprocessor = load_policy_and_processors(
        policy_path=policy_path,
        dataset_meta=dataset.meta,
        stats_meta=stats_meta,
        device=device,
    )

    device_obj = torch.device(device or policy.config.device)
    action_names = list(dataset.meta.features[ACTION]["names"])
    mode, _, _ = parse_frame_index_arg(frame_index)

    if mode == "range":
        if global_index is not None or episode_index is None or not isinstance(frame_index, str):
            raise ValueError("Range mode requires episode_index and frame_index=start:end, without global_index.")
        start_index, end_index = resolve_sample_range(
            dataset=dataset, episode_index=episode_index, frame_index=frame_index
        )
        samples = [_normalize_dataset_sample(dataset[idx]) for idx in range(start_index, end_index)]
        rows = []
        for sample in samples:
            if reset_policy and hasattr(policy, "reset"):
                policy.reset()
            metadata, prediction_map, label_map, delta_map = _predict_single_sample(
                sample=sample,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device_obj=device_obj,
                action_names=action_names,
            )
            rows.append(
                {
                    "policy_path": str(policy_path),
                    "dataset_repo_id": dataset_repo_id,
                    "dataset_root": str(dataset_root),
                    "stats_dataset_repo_id": stats_dataset_repo_id or dataset_repo_id,
                    "stats_dataset_root": str(stats_dataset_root or dataset_root),
                    **metadata,
                    "prediction": prediction_map,
                    "label": label_map,
                    "delta": delta_map,
                }
            )
        exported_paths = save_range_artifacts(output_dir=output_dir, samples=samples, comparison_rows=rows, fps=30)
        return {
            "policy_path": str(policy_path),
            "dataset_repo_id": dataset_repo_id,
            "dataset_root": str(dataset_root),
            "stats_dataset_repo_id": stats_dataset_repo_id or dataset_repo_id,
            "stats_dataset_root": str(stats_dataset_root or dataset_root),
            "episode_index": episode_index,
            "range": {"start": rows[0]["frame_index"], "end": rows[-1]["frame_index"] + 1},
            "total_frames": len(rows),
            "exported_paths": exported_paths,
        }

    sample_index = resolve_sample_index(
        dataset=dataset,
        global_index=global_index,
        episode_index=episode_index,
        frame_index=int(frame_index) if isinstance(frame_index, str) else frame_index,
    )
    sample = _normalize_dataset_sample(dataset[sample_index])
    if reset_policy and hasattr(policy, "reset"):
        policy.reset()
    metadata, prediction_map, label_map, delta_map = _predict_single_sample(
        sample=sample,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device_obj=device_obj,
        action_names=action_names,
    )

    comparison = {
        "policy_path": str(policy_path),
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": str(dataset_root),
        "stats_dataset_repo_id": stats_dataset_repo_id or dataset_repo_id,
        "stats_dataset_root": str(stats_dataset_root or dataset_root),
        **metadata,
        "action_names": action_names,
        "prediction": prediction_map,
        "label": label_map,
        "delta": delta_map,
    }
    save_artifacts(output_dir=output_dir, sample=sample, comparison=comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one dataset frame with offline policy inference.")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-index", type=int)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--frame-index", type=str)
    parser.add_argument("--stats-dataset-repo-id", type=str)
    parser.add_argument("--stats-dataset-root", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--use-imagenet-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-policy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_inspection(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        global_index=args.global_index,
        episode_index=args.episode_index,
        frame_index=args.frame_index,
        stats_dataset_repo_id=args.stats_dataset_repo_id,
        stats_dataset_root=args.stats_dataset_root,
        device=args.device,
        use_imagenet_stats=args.use_imagenet_stats,
        reset_policy=args.reset_policy,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
