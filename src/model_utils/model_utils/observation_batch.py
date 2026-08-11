"""Portable raw observation batches backed by Safetensors."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_NAME = "ibrobot-observation-batch"
SCHEMA_VERSION = 1
FORMAT_KEY = "__ibrobot.format"
VERSION_KEY = "__ibrobot.schema_version"
FIELDS_KEY = "__ibrobot.fields"
TASKS_KEY = "__ibrobot.tasks"
PROVENANCE_KEY = "__ibrobot.provenance"
TASK_INDEX_TENSOR = "__ibrobot.task_index"
EPISODE_INDEX_TENSOR = "__ibrobot.episode_index"
FRAME_INDEX_TENSOR = "__ibrobot.frame_index"
DATASET_INDEX_TENSOR = "__ibrobot.dataset_index"
_RESERVED_PREFIX = "__ibrobot."
_SUPPORTED_DTYPES = {
    "bool": np.dtype("bool"),
    "uint8": np.dtype("uint8"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
}


@dataclass(frozen=True)
class FieldSpec:
    """Explicit numeric field contract and optional random generation range."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | int = 0
    maximum: float | int = 1
    semantic: str = "tensor"
    layout: str = ""


@dataclass
class ObservationBatch:
    """Loaded samples and file-level provenance."""

    samples: list[dict[str, Any]]
    fields: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    sample_provenance: dict[str, np.ndarray]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _dtype_name(dtype: np.dtype[Any]) -> str:
    name = np.dtype(dtype).name
    if name not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {name}")
    return name


def _field_spec_map(
    fields: Sequence[FieldSpec | Mapping[str, Any]], *, require_fields: bool = True
) -> dict[str, FieldSpec]:
    specs = [field if isinstance(field, FieldSpec) else FieldSpec(**field) for field in fields]
    if require_fields and not specs:
        raise ValueError("fields must contain at least one specification")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("fields must contain unique names")
    for spec in specs:
        if (
            not isinstance(spec.name, str)
            or not spec.name
            or spec.name.startswith(_RESERVED_PREFIX)
            or not isinstance(spec.shape, tuple | list)
            or any(type(dimension) is not int or dimension <= 0 for dimension in spec.shape)
            or spec.dtype not in _SUPPORTED_DTYPES
        ):
            raise ValueError(f"Invalid field specification: {spec.name!r}")
        dtype = _SUPPORTED_DTYPES[spec.dtype]
        if spec.semantic == "image":
            if spec.layout not in ("CHW", "HWC"):
                raise ValueError(f"Image field {spec.name!r} must declare layout CHW or HWC")
            channel_axis = 0 if spec.layout == "CHW" else -1
            if len(spec.shape) != 3 or spec.shape[channel_axis] not in (1, 3, 4):
                raise ValueError(f"Image field {spec.name!r} must have 1, 3, or 4 channels in {spec.layout} layout")
            if dtype != np.dtype("uint8") and not np.issubdtype(dtype, np.floating):
                raise ValueError(f"Image field {spec.name!r} must use a floating-point or uint8 dtype")
        elif spec.semantic != "tensor" or spec.layout:
            raise ValueError(f"Non-image field {spec.name!r} must use semantic='tensor' and an empty layout")
    return {spec.name: spec for spec in specs}


def _validate_declared_array(name: str, array: np.ndarray, spec: FieldSpec) -> None:
    if array.shape != tuple(spec.shape) or array.dtype != _SUPPORTED_DTYPES[spec.dtype]:
        raise ValueError(
            f"Field {name!r} does not match declared shape/dtype: expected {tuple(spec.shape)} {spec.dtype}, "
            f"got {array.shape} {array.dtype}"
        )
    if (
        spec.semantic == "image"
        and np.issubdtype(array.dtype, np.floating)
        and (not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1))
    ):
        raise ValueError(f"Floating-point image field {name!r} must contain finite values in [0, 1]")


def _canonical_image(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "CHW":
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(array * 255)
    return np.ascontiguousarray(array, dtype=np.uint8)


def _normalize_samples(samples: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    materialized = [dict(sample) for sample in samples]
    if not materialized:
        raise ValueError("Observation batch must contain at least one sample")
    fields = list(materialized[0])
    if not fields:
        raise ValueError("Observation samples must contain at least one field")
    if any(name.startswith(_RESERVED_PREFIX) for name in fields):
        raise ValueError(f"Field names beginning with {_RESERVED_PREFIX!r} are reserved")
    expected = set(fields)
    for index, sample in enumerate(materialized):
        if set(sample) != expected:
            raise ValueError(f"Sample {index} fields differ from the first sample")
    return materialized, fields


def _string_tensor(values: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    table: list[str] = []
    indices: dict[str, int] = {}
    encoded = []
    for value in values:
        if value not in indices:
            indices[value] = len(table)
            table.append(value)
        encoded.append(indices[value])
    return np.asarray(encoded, dtype=np.int64), table


def save_observation_batch(
    path: str | os.PathLike[str],
    samples: Iterable[Mapping[str, Any]],
    *,
    field_specs: Sequence[FieldSpec | Mapping[str, Any]],
    force: bool = False,
    provenance: Mapping[str, Any] | None = None,
    sample_provenance: Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Validate and atomically save raw samples in schema v1."""
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise RuntimeError("Saving observation batches requires safetensors") from exc

    destination = Path(path)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, names = _normalize_samples(samples)
    specs = _field_spec_map(field_specs, require_fields=False)
    tensors: dict[str, np.ndarray] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    task_values: list[str] | None = None
    numeric_fields: set[str] = set()

    for name in names:
        values = [row[name] for row in rows]
        if all(isinstance(value, str) for value in values):
            if name in specs:
                raise ValueError(f"Field spec {name!r} describes a string field")
            encoded, table = _string_tensor(values)
            if name == "task":
                task_values = table
                tensors[TASK_INDEX_TENSOR] = encoded
                descriptors[name] = {
                    "shape": [],
                    "dtype": "string",
                    "semantic": "task",
                    "layout": "",
                    "value_encoding": "task-index",
                }
            else:
                tensors[name] = encoded
                descriptors[name] = {
                    "shape": [],
                    "dtype": "string",
                    "semantic": "text",
                    "layout": "",
                    "value_encoding": "string-table",
                    "values": table,
                }
            continue
        if any(isinstance(value, str) for value in values):
            raise ValueError(f"Field {name!r} mixes strings and tensors")
        if name not in specs:
            raise ValueError(f"Numeric field {name!r} requires an explicit field spec")
        spec = specs[name]
        numeric_fields.add(name)
        arrays = [_numpy(value) for value in values]
        for array in arrays:
            _validate_declared_array(name, array, spec)
        if spec.semantic == "image":
            arrays = [_canonical_image(array, spec.layout) for array in arrays]
            shape = arrays[0].shape
            dtype_name = "uint8"
        else:
            shape = tuple(spec.shape)
            dtype_name = spec.dtype
        tensors[name] = np.ascontiguousarray(np.stack(arrays))
        descriptors[name] = {
            "shape": list(shape),
            "dtype": dtype_name,
            "semantic": spec.semantic,
            "layout": "HWC" if spec.semantic == "image" else "",
            "value_encoding": "uint8-0-255" if spec.semantic == "image" else "raw",
        }
    if set(specs) != numeric_fields:
        raise ValueError(f"Field specs do not match numeric sample fields: {sorted(set(specs) - numeric_fields)}")

    provenance_tensors = (
        ("dataset_index", DATASET_INDEX_TENSOR),
        ("episode_index", EPISODE_INDEX_TENSOR),
        ("frame_index", FRAME_INDEX_TENSOR),
    )
    for key, tensor_name in provenance_tensors:
        if sample_provenance and key in sample_provenance:
            values = np.asarray(sample_provenance[key], dtype=np.int64)
            if values.shape != (len(rows),):
                raise ValueError(f"sample_provenance[{key!r}] must have shape ({len(rows)},)")
            tensors[tensor_name] = values

    metadata = {
        FORMAT_KEY: FORMAT_NAME,
        VERSION_KEY: str(SCHEMA_VERSION),
        FIELDS_KEY: json.dumps(descriptors, sort_keys=True, separators=(",", ":")),
        TASKS_KEY: json.dumps(task_values or [], ensure_ascii=False, separators=(",", ":")),
        PROVENANCE_KEY: json.dumps(dict(provenance or {}), sort_keys=True, ensure_ascii=False, separators=(",", ":")),
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata=metadata)
        load_observation_batch(temporary)
        if destination.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
        os.replace(temporary, destination)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _load_legacy_json(path: Path) -> ObservationBatch:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    provenance: dict[str, Any] = {}
    if isinstance(payload, dict):
        provenance = payload.get("provenance", {})
        payload = payload.get("samples", payload.get("observations"))
    if not isinstance(payload, list) or not all(isinstance(sample, dict) for sample in payload):
        raise ValueError("Legacy JSON must be a sample list or an object containing 'samples'")
    return ObservationBatch(samples=payload, fields={}, provenance=provenance, sample_provenance={})


def load_observation_batch(path: str | os.PathLike[str]) -> ObservationBatch:
    """Load and strictly validate Safetensors v1, or read a legacy JSON batch."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        return _load_legacy_json(source)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Loading Safetensors observation batches requires safetensors") from exc

    with safe_open(source, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        if metadata.get(FORMAT_KEY) != FORMAT_NAME:
            raise ValueError(f"Not an {FORMAT_NAME} file")
        if metadata.get(VERSION_KEY) != str(SCHEMA_VERSION):
            raise ValueError(f"Unsupported observation batch schema version: {metadata.get(VERSION_KEY)!r}")
        try:
            fields = json.loads(metadata[FIELDS_KEY])
            tasks = json.loads(metadata[TASKS_KEY])
            provenance = json.loads(metadata.get(PROVENANCE_KEY, "{}"))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid observation batch metadata") from exc
        if not isinstance(fields, dict) or not fields or not isinstance(provenance, dict):
            raise ValueError("Field metadata must be a non-empty object")
        tensor_names = handle.keys()
        tensors = {key: handle.get_tensor(key) for key in tensor_names}

    sample_count: int | None = None
    columns: dict[str, Any] = {}
    required_descriptor_keys = {"shape", "dtype", "semantic", "layout", "value_encoding"}
    for name, descriptor in fields.items():
        if (
            not isinstance(name, str)
            or not isinstance(descriptor, dict)
            or not required_descriptor_keys <= descriptor.keys()
        ):
            raise ValueError(f"Invalid descriptor for field {name!r}")
        tensor_name = TASK_INDEX_TENSOR if descriptor["value_encoding"] == "task-index" else name
        if tensor_name not in tensors:
            raise ValueError(f"Missing tensor for field {name!r}")
        tensor = tensors[tensor_name]
        if tensor.ndim < 1:
            raise ValueError(f"Tensor {tensor_name!r} has no leading N dimension")
        sample_count = tensor.shape[0] if sample_count is None else sample_count
        if tensor.shape[0] != sample_count:
            raise ValueError("All tensors must have the same leading N dimension")
        encoding = descriptor["value_encoding"]
        if encoding == "task-index":
            if (
                descriptor["dtype"] != "string"
                or not isinstance(tasks, list)
                or not all(isinstance(task, str) for task in tasks)
                or tensor.dtype != np.int64
                or tensor.ndim != 1
                or np.any(tensor < 0)
                or np.any(tensor >= len(tasks))
            ):
                raise ValueError("Invalid task mapping or task indices")
            columns[name] = [tasks[int(index)] for index in tensor]
        elif encoding == "string-table":
            values = descriptor.get("values")
            if (
                descriptor["dtype"] != "string"
                or not isinstance(values, list)
                or not all(isinstance(value, str) for value in values)
                or tensor.dtype != np.int64
                or tensor.ndim != 1
                or np.any(tensor < 0)
                or np.any(tensor >= len(values))
            ):
                raise ValueError(f"Invalid string mapping for field {name!r}")
            columns[name] = [values[int(index)] for index in tensor]
        elif encoding in ("raw", "uint8-0-255"):
            expected_shape = tuple(descriptor["shape"])
            if tensor.shape[1:] != expected_shape or _dtype_name(tensor.dtype) != descriptor["dtype"]:
                raise ValueError(f"Tensor {name!r} does not match its shape/dtype metadata")
            if descriptor["semantic"] == "image":
                if (
                    descriptor["layout"] != "HWC"
                    or descriptor["dtype"] != "uint8"
                    or descriptor["value_encoding"] != "uint8-0-255"
                    or len(expected_shape) != 3
                    or expected_shape[-1] not in (1, 3, 4)
                ):
                    raise ValueError(f"Image field {name!r} is not canonical HWC uint8")
            elif descriptor["semantic"] != "tensor" or descriptor["layout"] or encoding != "raw":
                raise ValueError(f"Non-image field {name!r} is not a raw tensor with empty layout")
            columns[name] = tensor
        else:
            raise ValueError(f"Unknown value encoding for field {name!r}: {encoding!r}")
    known_tensors = {
        TASK_INDEX_TENSOR,
        EPISODE_INDEX_TENSOR,
        FRAME_INDEX_TENSOR,
        DATASET_INDEX_TENSOR,
        *(name for name, descriptor in fields.items() if descriptor["value_encoding"] != "task-index"),
    }
    if set(tensors) - known_tensors:
        raise ValueError(f"Undeclared tensors: {sorted(set(tensors) - known_tensors)}")
    sample_provenance: dict[str, np.ndarray] = {}
    provenance_tensors = (
        ("dataset_index", DATASET_INDEX_TENSOR),
        ("episode_index", EPISODE_INDEX_TENSOR),
        ("frame_index", FRAME_INDEX_TENSOR),
    )
    for key, tensor_name in provenance_tensors:
        if tensor_name not in tensors:
            continue
        tensor = tensors[tensor_name]
        if tensor.dtype != np.int64 or tensor.shape != (sample_count,):
            raise ValueError(f"Provenance tensor {tensor_name!r} must be int64 with shape ({sample_count},)")
        sample_provenance[key] = tensor
    samples = [{name: values[index] for name, values in columns.items()} for index in range(sample_count or 0)]
    return ObservationBatch(
        samples=samples,
        fields=fields,
        provenance=provenance,
        sample_provenance=sample_provenance,
    )


def iter_observation_batch(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Iterate raw sample dictionaries; reserved provenance tensors are omitted."""
    yield from load_observation_batch(path).samples


def generate_random_observations(
    fields: Sequence[FieldSpec | Mapping[str, Any]], num_samples: int, *, seed: int = 0
) -> list[dict[str, Any]]:
    """Generate deterministic tensors from field shapes, dtypes, and inclusive ranges."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    specs = list(_field_spec_map(fields).values())
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    for spec in specs:
        dtype = _SUPPORTED_DTYPES.get(spec.dtype)
        if dtype is None or spec.minimum > spec.maximum:
            raise ValueError(f"Invalid dtype or range for field {spec.name!r}")
        if spec.semantic == "image" and np.issubdtype(dtype, np.floating) and (spec.minimum < 0 or spec.maximum > 1):
            raise ValueError(f"Floating-point image range for {spec.name!r} must be within [0, 1]")
        size = (num_samples, *spec.shape)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            low, high = int(spec.minimum), int(spec.maximum)
            if low < info.min or high > info.max:
                raise ValueError(f"Range for {spec.name!r} exceeds {spec.dtype}")
            columns[spec.name] = rng.integers(low, high, endpoint=True, size=size, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            columns[spec.name] = rng.integers(0, 2, size=size, dtype=np.uint8).astype(bool)
        else:
            columns[spec.name] = rng.uniform(float(spec.minimum), float(spec.maximum), size=size).astype(dtype)
    return [{name: column[index] for name, column in columns.items()} for index in range(num_samples)]


def _scalar(value: Any) -> int:
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar index, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def _episode_groups(dataset: Any) -> dict[int, list[int]]:
    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is not None:
        groups = {}
        for episode_index, episode in enumerate(episodes):
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])
            groups[episode_index] = list(range(start, stop))
        return groups
    source = getattr(dataset, "hf_dataset", None)
    if source is None and getattr(dataset, "reader", None) is not None:
        source = getattr(dataset.reader, "hf_dataset", None)
    if source is not None:
        episodes = source["episode_index"]
        return _group_episode_values(episodes)
    episodes = getattr(dataset, "episode_indices", None)
    if episodes is not None:
        return _group_episode_values(episodes)
    values = []
    for index in range(len(dataset)):
        item = dataset[index]
        if isinstance(item, tuple):
            item = item[0]
        values.append(_scalar(item["episode_index"]))
    return _group_episode_values(values)


def _group_episode_values(values: Iterable[Any]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(values):
        groups.setdefault(_scalar(value), []).append(index)
    return groups


def select_dataset_indices(
    episode_groups: Mapping[int, Sequence[int]],
    num_samples: int,
    *,
    seed: int = 0,
    strategy: str = "episode-stratified",
) -> list[int]:
    """Select unique frame indices with stratified, global-random, or global-uniform sampling."""
    all_indices = [index for indices in episode_groups.values() for index in indices]
    if num_samples <= 0 or num_samples > len(all_indices):
        raise ValueError(f"num_samples must be in [1, {len(all_indices)}]")
    rng = np.random.default_rng(seed)
    if strategy == "global-random":
        return [int(value) for value in rng.choice(all_indices, num_samples, replace=False)]
    if strategy == "global-uniform":
        positions = np.linspace(0, len(all_indices) - 1, num_samples).round().astype(int)
        return [all_indices[position] for position in positions]
    if strategy != "episode-stratified":
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    episodes = list(episode_groups)
    if num_samples < len(episodes):
        selected_episodes = [int(value) for value in rng.choice(episodes, num_samples, replace=False)]
        return [int(rng.choice(episode_groups[episode])) for episode in selected_episodes]
    allocation = {episode: 1 for episode in episodes}
    remaining_count = num_samples - len(episodes)
    while remaining_count:
        eligible = [episode for episode in episodes if allocation[episode] < len(episode_groups[episode])]
        if not eligible:
            raise ValueError("Not enough unique frames to satisfy episode-stratified sampling")
        for episode in rng.permutation(eligible):
            if not remaining_count:
                break
            allocation[int(episode)] += 1
            remaining_count -= 1
    selected = []
    for episode in episodes:
        frames = rng.choice(episode_groups[episode], allocation[episode], replace=False)
        selected.extend(sorted(int(frame) for frame in frames))
    return selected


def extract_lerobot_observations(
    root: str | os.PathLike[str],
    num_samples: int,
    *,
    policy_path: str | os.PathLike[str],
    fields: Sequence[str] | None = None,
    seed: int = 0,
    strategy: str = "episode-stratified",
    repo_id: str = "local/observation-batch",
) -> tuple[list[dict[str, Any]], list[FieldSpec], dict[str, Any], dict[str, list[int]]]:
    """Extract selected raw fields from a local LeRobot dataset using explicit PyAV decoding."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot extraction requires the lerobot package") from exc
    policy_root = Path(policy_path).resolve()
    config_path = policy_root / "config.json"
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"Policy bundle is missing config.json: {config_path}") from exc
    input_features = config.get("input_features") if isinstance(config, dict) else None
    if not isinstance(input_features, dict) or not input_features:
        raise ValueError(f"Policy config must contain a non-empty input_features object: {config_path}")
    policy_specs: list[FieldSpec] = []
    for name, feature in input_features.items():
        if not isinstance(name, str) or not name or not isinstance(feature, dict):
            raise ValueError(f"Invalid policy input feature in {config_path}: {name!r}")
        feature_type = feature.get("type")
        shape = feature.get("shape")
        if not isinstance(feature_type, str) or not isinstance(shape, list):
            raise ValueError(f"Policy input feature {name!r} must declare type and shape")
        normalized_type = feature_type.upper()
        if normalized_type not in {"VISUAL", "STATE", "ENV", "ACTION"}:
            raise ValueError(f"Unsupported policy input feature type for {name!r}: {feature_type!r}")
        is_visual = normalized_type == "VISUAL"
        policy_specs.append(
            FieldSpec(
                name=name,
                shape=tuple(shape),
                dtype="float32",
                semantic="image" if is_visual else "tensor",
                layout="CHW" if is_visual else "",
            )
        )
    policy_spec_map = _field_spec_map(policy_specs)
    if fields is not None:
        selected = list(fields)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Explicit fields must contain unique names")
        unknown = set(selected) - set(policy_spec_map) - {"task"}
        if unknown:
            raise ValueError(f"Selected fields are not policy input features: {sorted(unknown)}")
    else:
        selected = list(policy_spec_map)
    selected_specs = [policy_spec_map[name] for name in selected if name != "task"]

    dataset = LeRobotDataset(repo_id=repo_id, root=Path(root), video_backend="pyav")
    indices = select_dataset_indices(_episode_groups(dataset), num_samples, seed=seed, strategy=strategy)
    samples: list[dict[str, Any]] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    dataset_indices: list[int] = []
    include_default_task: bool | None = None
    for index in indices:
        item = dataset[index]
        if isinstance(item, tuple):
            item = next((part for part in item if isinstance(part, Mapping)), None)
        if not isinstance(item, Mapping):
            raise TypeError(f"LeRobot dataset item {index} is not a mapping")
        episode_indices.append(_scalar(item.get("episode_index", -1)))
        frame_indices.append(_scalar(item.get("frame_index", index)))
        dataset_indices.append(_scalar(item.get("index", index)))
        names = selected
        if fields is None:
            has_task = "task" in item
            if include_default_task is None:
                include_default_task = has_task
            elif has_task != include_default_task:
                raise KeyError("Dataset items have inconsistent task fields")
            if include_default_task:
                names = [*selected, "task"]
        missing = set(names) - set(item)
        if missing:
            raise KeyError(f"Dataset item is missing selected fields: {sorted(missing)}")
        sample = {name: item[name] for name in names}
        for name, spec in policy_spec_map.items():
            if name in sample:
                _validate_declared_array(name, _numpy(sample[name]), spec)
        samples.append(sample)
    provenance = {
        "source": "lerobot-local",
        "root": str(Path(root).resolve()),
        "policy_path": str(policy_root),
        "repo_id": repo_id,
        "sampling_strategy": strategy,
        "seed": seed,
        "selected_indices": indices,
        "video_backend": "pyav",
    }
    return (
        samples,
        selected_specs,
        provenance,
        {
            "dataset_index": dataset_indices,
            "episode_index": episode_indices,
            "frame_index": frame_indices,
        },
    )
