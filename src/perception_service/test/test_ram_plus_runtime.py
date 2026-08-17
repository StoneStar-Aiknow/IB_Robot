import json

import numpy as np
import pytest

from inference_service.generic_runtime import DeploymentIdentity, NamedTensorResult, RuntimeLatency
from perception_service.ram_plus_adapter import (
    RAM_PLUS_POSTPROCESSING,
    RAM_PLUS_PREPROCESSING,
    RAMPlusAdapter,
    RecognizedTag,
    select_mask_tags,
)


def _write_assets(root, preprocessing=RAM_PLUS_PREPROCESSING) -> None:
    assets = root / "assets"
    assets.mkdir()
    (assets / "adapter.json").write_text(
        json.dumps(
            {
                "family": "ram_plus",
                "preprocessing": preprocessing,
                "postprocessing": RAM_PLUS_POSTPROCESSING,
                "torch_module_loader": "perception_service.torch_model_loaders:load_ram_plus",
            }
        ),
        encoding="utf-8",
    )
    (assets / "ram_tag_list.txt").write_text(
        "\n".join(["cup", "table", *(f"tag-{index}" for index in range(2, 4585))]) + "\n",
        encoding="utf-8",
    )
    thresholds = np.full(4585, 0.99, dtype=np.float32)
    thresholds[:2] = [0.4, 0.8]
    np.savetxt(assets / "ram_tag_list_threshold.txt", thresholds)


def _result(logits) -> NamedTensorResult:
    rows = np.asarray(logits, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    values = np.full((len(rows), 4585), -80.0, dtype=np.float32)
    values[:, : rows.shape[1]] = rows
    return NamedTensorResult(
        outputs={"tag_logits": values},
        deployment=DeploymentIdentity("bundle", "uuid", 1, "torch_cpu", "uuid", 1, "fingerprint", "torch"),
        latency=RuntimeLatency(1.0, 1.0),
    )


def test_ram_plus_adapter_preprocesses_and_decodes_deterministically(tmp_path) -> None:
    _write_assets(tmp_path)
    adapter = RAMPlusAdapter.from_bundle(tmp_path)
    tensor = adapter.preprocess(np.zeros((12, 20, 3), dtype=np.uint8))["observation.image"]
    tags = adapter.postprocess(_result([1.0, 2.0]))

    assert tensor.shape == (1, 3, 384, 384)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert [tag.label for tag in tags] == ["table", "cup"]
    assert adapter.postprocess(_result([])) == []


def test_ram_plus_adapter_preserves_dynamic_batch_order(tmp_path) -> None:
    _write_assets(tmp_path)
    adapter = RAMPlusAdapter.from_bundle(tmp_path)
    tensor = adapter.preprocess_batch([np.zeros((12, 20, 3), dtype=np.uint8), np.full((8, 9, 3), 255, dtype=np.uint8)])[
        "observation.image"
    ]
    decoded = adapter.postprocess_batch(_result([[2.0, -2.0], [-2.0, 2.0]]))

    assert tensor.shape == (2, 3, 384, 384)
    assert [[tag.label for tag in row] for row in decoded] == [["cup"], ["table"]]


def test_ram_plus_mask_tags_apply_policy_before_candidate_limit():
    values = [
        RecognizedTag("yellow", 0.99),
        RecognizedTag("banana", 0.95),
        RecognizedTag("fruit", 0.8),
        RecognizedTag("food", 0.7),
        RecognizedTag("produce", 0.6),
    ]

    assert [value.label for value in select_mask_tags(values, excluded_labels=["fruit", "food"], limit=2)] == [
        "banana",
        "produce",
    ]


def test_ram_plus_bundle_rejects_adapter_identity_drift(tmp_path) -> None:
    _write_assets(tmp_path, preprocessing="different-preprocessing")

    with pytest.raises(ValueError, match="adapter identity mismatch"):
        RAMPlusAdapter.from_bundle(tmp_path)
