import json

import numpy as np

from inference_service.backends import BackendState
from inference_service.generic_runtime import DeploymentIdentity, NamedTensorResult, RuntimeLatency
from perception_service.semantic_model_adapters import GroundingDINORawAdapter, SAM2PromptAdapter


def _result(outputs):
    return NamedTensorResult(
        outputs=outputs,
        deployment=DeploymentIdentity(
            bundle="test",
            bundle_uuid="bundle-uuid",
            bundle_revision=1,
            deployment="ascend_310p",
            deployment_uuid="deployment-uuid",
            deployment_revision=1,
            deployment_fingerprint="fingerprint",
            backend="ascend",
        ),
        latency=RuntimeLatency(total_ms=1.0, backend_ms=1.0),
        state=BackendState.READY,
    )


def _write_grounding_bundle(root):
    assets = root / "assets"
    vocab = assets / "bert-base-uncased" / "vocab.txt"
    vocab.parent.mkdir(parents=True)
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n.\n?\nbanana\n", encoding="utf-8")
    (assets / "adapter.json").write_text(
        json.dumps(
            {
                "family": "grounding_dino_raw",
                "preprocessing": "grounding-dino-swint-rgb720x1280-bert-seq8-v1",
                "postprocessing": "grounding-dino-raw-logits-cxcywh-v1",
            }
        ),
        encoding="utf-8",
    )
    np.save(assets / "encoder_tgt.npy", np.zeros((1, 900, 256), dtype=np.float32))


def test_grounding_dino_raw_adapter_uses_fixed_token_window_and_returns_boxes(tmp_path):
    _write_grounding_bundle(tmp_path)
    adapter = GroundingDINORawAdapter.from_bundle(tmp_path)
    inputs = adapter.preprocess((np.zeros((360, 640, 3), dtype=np.uint8), "banana", 0.3, 0.25))

    assert inputs["image"].shape == (1, 3, 720, 1280)
    assert inputs["input_ids"].shape == (1, 8)
    assert inputs["encoder_tgt"].shape == (1, 900, 256)

    logits = np.full((1, 2, 256), -10.0, dtype=np.float32)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 20] = 10.0
    records = adapter.postprocess(
        _result(
            {
                "pred_logits": logits,
                "pred_boxes": np.asarray([[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.2, 0.2]]], dtype=np.float32),
            }
        ),
        image_shape=(360, 640),
        prompt="banana",
        box_threshold=0.3,
        text_threshold=0.25,
    )

    assert len(records) == 1
    assert records[0].label == "banana"
    np.testing.assert_allclose(records[0].bbox_xyxy, [160.0, 90.0, 480.0, 270.0])
    assert records[0].mask.shape == (360, 640)
    assert not records[0].mask.any()


def test_sam2_prompt_postprocess_crops_top_left_padding_before_resize():
    adapter = SAM2PromptAdapter()
    logits = np.full((1, 1, 256, 256), -1.0, dtype=np.float32)
    logits[:, :, 128:, :] = 1.0

    masks = adapter.postprocess(_result({"mask_logits": logits}), image_shape=(512, 1024), count=1)

    assert masks[0].shape == (512, 1024)
    assert not masks[0].any()
