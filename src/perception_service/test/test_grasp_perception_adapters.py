import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import Image

from ibrobot_msgs.msg import Detection2D, DetectionArray
from inference_service.backends import BackendState
from inference_service.generic_runtime import DeploymentIdentity, NamedTensorResult, RuntimeLatency
from perception_service.model_service_plugins import SegmentDetectionsPlugin
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


def _compiled(roles, execution):
    """Build a compiled deployment carrying only the bindings a test cares about."""
    from inference_manifest import ArtifactBindings, CompiledDeployment, DeploymentArtifact, DeploymentTarget

    return CompiledDeployment(
        backend="ascend",
        target=DeploymentTarget(soc="Ascend310P1", runtime="ascend"),
        artifacts={role: DeploymentArtifact(path=f"artifacts/test/{role}.om", format="om") for role in execution},
        execution=tuple(execution),
        bindings={
            role: ArtifactBindings(inputs=tuple(inputs), outputs=tuple(outputs))
            for role, (inputs, outputs) in roles.items()
        },
    )


def _write_sam2_prompt_bundle(root):
    assets = root / "assets"
    assets.mkdir(parents=True)
    (assets / "adapter.json").write_text(
        json.dumps(
            {
                "family": "sam2_prompt",
                "preprocessing": "sam2-longest-side1024-imagenet-box-prompt-v1",
                "postprocessing": "sam2-mask-logits-iou-v1",
            }
        ),
        encoding="utf-8",
    )


def _sam2_prompt_deployment(*, batch: int, canvas: int, mask_batch: int | None = None):
    from inference_manifest import TensorBinding

    mask = batch if mask_batch is None else mask_batch
    return _compiled(
        {
            "encoder": (
                [
                    TensorBinding(
                        semantic="image", index=0, dtype="float32", shape=(1, 3, canvas, canvas), layout="NCHW"
                    )
                ],
                [
                    TensorBinding(
                        semantic="internal.image_embed", index=0, dtype="float32", shape=(1, 256, 64, 64), layout="NCHW"
                    )
                ],
            ),
            "decoder": (
                [
                    TensorBinding(
                        semantic="internal.image_embed", index=0, dtype="float32", shape=(1, 256, 64, 64), layout="NCHW"
                    ),
                    TensorBinding(semantic="point_coords", index=1, dtype="float32", shape=(batch, 2, 2)),
                    TensorBinding(semantic="point_labels", index=2, dtype="int8", shape=(batch, 2)),
                    TensorBinding(
                        semantic="mask_input",
                        index=3,
                        dtype="float32",
                        shape=(mask, 1, canvas // 4, canvas // 4),
                        layout="NCHW",
                    ),
                ],
                [
                    TensorBinding(
                        semantic="mask_logits",
                        index=0,
                        dtype="float32",
                        shape=(batch, 1, canvas // 4, canvas // 4),
                        layout="NCHW",
                    )
                ],
            ),
        },
        ("encoder", "decoder"),
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
    assert records[0].mask is None


def test_grounding_dino_raw_adapter_derives_abi_shapes_from_manifest(tmp_path):
    from inference_manifest import ModelDescriptor, SemanticTensor

    _write_grounding_bundle(tmp_path)
    model = ModelDescriptor(
        kind="perception",
        family="grounding_dino_raw",
        inputs=(
            SemanticTensor(semantic="input_ids", dtype="int64", shape=(1, 8)),
            SemanticTensor(semantic="encoder_tgt", dtype="float32", shape=(1, 900, 256)),
        ),
        outputs=(SemanticTensor(semantic="pred_logits", dtype="float32", shape=(1, 900, 256)),),
    )
    adapter = GroundingDINORawAdapter.from_bundle(tmp_path, model=model)

    assert adapter.tokenizer.sequence_length == 8
    assert adapter.encoder_target.shape == (1, 900, 256)


def test_grounding_dino_raw_adapter_tokenizes_prompt_once_per_call(tmp_path, monkeypatch):
    _write_grounding_bundle(tmp_path)
    adapter = GroundingDINORawAdapter.from_bundle(tmp_path)
    calls = []
    original_encode = adapter.tokenizer.encode

    def _counting_encode(text):
        calls.append(text)
        return original_encode(text)

    monkeypatch.setattr(adapter.tokenizer, "encode", _counting_encode)

    logits = np.full((1, 2, 256), -10.0, dtype=np.float32)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 1] = 10.0
    adapter.postprocess(
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

    assert calls == ["banana"]


def test_grounding_dino_raw_adapter_ignores_padding_logit_columns(tmp_path):
    """Columns past the prompt token window are padding and must never score a box."""
    _write_grounding_bundle(tmp_path)
    adapter = GroundingDINORawAdapter.from_bundle(tmp_path)
    logits = np.full((1, 2, 256), -10.0, dtype=np.float32)
    logits[0, :, adapter.tokenizer.sequence_length :] = 10.0

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

    assert records == []


def test_grounding_dino_raw_adapter_scores_and_decodes_from_prompt_token_columns(tmp_path):
    """Column 1 of the seq-8 window holds the 'banana' token for this bundle vocabulary."""
    _write_grounding_bundle(tmp_path)
    adapter = GroundingDINORawAdapter.from_bundle(tmp_path)
    logits = np.full((1, 1, 256), -10.0, dtype=np.float32)
    # Columns 0/2/3 are [CLS]/./[SEP]; a high logit there must not leak into the label.
    logits[0, 0, [0, 2, 3]] = 10.0
    logits[0, 0, 1] = 10.0

    records = adapter.postprocess(
        _result(
            {
                "pred_logits": logits,
                "pred_boxes": np.asarray([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32),
            }
        ),
        image_shape=(360, 640),
        prompt="banana",
        box_threshold=0.3,
        text_threshold=0.25,
    )

    assert len(records) == 1
    assert records[0].label == "banana"
    assert records[0].confidence == pytest.approx(1.0 / (1.0 + np.exp(-10.0)), rel=1e-6)


def test_grounding_dino_raw_adapter_rejects_a_token_window_wider_than_the_logits(tmp_path):
    _write_grounding_bundle(tmp_path)
    adapter = GroundingDINORawAdapter.from_bundle(tmp_path)
    logits = np.zeros((1, 1, 4), dtype=np.float32)

    with pytest.raises(RuntimeError, match="prompt token columns"):
        adapter.postprocess(
            _result(
                {
                    "pred_logits": logits,
                    "pred_boxes": np.asarray([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32),
                }
            ),
            image_shape=(360, 640),
            prompt="banana",
        )


def test_grounding_dino_raw_adapter_feeds_the_image_in_the_declared_layout(tmp_path):
    """A deployment compiled NHWC at 360x640 must not be fed a 720x1280 NCHW tensor."""
    from inference_manifest import TensorBinding

    _write_grounding_bundle(tmp_path)
    deployment = _compiled(
        {
            "vision": (
                [
                    TensorBinding(semantic="image", index=0, dtype="float32", shape=(1, 360, 640, 3), layout="NHWC"),
                    TensorBinding(semantic="input_ids", index=1, dtype="int64", shape=(1, 8)),
                    TensorBinding(semantic="encoder_tgt", index=2, dtype="float32", shape=(1, 900, 256)),
                ],
                [TensorBinding(semantic="pred_boxes", index=0, dtype="float32", shape=(1, 900, 4))],
            )
        },
        ("vision",),
    )

    adapter = GroundingDINORawAdapter.from_bundle(tmp_path, deployment=deployment)
    inputs = adapter.preprocess((np.zeros((720, 1280, 3), dtype=np.uint8), "banana", 0.3, 0.25))

    assert adapter.image_layout == "NHWC"
    assert inputs["image"].shape == (1, 360, 640, 3)


def test_sam2_prompt_adapter_derives_batch_and_canvas_from_the_compiled_deployment(tmp_path):
    """`model.inputs` declares a -1 batch, so only the deployment bindings carry it."""
    _write_sam2_prompt_bundle(tmp_path)
    adapter = SAM2PromptAdapter.from_bundle(tmp_path, deployment=_sam2_prompt_deployment(batch=1, canvas=512))

    assert (adapter.batch_size, adapter.image_size) == (1, 512)

    inputs = adapter.preprocess((np.zeros((256, 512, 3), dtype=np.uint8), [[0, 0, 10, 10]]))

    assert inputs["image"].shape == (1, 3, 512, 512)
    assert inputs["point_coords"].shape == (1, 2, 2)
    assert inputs["point_labels"].shape == (1, 2)
    assert inputs["mask_input"].shape == (1, 1, 128, 128)
    assert inputs["has_mask_input"].shape == (1,)
    with pytest.raises(ValueError, match="only accepts 1"):
        adapter.preprocess((np.zeros((256, 512, 3), dtype=np.uint8), [[0, 0, 10, 10]] * 2))


def test_sam2_prompt_adapter_scales_prompts_and_masks_to_the_declared_canvas(tmp_path):
    _write_sam2_prompt_bundle(tmp_path)
    adapter = SAM2PromptAdapter.from_bundle(tmp_path, deployment=_sam2_prompt_deployment(batch=2, canvas=512))

    inputs = adapter.preprocess((np.zeros((256, 512, 3), dtype=np.uint8), [[0, 0, 128, 64]]))

    # A 512-wide source letterboxes onto a 512 canvas at scale 1.0, not 1024/512.
    np.testing.assert_allclose(inputs["point_coords"][0], [[0.0, 0.0], [128.0, 64.0]])

    logits = np.full((2, 1, 128, 128), -1.0, dtype=np.float32)
    logits[:, :, :64, :] = 1.0
    masks = adapter.postprocess(_result({"mask_logits": logits}), image_shape=(256, 512), count=1)

    # The source occupies the top half of the square canvas, so the whole crop is mask.
    assert masks[0].shape == (256, 512)
    assert masks[0].all()


def test_sam2_prompt_adapter_rejects_a_deployment_whose_mask_batch_disagrees(tmp_path):
    _write_sam2_prompt_bundle(tmp_path)
    deployment = _sam2_prompt_deployment(batch=4, canvas=1024, mask_batch=1)

    with pytest.raises(ValueError, match="mask_input batch"):
        SAM2PromptAdapter.from_bundle(tmp_path, deployment=deployment)


def test_sam2_prompt_preprocess_rejects_batch_larger_than_compiled_size():
    adapter = SAM2PromptAdapter()
    adapter.batch_size = 4
    image = np.zeros((512, 1024, 3), dtype=np.uint8)
    boxes = [[0, 0, 10, 10]] * 5

    with pytest.raises(ValueError, match="only accepts 4"):
        adapter.preprocess((image, boxes))


_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 512
_PROMPT_SCALE = 1024.0 / _IMAGE_WIDTH


class _Bridge:
    """cv_bridge stand-in covering only the rgb8/mono8 conversions the plugin performs."""

    def imgmsg_to_cv2(self, image, desired_encoding=None):
        del desired_encoding
        return np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(image.height, image.width, 3)

    def cv2_to_imgmsg(self, array, encoding=""):
        message = Image()
        message.height, message.width = array.shape[:2]
        message.encoding = encoding
        message.step = array.shape[1]
        message.data = array.tobytes()
        return message


class _StubSession:
    """Records every compiled-decoder invocation so batching can be asserted."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.prompts = []

    def infer(self, request):
        self.prompts.append(np.asarray(request.inputs["point_coords"]).copy())
        return _result({"mask_logits": np.ones((self.batch_size, 1, 256, 256), dtype=np.float32)})


def _segment_plugin(batch_size: int):
    adapter = SAM2PromptAdapter()
    adapter.batch_size = batch_size
    plugin = object.__new__(SegmentDetectionsPlugin)
    plugin.adapter = adapter
    plugin.host = SimpleNamespace(bridge=_Bridge())
    plugin.session = _StubSession(batch_size)
    plugin._requests = itertools.count(1)
    return plugin


def _detection_request(count: int):
    image = Image()
    image.height = _IMAGE_HEIGHT
    image.width = _IMAGE_WIDTH
    image.encoding = "rgb8"
    image.step = _IMAGE_WIDTH * 3
    image.data = bytes(_IMAGE_HEIGHT * _IMAGE_WIDTH * 3)
    detections = []
    for index in range(count):
        detection = Detection2D()
        detection.header = image.header
        detection.label = f"object-{index}"
        detection.confidence = 1.0 - index / 100.0
        detection.bbox = [float(index), float(index), float(index + 20), float(index + 20)]
        detections.append(detection)
    return SimpleNamespace(image=image, detections=DetectionArray(header=image.header, detections=detections))


def test_segment_detections_covers_every_box_in_compiled_decoder_batches():
    """5 boxes against a batch-4 decoder must run twice and keep the detection order."""
    plugin = _segment_plugin(4)
    request = _detection_request(5)
    response = SimpleNamespace()

    message = plugin.handle(request, response)

    assert message == "segmented 5 detections"
    assert len(plugin.session.prompts) == 2
    assert [detection.label for detection in response.detections.detections] == [
        "object-0",
        "object-1",
        "object-2",
        "object-3",
        "object-4",
    ]
    assert all(
        detection.mask.height == _IMAGE_HEIGHT and detection.mask.width == _IMAGE_WIDTH
        for detection in response.detections.detections
    )


def test_segment_detections_sends_each_box_to_its_own_decoder_slot():
    plugin = _segment_plugin(4)
    request = _detection_request(6)

    plugin.handle(request, SimpleNamespace())

    first, second = plugin.session.prompts
    boxes = [detection.bbox[0] * _PROMPT_SCALE for detection in request.detections.detections]
    assert [float(first[index][0][0]) for index in range(4)] == pytest.approx(boxes[:4])
    assert [float(second[index][0][0]) for index in range(2)] == pytest.approx(boxes[4:])
    # The compiled decoder always consumes batch_size slots; the tail repeats the last
    # real box so the padding never introduces an extra detection in the response.
    assert [float(second[index][0][0]) for index in range(2, 4)] == pytest.approx([boxes[5]] * 2)


def test_segment_detections_returns_an_empty_array_without_touching_the_decoder():
    plugin = _segment_plugin(4)
    response = SimpleNamespace()

    assert plugin.handle(_detection_request(0), response) == "segmented 0 detections"
    assert plugin.session.prompts == []
    assert list(response.detections.detections) == []


def test_segment_detections_rejects_more_detections_than_the_service_contract():
    plugin = _segment_plugin(4)

    with pytest.raises(ValueError, match="exceeds limit"):
        plugin.handle(_detection_request(17), SimpleNamespace())


def test_sam2_prompt_postprocess_crops_top_left_padding_before_resize():
    adapter = SAM2PromptAdapter()
    logits = np.full((1, 1, 256, 256), -1.0, dtype=np.float32)
    logits[:, :, 128:, :] = 1.0

    masks = adapter.postprocess(_result({"mask_logits": logits}), image_shape=(512, 1024), count=1)

    assert masks[0].shape == (512, 1024)
    assert not masks[0].any()
