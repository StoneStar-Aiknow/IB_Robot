from dataclasses import dataclass, field

import pytest

from perception_service.model_contracts import ModelManifest, validate_mask_batch, validate_ready, validate_text_batch


@dataclass
class _Stamp:
    sec: int = 1
    nanosec: int = 2


@dataclass
class _Header:
    stamp: _Stamp = field(default_factory=_Stamp)


@dataclass
class _Image:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes
    header: _Header = field(default_factory=_Header)


def _rgb():
    return _Image(2, 3, "bgr8", 9, bytes(18))


def _mask():
    return _Image(2, 3, "mono8", 3, bytes(6))


def test_model_manifest_fingerprint_is_stable_and_sensitive():
    manifest = ModelManifest("siglip2", "v1", "weights", "config", "cuda", "torch", "pre", 512, "l2")

    assert manifest.fingerprint() == manifest.fingerprint()
    assert (
        manifest.fingerprint()
        != ModelManifest("siglip2", "v2", "weights", "config", "cuda", "torch", "pre", 512, "l2").fingerprint()
    )


def test_mask_batch_accepts_matching_mono8_images():
    validate_mask_batch(_rgb(), [_mask()] * 8)


def test_mask_batch_rejects_ninth_mask():
    with pytest.raises(ValueError, match="exceeds limit"):
        validate_mask_batch(_rgb(), [_mask()] * 9)


def test_mask_batch_rejects_dimension_and_timestamp_mismatch():
    wrong_size = _mask()
    wrong_size.width = 2
    with pytest.raises(ValueError, match="dimensions"):
        validate_mask_batch(_rgb(), [wrong_size])

    wrong_stamp = _mask()
    wrong_stamp.header.stamp.sec = 4
    with pytest.raises(ValueError, match="timestamp"):
        validate_mask_batch(_rgb(), [wrong_stamp])


def test_readiness_fails_closed():
    with pytest.raises(RuntimeError, match="missing model"):
        validate_ready(False, "missing model")


def test_text_batch_is_non_empty_bounded_and_contains_text():
    validate_text_batch(["cup", "red bottle"])
    with pytest.raises(ValueError, match="at least one"):
        validate_text_batch([])
    with pytest.raises(ValueError, match="limit 16"):
        validate_text_batch(["cup"] * 17)
    with pytest.raises(ValueError, match="text 1"):
        validate_text_batch(["cup", "  "])
