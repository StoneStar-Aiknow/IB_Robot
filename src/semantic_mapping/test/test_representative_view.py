import numpy as np

from semantic_mapping.representative_view import OptionalCaptioner, RepresentativeViewStore


def _view(store, stamp, confidence, size=4):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6] = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2 : 2 + size, 2 : 2 + size] = 1
    return store.create("object", stamp, confidence, image, mask, np.array([2, 2, 6, 6]))


def test_representative_selection_uses_confidence_area_sharpness_and_timestamp():
    store = RepresentativeViewStore()
    assert store.consider(_view(store, 20, 0.8, 2))
    assert store.consider(_view(store, 30, 0.9, 2))
    assert not store.consider(_view(store, 10, 0.8, 4))

    assert store.get("object").confidence == 0.9
    assert store.get("object").stamp_ns == 30


def test_caption_failure_is_recorded_without_raising():
    class Client:
        @staticmethod
        def chat(prompt, image, clear_history):
            raise RuntimeError("VLM unavailable")

    store = RepresentativeViewStore()
    record = OptionalCaptioner(Client(), "vlm-hash").caption("object", _view(store, 20, 0.8), "describe")

    assert not record.success
    assert record.caption == ""
    assert record.message == "VLM unavailable"
