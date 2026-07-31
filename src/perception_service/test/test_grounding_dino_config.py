import sys
from types import ModuleType

import perception_service.grounded_sam2_wrapper as wrapper_module
from perception_service.grounding_dino_config import GROUNDING_DINO_SWINT_OGC_CONFIG


def test_grounding_dino_swint_ogc_config_matches_checkpoint_architecture() -> None:
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["backbone"] == "swin_T_224_1k"
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["enc_layers"] == 6
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["dec_layers"] == 6
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["hidden_dim"] == 256
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["num_queries"] == 900
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["text_encoder_type"] == "bert-base-uncased"


def test_grounding_model_builds_from_a_copy_of_source_config(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeSLConfig:
        def __init__(self, values):
            captured["values"] = values
            self.__dict__.update(values)

    class FakeModel:
        def load_state_dict(self, state, strict):
            captured["state"] = state
            captured["strict"] = strict

        def eval(self):
            captured["evaluated"] = True

    groundingdino = ModuleType("groundingdino")
    models = ModuleType("groundingdino.models")
    util = ModuleType("groundingdino.util")
    slconfig = ModuleType("groundingdino.util.slconfig")
    utils = ModuleType("groundingdino.util.utils")
    fake_model = FakeModel()

    def build_model(args):
        captured["args"] = args
        return fake_model

    models.build_model = build_model
    slconfig.SLConfig = FakeSLConfig
    utils.clean_state_dict = lambda state: state
    monkeypatch.setitem(sys.modules, "groundingdino", groundingdino)
    monkeypatch.setitem(sys.modules, "groundingdino.models", models)
    monkeypatch.setitem(sys.modules, "groundingdino.util", util)
    monkeypatch.setitem(sys.modules, "groundingdino.util.slconfig", slconfig)
    monkeypatch.setitem(sys.modules, "groundingdino.util.utils", utils)
    monkeypatch.setattr(wrapper_module, "_patch_transformers_bert_for_groundingdino", lambda: None)
    monkeypatch.setattr(wrapper_module.torch, "load", lambda *_args, **_kwargs: {"model": {}})

    model = wrapper_module.GroundedSAM2Wrapper._load_grounding_model(
        tmp_path / "groundingdino.pth", "/models/bert-base-uncased", "cpu"
    )

    assert model is fake_model
    assert captured["values"] is not GROUNDING_DINO_SWINT_OGC_CONFIG
    assert captured["args"].device == "cpu"
    assert captured["args"].text_encoder_type == "/models/bert-base-uncased"
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["text_encoder_type"] == "bert-base-uncased"
    assert captured["strict"] is False
    assert captured["evaluated"] is True
