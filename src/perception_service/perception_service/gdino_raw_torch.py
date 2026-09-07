"""Torch module loader for the raw Grounding-DINO Swin-T OGC deployment.

The ``torch_module_loader`` entry in a bundle's ``assets/adapter.json`` points
at :func:`load_gdino_raw_module`.  The returned callable consumes the raw
Grounding-DINO contract inputs (``image``/``input_ids``/``token_type_ids``/
``position_ids``/``text_self_attention_masks``/``text_token_mask``) and returns
``{"pred_logits", "pred_boxes"}``, matching the compiled 310P artifacts in the
same bundle so both deployments share one adapter
(:class:`perception_service.semantic_model_adapters.GroundingDINORawAdapter`).

``encoder_tgt`` is a compiled-graph constant of the Ascend artifacts; the Torch
model owns its query embeddings, so the input is ignored here.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import torch


def _load_grounding_dino(weights_path: Path, device: str) -> Any:
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    with as_file(files("groundingdino").joinpath("config", "GroundingDINO_SwinT_OGC.py")) as config_path:
        args = SLConfig.fromfile(str(config_path))
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_gdino_raw_module(context: Any):
    """Build the raw Grounding-DINO callable for a validated Torch deployment."""
    from groundingdino.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list

    bundle_root = Path(context.validated_manifest.bundle_root)
    weights_path = bundle_root / "assets" / "groundingdino_swint_ogc.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Grounding-DINO weights missing: {weights_path}")
    model = _load_grounding_dino(weights_path, "cpu")

    def run(inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        image = inputs["image"]
        input_ids = inputs["input_ids"].long()
        token_type_ids = inputs["token_type_ids"].long()
        position_ids = inputs["position_ids"].long()
        text_self_attention_masks = inputs["text_self_attention_masks"]
        text_token_mask = inputs["text_token_mask"].bool()

        # Text encoding: feed the pre-tokenized contract inputs directly instead of
        # re-tokenizing a caption; the manifest tokenizer owns the word pieces.
        bert_output = model.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=text_self_attention_masks,
            position_ids=position_ids,
        )
        encoded_text = model.feat_map(bert_output["last_hidden_state"])
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks.bool(),
        }

        samples = nested_tensor_from_tensor_list(image)
        model.set_image_tensor(samples)
        try:
            srcs = []
            masks = []
            for level, feat in enumerate(model.features):
                src, mask = feat.decompose()
                srcs.append(model.input_proj[level](src))
                masks.append(mask)
            if model.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for level in range(_len_srcs, model.num_feature_levels):
                    if level == _len_srcs:
                        src = model.input_proj[level](model.features[-1].tensors)
                    else:
                        src = model.input_proj[level](srcs[-1])
                    mask = torch.nn.functional.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(
                        torch.bool
                    )[0]
                    pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    model.poss.append(pos_l)

            input_query_bbox = input_query_label = attn_mask = None
            hs, reference, _hs_enc, _ref_enc, _init_box_proposal = model.transformer(
                srcs, masks, input_query_bbox, model.poss, input_query_label, attn_mask, text_dict
            )

            outputs_coord_list = []
            for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], model.bbox_embed, hs, strict=False):
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                outputs_coord_list.append(layer_outputs_unsig.sigmoid())
            outputs_coord_list = torch.stack(outputs_coord_list)
            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs, text_dict)
                    for layer_cls_embed, layer_hs in zip(model.class_embed, hs, strict=False)
                ]
            )
            pred_logits = outputs_class[-1]
            # ContrastiveEmbed pads pred_logits to max_text_len=256 columns with
            # -inf. The compiled 310P artifacts emit finite values for that
            # padding, and the raw postprocess clips logits at +-50 before the
            # sigmoid, so saturate to -50 to keep the same finite contract.
            pred_logits = torch.nan_to_num(pred_logits, nan=0.0, neginf=-50.0, posinf=50.0)
            return {
                "pred_logits": pred_logits,
                "pred_boxes": outputs_coord_list[-1],
            }
        finally:
            model.unset_image_tensor()

    class _Callable:
        """Expose the model for the session's ``.to(device)``/``.eval()`` protocol."""

        def __init__(self, inner, module):
            self._inner = inner
            self._model = module

        def __call__(self, inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
            return self._inner(inputs)

        def to(self, device: torch.device):
            self._model.to(device=device)
            return self

        def eval(self):
            self._model.eval()
            return self

    return _Callable(run, model)
