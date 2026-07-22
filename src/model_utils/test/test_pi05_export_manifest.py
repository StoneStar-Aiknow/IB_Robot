from __future__ import annotations

import json
from pathlib import Path

from inference_manifest import load_inference_manifest
from model_utils.pi05_export.convert_om import build_arg_parser, write_pi05_ascend_deployment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_pi05_bundle(root: Path) -> None:
    _write_json(
        root / "config.json",
        {
            "type": "pi05",
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            "chunk_size": 2,
            "max_action_dim": 8,
            "num_inference_steps": 2,
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})


def test_convert_om_parser_uses_unified_manifest_options():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--pretrained-policy-path",
            "/tmp/policy",
            "--soc-version",
            "Ascend310P3",
            "--bundle-root",
            "/tmp/bundle",
            "--skip-manifest",
        ]
    )

    assert args.bundle_root == "/tmp/bundle"
    assert args.skip_manifest is True
    assert "--skip-om-manifest" not in parser.format_help()
    assert "--om-manifest-dir" not in parser.format_help()


def test_write_pi05_ascend_deployment_uses_compiled_abis_and_strict_loader(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _create_pi05_bundle(bundle)
    compiler_output = tmp_path / "compiler"
    compiler_output.mkdir()
    vlm_om = compiler_output / "vlm.om"
    action_om = compiler_output / "action.om"
    vlm_om.write_bytes(b"vlm")
    action_om.write_bytes(b"action")
    vlm_abi = compiler_output / "vlm.abi.json"
    action_abi = compiler_output / "action.abi.json"
    _write_json(
        vlm_abi,
        {
            "inputs": [
                {
                    "name": "observation.images.top",
                    "index": 0,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 24],
                    "layout": "NCHW",
                },
                {"name": "lang_tokens", "index": 1, "dtype": "int64", "shape": [1, 4]},
                {"name": "lang_masks", "index": 2, "dtype": "bool", "shape": [1, 4]},
            ],
            "outputs": [
                {"name": "past_kv_tensor", "index": 0, "dtype": "float16", "shape": [1, 2]},
                {"name": "prefix_pad_masks", "index": 1, "dtype": "bool", "shape": [1, 4]},
            ],
        },
    )
    _write_json(
        action_abi,
        {
            "inputs": [
                {"name": "past_kv_tensor", "index": 0, "dtype": "float16", "shape": [1, 2]},
                {"name": "prefix_pad_masks", "index": 1, "dtype": "bool", "shape": [1, 4]},
                {"name": "time", "index": 2, "dtype": "float32", "shape": [1]},
                {"name": "noise", "index": 3, "dtype": "float32", "shape": [1, 2, 8]},
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 2, 8]}],
        },
    )

    manifest_path = write_pi05_ascend_deployment(
        bundle,
        "ascend",
        "Ascend310P3",
        vlm_abi,
        vlm_om,
        action_abi,
        action_om,
    )
    validated = load_inference_manifest(bundle, "ascend")

    assert manifest_path == bundle / "inference_manifest.json"
    assert validated.deployment.execution == ("vlm", "action_expert")
    assert {link.semantic for link in validated.deployment.device_links} == {
        "internal.past_kv",
        "internal.prefix_pad_masks",
    }
    assert validated.deployment.bindings["vlm"].inputs[1].semantic == "observation.language.tokens"
    assert all(
        artifact.path.startswith("artifacts/ascend/ascend/") for artifact in validated.deployment.artifacts.values()
    )
