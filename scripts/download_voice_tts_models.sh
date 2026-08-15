#!/usr/bin/env bash
# Download ZipVoice-Distill ONNX models for Ubuntu deployment.
#
# Downloads text_encoder.onnx + fm_decoder.onnx from ModelScope
# (k2-fsa/ZipVoice) and reuses tokens/Vocos/prompt assets from the
# 310P bundle when available.  Also generates the zipvoice_onnx.json
# runtime config and the inference_manifest.json with the ubuntu_onnx
# deployment so the bundle is ready for `deployment:=ubuntu_onnx`.
#
# Usage:
#   ./scripts/download_voice_tts_models.sh [--bundle-dir DIR]
#
# Defaults:
#   BUNDLE_DIR = $WORKSPACE/models/voice_tts/zipvoice
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
BUNDLE_DIR="${1:-${WORKSPACE}/models/voice_tts/zipvoice}"
MODELSCOPE_BASE="https://www.modelscope.cn/models/k2-fsa/ZipVoice/resolve/master/zipvoice_distill"

ONNX_DIR="${BUNDLE_DIR}/artifacts/onnx"
ASSETS_DIR="${BUNDLE_DIR}/assets"
mkdir -p "${ONNX_DIR}" "${ASSETS_DIR}/vocos" "${ASSETS_DIR}/prompts"

echo "[1/7] Downloading text_encoder.onnx..."
if [ ! -f "${ONNX_DIR}/text_encoder.onnx" ]; then
    curl -fL -o "${ONNX_DIR}/text_encoder.onnx" "${MODELSCOPE_BASE}/text_encoder.onnx"
fi

echo "[2/7] Downloading fm_decoder.onnx..."
if [ ! -f "${ONNX_DIR}/fm_decoder.onnx" ]; then
    curl -fL -o "${ONNX_DIR}/fm_decoder.onnx" "${MODELSCOPE_BASE}/fm_decoder.onnx"
fi

echo "[3/7] Downloading tokens.txt..."
if [ ! -f "${ASSETS_DIR}/tokens.txt" ]; then
    curl -fL -o "${ASSETS_DIR}/tokens.txt" "${MODELSCOPE_BASE}/tokens.txt"
fi

echo "[4/7] Downloading Vocos checkpoint..."
if [ ! -f "${ASSETS_DIR}/vocos/pytorch_model.bin" ]; then
    curl -fL -o "${ASSETS_DIR}/vocos/pytorch_model.bin" "${MODELSCOPE_BASE}/vocos/pytorch_model.bin"
fi

echo "[5/7] Downloading default prompt profile..."
if [ ! -f "${ASSETS_DIR}/prompts/default.npz" ]; then
    curl -fL -o "${ASSETS_DIR}/prompts/default.npz" "${MODELSCOPE_BASE}/prompts/default.npz"
fi

echo "[6/7] Generating zipvoice_onnx.json..."
cat > "${ASSETS_DIR}/zipvoice_onnx.json" <<'JSON'
{
  "text_encoder_path": "artifacts/onnx/text_encoder.onnx",
  "fm_decoder_path": "artifacts/onnx/fm_decoder.onnx",
  "tokens_path": "assets/tokens.txt",
  "vocos_checkpoint_path": "assets/vocos/pytorch_model.bin",
  "prompt_profiles": {"default": "assets/prompts/default.npz"},
  "text_capacity": 256,
  "num_steps": 8,
  "t_shift": 0.5,
  "guidance_scale": 3.0,
  "speed": 1.0,
  "feature_scale": 0.1,
  "sample_rate": 24000,
  "cross_fade_sec": 0.1,
  "seed": 42,
  "inter_op_num_threads": 4,
  "intra_op_num_threads": 4,
  "providers": ["CPUExecutionProvider"]
}
JSON

echo "[7/7] Generating inference_manifest.json..."
python3 - "$BUNDLE_DIR" <<'PY'
import json
import sys
import uuid
from pathlib import Path

bundle_dir = Path(sys.argv[1])
files = sorted(
    [p.relative_to(bundle_dir).as_posix() for p in bundle_dir.joinpath("assets").rglob("*") if p.is_file()]
    + [p.relative_to(bundle_dir).as_posix() for p in bundle_dir.joinpath("artifacts").rglob("*") if p.is_file()]
)

manifest = {
    "schema_version": 2,
    "bundle": {
        "uuid": str(uuid.uuid4()),
        "revision": 1,
        "name": "zipvoice-distill-ubuntu-onnx",
        "files": [{"path": f} for f in files],
    },
    "model": {
        "kind": "generic",
        "family": "zipvoice",
        "inputs": [
            {"semantic": "tts.text", "dtype": "uint8", "shape": [-1]},
            {"semantic": "tts.prompt_audio", "dtype": "float32", "shape": [-1]},
            {"semantic": "tts.prompt_sample_rate", "dtype": "int64", "shape": []},
            {"semantic": "tts.prompt_text", "dtype": "uint8", "shape": [-1]},
        ],
        "outputs": [{"semantic": "tts.audio", "dtype": "float32", "shape": [-1]}],
        "semantic_identity": {
            "logical_model_revision": "zipvoice-distill-onnx-dynamic-2026-08-15",
            "preprocessing_contract": "emilia-zh-cn2an-jieba-pypinyin-fixed-golden-prompt-v1",
            "output_semantics": "mono-float32-pcm-24000hz-vocos-cpu",
        },
    },
    "deployments": {
        "ubuntu_onnx": {
            "uuid": str(uuid.uuid4()),
            "revision": 1,
            "backend": "torch",
            "device": "cpu",
        }
    },
}
manifest_path = bundle_dir / "inference_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"manifest written to {manifest_path}")
PY

echo "Done. Models and bundle metadata generated in ${BUNDLE_DIR}"
