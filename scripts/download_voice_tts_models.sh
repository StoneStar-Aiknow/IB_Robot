#!/usr/bin/env bash
# Download ZipVoice-Distill ONNX models for Ubuntu deployment.
#
# Downloads text_encoder.onnx + fm_decoder.onnx from ModelScope
# (k2-fsa/ZipVoice) and reuses tokens/Vocos/prompt assets from the
# 310P bundle when available.  Also generates the zipvoice_onnx.json
# runtime config and merges the ubuntu_onnx deployment into
# inference_manifest.json so the bundle is ready for `deployment:=ubuntu_onnx`.
#
# Usage:
#   ./scripts/download_voice_tts_models.sh [--bundle-dir DIR]
#
# Defaults:
#   BUNDLE_DIR = $WORKSPACE/models/voice_tts/zipvoice
#
# The manifest merge preserves an existing bundle uuid and any existing
# deployments (e.g. ascend_310p).  Run this script in the same bundle
# directory that already contains a 310P bundle to make both deployments
# available; run it in a fresh directory to create an ONNX-only bundle.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
BUNDLE_DIR="${WORKSPACE}/models/voice_tts/zipvoice"

while [ $# -gt 0 ]; do
    case "$1" in
        --bundle-dir)
            [ $# -ge 2 ] || { echo "error: --bundle-dir requires a value" >&2; exit 1; }
            BUNDLE_DIR="$2"
            shift 2
            ;;
        --bundle-dir=*)
            BUNDLE_DIR="${1#--bundle-dir=}"
            shift
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            echo "usage: $0 [--bundle-dir DIR]" >&2
            exit 1
            ;;
    esac
done

MODELSCOPE_BASE="https://www.modelscope.cn/models/k2-fsa/ZipVoice/resolve/master/zipvoice_distill"

ONNX_DIR="${BUNDLE_DIR}/artifacts/onnx"
ASSETS_DIR="${BUNDLE_DIR}/assets"
mkdir -p "${ONNX_DIR}" "${ASSETS_DIR}/vocos" "${ASSETS_DIR}/prompts"

# Expected SHA-256 digests for downloaded assets.  Leave empty to skip
# verification on first run; the script prints the computed digest so you
# can record it here after a trusted download.  Mirrors
# scripts/download_speech_direction_models.sh convention.
TEXT_ENCODER_SHA256=""
FM_DECODER_SHA256=""
TOKENS_SHA256=""
VOCOS_SHA256=""
PROMPT_SHA256=""

verify_file() {
    local path="$1" expected_sha="$2" desc="$3"
    local actual_sha actual_size
    actual_sha=$(sha256sum "${path}" | cut -d' ' -f1)
    actual_size=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}")
    if [ -n "${expected_sha}" ]; then
        if [ "${actual_sha}" != "${expected_sha}" ]; then
            echo "[error] ${desc} SHA-256 mismatch: actual=${actual_sha} expected=${expected_sha}" >&2
            exit 1
        fi
        echo "[verify] ${desc} size=${actual_size} SHA-256 OK"
    else
        echo "[info] ${desc} size=${actual_size} SHA-256=${actual_sha}"
        echo "       record this value in the script to enable verification on future runs"
    fi
}

download_and_verify() {
    local url="$1" dest="$2" expected_sha="$3" desc="$4"
    if [ ! -f "${dest}" ]; then
        echo "  downloading ${desc}..."
        curl -fL -o "${dest}" "${url}"
    fi
    verify_file "${dest}" "${expected_sha}" "${desc}"
}

echo "[1/7] Downloading text_encoder.onnx..."
download_and_verify "${MODELSCOPE_BASE}/text_encoder.onnx" "${ONNX_DIR}/text_encoder.onnx" "${TEXT_ENCODER_SHA256}" "text_encoder.onnx"

echo "[2/7] Downloading fm_decoder.onnx..."
download_and_verify "${MODELSCOPE_BASE}/fm_decoder.onnx" "${ONNX_DIR}/fm_decoder.onnx" "${FM_DECODER_SHA256}" "fm_decoder.onnx"

echo "[3/7] Downloading tokens.txt..."
download_and_verify "${MODELSCOPE_BASE}/tokens.txt" "${ASSETS_DIR}/tokens.txt" "${TOKENS_SHA256}" "tokens.txt"

echo "[4/7] Downloading Vocos checkpoint..."
download_and_verify "${MODELSCOPE_BASE}/vocos/pytorch_model.bin" "${ASSETS_DIR}/vocos/pytorch_model.bin" "${VOCOS_SHA256}" "vocos/pytorch_model.bin"

echo "[5/7] Downloading default prompt profile..."
download_and_verify "${MODELSCOPE_BASE}/prompts/default.npz" "${ASSETS_DIR}/prompts/default.npz" "${PROMPT_SHA256}" "prompts/default.npz"

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

echo "[7/7] Merging ubuntu_onnx deployment into inference_manifest.json..."
python3 - "$BUNDLE_DIR" <<'PY'
import hashlib
import json
import sys
import uuid
from pathlib import Path

bundle_dir = Path(sys.argv[1])
manifest_path = bundle_dir / "inference_manifest.json"

UBUNTU_ONNX_DEPLOYMENT_NAME = "ubuntu_onnx"
BUNDLE_NAME_ONNX_ONLY = "zipvoice-distill-ubuntu-onnx"


def canonical_bundle_digest(bundle_uuid, bundle_revision, bundle_name, file_paths):
    """Hash the lightweight bundle declaration without reading bundle files.

    Mirrors inference_manifest.integrity.canonical_bundle_digest so the
    value matches the official loader.  Prefers the installed package;
    falls back to the inline implementation below.
    """
    try:
        from inference_manifest import canonical_bundle_digest as _impl
        from inference_manifest.models import BundleFile
        files = [BundleFile(path=p) for p in file_paths]
        return _impl(bundle_uuid, bundle_revision, bundle_name, files)
    except ImportError:
        payload = {
            "format": "ibrobot.bundle-structure-v2",
            "uuid": bundle_uuid,
            "revision": bundle_revision,
            "name": bundle_name,
            "files": sorted(file_paths),
        }
        payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def deployment_fingerprint(schema_version, bundle_digest, deployment_name, deployment):
    """Mirrors inference_manifest.integrity.deployment_fingerprint."""
    try:
        from inference_manifest import deployment_fingerprint as _impl
        return _impl(schema_version, bundle_digest, deployment_name, deployment)
    except ImportError:
        payload = {
            "format": "ibrobot.deployment-structure-v2",
            "schema_version": schema_version,
            "bundle_digest": bundle_digest,
            "deployment_name": deployment_name,
            "deployment": deployment,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()


file_paths = sorted(
    [p.relative_to(bundle_dir).as_posix() for p in bundle_dir.joinpath("assets").rglob("*") if p.is_file()]
    + [p.relative_to(bundle_dir).as_posix() for p in bundle_dir.joinpath("artifacts").rglob("*") if p.is_file()]
)

ubuntu_onnx_deployment = {
    "uuid": str(uuid.uuid4()),
    "revision": 1,
    "backend": "torch",
    "device": "cpu",
}

model_block = {
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
}

if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = manifest.setdefault("bundle", {})
    bundle_uuid = bundle.get("uuid", str(uuid.uuid4()))
    bundle_revision = bundle.get("revision", 1)
    bundle_name = bundle.get("name", BUNDLE_NAME_ONNX_ONLY)
    manifest.setdefault("schema_version", 2)
    manifest.setdefault("model", model_block)
    deployments = manifest.setdefault("deployments", {})
    preserved = list(deployments.keys() - {UBUNTU_ONNX_DEPLOYMENT_NAME})
    deployments[UBUNTU_ONNX_DEPLOYMENT_NAME] = ubuntu_onnx_deployment
    print(f"merging into existing manifest (preserved deployments: {preserved or 'none'})")
else:
    bundle_uuid = str(uuid.uuid4())
    bundle_revision = 1
    bundle_name = BUNDLE_NAME_ONNX_ONLY
    manifest = {
        "schema_version": 2,
        "model": model_block,
    }
    manifest["deployments"] = {UBUNTU_ONNX_DEPLOYMENT_NAME: ubuntu_onnx_deployment}
    print("creating new manifest (ubuntu_onnx only)")

bundle_digest = canonical_bundle_digest(bundle_uuid, bundle_revision, bundle_name, file_paths)
fingerprint = deployment_fingerprint(
    manifest.get("schema_version", 2),
    bundle_digest,
    UBUNTU_ONNX_DEPLOYMENT_NAME,
    ubuntu_onnx_deployment,
)

manifest["bundle"] = {
    "uuid": bundle_uuid,
    "revision": bundle_revision,
    "name": bundle_name,
    "files": [{"path": f} for f in file_paths],
    "digest": {
        "algorithm": "sha256",
        "scope": "structure",
        "value": bundle_digest,
    },
}

ubuntu_onnx_deployment["fingerprint"] = fingerprint

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"manifest written to {manifest_path}")
print(f"bundle uuid: {bundle_uuid}")
print(f"bundle digest: {bundle_digest}")
print(f"ubuntu_onnx fingerprint: {fingerprint}")
PY

echo "Done. Models and bundle metadata generated in ${BUNDLE_DIR}"
