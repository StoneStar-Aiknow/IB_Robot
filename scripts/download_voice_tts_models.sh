#!/usr/bin/env bash
# Download ZipVoice-Distill ONNX models for Ubuntu deployment.
#
# Downloads text_encoder.onnx + fm_decoder.onnx from ModelScope
# (k2-fsa/ZipVoice) and reuses tokens/Vocos/prompt assets from the
# 310P bundle when available.
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

echo "[1/5] Downloading text_encoder.onnx..."
if [ ! -f "${ONNX_DIR}/text_encoder.onnx" ]; then
    curl -fL -o "${ONNX_DIR}/text_encoder.onnx" "${MODELSCOPE_BASE}/text_encoder.onnx"
fi

echo "[2/5] Downloading fm_decoder.onnx..."
if [ ! -f "${ONNX_DIR}/fm_decoder.onnx" ]; then
    curl -fL -o "${ONNX_DIR}/fm_decoder.onnx" "${MODELSCOPE_BASE}/fm_decoder.onnx"
fi

echo "[3/5] Downloading tokens.txt..."
if [ ! -f "${ASSETS_DIR}/tokens.txt" ]; then
    curl -fL -o "${ASSETS_DIR}/tokens.txt" "${MODELSCOPE_BASE}/tokens.txt"
fi

echo "[4/5] Downloading Vocos checkpoint..."
if [ ! -f "${ASSETS_DIR}/vocos/pytorch_model.bin" ]; then
    curl -fL -o "${ASSETS_DIR}/vocos/pytorch_model.bin" "${MODELSCOPE_BASE}/vocos/pytorch_model.bin"
fi

echo "[5/5] Downloading default prompt profile..."
if [ ! -f "${ASSETS_DIR}/prompts/default.npz" ]; then
    curl -fL -o "${ASSETS_DIR}/prompts/default.npz" "${MODELSCOPE_BASE}/prompts/default.npz"
fi

echo "Done. Models downloaded to ${BUNDLE_DIR}"
