#!/bin/bash
# download_perception_models.sh - Download perception model assets
# Downloads SAM2, Grounding-DINO, and the BERT text encoder used by perception_service.
#
# Usage:
#   ./scripts/download_perception_models.sh              # Download all
#   ./scripts/download_perception_models.sh --sam-only   # Download SAM2 only
#   ./scripts/download_perception_models.sh --gdino-only # Download Grounding-DINO only
#   ./scripts/download_perception_models.sh --target /custom/path  # Custom target directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL_DIR="${WORKSPACE}/models/perception"
MODEL_DIR="${DEFAULT_MODEL_DIR}"

SAM_ONLY=false
GDINO_ONLY=false

SAM_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM_CHECKPOINT="sam2.1_hiera_tiny.pt"
SAM_URL="${SAM_BASE_URL}/${SAM_CHECKPOINT}"

GDINO_BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download"
GDINO_CHECKPOINT="groundingdino_swint_ogc.pth"
GDINO_URL="${GDINO_BASE_URL}/v0.1.0-alpha/${GDINO_CHECKPOINT}"
BERT_MODEL_ID="bert-base-uncased"
BERT_DIR="${BERT_MODEL_ID}"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "${WORKSPACE}/venv/bin/python3" ]]; then
        PYTHON_BIN="${WORKSPACE}/venv/bin/python3"
    else
        PYTHON_BIN="python3"
    fi
fi

usage() {
    cat <<'EOF'
Download perception model assets for IB-Robot.

Usage:
  ./scripts/download_perception_models.sh [OPTIONS]

Options:
  --sam-only       Download SAM2 model only
  --gdino-only     Download Grounding-DINO model only
  --target DIR     Target directory (default: models/perception/)
  -h, --help       Show this help

Models downloaded:
  - SAM 2.1 Hiera Tiny (~150 MB)
  - Grounding-DINO SwinT OGC (~660 MB)
  - BERT base uncased text encoder for Grounding-DINO (~420 MB)
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --sam-only) SAM_ONLY=true ;;
            --gdino-only) GDINO_ONLY=true ;;
            --target)
                shift
                if [[ $# -eq 0 ]]; then echo "Error: --target requires a path"; exit 1; fi
                MODEL_DIR="$1"
                ;;
            -h|--help) usage; exit 0 ;;
            *) echo "Unknown argument: $1"; usage; exit 1 ;;
        esac
        shift
    done
}

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [[ -f "${dest}" ]]; then
        echo "[skip] ${desc} already exists: ${dest}"
        return 0
    fi

    echo "[download] ${desc} -> ${dest}"
    local tmp="${dest}.part"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${tmp}" "${url}"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "${tmp}" "${url}"
    else
        echo "Error: wget or curl required"
        exit 1
    fi
    mv "${tmp}" "${dest}"
    echo "[done] ${desc}"
}

download_bert_text_encoder() {
    local dest="${MODEL_DIR}/${BERT_DIR}"

    if [[ -f "${dest}/config.json" ]] && [[ -f "${dest}/vocab.txt" ]] &&
       { [[ -f "${dest}/model.safetensors" ]] || [[ -f "${dest}/pytorch_model.bin" ]]; }; then
        echo "[skip] BERT text encoder already exists: ${dest}"
        return 0
    fi

    echo "[download] BERT text encoder -> ${dest}"
    "${PYTHON_BIN}" - "$BERT_MODEL_ID" "$dest" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Error: huggingface_hub is required to download bert-base-uncased. "
        "Install perception dependencies first with ./scripts/setup.sh --with-perception."
    ) from exc

repo_id = sys.argv[1]
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(dest),
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
)
PY
    echo "[done] BERT text encoder"
}

main() {
    parse_args "$@"

    local download_sam=true
    local download_gdino=true

    if [[ "${SAM_ONLY}" == true ]]; then download_gdino=false; fi
    if [[ "${GDINO_ONLY}" == true ]]; then download_sam=false; fi

    mkdir -p "${MODEL_DIR}"

    echo "Detection model directory: ${MODEL_DIR}"
    echo ""

    if [[ "${download_sam}" == true ]]; then
        download_file "${SAM_URL}" "${MODEL_DIR}/${SAM_CHECKPOINT}" "SAM 2.1 Hiera Tiny"
    fi

    if [[ "${download_gdino}" == true ]]; then
        download_file "${GDINO_URL}" "${MODEL_DIR}/${GDINO_CHECKPOINT}" "Grounding-DINO SwinT OGC"
        download_bert_text_encoder
    fi

    echo ""
    echo "All requested models downloaded to: ${MODEL_DIR}"
    if [[ "${download_sam}" == true ]]; then
        echo "  SAM2:  ${MODEL_DIR}/${SAM_CHECKPOINT}"
    fi
    if [[ "${download_gdino}" == true ]]; then
        echo "  GDINO: ${MODEL_DIR}/${GDINO_CHECKPOINT}"
        echo "  BERT:  ${MODEL_DIR}/${BERT_DIR}"
    fi
}

main "$@"
