#!/bin/bash
# download_perception_models.sh - Download perception model assets
# Downloads SAM2, Grounding-DINO, BERT, RAM++, and SigLIP2 assets used by perception services.
#
# Usage:
#   ./scripts/download_perception_models.sh              # Download all
#   ./scripts/download_perception_models.sh --sam-only   # Download SAM2 only
#   ./scripts/download_perception_models.sh --gdino-only # Download Grounding-DINO only
#   ./scripts/download_perception_models.sh --ram-only   # Download RAM++ only
#   ./scripts/download_perception_models.sh --siglip2-only # Download SigLIP2 only
#   ./scripts/download_perception_models.sh --target /custom/path  # Custom target directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL_DIR="${WORKSPACE}/models"
MODEL_DIR="${DEFAULT_MODEL_DIR}"

ONLY_MODEL=""

SAM_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM_CHECKPOINT="sam2.1_hiera_tiny.pt"
SAM_URL="${SAM_BASE_URL}/${SAM_CHECKPOINT}"
SAM_BUNDLE="sam2.1_hiera_tiny"
SAM_DIR="${SAM_BUNDLE}/assets"

GDINO_CHECKPOINT="groundingdino_swint_ogc.pth"
GDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/${GDINO_CHECKPOINT}"
GDINO_BUNDLE="grounded_sam2_swint_ogc"
GDINO_DIR="${GDINO_BUNDLE}/assets"
BERT_MODEL_ID="bert-base-uncased"
BERT_DIR="${GDINO_BUNDLE}/assets/${BERT_MODEL_ID}"
RAM_MODEL_ID="xinyu1205/recognize-anything-plus-model"
RAM_CHECKPOINT="ram_plus_swin_large_14m.pth"
RAM_BUNDLE="ram_plus_swin_large_14m"
RAM_DIR="${RAM_BUNDLE}/assets"
RAM_BERT_DIR="${RAM_BUNDLE}/assets/${BERT_MODEL_ID}"
SIGLIP_MODEL_ID="google/siglip2-so400m-patch14-384"
SIGLIP_BUNDLE="siglip2_so400m_patch14_384"
SIGLIP_DIR="${SIGLIP_BUNDLE}/assets/model"

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
  --ram-only       Download RAM++ model only
  --siglip2-only   Download SigLIP2 model only
  --target DIR     Bundle parent directory (default: models/)
  -h, --help       Show this help

Models downloaded:
  - SAM 2.1 Hiera Tiny (~150 MB)
  - Grounding-DINO SwinT OGC (~660 MB)
  - BERT base uncased text encoder for Grounding-DINO (~420 MB)
  - RAM++ Swin Large 14M (~3 GB)
  - SigLIP2 SO400M patch14-384 for image/text embeddings
EOF
}

select_only() {
    local model="$1"
    if [[ -n "${ONLY_MODEL}" ]]; then
        echo "Error: only one --*-only option may be specified"
        exit 1
    fi
    ONLY_MODEL="${model}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --sam-only) select_only sam2 ;;
            --gdino-only) select_only grounding_dino ;;
            --ram-only) select_only ram_plus ;;
            --siglip2-only) select_only siglip2 ;;
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

    mkdir -p "$(dirname "${dest}")"
    if [[ -s "${dest}" ]]; then
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
    local dest="$1"

    if [[ -s "${dest}/config.json" ]] && [[ -s "${dest}/vocab.txt" ]] &&
       { [[ -s "${dest}/model.safetensors" ]] || [[ -s "${dest}/pytorch_model.bin" ]]; }; then
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
        "Install perception dependencies first with ./scripts/setup.sh."
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

download_siglip_encoder() {
    local dest="${MODEL_DIR}/${SIGLIP_DIR}"
    if [[ -s "${dest}/config.json" ]] && [[ -s "${dest}/preprocessor_config.json" ]] \
        && { [[ -s "${dest}/model.safetensors" ]] || [[ -s "${dest}/pytorch_model.bin" ]]; }; then
        echo "[skip] SigLIP2 encoder already exists: ${dest}"
        return 0
    fi

    echo "[download] SigLIP2 encoder -> ${dest}"
    "${PYTHON_BIN}" - "$SIGLIP_MODEL_ID" "$dest" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
dest = Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=str(dest))
PY
    echo "[done] SigLIP2 encoder"
}

download_ram_plus() {
    local dest="${MODEL_DIR}/${RAM_DIR}"
    local checkpoint="${dest}/${RAM_CHECKPOINT}"
    if [[ -s "${checkpoint}" ]]; then
        echo "[skip] RAM++ checkpoint already exists: ${checkpoint}"
        return 0
    fi

    echo "[download] RAM++ checkpoint -> ${checkpoint}"
    "${PYTHON_BIN}" - "$RAM_MODEL_ID" "$RAM_CHECKPOINT" "$dest" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, checkpoint, destination = sys.argv[1:]
dest = Path(destination)
dest.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=str(dest), allow_patterns=[checkpoint])
PY
    echo "[done] RAM++ checkpoint"
}

prepare_bundle_metadata() {
    local family="$1"
    case "${family}" in
        sam2)
            ;;
        grounding_dino)
            download_file "${SAM_URL}" "${MODEL_DIR}/${GDINO_DIR}/${SAM_CHECKPOINT}" \
                "Grounded-SAM2 SAM 2.1 Hiera Tiny"
            ;;
        ram_plus)
            "${PYTHON_BIN}" - "${MODEL_DIR}/${RAM_DIR}" <<'PY'
import sys
from importlib.resources import as_file, files
from pathlib import Path
from shutil import copyfile

destination = Path(sys.argv[1])
for name in ("ram_tag_list.txt", "ram_tag_list_threshold.txt"):
    with as_file(files("ram").joinpath("data", name)) as source:
        copyfile(source, destination / name)
PY
            download_bert_text_encoder "${MODEL_DIR}/${RAM_BERT_DIR}"
            ;;
        siglip2)
            ;;
    esac
}

finalize_bundle() {
    local family="$1"
    PYTHONPATH="${WORKSPACE}/src/perception_service:${WORKSPACE}/src/inference_manifest:${WORKSPACE}/src/inference_service:${PYTHONPATH:-}" \
        "${PYTHON_BIN}" -m perception_service.package_perception_bundles \
        --models-root "${MODEL_DIR}" --family "${family}"
}

main() {
    parse_args "$@"

    local download_sam=false
    local download_gdino=false
    local download_ram=false
    local download_siglip=false

    if [[ -z "${ONLY_MODEL}" || "${ONLY_MODEL}" == "sam2" ]]; then download_sam=true; fi
    if [[ -z "${ONLY_MODEL}" || "${ONLY_MODEL}" == "grounding_dino" ]]; then download_gdino=true; fi
    if [[ -z "${ONLY_MODEL}" || "${ONLY_MODEL}" == "ram_plus" ]]; then download_ram=true; fi
    if [[ -z "${ONLY_MODEL}" || "${ONLY_MODEL}" == "siglip2" ]]; then download_siglip=true; fi

    mkdir -p "${MODEL_DIR}"

    echo "Detection model directory: ${MODEL_DIR}"
    echo ""

    if [[ "${download_sam}" == true ]]; then
        download_file "${SAM_URL}" "${MODEL_DIR}/${SAM_DIR}/${SAM_CHECKPOINT}" "SAM 2.1 Hiera Tiny"
        prepare_bundle_metadata sam2
        finalize_bundle sam2
    fi

    if [[ "${download_gdino}" == true ]]; then
        download_file "${GDINO_URL}" "${MODEL_DIR}/${GDINO_DIR}/${GDINO_CHECKPOINT}" "Grounding-DINO SwinT OGC"
        download_bert_text_encoder "${MODEL_DIR}/${BERT_DIR}"
        prepare_bundle_metadata grounding_dino
        finalize_bundle grounded_sam2
    fi
    if [[ "${download_ram}" == true ]]; then
        download_ram_plus
        prepare_bundle_metadata ram_plus
        finalize_bundle ram_plus
    fi
    if [[ "${download_siglip}" == true ]]; then
        download_siglip_encoder
        prepare_bundle_metadata siglip2
        finalize_bundle siglip2
    fi

    echo ""
    echo "All requested models downloaded to: ${MODEL_DIR}"
    if [[ "${download_sam}" == true ]]; then
        echo "  SAM2:    ${MODEL_DIR}/${SAM_DIR}/${SAM_CHECKPOINT}"
    fi
    if [[ "${download_gdino}" == true ]]; then
        echo "  GDINO:   ${MODEL_DIR}/${GDINO_DIR}/${GDINO_CHECKPOINT}"
        echo "  BERT:    ${MODEL_DIR}/${BERT_DIR}"
    fi
    if [[ "${download_ram}" == true ]]; then
        echo "  RAM++:   ${MODEL_DIR}/${RAM_DIR}/${RAM_CHECKPOINT}"
    fi
    if [[ "${download_siglip}" == true ]]; then
        echo "  SigLIP2: ${MODEL_DIR}/${SIGLIP_DIR}"
    fi
}

main "$@"
