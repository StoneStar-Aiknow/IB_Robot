#!/bin/bash
# download_speech_direction_models.sh - Download model assets for speech_direction
#
# 下载 speech_direction 链路所需的模型(均不入库,放 models/ 下):
#   1. Silero VAD v5 ONNX    — models/voice_asr/silero-vad/silero_vad_v5.onnx (~2.3MB)
#                              与 voice_asr_service 共用,若已存在则跳过
#   2. FullSubNet 源码仓      — models/fullsubnet_repo/ (git clone,需 model.py)
#   3. FullSubNet ckpt        — models/fullsubnet/fullsubnet_best_model_58epochs.tar (~67MB)
#
# 仿照 scripts/download_perception_models.sh 的风格:已存在则跳过,wget/curl 下载。
#
# Usage:
#   ./scripts/download_speech_direction_models.sh                 # 下载全部
#   ./scripts/download_speech_direction_models.sh --silero-only   # 仅 Silero VAD
#   ./scripts/download_speech_direction_models.sh --fullsubnet-only # 仅 FullSubNet(repo + ckpt)
#   ./scripts/download_speech_direction_models.sh --target /custom/path  # 自定义 models 根目录
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODELS_DIR="${WORKSPACE}/models"
MODELS_DIR="${DEFAULT_MODELS_DIR}"

SILERO_ONLY=false
FULLSUBNET_ONLY=false

# Silero VAD v5(snakers4/silero-vad 官方 release)
SILERO_URL="https://github.com/snakers4/silero-vad/releases/download/v5.0/silero_vad_v5.onnx"

# FullSubNet 源码仓及与 v0.2 checkpoint 验证配套的不可变提交。
FULLSUBNET_REPO_URL="https://github.com/Audio-WestlakeU/FullSubNet.git"
FULLSUBNET_REPO_REV="e97448375cd1e883276ad583317b1828318910dc"

# FullSubNet ckpt(Audio-WestlakeU/FullSubNet v0.2 release,58epochs 版本)
FULLSUBNET_CKPT_URL="https://github.com/Audio-WestlakeU/FullSubNet/releases/download/v0.2/fullsubnet_best_model_58epochs.tar"

usage() {
    cat <<'EOF'
Download speech_direction model assets for IB-Robot.

Usage:
  ./scripts/download_speech_direction_models.sh [OPTIONS]

Options:
  --silero-only         Download Silero VAD v5 only
  --fullsubnet-only     Download FullSubNet (repo + ckpt) only
  --target DIR          Target models directory (default: models/)
  -h, --help            Show this help

Models downloaded:
  - Silero VAD v5 ONNX (~2.3 MB, shared with voice_asr_service)
  - FullSubNet source repo (git clone, ~tens of MB)
  - FullSubNet ckpt (~67 MB)

Models already present are skipped.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --silero-only) SILERO_ONLY=true ;;
            --fullsubnet-only) FULLSUBNET_ONLY=true ;;
            --target)
                shift
                if [[ $# -eq 0 ]]; then echo "Error: --target requires a path"; exit 1; fi
                MODELS_DIR="$1"
                ;;
            -h|--help) usage; exit 0 ;;
            *) echo "Unknown argument: $1"; usage; exit 1 ;;
        esac
        shift
    done
}

# 下载单个文件(已存在则跳过),仿 download_perception_models.sh
download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [[ -f "${dest}" ]]; then
        echo "[skip] ${desc} already exists: ${dest}"
        return 0
    fi

    echo "[download] ${desc} -> ${dest}"
    mkdir -p "$(dirname "${dest}")"
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

download_silero() {
    local dest="${MODELS_DIR}/voice_asr/silero-vad/silero_vad_v5.onnx"
    download_file "${SILERO_URL}" "${dest}" "Silero VAD v5"
}

download_fullnet_repo() {
    local dest="${MODELS_DIR}/fullsubnet_repo"
    local tmp="${dest}.part"
    local backup="${dest}.backup"
    local model_rel="recipes/dns_interspeech_2020/fullsubnet/model.py"

    # 只有关键文件和固定 revision 同时匹配才视为已就绪。
    if [[ -f "${dest}/${model_rel}" ]] \
        && [[ "$(git -C "${dest}" rev-parse HEAD 2>/dev/null || true)" == "${FULLSUBNET_REPO_REV}" ]]; then
        echo "[skip] FullSubNet repo already exists at ${FULLSUBNET_REPO_REV}: ${dest}"
        return 0
    fi

    echo "[clone] FullSubNet repo ${FULLSUBNET_REPO_REV} -> ${dest}"
    mkdir -p "${MODELS_DIR}"
    rm -rf "${tmp}" "${backup}"
    git init "${tmp}"
    git -C "${tmp}" remote add origin "${FULLSUBNET_REPO_URL}"
    git -C "${tmp}" fetch --depth 1 origin "${FULLSUBNET_REPO_REV}"
    git -C "${tmp}" checkout --detach FETCH_HEAD

    # 安装前同时校验 revision 和运行时实际导入的关键文件。
    if [[ "$(git -C "${tmp}" rev-parse HEAD)" != "${FULLSUBNET_REPO_REV}" ]]; then
        echo "[error] FullSubNet repo revision 校验失败"
        rm -rf "${tmp}"
        return 1
    fi
    if [[ ! -f "${tmp}/${model_rel}" ]]; then
        echo "[error] FullSubNet repo 缺少 ${model_rel}"
        rm -rf "${tmp}"
        return 1
    fi

    # 先保留旧目录，替换成功后再删除；失败时可恢复，不在目标路径留下半成品。
    if [[ -e "${dest}" ]]; then
        mv "${dest}" "${backup}"
    fi
    if mv "${tmp}" "${dest}"; then
        rm -rf "${backup}"
    else
        if [[ -e "${backup}" ]]; then
            mv "${backup}" "${dest}"
        fi
        return 1
    fi
    echo "[done] FullSubNet repo @ ${FULLSUBNET_REPO_REV}"
}

download_fullnet_ckpt() {
    local dest="${MODELS_DIR}/fullsubnet/fullsubnet_best_model_58epochs.tar"
    download_file "${FULLSUBNET_CKPT_URL}" "${dest}" "FullSubNet ckpt (~67MB)"
}

main() {
    parse_args "$@"

    local download_silero=true
    local download_fullsubnet=true

    if [[ "${SILERO_ONLY}" == true ]]; then download_fullsubnet=false; fi
    if [[ "${FULLSUBNET_ONLY}" == true ]]; then download_silero=false; fi

    mkdir -p "${MODELS_DIR}"
    echo "Models directory: ${MODELS_DIR}"
    echo ""

    if [[ "${download_silero}" == true ]]; then
        download_silero
    fi

    if [[ "${download_fullsubnet}" == true ]]; then
        download_fullnet_repo
        download_fullnet_ckpt
    fi

    echo ""
    echo "Speech direction models setup complete. Directory: ${MODELS_DIR}"
    if [[ "${download_silero}" == true ]]; then
        echo "  Silero VAD:      ${MODELS_DIR}/voice_asr/silero-vad/silero_vad_v5.onnx"
    fi
    if [[ "${download_fullsubnet}" == true ]]; then
        echo "  FullSubNet repo: ${MODELS_DIR}/fullsubnet_repo/"
        echo "  FullSubNet ckpt: ${MODELS_DIR}/fullsubnet/fullsubnet_best_model_58epochs.tar"
    fi
}

main "$@"
