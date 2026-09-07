#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMAGE="${HOUMO_IMAGE:-harbor.houmo.ai/toolchain/release:Dadao-xh2-v1.3.0-ubuntu24.04-x86.64}"
OUTPUT_REL="${PI05_HMM_OUTPUT:-models/pi05_hmm_standard}"
WORK_REL="${PI05_HMM_WORK:-models/_work/${OUTPUT_REL##*/}}"
PIP_CACHE="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"

if [[ -z "${MODEL_BUNDLE_ROOT:-}" ]]; then
    printf 'MODEL_BUNDLE_ROOT is required and must be workspace-relative\n' >&2
    exit 2
fi
MODEL_BUNDLE_REL="${MODEL_BUNDLE_ROOT}"
if [[ "${MODEL_BUNDLE_REL}" = /* || "${OUTPUT_REL}" = /* ]]; then
    printf 'MODEL_BUNDLE_ROOT and PI05_HMM_OUTPUT must be workspace-relative\n' >&2
    exit 2
fi
if [[ ! -d "${WORKSPACE}/${MODEL_BUNDLE_REL}" ]]; then
    printf 'Model bundle does not exist: %s\n' "${WORKSPACE}/${MODEL_BUNDLE_REL}" >&2
    exit 2
fi
if [[ -e "${WORKSPACE}/${OUTPUT_REL}" ]]; then
    printf 'PI05_HMM_OUTPUT already exists; use a new path or remove it explicitly: %s\n' "${WORKSPACE}/${OUTPUT_REL}" >&2
    exit 2
fi

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
LEROBOT_BRANCH="$(git -C "${WORKSPACE}/libs/lerobot" branch --show-current)"
LEROBOT_HEAD="$(git -C "${WORKSPACE}/libs/lerobot" rev-parse HEAD)"
if [[ "${LEROBOT_BRANCH}" != "ibrobot/lerobot-v0.6.0-patched" ]]; then
    printf 'Expected patched LeRobot branch, got %s\n' "${LEROBOT_BRANCH}" >&2
    exit 1
fi
if [[ -n "$(git -C "${WORKSPACE}/libs/lerobot" status --short)" ]]; then
    printf 'libs/lerobot must be clean before HMM conversion\n' >&2
    exit 1
fi
if ! git -C "${WORKSPACE}/libs/lerobot" merge-base --is-ancestor \
    30da8e687a6dfc617fcd94afc367ac7071c376ce "${LEROBOT_HEAD}"; then
    printf 'LeRobot HEAD is not based on the manifest-pinned v0.6.0 commit\n' >&2
    exit 1
fi

mkdir -p "${WORKSPACE}/${WORK_REL}/outputs"

docker run --rm --device nvidia.com/gpu=all --ipc=host \
    -e IBR_HOUMO_IMAGE_ID="${IMAGE_ID}" \
    -e PYTHONNOUSERSITE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "${WORKSPACE}:/workspace" \
    -v "${PIP_CACHE}:/root/.cache/pip" \
    -v "${WORKSPACE}/${MODEL_BUNDLE_REL}:/work/models/pi05_libero_finetuned:ro" \
    -w /workspace \
    "${IMAGE_ID}" bash -lc "
        set -euo pipefail
        python3 -m pip uninstall -y torchao >/dev/null 2>&1 || true
        python3 -m pip install --disable-pip-version-check '/workspace/libs/lerobot[pi]'
        python3 -m pip install --disable-pip-version-check transformers==5.3.0
        export PYTHONPATH=/workspace/libs/lerobot/src:/workspace/src/model_utils:/workspace/src/inference_manifest
        python3 -m model_utils.pi05_export.export_hmm_modules \
            --repo-root /workspace \
            --model-path /work/models/pi05_libero_finetuned \
            --lerobot-src /workspace/libs/lerobot/src \
            --output-dir /workspace/${WORK_REL}/outputs/xh2 \
            --bundle-root /workspace/${OUTPUT_REL}
        python3 -m model_utils.pi05_export.build_hmm_modules \
            --output-dir /workspace/${WORK_REL}/outputs/xh2
        python3 -m model_utils.pi05_export.package_hmm_modules \
            --bundle-root /workspace/${OUTPUT_REL} \
            --output-dir /workspace/${WORK_REL}/outputs/xh2 \
            --deployment hmm \
            --target-soc lq50 \
            --target-runtime tcim
    "
