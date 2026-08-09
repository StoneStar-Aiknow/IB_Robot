#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
IMAGE="${HOUMO_IMAGE:-harbor.houmo.ai/toolchain/release:Dadao-xh2-v1.3.0-ubuntu24.04-x86.64}"
OUTPUT_REL="${SMOLVLA_HMM_OUTPUT:-models/smolvla_hmm_standard}"
WORK_REL="${SMOLVLA_HMM_WORK:-models/_work/${OUTPUT_REL##*/}}"
DEVICE="${SMOLVLA_EXPORT_DEVICE:-cuda}"
PIP_CACHE="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"
TRANSFORMERS_CANDIDATES="${SMOLVLA_TRANSFORMERS_CANDIDATES:-5.3.0 4.57.1}"

if [[ -z "${MODEL_BUNDLE_ROOT:-}" ]]; then
    printf 'MODEL_BUNDLE_ROOT is required and must be workspace-relative\n' >&2
    exit 2
fi
MODEL_BUNDLE_REL="${MODEL_BUNDLE_ROOT}"
if [[ "${MODEL_BUNDLE_REL}" = /* ]]; then
    printf 'MODEL_BUNDLE_ROOT must be workspace-relative: %s\n' "${MODEL_BUNDLE_REL}" >&2
    exit 2
fi
if [[ ! -d "${WORKSPACE}/${MODEL_BUNDLE_REL}" ]]; then
    printf 'Model bundle does not exist: %s\n' "${WORKSPACE}/${MODEL_BUNDLE_REL}" >&2
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

docker run --rm --device nvidia.com/gpu=all \
    --ipc=host \
    -e IBR_HOUMO_IMAGE_ID="${IMAGE_ID}" \
    -e PYTHONNOUSERSITE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e SMOLVLA_TRANSFORMERS_CANDIDATES="${TRANSFORMERS_CANDIDATES}" \
    -v "${WORKSPACE}:/workspace" \
    -v "${PIP_CACHE}:/root/.cache/pip" \
    -w /workspace \
    "${IMAGE_ID}" \
    bash -lc "
        set -euo pipefail
        python3 -m pip uninstall -y torchao >/dev/null 2>&1 || true
        python3 -m pip install --disable-pip-version-check '/workspace/libs/lerobot[smolvla]'
        export PYTHONPATH=/workspace/libs/lerobot/src:/workspace/src/model_utils:/workspace/src/inference_manifest
        selected_transformers=''
        for candidate in \${SMOLVLA_TRANSFORMERS_CANDIDATES}; do
            printf 'Testing transformers==%s...\\n' "\${candidate}"
            if python3 -m pip install --disable-pip-version-check \
                "transformers==\${candidate}" && \
                IBR_TRANSFORMERS_CANDIDATE="\${candidate}" python3 - <<'PY'
import inspect
from pathlib import Path

import lerobot
import transformers
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

expected = Path('/workspace/libs/lerobot/src').resolve()
actual = Path(inspect.getfile(lerobot)).resolve()
if expected not in actual.parents:
    raise SystemExit(f'unexpected LeRobot import: {actual}')

config = PreTrainedConfig.from_pretrained('/workspace/${MODEL_BUNDLE_REL}')
config.vlm_model_name = '/workspace/${MODEL_BUNDLE_REL}/HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
config.device = 'cpu'
policy = SmolVLAPolicy.from_pretrained('/workspace/${MODEL_BUNDLE_REL}', config=config, strict=False)
print(
    f'Preflight passed: transformers={transformers.__version__}, '
    f'policy={type(policy).__name__}, layers={policy.config.num_vlm_layers}'
)
PY
            then
                selected_transformers="\${candidate}"
                break
            fi
        done
        if [[ -z "\${selected_transformers}" ]]; then
            printf 'No Transformers candidate passed SmolVLA loading preflight\\n' >&2
            exit 1
        fi
        export IBR_TRANSFORMERS_SELECTED="\${selected_transformers}"
        python3 - <<'PY'
import inspect
from pathlib import Path
import lerobot
from lerobot.policies.smolvla import modeling_smolvla

expected = Path('/workspace/libs/lerobot/src').resolve()
actual = Path(inspect.getfile(lerobot)).resolve()
smolvla = Path(inspect.getfile(modeling_smolvla)).resolve()
if expected not in actual.parents or expected not in smolvla.parents:
    raise SystemExit(f'unexpected LeRobot import: {actual}, {smolvla}')
print(f'LeRobot import verified: {actual}')
print(f'SmolVLA import verified: {smolvla}')
PY
        python3 -m model_utils.smolvla_export.export_hmm_modules \
            --repo-root /workspace \
            --model-path /workspace/${MODEL_BUNDLE_REL} \
            --lerobot-src /workspace/libs/lerobot/src \
            --output-dir /workspace/${WORK_REL} \
            --bundle-root /workspace/${OUTPUT_REL} \
            --device ${DEVICE}
        python3 -m model_utils.smolvla_export.build_hmm_modules \
            --output-dir /workspace/${WORK_REL}
        python3 -m model_utils.smolvla_export.package_hmm_modules \
            --bundle-root /workspace/${OUTPUT_REL} \
            --output-dir /workspace/${WORK_REL} \
            --deployment hmm \
            --target-soc lq50 \
            --target-runtime tcim-lite
    "
