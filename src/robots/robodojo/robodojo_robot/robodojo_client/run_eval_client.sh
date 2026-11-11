#!/usr/bin/env bash
# Start RoboDojo's eval client against the robodojo_runtime ROS node (the policy server).
#
#   ROBODOJO_ROOT=<RoboDojo checkout> bash run_eval_client.sh make-wrapper   # XPolicyLab/policy/<POLICY_NAME>
#   ROBODOJO_ROOT=... TASK=general_pickup SEED=0 EVAL_NUM=3 PORT=34950 bash run_eval_client.sh
#
# Required: ROBODOJO_ROOT. The RoboDojo conda env is activated from CONDA_SH (default: the
# conda found on PATH) and CONDA_ENV (default RoboDojo).
#
# RoboDojo keeps its own main loop (src/eval_client/main.py -> run_eval -> deploy.eval_one_episode);
# EVAL_NUM episodes run back to back in one Isaac process. Results land in RoboDojo's usual
# eval_result/RoboDojo/... plus per-episode artifacts under eval_result/ibrobot_virtual_runtime/.
set -euo pipefail

# RoboDojo runs its own interpreters (conda base for `conda info`, the RoboDojo env for Isaac).
# A ROS / venv shell exports PYTHONPATH, VIRTUAL_ENV and overlay paths that make those
# interpreters import the wrong (Python 3.10) packages, so re-exec once with an allowlisted
# environment. Proxy, display, GPU and the variables of this script are kept.
if [[ "${IBROBOT_CLIENT_ENV_CLEAN:-}" != 1 ]]; then
  keep=(HOME USER LOGNAME SHELL TERM LANG LC_ALL TZ DISPLAY XAUTHORITY XDG_RUNTIME_DIR TMPDIR
        CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY
        NO_PROXY all_proxy ALL_PROXY OMNI_KIT_ACCEPT_EULA
        ROBODOJO_ROOT CONDA_SH CONDA_ENV POLICY_NAME TASK SEED PORT HOST GPU EVAL_NUM ROBODOJO_RUN_ID
        IBROBOT_RUNTIME_ARTIFACT_DIR IBROBOT_RUNTIME_GRIPPER_DEFAULT AGENT_EVAL_RUN_ID AGENT_EVAL_LAYOUT)
  args=(IBROBOT_CLIENT_ENV_CLEAN=1 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
  for name in "${keep[@]}"; do
    [[ -n "${!name+x}" ]] && args+=("${name}=${!name}")
  done
  exec env -i "${args[@]}" bash "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi
if [[ "${1:-}" == "print-env" ]]; then
  env | sort
  exit 0
fi

CLIENT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:?set ROBODOJO_ROOT to the RoboDojo checkout}"
POLICY_NAME="${POLICY_NAME:-ibrobot_direct_control}"
WRAPPER="${ROBODOJO_ROOT}/XPolicyLab/policy/${POLICY_NAME}"

make_wrapper() {
  [[ -f "${ROBODOJO_ROOT}/XPolicyLab/setup_policy_server.py" ]] || { echo "not a RoboDojo root: ${ROBODOJO_ROOT}" >&2; exit 1; }
  mkdir -p "${WRAPPER}"
  ln -sfn "${CLIENT_DIR}/__init__.py" "${WRAPPER}/__init__.py"
  ln -sfn "${CLIENT_DIR}/deploy.py" "${WRAPPER}/deploy.py"
  ln -sfn "${CLIENT_DIR}/artifacts.py" "${WRAPPER}/artifacts.py"
  sed "s/^policy_name:.*/policy_name: ${POLICY_NAME}/" "${CLIENT_DIR}/deploy.yml" > "${WRAPPER}/deploy.yml"
  echo "wrapper ready: ${WRAPPER}"
  ls -l "${WRAPPER}"
}

if [[ "${1:-}" == "make-wrapper" ]]; then
  make_wrapper
  exit 0
fi
# Re-link when the wrapper points at another checkout (e.g. an older worktree).
[[ "$(readlink -f "${WRAPPER}/deploy.py" 2>/dev/null)" == "${CLIENT_DIR}/deploy.py" ]] || make_wrapper

TASK="${TASK:-general_pickup}"
SEED="${SEED:-0}"
PORT="${PORT:-34950}"
HOST="${HOST:-localhost}"
GPU="${GPU:-0}"
export EVAL_NUM="${EVAL_NUM:-3}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-vrt_${TASK}_s${SEED}_$(date +%m%d_%H%M%S)}"

# shellcheck disable=SC1091
if [[ -z "${CONDA_SH:-}" ]]; then
  for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3" /opt/conda; do
    [[ -f "${candidate}/etc/profile.d/conda.sh" ]] && { CONDA_SH="${candidate}/etc/profile.d/conda.sh"; break; }
  done
  [[ -n "${CONDA_SH:-}" ]] || { echo "set CONDA_SH (conda.sh of the RoboDojo conda install)" >&2; exit 1; }
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-RoboDojo}"
cd "${ROBODOJO_ROOT}"
export PYTHONPATH="${ROBODOJO_ROOT}:${ROBODOJO_ROOT}/XPolicyLab${PYTHONPATH:+:${PYTHONPATH}}"
echo "[virtual-runtime client] run_id=${ROBODOJO_RUN_ID} task=${TASK} seed=${SEED} episodes=${EVAL_NUM} port=${PORT}"
exec bash XPolicyLab/utils/setup_env_client.sh "${ROBODOJO_ROOT}/XPolicyLab/utils" "${WRAPPER}/deploy.yml" \
  RoboDojo "${PORT}" RoboDojo "${TASK}" arx_x5 "${POLICY_NAME}" "ckpt_name=ibrobot,action_type=joint" \
  "${ROBODOJO_ROOT}" "${SEED}" "${GPU}" "${HOST}" ws
