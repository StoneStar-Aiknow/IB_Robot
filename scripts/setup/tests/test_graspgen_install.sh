#!/usr/bin/env bash
# Regression harness for the GraspGen wheel install path: the Torch/torchcodec
# ABI pins in install_graspgen_torch_abi (including healing environments that
# were set up before the torchcodec pin existed) and the pointnet2_ops
# install/build/skip decision in graspgen_pointnet2ops_plan.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SETUP_LIB="${REPO_ROOT}/scripts/setup/python_venv.sh"
GRASPGEN_LIB="${REPO_ROOT}/scripts/setup/install_graspgen_pip.sh"
BENCHMARK_LIB="${REPO_ROOT}/scripts/setup/benchmark_profile.sh"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

WORKSPACE="${TMP_ROOT}/ws"
mkdir -p "${WORKSPACE}"

# Profile predicates are production dependencies of the ABI helper.
source "${REPO_ROOT}/scripts/setup/common.sh"
SETUP_PROFILE=full

log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

# shellcheck disable=SC1090
source "${SETUP_LIB}"
# shellcheck disable=SC1090
source "${GRASPGEN_LIB}"
# shellcheck disable=SC1090
source "${BENCHMARK_LIB}"

PASS=0
FAIL=0

assert_eq() {
    local name="$1"
    local expected="$2"
    local actual="$3"

    if [[ "${actual}" == "${expected}" ]]; then
        printf '  PASS  %s\n' "${name}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected: %s\n' "${expected}" >&2
        printf '    actual:   %s\n' "${actual}" >&2
        FAIL=$((FAIL + 1))
    fi
}

# The architecture guards must be testable on any host, and the accelerator
# hook must not depend on the machine running the suite.
uname() { echo "${FAKE_UNAME_M:-x86_64}"; }
accel_supports_torch_grasp() { return 0; }
SETUP_PIP_INDEX_URL="https://example.invalid/simple"
nvidia-smi() {
    case "$*" in
        *--query-gpu=compute_cap*)
            printf '%s\n' "${FAKE_GPU_CAPS:-}"
            ;;
        *-L*)
            [[ -n "${FAKE_GPU_CAPS:-}" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

# Fake venv interpreter: dispatches on the -c probe, or on heredoc code read
# from stdin, and records every probe so tests can assert the unconditional
# import verification actually ran.
FAKE_PY="${TMP_ROOT}/fake-python"
FAKE_PY_LOG="${TMP_ROOT}/fake-python-calls"
: > "${FAKE_PY_LOG}"
cat > "${FAKE_PY}" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
    printf '%s\n' "$2" >>"${FAKE_PY_LOG:?}"
    case "$2" in
        *'import torch; print(torch.__version__)'*)
            echo "${FAKE_TORCH_VERSION:-}" ;;
        *'m.version("torchcodec")'*)
            echo "${FAKE_TORCHCODEC_VERSION:-}" ;;
        *'import torchvision; print(torchvision.__version__)'*)
            echo "${FAKE_TORCHVISION_VERSION:-}" ;;
        *'torchcodec._core'*)
            exit "${FAKE_TORCHCODEC_IMPORT_RC:-0}" ;;
        *'sys.version_info'*)
            echo "${FAKE_PYTHON_VERSION:-3.10}" ;;
        *'.split(".")[0]'*)
            echo "${FAKE_TORCH_CUDA_MAJOR:-12}" ;;
        *'print(torch.version.cuda or "")'*)
            echo "${FAKE_TORCH_CUDA:-12.6}" ;;
    esac
    exit 0
fi
code="$(cat)"
printf '%s\n' "${code}" >>"${FAKE_PY_LOG:?}"
case "${code}" in
    *get_device_capability*)
        if [[ -n "${FAKE_GPU_CAPS:-}" ]]; then printf '%s\n' "${FAKE_GPU_CAPS}"; fi ;;
esac
exit 0
EOF
chmod +x "${FAKE_PY}"
VENV_PYTHON="${FAKE_PY}"
export FAKE_PY_LOG FAKE_TORCH_VERSION FAKE_TORCHCODEC_VERSION FAKE_TORCHVISION_VERSION \
    FAKE_TORCHCODEC_IMPORT_RC FAKE_PYTHON_VERSION FAKE_TORCH_CUDA FAKE_TORCH_CUDA_MAJOR \
    FAKE_GPU_CAPS FAKE_NVCC

CAPTURED_COMMANDS=""
run_cmd() {
    CAPTURED_COMMANDS+="$(printf '%q ' "$@")"$'\n'
    # Stand-in for scripts/build_pointnet2ops_wheel.sh: produce the wheel file
    # the real build would leave behind in the requested output directory.
    if [[ "${1:-}" == "env" ]]; then
        local arg outdir=""
        for arg in "$@"; do
            [[ "${arg}" == POINTNET2OPS_WHEEL_OUT_DIR=* ]] && outdir="${arg#*=}"
        done
        if [[ -n "${outdir}" ]]; then
            mkdir -p "${outdir}"
            : >"${outdir}/pointnet2_ops-3.0.0+ibrobot.1-cp310-cp310-manylinux_2_17_x86_64.whl"
            (
                cd "${outdir}"
                sha256sum pointnet2_ops-3.0.0+ibrobot.1-cp310-cp310-manylinux_2_17_x86_64.whl > SHA256SUMS
            )
        fi
    fi
}

graspgen_find_nvcc() { printf '%s\n' "${FAKE_NVCC:-}"; }
FAKE_NVCC_BIN="${TMP_ROOT}/cuda-12.6/bin/nvcc"
mkdir -p "${TMP_ROOT}/cuda-12.6/bin"
cat > "${FAKE_NVCC_BIN}" <<'EOF'
#!/usr/bin/env bash
echo "Cuda compilation tools, release 12.6, V12.6.85"
EOF
chmod +x "${FAKE_NVCC_BIN}"

POINTNET2OPS_WHEEL_NAME="pointnet2_ops-3.0.0+ibrobot.1-cp310-cp310-manylinux_2_17_x86_64.whl"

echo "== torch and torchcodec are pinned independently =="

# The regression this guards against: environments set up by master's no-nvcc
# path already carry 2.7.1+cu126 with a resolver-picked torchcodec, and a
# re-run must heal them instead of short-circuiting past the torchcodec pin.
FAKE_TORCH_VERSION="2.7.1+cu126"
FAKE_TORCHVISION_VERSION="0.22.1+cu126"
FAKE_TORCHCODEC_VERSION="0.11.1"
FAKE_TORCHCODEC_IMPORT_RC=0
CAPTURED_COMMANDS=""
: >"${FAKE_PY_LOG}"
install_graspgen_torch_abi "ok:test"
assert_eq "matching Torch installs no Torch" \
    "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'torch==2.7.1+cu126' || true)"
assert_eq "the resolver-picked torchcodec is replaced" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'torchcodec==0.5' || true)"
assert_eq "the torchcodec ABI check ran" \
    "1" "$(grep -c 'torchcodec._core' "${FAKE_PY_LOG}" || true)"

# Both versions already correct: nothing reinstalls, but the import check must
# still run — it is the only thing that catches a broken-but-versioned install.
FAKE_TORCHCODEC_VERSION="0.5"
CAPTURED_COMMANDS=""
: >"${FAKE_PY_LOG}"
install_graspgen_torch_abi "ok:test"
assert_eq "healthy environment reinstalls nothing" \
    "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'pip install' || true)"
assert_eq "the torchcodec ABI check still runs" \
    "1" "$(grep -c 'torchcodec._core' "${FAKE_PY_LOG}" || true)"

# Fresh venv: neither package present, both get pinned.
FAKE_TORCH_VERSION=""
FAKE_TORCHVISION_VERSION=""
FAKE_TORCHCODEC_VERSION=""
CAPTURED_COMMANDS=""
install_graspgen_torch_abi "ok:test"
assert_eq "missing Torch is pinned" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'torch==2.7.1+cu126' || true)"
assert_eq "missing torchcodec is pinned" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'torchcodec==0.5' || true)"

# A torchcodec that reports the right version but cannot load fails the step.
FAKE_TORCH_VERSION="2.7.1+cu126"
FAKE_TORCHVISION_VERSION="0.22.1+cu126"
FAKE_TORCHCODEC_VERSION="0.5"
FAKE_TORCHCODEC_IMPORT_RC=1
torch_abi_rc=0
install_graspgen_torch_abi "ok:test" || torch_abi_rc=$?
assert_eq "an unloadable torchcodec fails the ABI pin" "1" "${torch_abi_rc}"
FAKE_TORCHCODEC_IMPORT_RC=0

# The stack is x86_64-only (audited wheel, build script, torchcodec wheels).
FAKE_UNAME_M="aarch64"
FAKE_TORCH_VERSION=""
FAKE_TORCHVISION_VERSION=""
FAKE_TORCHCODEC_VERSION=""
CAPTURED_COMMANDS=""
: >"${FAKE_PY_LOG}"
plan="$(graspgen_pointnet2ops_plan)"
install_graspgen_torch_abi "${plan}"
assert_eq "aarch64 pins nothing" \
    "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'pip install' || true)"
assert_eq "aarch64 probes nothing" "0" "$(wc -l <"${FAKE_PY_LOG}")"
FAKE_UNAME_M="x86_64"

# Disabled profiles and non-CUDA accelerators produce skip plans and keep the
# LeRobot Torch. The ABI helper treats any skip plan as a no-op.
FAKE_TORCH_VERSION=""
FAKE_TORCHVISION_VERSION=""
FAKE_TORCHCODEC_VERSION=""
for guard_case in inference accel; do
    CAPTURED_COMMANDS=""
    install_graspgen_torch_abi "skip:${guard_case}"
    assert_eq "${guard_case} keeps the lerobot Torch" \
        "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'pip install' || true)"
done

# A mismatched TorchVision must heal the whole Torch pair, even when torch
# itself already has the expected version.
FAKE_TORCH_VERSION="2.7.1+cu126"
FAKE_TORCHVISION_VERSION="0.21.0+cu126"
FAKE_TORCHCODEC_VERSION="0.5"
CAPTURED_COMMANDS=""
install_graspgen_torch_abi "ok:compatible"
assert_eq "mismatched TorchVision reinstalls the Torch pair" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'torch==2.7.1+cu126' || true)"

# A skip plan must not downgrade a runtime that the selected GPU can use.
FAKE_TORCH_VERSION=""
FAKE_TORCHVISION_VERSION=""
FAKE_TORCHCODEC_VERSION=""
CAPTURED_COMMANDS=""
install_graspgen_torch_abi "skip:Blackwell"
assert_eq "a skipped Blackwell stack leaves Torch untouched" \
    "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'pip install' || true)"

echo "== the torchcodec version matches the validated benchmark pair =="

assert_eq "grasp and benchmark ride the same torchcodec for torch 2.7.1+cu126" \
    "${BENCHMARK_TORCHCODEC_VERSION}" "${GRASPGEN_TORCHCODEC_VERSION}"

echo "== the pointnet2_ops plan decides install, build, or skip =="

# No visible GPU (the verification-container norm): install the audited wheel.
FAKE_GPU_CAPS=""
FAKE_NVCC=""
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "no visible GPU installs the audited wheel" "ok" "${plan%%:*}"

# sm_86/87/89 run the sm_86 cubins.
FAKE_GPU_CAPS="$(printf '8.6\n8.9')"
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "sm_86 and sm_89 install the audited wheel" "ok" "${plan%%:*}"

# Incompatible pre-Blackwell GPU without a toolkit: skip with a way out.
FAKE_GPU_CAPS="8.0"
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "A100 without nvcc skips grasp" "skip" "${plan%%:*}"
assert_eq "the skip names the toolkit way out" \
    "1" "$(printf '%s' "${plan#skip:}" | grep -c 'install a CUDA 12.x toolkit' || true)"

# Incompatible pre-Blackwell GPU with a matching-major toolkit: build locally.
FAKE_NVCC="${FAKE_NVCC_BIN}"
FAKE_TORCH_CUDA_MAJOR="12"
FAKE_GPU_CAPS="8.0"
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "A100 with CUDA 12.x nvcc builds a local wheel" "build:8.0" "${plan}"

# The nvcc major must match the pinned Torch's CUDA major, not merely exist.
FAKE_TORCH_CUDA_MAJOR="12"
FAKE_GPU_CAPS="9.0"
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "H100 with CUDA 12.x nvcc builds a local wheel" "build:9.0" "${plan}"

# A mixed A100/H100 host must compile one wheel containing both architectures.
FAKE_GPU_CAPS="$(printf '9.0\n8.0\n9.0')"
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "mixed GPU hosts build every unique architecture" "build:8.0;9.0" "${plan}"

# Blackwell is beyond the pinned Torch itself: skip without an nvcc detour.
FAKE_GPU_CAPS="12.0"
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "RTX 50 skips grasp" "skip" "${plan%%:*}"
assert_eq "the Blackwell skip does not point at a toolkit" \
    "0" "$(printf '%s' "${plan#skip:}" | grep -c 'toolkit' || true)"

# Python 3.11 is supported by the workspace generally, but not by this cp310
# extension; GraspGen must be skipped without touching Torch.
FAKE_PYTHON_VERSION="3.11"
FAKE_GPU_CAPS="8.6"
CAPTURED_COMMANDS=""
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "Python 3.11 skips the cp310 grasp extension" "skip" "${plan%%:*}"
install_graspgen_torch_abi "${plan}"
assert_eq "Python 3.11 leaves Torch untouched" \
    "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'pip install' || true)"
FAKE_PYTHON_VERSION="3.10"

# Even compatible GPUs must be included when an incompatible GPU makes the
# setup choose a local wheel; that one wheel replaces the audited build.
FAKE_GPU_CAPS="$(printf '8.9\n8.0')"
plan="$(graspgen_pointnet2ops_plan)"
assert_eq "a mixed fleet builds for every GPU" "build:8.0;8.9" "${plan}"

echo "== the local wheel build caches per architecture =="

unset GRASPGEN_POINTNET2OPS_WHEEL
CAPTURED_COMMANDS=""
FAKE_GPU_CAPS="$(printf '8.0\n9.0')"
graspgen_pointnet2ops_build "8.0;9.0"
assert_eq "the build exports the local wheel path" \
    "${WORKSPACE}/venv/.cache/pointnet2_ops/sm_80_90/${POINTNET2OPS_WHEEL_NAME}" \
    "${GRASPGEN_POINTNET2OPS_WHEEL}"
assert_eq "the build pins every requested architecture" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'TORCH_CUDA_ARCH_LIST=8.0\\;9.0' || true)"
assert_eq "the build keeps the audited wheel directory untouched" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'build_pointnet2ops_wheel.sh' || true)"

CAPTURED_COMMANDS=""
graspgen_pointnet2ops_build "8.0;9.0"
assert_eq "a cached wheel is reused without rebuilding" \
    "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'build_pointnet2ops_wheel.sh' || true)"

# A partial or modified cache entry must never be trusted solely by filename.
printf 'corrupt' >>"${GRASPGEN_POINTNET2OPS_WHEEL}"
CAPTURED_COMMANDS=""
graspgen_pointnet2ops_build "8.0;9.0"
assert_eq "a corrupt cached wheel is rebuilt" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'build_pointnet2ops_wheel.sh' || true)"

echo "== the wheel installer honors the local build =="

FAKE_TORCH_VERSION="2.7.1+cu126"
FAKE_TORCH_CUDA="12.6"
FAKE_PYTHON_VERSION="3.10"
CAPTURED_COMMANDS=""
GRASPGEN_POINTNET2OPS_WHEEL="${WORKSPACE}/venv/.cache/pointnet2_ops/sm_80_90/${POINTNET2OPS_WHEEL_NAME}" \
    install_graspgen_pointnet2ops_wheel fake-pip -m pip install
assert_eq "the local wheel is installed instead of the audited one" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c "venv/.cache/pointnet2_ops/sm_80_90/${POINTNET2OPS_WHEEL_NAME}" || true)"
assert_eq "the local wheel replacement is forced" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c -- '--force-reinstall' || true)"
unset GRASPGEN_POINTNET2OPS_WHEEL

# Run the actual pre-resolution plan block with strict unset-variable checking.
# This catches newly merged upstream uses of removed profile booleans.
echo "== the early GraspGen plan respects the capability table =="
plan_block="$(awk '/^    local pointnet2ops_plan=/{capture=1} capture && /^    if \[\[ .*INSTALL_BENCHMARK_DEPS.*SETUP_PLATFORM_ID/{exit} capture{print}' "${SETUP_LIB}")"
for p in full inference training; do
    plan_output="$(
        set -eu
        unset full_profile training_profile
        SETUP_PROFILE=$p
        INSTALL_BENCHMARK_DEPS=false
        # Disable the hardware branch so this tests admission, not a GPU.
        accel_supports_torch_grasp() { echo ACCEL_CHECK; return 1; }
        run_plan_block() {
            eval "${plan_block}"
            echo "PLAN=${pointnet2ops_plan}"
        }
        run_plan_block
    )"
    expected=0
    [[ "$p" == full ]] && expected=1
    assert_eq "$p early plan consults accelerator only when grasp is enabled" \
        "$expected" "$(printf '%s' "${plan_output}" | grep -c ACCEL_CHECK || true)"
done

echo "== the grasp section follows the plan instead of aborting setup =="

venv_fn="$(awk '/^setup_python_venv\(\) \{/,/^\}/' "${SETUP_LIB}")"
assert_eq "the grasp install is driven by the pointnet2ops plan" \
    "1" "$(printf '%s\n' "${venv_fn}" | grep -c 'graspgen_pointnet2ops_plan')"
assert_eq "a skip plan installs nothing" \
    "1" "$(printf '%s\n' "${venv_fn}" | grep -c 'pointnet2ops_action}" == "skip"')"
assert_eq "a failed build degrades to a skip" \
    "1" "$(printf '%s\n' "${venv_fn}" | grep -c 'pointnet2ops_action}" == "build"')"
assert_eq "the preflight skips non-x86_64 hosts" \
    "1" "$(grep -c 'pointnet2_ops wheel and its build script are x86_64-only' "${GRASPGEN_LIB}")"
assert_eq "the recorded install still gates the smoke test" \
    "1" "$(printf '%s\n' "${venv_fn}" | grep -c 'grasp_installed=true')"

echo
if [[ "${FAIL}" -eq 0 ]]; then
    echo "All ${PASS} assertions passed."
    exit 0
fi
echo "${FAIL} of $((PASS + FAIL)) assertions failed."
exit 1
