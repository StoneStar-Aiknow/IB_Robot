#!/usr/bin/env bash
# Regression harness for the accelerator axis: detection, the ambiguous-host
# refusal, and the accelerator hooks that gate the Torch grasp stack.
#
# The accelerator axis is independent of the OS axis, so none of these cases
# set SETUP_PLATFORM_ID: an Ascend NPU is deployed on both openEuler and
# Ubuntu, and the Torch build cannot be derived from the OS.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

WORKSPACE="${REPO_ROOT}"
# detect.sh resolves the accelerator impl relative to SCRIPT_DIR/setup.
SCRIPT_DIR="${REPO_ROOT}/scripts"

log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

# shellcheck disable=SC1090
source "${REPO_ROOT}/scripts/setup/detect.sh"

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

# The host running this test may itself have a real NVIDIA GPU or CANN
# toolkit, so every case stubs both probes. They are stubbed as shell
# functions rather than as a fake nvidia-smi on PATH because the board targets
# mount /tmp noexec, where an executable fixture cannot run.
NO_CANN="${TMP_ROOT}/absent-cann"
CANN_81="${TMP_ROOT}/cann-8.1"
CANN_82="${TMP_ROOT}/cann-8.2"
CANN_83="${TMP_ROOT}/cann-8.3"
CANN_85="${TMP_ROOT}/cann-8.5"
CANN_77="${TMP_ROOT}/cann-7.0.0"
mkdir -p "${CANN_81}" "${CANN_82}" "${CANN_83}" "${CANN_85}" "${CANN_77}"
printf 'version=8.1.RC1\n' > "${CANN_81}/version.cfg"
printf 'version=8.2.RC1\n' > "${CANN_82}/version.cfg"
printf 'version=8.3.RC1\n' > "${CANN_83}/version.cfg"
printf 'version=8.5.RC1\n' > "${CANN_85}/version.cfg"
printf 'version=7.0.0\n' > "${CANN_77}/version.cfg"

# $1/$2: "yes" or "no" for the NVIDIA and Ascend probes.
accel_id_for() {
    local gpu="$1" npu="$2"
    (
        _accel_has_nvidia_gpu() { [[ "${gpu}" == "yes" ]]; }
        _accel_has_ascend_npu() { [[ "${npu}" == "yes" ]]; }
        ACCEL_OVERRIDE="" detect_accel_id 2>/dev/null || echo "DETECT_FAILED"
    )
}

# Load an accelerator impl in a subshell and report one hook's result, so the
# three impls cannot leak function definitions into each other.
grasp_support_for() {
    local accel_id="$1"
    (
        SETUP_ACCEL_ID="${accel_id}"
        load_accel_impl
        if accel_supports_torch_grasp; then echo "yes"; else echo "no"; fi
    )
}

constraints_for() {
    local accel_id="$1" cann_root="${2:-${NO_CANN}}"
    (
        SETUP_ACCEL_ID="${accel_id}"
        IBR_CANN_TOOLKIT_ROOT="${cann_root}"
        load_accel_impl
        # stdout is the constraint path; warnings go to stderr.
        accel_pip_constraints 2>/dev/null
    )
}

echo "== accelerator detection =="
assert_eq "NVIDIA GPU alone resolves to cuda" \
    "cuda" "$(accel_id_for yes no)"
assert_eq "Ascend NPU alone resolves to cann" \
    "cann" "$(accel_id_for no yes)"
assert_eq "no accelerator resolves to cpu" \
    "cpu" "$(accel_id_for no no)"
assert_eq "a host with both refuses to guess" \
    "DETECT_FAILED" "$(accel_id_for yes yes)"

echo "== the Ascend probe reads devices, not the CANN toolkit =="
# A CANN toolkit with no NPU is a real configuration: a workstation that only
# converts OM bundles. Probing the toolkit made such a host, if it also had an
# NVIDIA GPU, look like it had two accelerators and refuse to run setup.
mkdir -p "${TMP_ROOT}/dev-npu" "${TMP_ROOT}/dev-none"
: > "${TMP_ROOT}/dev-npu/davinci0"
# The manager node exists on hosts without a compute device; it must not count.
: > "${TMP_ROOT}/dev-none/davinci_manager"
probe_ascend() {
    if IBR_ASCEND_DEVICE_GLOB="$1" IBR_CANN_TOOLKIT_ROOT="$2" _accel_has_ascend_npu; then
        echo "yes"
    else
        echo "no"
    fi
}
assert_eq "a davinci compute device is detected as an NPU" \
    "yes" "$(probe_ascend "${TMP_ROOT}/dev-npu/davinci[0-9]*" "${CANN_81}")"
assert_eq "a CANN toolkit without a device is not an NPU" \
    "no" "$(probe_ascend "${TMP_ROOT}/dev-none/davinci[0-9]*" "${CANN_81}")"
assert_eq "the davinci manager node alone is not a compute device" \
    "no" "$(probe_ascend "${TMP_ROOT}/dev-none/davinci[0-9]*" "${NO_CANN}")"

echo "== explicit override =="
assert_eq "ACCEL_OVERRIDE wins over detection" \
    "cann" "$(ACCEL_OVERRIDE="cann" detect_accel_id)"
assert_eq "ACCEL_OVERRIDE resolves an ambiguous host" \
    "cuda" "$(ACCEL_OVERRIDE="cuda" detect_accel_id)"

echo "== every accelerator profile implements the whole hook contract =="
# Assert the hooks are DEFINED, not merely quiet. An undefined hook is
# indistinguishable from a no-op one when a test compares stdout only: bash
# reports "command not found" on stderr and leaves stdout empty. That is how a
# missing accel_lerobot_requirements in cuda.sh passed this suite while
# aborting a real `--accel cuda` setup run at exit 127.
ACCEL_HOOKS=(
    accel_supports_torch_grasp
    accel_pip_constraints
    accel_requirements
    accel_verify_runtime
    accel_post_requirements
)
for accel_id in cuda cann cpu; do
    undefined_hooks="$(
        (
            SETUP_ACCEL_ID="${accel_id}"
            load_accel_impl
            for hook in "${ACCEL_HOOKS[@]}"; do
                declare -F "${hook}" >/dev/null 2>&1 || printf '%s\n' "${hook}"
            done
        ) | sort | paste -sd, -
    )"
    assert_eq "the ${accel_id} profile defines every accelerator hook" \
        "" "${undefined_hooks}"
done

echo "== Torch grasp stack is gated on the accelerator, not the OS =="
assert_eq "cuda installs the Torch grasp stack" "yes" "$(grasp_support_for cuda)"
assert_eq "cann skips the Torch grasp stack" "no" "$(grasp_support_for cann)"
assert_eq "cpu skips the Torch grasp stack" "no" "$(grasp_support_for cpu)"

echo "== Torch ABI constraints =="
assert_eq "cuda needs no global constraint" \
    "" "$(constraints_for cuda)"
assert_eq "cpu needs no global constraint" \
    "" "$(constraints_for cpu)"
assert_eq "CANN 8.1 pins its Torch ABI constraint" \
    "${REPO_ROOT}/requirements/constraints-cann-8.1.txt" \
    "$(constraints_for cann "${CANN_81}")"
assert_eq "CANN 8.3 pins its Torch ABI constraint" \
    "${REPO_ROOT}/requirements/constraints-cann-8.3.txt" \
    "$(constraints_for cann "${CANN_83}")"
assert_eq "an unsupported CANN release pins nothing" \
    "" "$(constraints_for cann "${CANN_82}")"

echo "== an unsupported ABI installs nothing but never aborts setup =="
# Installing some other CANN release's torch_npu resolves cleanly and only
# fails at runtime, so the hook must install nothing rather than guess. It
# must also NOT fail: by this point LeRobot is already installed, and an abort
# strands a half-installed workspace. No toolkit at all is the other empty
# case — the release is unknowable, so nothing is installed either.
accel_requirements_rc_for() {
    local cann_root="$1" rc=0 out
    out="$(
        SETUP_ACCEL_ID="cann"
        IBR_CANN_TOOLKIT_ROOT="${cann_root}"
        load_accel_impl
        accel_requirements 2>/dev/null
    )" || rc=$?
    printf 'rc=%s out=%s\n' "${rc}" "${out##*/}"
}
assert_eq "CANN 8.1 selects its requirements file" \
    "rc=0 out=accel-cann-8.1.txt" "$(accel_requirements_rc_for "${CANN_81}")"
assert_eq "CANN 8.3 selects its requirements file" \
    "rc=0 out=accel-cann-8.3.txt" "$(accel_requirements_rc_for "${CANN_83}")"
assert_eq "an unsupported CANN release installs nothing and continues" \
    "rc=0 out=" "$(accel_requirements_rc_for "${CANN_82}")"
assert_eq "CANN 8.5 installs nothing and continues" \
    "rc=0 out=" "$(accel_requirements_rc_for "${CANN_85}")"
assert_eq "a release absent from the compatibility table continues" \
    "rc=0 out=" "$(accel_requirements_rc_for "${CANN_77}")"
assert_eq "no detected toolkit installs nothing instead of guessing" \
    "rc=0 out=" "$(accel_requirements_rc_for "${NO_CANN}")"

echo "== diagnostics land on stderr, never on the hook's stdout =="
# accel_requirements' stdout IS the requirements path the caller installs, so a
# warning or an error there would be read as a file name.
cann_stderr_for() {
    (
        SETUP_ACCEL_ID="cann"
        IBR_CANN_TOOLKIT_ROOT="$1"
        load_accel_impl
        accel_requirements 2>&1 >/dev/null | grep -c "supported Torch ABI"
    )
}
assert_eq "no detected toolkit explains itself on stderr" \
    "1" "$(cann_stderr_for "${NO_CANN}")"
assert_eq "an unsupported release explains itself on stderr" \
    "1" "$(cann_stderr_for "${CANN_82}")"
assert_eq "CANN 8.1 emits no diagnostic" \
    "0" "$(cann_stderr_for "${CANN_81}")"
assert_eq "CANN 8.3 emits no diagnostic" \
    "0" "$(cann_stderr_for "${CANN_83}")"
# The deferral must come with instructions an operator can follow without
# reading the source: the Ascend compatibility lookup and the venv pip.
cann_guidance_for() {
    (
        SETUP_ACCEL_ID="cann"
        IBR_CANN_TOOLKIT_ROOT="$1"
        load_accel_impl
        accel_requirements 2>&1 >/dev/null | grep -c "hiascend.com/developer/download/compatibility"
    )
}
assert_eq "an unsupported release points to the Ascend compatibility table" \
    "1" "$(cann_guidance_for "${CANN_82}")"
assert_eq "an unknown release points to the Ascend compatibility table" \
    "1" "$(cann_guidance_for "${CANN_77}")"
assert_eq "a host without a toolkit gets no pairing lookup to follow" \
    "0" "$(cann_guidance_for "${NO_CANN}")"
assert_eq "CANN 8.1 needs no manual guidance" \
    "0" "$(cann_guidance_for "${CANN_81}")"
assert_eq "CANN 8.3 needs no manual guidance" \
    "0" "$(cann_guidance_for "${CANN_83}")"
assert_eq "CANN 8.1 constraints go to stdout alone" \
    "${REPO_ROOT}/requirements/constraints-cann-8.1.txt" \
    "$( ( SETUP_ACCEL_ID=cann; IBR_CANN_TOOLKIT_ROOT="${CANN_81}"; load_accel_impl; accel_pip_constraints 2>/dev/null ) )"

echo "== ACCEL_OVERRIDE reaches detection from the environment =="
# The ambiguous-host error tells the user to set this variable, so setup.sh
# must not clear it during its own initialization.
#
# Point WORKSPACE at an empty directory, which setup.sh honors. The accelerator
# is resolved before the first mutating step (update_submodules), so a correct
# setup.sh never reaches it; should this guard ever regress, that step runs git
# in a non-repository and fails instead of touching the real checkout.
SAFE_WORKSPACE="${TMP_ROOT}/empty-workspace"
mkdir -p "${SAFE_WORKSPACE}"
override_rc=0
override_output="$(env -i HOME="${HOME}" USER="${USER:-$(id -un)}" TERM=dumb \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    WORKSPACE="${SAFE_WORKSPACE}" \
    ACCEL_OVERRIDE="definitely-not-an-accel" \
    timeout 60 bash "${REPO_ROOT}/scripts/setup.sh" --yes --no-sudo 2>&1)" || override_rc=$?
assert_eq "an invalid ACCEL_OVERRIDE from the environment fails setup closed" \
    "1" "${override_rc}"
assert_eq "the failure names the inherited accelerator, so it was not cleared" \
    "1" "$(printf '%s' "${override_output}" | grep -c "Unsupported accelerator 'definitely-not-an-accel'")"

echo "== setup stays usable on a minimal rootfs =="
# Deliberately a structural assertion. bash implements process substitution
# `< <(...)` through /dev/fd, which requires /proc to be mounted; a minimal
# openEuler Embedded rootfs guarantees neither. Introducing one into the
# accelerator path aborted the openEuler verification outright with
# `/dev/fd/63: No such file or directory`, so guard the pattern rather than the
# symptom — the dev host always has /dev/fd, so the failure cannot be
# reproduced here behaviourally. Use a here-string (temporary file) instead.
#
# benchmark_profile.sh is exempt: the benchmark profile refuses to run anywhere
# but Ubuntu 22.04 x86_64, so it never reaches a board rootfs.
procsub_offenders="$(
    for script in \
        "${REPO_ROOT}/scripts/setup.sh" \
        "${REPO_ROOT}"/scripts/setup/*.sh \
        "${REPO_ROOT}"/scripts/setup/accels/*.sh \
        "${REPO_ROOT}"/scripts/setup/platforms/*.sh
    do
        [[ -f "${script}" ]] || continue
        case "${script}" in */benchmark_profile.sh) continue ;; esac
        # One grep, no pipe: `sed ... | grep -q` would make grep exit on first
        # match, SIGPIPE the sed, and under this file's `set -o pipefail` the
        # pipeline would report failure — silently reporting no offenders.
        # `^[^#]*<` instead requires the pattern to appear before any '#', so
        # the comment documenting this rule does not match itself.
        if grep -qE '^[^#]*<[[:space:]]*\(' "${script}"; then
            basename "${script}"
        fi
    done | sort | paste -sd, -
)"
assert_eq "no process substitution on the board-reachable setup path" \
    "" "${procsub_offenders}"

echo "== the grasp smoke test follows the install, not the OS =="
# The install moved to the accelerator axis but the smoke test kept an OS test,
# so a full setup on Ubuntu + Ascend or Ubuntu + CPU skipped the install and
# then failed importing grasp_gen — breaking the very combination this axis
# exists to support. No gate catches it: the Ubuntu gate passes --accel cuda so
# grasp installs, and the openEuler gate and the 310P are both openEuler, which
# the old OS test excluded. A full setup_python_venv run cannot be driven from a
# fixture, so assert the structure that made drift possible: the smoke test must
# key off the recorded install, and must not re-derive a platform condition.
venv_fn="$(awk '/^setup_python_venv\(\) \{/,/^\}/' "${REPO_ROOT}/scripts/setup/python_venv.sh")"
smoke_gate="$(printf '%s\n' "${venv_fn}" | grep -n 'Missing grasp_gen after default grasp install' -B 12 | grep -E '^\s*[0-9]+-\s*if \[\[' | tail -1)"
assert_eq "the grasp smoke test is gated on the recorded install" \
    "1" "$(printf '%s' "${smoke_gate}" | grep -c 'grasp_installed')"
assert_eq "the grasp smoke test no longer tests the OS" \
    "0" "$(printf '%s' "${smoke_gate}" | grep -c 'SETUP_PLATFORM_ID')"
# And the flag may only be raised where the install actually runs.
assert_eq "the install fact is set only next to the install itself" \
    "1" "$(printf '%s\n' "${venv_fn}" | grep -c 'grasp_installed=true')"

echo "== unknown accelerator =="
unknown_rc=0
( SETUP_ACCEL_ID="tpu"; load_accel_impl ) >/dev/null 2>&1 || unknown_rc=$?
assert_eq "an unknown accelerator ID is rejected" "1" "${unknown_rc}"

echo "== the pending-manual-config state is recorded by the caller, not the hook =="
# accel_requirements runs inside a $( ) command substitution, so a variable it
# assigns never reaches the caller — the state must be recorded by
# install_accel_requirements from the empty result, and only for cann (cuda
# and cpu have no NPU backend to defer). setup.sh's end summary then reads it.
assert_eq "the cann profile does not set the pending state itself" \
    "0" "$(grep -c 'ACCEL_NPU_PENDING_MANUAL_CONFIG=' "${REPO_ROOT}/scripts/setup/accels/cann.sh")"
assert_eq "the caller records the pending state once, from the empty result" \
    "1" "$(grep -c 'ACCEL_NPU_PENDING_MANUAL_CONFIG=true' "${REPO_ROOT}/scripts/setup/python_venv.sh")"
assert_eq "the setup summary surfaces the manual-configuration notice" \
    "1" "$(grep -c 'Ascend NPU runtime is NOT configured' "${REPO_ROOT}/scripts/setup.sh")"
assert_eq "the setup summary points to the Ascend compatibility lookup" \
    "1" "$(grep -c 'hiascend.com/developer/download/compatibility' "${REPO_ROOT}/scripts/setup.sh")"
assert_eq "the manual install goes through the workspace venv pip" \
    "1" "$(grep -c 'venv/bin/python -m pip install torch==' "${REPO_ROOT}/scripts/setup.sh")"

printf '\n== summary: %d passed, %d failed ==\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
