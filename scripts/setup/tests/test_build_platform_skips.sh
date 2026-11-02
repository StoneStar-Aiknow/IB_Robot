#!/usr/bin/env bash
# Exercise build.sh's actual skip arguments without running its build/setup steps.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# Only evaluate this bounded block; never source the side-effecting build script.
PLATFORM_SKIP_CODE="$(sed -n '/^# Platform-specific package skips$/,/^fi$/p' "${REPO_ROOT}/scripts/build.sh")"
[[ -n "${PLATFORM_SKIP_CODE}" ]]

unset IBR_BUILD_INCLUDE_FAST_CALIB_ON_OPENEULER
unset IBR_BUILD_INCLUDE_SIM_MODELS_ON_OPENEULER
unset IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER

check_args() (
    local name="$1" platform="$2"
    shift 2
    detect_host_metadata() { :; }
    detect_platform_id() { printf '%s\n' "${platform}"; }
    log_info() { :; }
    eval "${PLATFORM_SKIP_CODE}"
    if [[ ${#PLATFORM_ARGS[@]} -ne $# || "${PLATFORM_ARGS[*]}" != "$*" ]]; then
        printf '  FAIL  %s: expected [%s], got [%s]\n' "${name}" "$*" "${PLATFORM_ARGS[*]}" >&2
        exit 1
    fi
    printf '  PASS  %s\n' "${name}"
)

openeuler=openeuler-embedded-24.03
check_args "openEuler defaults" "${openeuler}" --packages-ignore fast_calib sim_models aimdk_mujoco
IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER=1 \
    check_args "standalone simulator opt-in only" "${openeuler}" --packages-ignore fast_calib sim_models
IBR_BUILD_INCLUDE_FAST_CALIB_ON_OPENEULER=1 \
    check_args "fast_calib opt-in only" "${openeuler}" --packages-ignore sim_models aimdk_mujoco
IBR_BUILD_INCLUDE_SIM_MODELS_ON_OPENEULER=1 \
    check_args "sim_models opt-in only" "${openeuler}" --packages-ignore fast_calib aimdk_mujoco
IBR_BUILD_INCLUDE_FAST_CALIB_ON_OPENEULER=1 IBR_BUILD_INCLUDE_SIM_MODELS_ON_OPENEULER=1 \
    check_args "existing opt-ins do not include standalone simulator" "${openeuler}" --packages-ignore aimdk_mujoco
IBR_BUILD_INCLUDE_FAST_CALIB_ON_OPENEULER=1 IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER=1 \
    check_args "fast_calib and standalone simulator opt-ins" "${openeuler}" --packages-ignore sim_models
IBR_BUILD_INCLUDE_SIM_MODELS_ON_OPENEULER=1 IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER=1 \
    check_args "sim_models and standalone simulator opt-ins" "${openeuler}" --packages-ignore fast_calib
IBR_BUILD_INCLUDE_FAST_CALIB_ON_OPENEULER=1 IBR_BUILD_INCLUDE_SIM_MODELS_ON_OPENEULER=1 \
    IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER=1 \
    check_args "all opt-ins omit the ignore flag" "${openeuler}"
IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER=true \
    check_args "standalone simulator requires exactly 1" "${openeuler}" --packages-ignore fast_calib sim_models aimdk_mujoco
check_args "Ubuntu defaults unchanged" ubuntu-22.04
IBR_BUILD_INCLUDE_FAST_CALIB_ON_OPENEULER=1 IBR_BUILD_INCLUDE_SIM_MODELS_ON_OPENEULER=1 \
    IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER=1 \
    check_args "Ubuntu ignores openEuler opt-ins" ubuntu-22.04

printf 'All platform skip checks passed.\n'
