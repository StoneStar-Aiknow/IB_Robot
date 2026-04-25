#!/usr/bin/env bash
# Regression harness for scripts/setup/lerobot_filter_series.py.
#
# Mocks the host facts via environment variables and asserts the filtered
# series output matches canonical fixtures. The harness purposefully does NOT
# invoke `git am`; it exercises the filter logic in isolation so it can run
# inside pre-commit and lightweight CI containers without a lerobot checkout.
#
# Usage:
#   scripts/setup/tests/test_lerobot_filter.sh
#
# Exit status:
#   0  — all fixtures pass.
#   1  — at least one fixture failed; details are printed to stderr.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
FILTER="${REPO_ROOT}/scripts/setup/lerobot_filter_series.py"
MANIFEST="${REPO_ROOT}/third_party/patches/lerobot/v0.5.1/manifest.yaml"
DEFAULT_SERIES="${REPO_ROOT}/third_party/patches/lerobot/v0.5.1/series.txt"

PASS=0
FAIL=0

PYTHON_BIN="${PYTHON_BIN:-python3}"

# run_fixture <name> <expected-series-csv> -- env=val [env=val...]
# Captures stdout from the filter and compares against the expected CSV.
run_fixture() {
    local name="$1"
    local expected="$2"
    shift 2

    local env_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        env_args+=("$1")
        shift
    done
    [[ "${1:-}" == "--" ]] && shift

    local series_file="${1:-${DEFAULT_SERIES}}"

    local actual
    actual="$(env -i PATH="${PATH}" HOME="${HOME}" \
        "${env_args[@]}" \
        "${PYTHON_BIN}" "${FILTER}" \
            --manifest "${MANIFEST}" \
            --series "${series_file}" 2>/dev/null \
        | paste -sd, -)"

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

# run_negative <name> <expected-exit> -- env=val [env=val...] -- args...
# Asserts the filter exits with the expected non-zero status.
run_negative() {
    local name="$1"
    local expected_exit="$2"
    shift 2

    local env_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        env_args+=("$1")
        shift
    done
    [[ "${1:-}" == "--" ]] && shift

    local extra_args=("$@")

    local actual_exit=0
    env -i PATH="${PATH}" HOME="${HOME}" \
        "${env_args[@]}" \
        "${PYTHON_BIN}" "${FILTER}" "${extra_args[@]}" >/dev/null 2>&1 \
        || actual_exit=$?

    if [[ "${actual_exit}" -eq "${expected_exit}" ]]; then
        printf '  PASS  %s (exit %d)\n' "${name}" "${actual_exit}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected exit: %d\n' "${expected_exit}" >&2
        printf '    actual exit:   %d\n' "${actual_exit}" >&2
        FAIL=$((FAIL + 1))
    fi
}

echo "== lerobot_filter_series regression harness =="

# Ubuntu 22.04 / Python 3.10 — must keep the legacy 5-patch series intact so
# PR #96's baseline output does not regress.
run_fixture "ubuntu-22.04 / py3.10 glibc / desktop profile" \
    "0001-python-compat-syntax-and-metadata.patch,0002-python-compat-min-version-3.10.patch,0003-python-compat-typing-unpack.patch,0005-compat-add-npu-device-detection.patch,0006-compat-add-ascend-om-config-fields.patch" \
    IBR_HOST_PYTHON_VERSION=3.10 \
    IBR_HOST_LIBC=glibc \
    IBR_LEROBOT_PROFILES=core,ros,hardware,dev \
    --

# openEuler Embedded 24.03 / Python 3.11 — must drop 0001-0003 (down-grade
# patches) but keep the Ascend-side compat patches.
run_fixture "openeuler-embedded-24.03 / py3.11 glibc" \
    "0005-compat-add-npu-device-detection.patch,0006-compat-add-ascend-om-config-fields.patch" \
    IBR_HOST_PYTHON_VERSION=3.11 \
    IBR_HOST_LIBC=glibc \
    IBR_LEROBOT_PROFILES=core,ros,hardware,openeuler \
    --

# OpenHarmony 5.1.0 / Python 3.12 / musl — must drop both the down-grade and
# the glibc-bound Ascend compat patches.
run_fixture "openharmony-5.1.0 / py3.12 musl" \
    "" \
    IBR_HOST_PYTHON_VERSION=3.12 \
    IBR_HOST_LIBC=musl \
    IBR_LEROBOT_PROFILES=core,openharmony,musl \
    --

# Force-Ascend on a glibc host — used for hardware bring-up scenarios where
# the operator wants the master-parity Ascend patches as well. We point the
# filter at the master-parity series file because the default series only
# contains 0001-0006.
MASTER_PARITY_SERIES="${REPO_ROOT}/third_party/patches/lerobot/v0.5.1/series.master-parity-candidates.txt"
if [[ -f "${MASTER_PARITY_SERIES}" ]]; then
    run_fixture "ascend-forced / py3.10 glibc / master-parity series" \
        "0007-ascend-om-act-runtime.patch,0008-ascend-3403-actwrapper.patch,0009-weighted-training.patch,0010-knowledge-distillation.patch,0011-mt-act-model.patch,0012-attention-visualization-tools.patch" \
        IBR_HOST_PYTHON_VERSION=3.10 \
        IBR_HOST_LIBC=glibc \
        IBR_LEROBOT_PROFILES=core,ascend,master-parity-candidates,om,training,distillation,models,mt-act,visualization,tooling \
        -- \
        "${MASTER_PARITY_SERIES}"
else
    printf '  SKIP  ascend-forced fixture (master-parity series file missing)\n'
fi

# Negative: malformed manifest must exit non-zero with parser error.
TMP_BAD_MANIFEST="$(mktemp)"
trap 'rm -f "${TMP_BAD_MANIFEST}"' EXIT
printf 'patches:\n  - file: 0001\n      bad-indent: true\n' > "${TMP_BAD_MANIFEST}"
run_negative "malformed manifest exits 1" 1 \
    IBR_HOST_PYTHON_VERSION=3.10 \
    IBR_HOST_LIBC=glibc \
    IBR_LEROBOT_PROFILES=core \
    -- \
    --manifest "${TMP_BAD_MANIFEST}" --series "${DEFAULT_SERIES}"

echo
echo "== summary: ${PASS} passed, ${FAIL} failed =="
[[ ${FAIL} -eq 0 ]]
