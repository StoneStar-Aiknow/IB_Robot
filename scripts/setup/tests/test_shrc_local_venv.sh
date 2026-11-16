#!/usr/bin/env bash
# Regression harness for the IBROBOT_VENV selector in .shrc_local.
#
# Covered behaviour:
#   - unset IBROBOT_VENV keeps the workspace-local venv (backward compatible)
#   - explicit IBROBOT_VENV wins over a local venv and over an active venv
#   - invalid IBROBOT_VENV fails closed: non-zero and no environment mutation
#   - empty IBROBOT_VENV behaves like unset
#   - repeated source is idempotent (venv bin on PATH once, PYTHONPATH entries once)
#   - sourcing does not leak `set -e` into the caller
#   - bash and zsh behave the same (zsh checks are skipped with a notice if absent)
#
# Hermetic: temp fixtures only, no network, no package installs. Fixture venvs
# are created with `python3 -m venv --without-pip`; no shared venv is touched.
#
# Usage:
#   scripts/setup/tests/test_shrc_local_venv.sh
#
# Exit status:
#   0 — all checks pass (skipped zsh checks do not fail the run).
#   1 — at least one check failed, or fixture creation failed.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SHRL="${REPO_ROOT}/.shrc_local"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

PASS=0
FAIL=0
SKIP=0

ok() { PASS=$((PASS + 1)); printf '  PASS  %s\n' "$1"; }

ko() {
    FAIL=$((FAIL + 1))
    printf '  FAIL  %s\n' "$1" >&2
    if [[ $# -gt 1 ]]; then
        printf '        %s\n' "$2" >&2
    fi
}

skip() { SKIP=$((SKIP + 1)); printf '  SKIP  %s\n' "$1"; }

assert_contains() {
    if grep -qF -- "$2" <<<"$3"; then
        ok "$1"
    else
        ko "$1" "missing: $2"
        printf '        ---- actual ----\n%s\n' "$3" >&2
    fi
}

assert_not_contains() {
    if grep -qF -- "$2" <<<"$3"; then
        ko "$1" "unexpected: $2"
        printf '        ---- actual ----\n%s\n' "$3" >&2
    else
        ok "$1"
    fi
}

assert_eq() {
    if [[ "$2" == "$3" ]]; then
        ok "$1"
    else
        ko "$1" "expected '$2', got '$3'"
    fi
}

PROBE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[[ -f "${SHRL}" ]] || { ko ".shrc_local exists"; exit 1; }

# 0. no machine-specific venv path is baked into the tracked script.
if grep -q '/home/' "${SHRL}"; then
    ko ".shrc_local carries no hardcoded /home/ path" "$(grep -n '/home/' "${SHRL}")"
else
    ok ".shrc_local carries no hardcoded /home/ path"
fi

# ---------------------------------------------------------------------------
# Fixtures: a stub workspace (copy of .shrc_local + local venv) and a second
# "shared" venv outside of it. Both are real venvs but pip-less.
# ---------------------------------------------------------------------------
WS="${TMP_ROOT}/ws"
SHARED="${TMP_ROOT}/shared"
mkdir -p "${WS}"
cp "${SHRL}" "${WS}/.shrc_local"

create_venv() {
    if ! python3 -m venv --without-pip "$1" >/dev/null 2>&1; then
        ko "create fixture venv: $1"
        exit 1
    fi
}

create_venv "${WS}/venv"
create_venv "${SHARED}"

PY_VER="python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
LOCAL_SITE="${WS}/venv/lib/${PY_VER}/site-packages"
SHARED_SITE="${SHARED}/lib/${PY_VER}/site-packages"

printf 'fixtures: ws=%s shared=%s\n' "${WS}" "${SHARED}"

# Probe body: <extra lines before source> + source + machine-readable state.
write_probe() {
    local name="$1" extra="$2" file="${TMP_ROOT}/$1.sh"
    {
        printf '%s\n' 'cd "$PROBE_WS" || exit 90'
        printf '%s\n' "${extra}"
        cat <<'PROBE'
source ./.shrc_local
printf 'RC=%s\n' "$?"
printf 'VIRTUAL_ENV=%s\n' "${VIRTUAL_ENV:-<unset>}"
printf 'PYTHON3=%s\n' "$(command -v python3)"
printf 'WORKSPACE=%s\n' "${WORKSPACE:-<unset>}"
printf 'PYTHONPATH=%s\n' "${PYTHONPATH:-<unset>}"
printf 'OLD_BIN_COUNT=%s\n' "$(printf '%s' "$PATH" | tr ':' '\n' | grep -Fxc "${PROBE_OLD_BIN}" || true)"
printf 'NEW_BIN_COUNT=%s\n' "$(printf '%s' "$PATH" | tr ':' '\n' | grep -Fxc "${PROBE_SHARED}/bin" || true)"
printf 'LEROBOT_COUNT=%s\n' "$(printf '%s' "${PYTHONPATH:-}" | tr ':' '\n' | grep -Fxc "${PROBE_WS}/libs/lerobot/src" || true)"
printf 'SITE_COUNT=%s\n' "$(printf '%s' "${PYTHONPATH:-}" | tr ':' '\n' | grep -Fxc "${VIRTUAL_ENV:-/nonexistent}/lib/${PROBE_PY_VER}/site-packages" || true)"
printf 'SET_E=%s\n' "$(case $- in *e*) echo yes;; *) echo no;; esac)"
PROBE
    } >"${file}"
}

run_probe() {  # <shell> <probe name> [extra env assignments...]
    local shell="$1" name="$2"
    shift 2
    local -a flags
    if [[ "${shell}" == "bash" ]]; then
        flags=(--noprofile --norc)
    else
        flags=(-f)
    fi
    env -i HOME="${HOME}" PATH="${PROBE_PATH}" TERM=dumb LANG=C.UTF-8 \
        PROBE_WS="${WS}" PROBE_SHARED="${SHARED}" PROBE_OLD_BIN="${WS}/venv/bin" \
        PROBE_PY_VER="${PY_VER}" \
        "$@" "${shell}" "${flags[@]}" "${TMP_ROOT}/${name}.sh" 2>&1
}

# ---------------------------------------------------------------------------
# 1. unset IBROBOT_VENV: the workspace-local venv stays the default.
# ---------------------------------------------------------------------------
write_probe probe_local ""
LOCAL_OUT="$(run_probe bash probe_local)"

assert_contains "local venv: source returns 0" "RC=0" "${LOCAL_OUT}"
assert_contains "local venv: VIRTUAL_ENV points at the local venv" "VIRTUAL_ENV=${WS}/venv" "${LOCAL_OUT}"
assert_contains "local venv: python3 resolves inside the local venv" "PYTHON3=${WS}/venv/bin/python3" "${LOCAL_OUT}"
assert_contains "local venv: WORKSPACE is the workspace root" "WORKSPACE=${WS}" "${LOCAL_OUT}"
assert_contains "local venv: PYTHONPATH exposes the local site-packages" "${LOCAL_SITE}" "${LOCAL_OUT}"
assert_contains "local venv: sourcing does not leak set -e" "SET_E=no" "${LOCAL_OUT}"

# ---------------------------------------------------------------------------
# 2. explicit IBROBOT_VENV wins over the local venv.
# ---------------------------------------------------------------------------
write_probe probe_shared ""
SHARED_OUT="$(run_probe bash probe_shared "IBROBOT_VENV=${SHARED}")"

assert_contains "shared venv: source returns 0" "RC=0" "${SHARED_OUT}"
assert_contains "shared venv: VIRTUAL_ENV points at the shared venv" "VIRTUAL_ENV=${SHARED}" "${SHARED_OUT}"
assert_contains "shared venv: python3 resolves inside the shared venv" "PYTHON3=${SHARED}/bin/python3" "${SHARED_OUT}"
assert_contains "shared venv: WORKSPACE still points at the worktree" "WORKSPACE=${WS}" "${SHARED_OUT}"
assert_contains "shared venv: PYTHONPATH exposes the shared site-packages" "${SHARED_SITE}" "${SHARED_OUT}"
assert_not_contains "shared venv: local venv site-packages is not injected" "${LOCAL_SITE}" "${SHARED_OUT}"

# ---------------------------------------------------------------------------
# 3. empty IBROBOT_VENV is treated as unset.
# ---------------------------------------------------------------------------
EMPTY_OUT="$(run_probe bash probe_local "IBROBOT_VENV=")"
assert_contains "empty selector: local venv still wins" "VIRTUAL_ENV=${WS}/venv" "${EMPTY_OUT}"
assert_contains "empty selector: source returns 0" "RC=0" "${EMPTY_OUT}"

# ---------------------------------------------------------------------------
# 4. invalid IBROBOT_VENV fails closed before touching the environment.
# ---------------------------------------------------------------------------
mkdir -p "${TMP_ROOT}/not-a-venv"
: >"${TMP_ROOT}/plain-file"

for entry in "missing:${TMP_ROOT}/does-not-exist" "no-activate:${TMP_ROOT}/not-a-venv" "plain-file:${TMP_ROOT}/plain-file"; do
    label="${entry%%:*}"
    path="${entry#*:}"
    name="probe_invalid_${label}"
    write_probe "${name}" ""
    out="$(run_probe bash "${name}" "IBROBOT_VENV=${path}")"
    assert_not_contains "invalid selector (${label}): non-zero status is reported" "RC=0" "${out}"
    assert_contains "invalid selector (${label}): clearly names IBROBOT_VENV" "IBROBOT_VENV=${path}" "${out}"
    assert_contains "invalid selector (${label}): explains the valid-venv requirement" "不是有效的 venv" "${out}"
    assert_contains "invalid selector (${label}): WORKSPACE is not exported" "WORKSPACE=<unset>" "${out}"
    assert_contains "invalid selector (${label}): PYTHONPATH is not exported" "PYTHONPATH=<unset>" "${out}"
    assert_contains "invalid selector (${label}): falls back to no venv activation" "VIRTUAL_ENV=<unset>" "${out}"
done

# ---------------------------------------------------------------------------
# 5. an already-active different venv is replaced by the selector.
# ---------------------------------------------------------------------------
write_probe probe_active 'source "${PROBE_WS}/venv/bin/activate"'
ACTIVE_OUT="$(run_probe bash probe_active "IBROBOT_VENV=${SHARED}")"

assert_contains "active venv: selector wins over the active venv" "VIRTUAL_ENV=${SHARED}" "${ACTIVE_OUT}"
assert_contains "active venv: old venv bin leaves PATH" "OLD_BIN_COUNT=0" "${ACTIVE_OUT}"
assert_contains "active venv: new venv bin is on PATH exactly once" "NEW_BIN_COUNT=1" "${ACTIVE_OUT}"

# ---------------------------------------------------------------------------
# 6. repeated source is idempotent for the venv and for the injected entries.
# ---------------------------------------------------------------------------
write_probe probe_repeat 'source ./.shrc_local >/dev/null 2>&1'
REPEAT_OUT="$(run_probe bash probe_repeat "IBROBOT_VENV=${SHARED}")"

assert_contains "repeat: second source still returns 0" "RC=0" "${REPEAT_OUT}"
assert_contains "repeat: venv bin is on PATH exactly once" "NEW_BIN_COUNT=1" "${REPEAT_OUT}"
assert_contains "repeat: libs/lerobot/src appears once" "LEROBOT_COUNT=1" "${REPEAT_OUT}"
assert_contains "repeat: venv site-packages appears once" "SITE_COUNT=1" "${REPEAT_OUT}"

# ---------------------------------------------------------------------------
# 7. zsh parity for the main cases (skipped when zsh is unavailable).
# ---------------------------------------------------------------------------
if command -v zsh >/dev/null 2>&1; then
    ZSH_LOCAL="$(run_probe zsh probe_local)"
    assert_contains "zsh local venv: VIRTUAL_ENV points at the local venv" "VIRTUAL_ENV=${WS}/venv" "${ZSH_LOCAL}"
    assert_contains "zsh local venv: python3 resolves inside the local venv" "PYTHON3=${WS}/venv/bin/python3" "${ZSH_LOCAL}"

    ZSH_SHARED="$(run_probe zsh probe_shared "IBROBOT_VENV=${SHARED}")"
    assert_contains "zsh shared venv: VIRTUAL_ENV points at the shared venv" "VIRTUAL_ENV=${SHARED}" "${ZSH_SHARED}"
    assert_contains "zsh shared venv: PYTHONPATH exposes the shared site-packages" "${SHARED_SITE}" "${ZSH_SHARED}"

    ZSH_INVALID="$(run_probe zsh probe_invalid_missing "IBROBOT_VENV=${TMP_ROOT}/does-not-exist")"
    assert_not_contains "zsh invalid selector: non-zero status is reported" "RC=0" "${ZSH_INVALID}"
    assert_contains "zsh invalid selector: WORKSPACE is not exported" "WORKSPACE=<unset>" "${ZSH_INVALID}"

    ZSH_REPEAT="$(run_probe zsh probe_repeat "IBROBOT_VENV=${SHARED}")"
    assert_contains "zsh repeat: venv bin is on PATH exactly once" "NEW_BIN_COUNT=1" "${ZSH_REPEAT}"
else
    skip "zsh not installed; bash-only run"
fi

printf '\nsummary: %d passed, %d failed, %d skipped\n' "${PASS}" "${FAIL}" "${SKIP}"
((FAIL == 0))
