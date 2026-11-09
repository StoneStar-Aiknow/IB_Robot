#!/usr/bin/env bash
# Regression harness for the Unitree SDK installer (install_unitree_sdk.sh):
# architecture gating, pinned source acquisition, pip invocation shape,
# verification, failure paths and the python_venv.sh hardware-branch wiring.
# Fully hermetic: git, the venv python and pip are stubbed; nothing touches
# the network, the real venv or a robot.
#
# Run: scripts/setup/tests/test_unitree_sdk_install.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

DRY_RUN=false
VERBOSE=false
source "${REPO_ROOT}/scripts/setup/common.sh"

PASS=0
FAIL=0

assert_eq() {
    local name="$1" expected="$2" actual="$3"
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

assert_contains() {
    local name="$1" haystack="$2" needle="$3"
    if [[ "${haystack}" == *"${needle}"* ]]; then
        printf '  PASS  %s\n' "${name}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    missing needle: %s\n' "${needle}" >&2
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    local name="$1" haystack="$2" needle="$3"
    if [[ "${haystack}" != *"${needle}"* ]]; then
        printf '  PASS  %s\n' "${name}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    unexpected needle: %s\n' "${needle}" >&2
        FAIL=$((FAIL + 1))
    fi
}

reset_fixtures() {
    GIT_LOG="${TMP_ROOT}/git-calls"
    PIP_LOG="${TMP_ROOT}/pip-calls"
    PY_LOG="${TMP_ROOT}/py-calls"
    : > "${GIT_LOG}"
    : > "${PIP_LOG}"
    : > "${PY_LOG}"
    SDK_HOME="${TMP_ROOT}/cache/unitree_sdk2_python"
    rm -rf "${SDK_HOME}"
    export UNITREE_SDK2_PYTHON_HOME="${SDK_HOME}"
    # The fake python runs as a subprocess: its log paths and behavior flags
    # must cross the process boundary.
    export GIT_LOG PIP_LOG PY_LOG
    export FAKE_DIRTY=0
    export FAKE_HAS_REVISION=0
    export FAKE_GIT_HEAD="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    export FAKE_CLONE_RC=0
    export FAKE_FETCH_RC=0
    export FAKE_PIP_RC=0
    export FAKE_VERIFY_RC=0
    FAKE_UNAME_M="x86_64"
}

# --- stubs --------------------------------------------------------------------

uname() { echo "${FAKE_UNAME_M}"; }

# Fake git: records every invocation; simulates clone/checkout state via
# FAKE_GIT_HEAD written into <dir>/.git/HEAD.
git() {
    echo "git $*" >> "${GIT_LOG}"
    local dir=""
    if [[ "${1:-}" == "-C" ]]; then
        dir="${2}"
        shift 2
    fi
    case "${1:-}" in
        clone)
            if [[ "${FAKE_CLONE_RC}" != "0" ]]; then
                return "${FAKE_CLONE_RC}"
            fi
            mkdir -p "${3}/.git"
            echo "${FAKE_GIT_HEAD}" > "${3}/.git/HEAD"
            return 0
            ;;
        diff)
            [[ "${FAKE_DIRTY}" == "1" ]] && return 1
            return 0
            ;;
        cat-file)
            [[ "${FAKE_HAS_REVISION}" == "1" ]] && return 0
            return 1
            ;;
        fetch)
            [[ "${FAKE_FETCH_RC}" != "0" ]] && return "${FAKE_FETCH_RC}"
            return 0
            ;;
        checkout)
            echo "${UNITREE_SDK2_PYTHON_REVISION}" > "${dir}/.git/HEAD"
            return 0
            ;;
        rev-parse)
            if [[ -f "${dir}/.git/HEAD" ]]; then
                cat "${dir}/.git/HEAD"
                return 0
            fi
            return 1
            ;;
    esac
    return 0
}

# Fake venv python: records -c probes and heredoc verification, and plays pip.
# It lives at <tmp>/fake-venv/bin/python so the venv-root derivation (dirname
# + ..) resolves exactly like a real "${VENV_PYTHON}" would.
FAKE_PY="${TMP_ROOT}/fake-venv/bin/python"
mkdir -p "${TMP_ROOT}/fake-venv/bin"
cat > "${FAKE_PY}" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "-" ]]; then
    cat > /dev/null   # consume the heredoc verification script
    echo "VERIFY" >> "${PY_LOG:?}"
    exit "${FAKE_VERIFY_RC:-0}"
fi
if [[ "${1:-}" == "-c" ]]; then
    echo "-c ${2}" >> "${PY_LOG:?}"
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
    shift 2
    echo "pip $*" >> "${PIP_LOG:?}"
    exit "${FAKE_PIP_RC:-0}"
fi
echo "fake-python: unexpected invocation: $*" >&2
exit 64
EOF
chmod +x "${FAKE_PY}"
VENV_PYTHON="${FAKE_PY}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/setup/install_unitree_sdk.sh"

pip_runner=("${FAKE_PY}" -m pip install)
PIN="${UNITREE_SDK2_PYTHON_REVISION}"

# --- cases --------------------------------------------------------------------

echo "== x86_64 fresh install: clone at the pin, cyclonedds pin, editable, verified"
reset_fixtures
install_unitree_sdk "${pip_runner[@]}"
assert_eq "fresh install succeeds" "0" "$?"
assert_contains "clones the mirror" "$(cat "${GIT_LOG}")" "clone"
assert_contains "checks out the pinned revision" "$(cat "${GIT_LOG}")" "checkout --detach ${PIN}"
assert_contains "installs the pinned cyclonedds" "$(cat "${PIP_LOG}")" "cyclonedds==0.10.2"
assert_contains "installs the SDK editable without deps" "$(cat "${PIP_LOG}")" "--no-deps -e ${SDK_HOME}"
assert_contains "import-verifies the install" "$(cat "${PY_LOG}")" "VERIFY"

echo "== repeat install with a ready pinned checkout: no clone, no fetch, pip re-run"
reset_fixtures
mkdir -p "${SDK_HOME}/.git"
echo "${PIN}" > "${SDK_HOME}/.git/HEAD"
FAKE_HAS_REVISION=1
install_unitree_sdk "${pip_runner[@]}"
assert_eq "ready-cache install succeeds" "0" "$?"
assert_not_contains "no clone when the pin is already checked out" "$(cat "${GIT_LOG}")" "clone"
assert_not_contains "no fetch when the pin is already checked out" "$(cat "${GIT_LOG}")" "fetch"
assert_contains "pip still re-runs (venv repair)" "$(cat "${PIP_LOG}")" "--no-deps"

echo "== aarch64 is reported unsupported and installs nothing"
reset_fixtures
FAKE_UNAME_M="aarch64"
OUTPUT="$(install_unitree_sdk "${pip_runner[@]}" 2>&1)"
assert_eq "aarch64 install is a clean skip" "0" "$?"
assert_contains "names the architecture as unsupported" "${OUTPUT}" "unsupported"
assert_eq "no git calls on unsupported arch" "" "$(cat "${GIT_LOG}")"
assert_eq "no pip calls on unsupported arch" "" "$(cat "${PIP_LOG}")"
assert_eq "no verification on unsupported arch" "" "$(cat "${PY_LOG}")"

echo "== dirty checkout: refuse instead of discarding"
reset_fixtures
mkdir -p "${SDK_HOME}/.git"
echo "${PIN}" > "${SDK_HOME}/.git/HEAD"
FAKE_DIRTY=1 FAKE_HAS_REVISION=1
OUTPUT="$(install_unitree_sdk "${pip_runner[@]}" 2>&1)"
assert_eq "dirty checkout fails the install" "1" "$?"
assert_contains "error names the tracked changes" "${OUTPUT}" "tracked local changes"
assert_eq "no pip calls on a dirty checkout" "" "$(cat "${PIP_LOG}")"

echo "== mismatched checkout: fetch the pinned revision and detach"
reset_fixtures
mkdir -p "${SDK_HOME}/.git"
echo "1111111111111111111111111111111111111111" > "${SDK_HOME}/.git/HEAD"
FAKE_HAS_REVISION=0
install_unitree_sdk "${pip_runner[@]}"
assert_eq "mismatched checkout is repaired" "0" "$?"
assert_contains "fetches the pinned revision" "$(cat "${GIT_LOG}")" "fetch --depth 1 origin ${PIN}"
assert_contains "detaches onto the pin" "$(cat "${GIT_LOG}")" "checkout --detach ${PIN}"

echo "== verification failure fails the install"
reset_fixtures
mkdir -p "${SDK_HOME}/.git"
echo "${PIN}" > "${SDK_HOME}/.git/HEAD"
FAKE_HAS_REVISION=1 FAKE_VERIFY_RC=1
OUTPUT="$(install_unitree_sdk "${pip_runner[@]}" 2>&1)"
assert_eq "verification failure fails the install" "1" "$?"
assert_contains "error tells the user to fix and re-run" "${OUTPUT}" "re-run setup"

echo "== dry-run: the plan is printed, nothing executes"
reset_fixtures
DRY_RUN=true
install_unitree_sdk "${pip_runner[@]}"
assert_eq "dry-run succeeds" "0" "$?"
assert_eq "no git calls under dry-run" "" "$(cat "${GIT_LOG}")"
assert_eq "no pip calls under dry-run" "" "$(cat "${PIP_LOG}")"
assert_eq "no verification under dry-run" "" "$(cat "${PY_LOG}")"
DRY_RUN=false

echo "== malformed pin refuses before any side effect"
reset_fixtures
UNITREE_SDK2_PYTHON_REVISION="814556d" install_unitree_sdk "${pip_runner[@]}"
assert_eq "short revision fails" "1" "$?"
assert_eq "no git calls for a malformed pin" "" "$(cat "${GIT_LOG}")"
assert_eq "no pip calls for a malformed pin" "" "$(cat "${PIP_LOG}")"
# The unset above must not poison later cases: re-source the helper so the
# file-scope defaults exist again, exactly as production setup.sh sees them.
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/setup/install_unitree_sdk.sh"

echo "== default cache derives from the active venv (pointnet2_ops convention)"
reset_fixtures
unset UNITREE_SDK2_PYTHON_HOME
unitree_sdk_default_home > "${TMP_ROOT}/default-home.txt"
assert_eq "default home is under the venv root" \
    "${TMP_ROOT}/fake-venv/.cache/unitree_sdk2_python" \
    "$(cat "${TMP_ROOT}/default-home.txt")"
install_unitree_sdk "${pip_runner[@]}"
assert_eq "install with derived default succeeds" "0" "$?"
assert_contains "source lands under the venv cache" "$(cat "${GIT_LOG}")" "clone"
assert_contains "editable install points at the venv cache" \
    "$(cat "${PIP_LOG}")" "--no-deps -e ${TMP_ROOT}/fake-venv/.cache/unitree_sdk2_python"

echo "== wiring: python_venv.sh exposes the helper and calls it in the hardware branch"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/setup/python_venv.sh"
assert_eq "helper sourced with python_venv.sh" "install_unitree_sdk" "$(declare -F install_unitree_sdk)"
assert_contains "helper invoked from the hardware branch" \
    "$(sed -n '/setup_profile_enables hardware/,/^    fi/p' "${REPO_ROOT}/scripts/setup/python_venv.sh")" \
    'install_unitree_sdk "${pip_install[@]}"'

# ------------------------------------------------------------------------------

echo
echo "passed: ${PASS}, failed: ${FAIL}"
[[ "${FAIL}" == "0" ]]
