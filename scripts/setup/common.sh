#!/bin/bash
# common.sh - Shared infrastructure for IB_Robot setup scripts

# Defensive defaults for dependent variables
SUDO_AUTH_READY="${SUDO_AUTH_READY:-false}"
USE_SUDO="${USE_SUDO:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_done()    { echo -e "${GREEN}[DONE]${NC} $*"; }

# Helper to run commands with or without sudo
run_sudo() {
    if [[ "${USE_SUDO}" == true ]]; then
        sudo "$@"
    else
        "$@"
    fi
}

run_privileged_with_live_output() {
    local msg="$1"
    local command_status
    local errexit_enabled=false
    shift
    log_info "${msg}"
    if [[ "${VERBOSE:-0}" == "1" ]]; then
        run_sudo "$@"
    else
        [[ $- == *e* ]] && errexit_enabled=true
        set +e
        run_sudo "$@" 2>&1 | grep -vE "^[[:space:]]*$"
        command_status=${PIPESTATUS[0]}
        ${errexit_enabled} && set -e
        return "${command_status}"
    fi
}

run_with_live_output() {
    local msg="$1"
    local command_status
    local errexit_enabled=false
    shift
    log_info "${msg}"
    if [[ "${VERBOSE:-0}" == "1" ]]; then
        "$@"
    else
        [[ $- == *e* ]] && errexit_enabled=true
        set +e
        "$@" 2>&1 | grep -vE "^[[:space:]]*$"
        command_status=${PIPESTATUS[0]}
        ${errexit_enabled} && set -e
        return "${command_status}"
    fi
}

ensure_sudo_session() {
    if [[ "${USE_SUDO}" != true || "${SUDO_AUTH_READY}" == true ]]; then
        return 0
    fi

    log_info "Sudo authentication required for upcoming system package operations."
    if ! sudo -n true 2>/dev/null; then
        sudo -v
    fi
    SUDO_AUTH_READY=true
}

print_cmd() {
    printf '%q ' "$@"
    printf '
'
}
run_cmd() {
    if [[ "${DRY_RUN}" == true ]]; then
        echo -n "[DRY-RUN] "
        print_cmd "$@"
        return 0
    fi
    [[ "${VERBOSE}" == true ]] && { echo -n "[CMD] "; print_cmd "$@"; }
    "$@"
}
ask_yn() {
    local prompt="$1"
    local default="${2:-n}"
    if [[ "${AUTO_YES}" == true ]]; then
        echo -e "${prompt} [auto-yes: YES]"
        return 0
    fi

    local hint
    if [[ "${default}" == "y" ]]; then hint="Y/n"; else hint="y/N"; fi
    read -r -p "${prompt} [${hint}]: " REPLY
    REPLY="${REPLY:-${default}}"
    [[ "${REPLY}" == "y" || "${REPLY}" == "Y" ]]
}
ui_render_block() {
    local content="$1"
    echo "${content}"
    echo ""
}

# ---------------------------------------------------------------------------
# Setup dependency-scope profiles (SETUP_PROFILE)
#
# A profile is the role of the host being prepared: a complete development
# workspace ("full"), an edge-cloud inference verification host ("inference"),
# or a LeRobot training host ("training"). Every gating decision in setup.sh
# and its modules is a lookup into this table, so adding a profile means adding
# one row, and a profile cannot cover only part of the matrix because the
# dependency scope, the submodule scope, and the verification strictness are
# columns of the same row.
#
# Format: a row starts in column 1, '|' separates its five cells, and an
# indented line continues the row above (the physical lines are glued together
# before the cells are split, so any cell may be wrapped after a comma; cells
# are whitespace-normalised afterwards). Column alignment is cosmetic.
#
#   profile     row key, matched against SETUP_PROFILE
#   rosdep      <scope>:<paths>, where scope is full | narrow | skip and paths
#               is the comma-separated rosdep --from-paths list ("-" for a
#               profile that must not run rosdep at all); scope and paths share
#               one cell so "skip but still resolve a path set" is
#               unrepresentable
#   submodules  all  | lerobot         submodule initialisation scope
#   verify      full | inference | training
#               verification strictness; "training" selects
#               verify_training_env(), "inference" and "training" skip tracing
#   features    comma-separated capabilities, queried with
#               setup_profile_enables <feature>
#
# Unrelated to the LeRobot patch-series profiles (IBR_LEROBOT_PROFILES):
# "training" exists in both namespaces, selecting the install scope here and
# filtering patches there.
# ---------------------------------------------------------------------------
_SETUP_PROFILE_TABLE='
# profile | rosdep | submodules | verify | features
full | full:src | all | full | ros_runtime,colcon,ros_patches,ros_compat_pins,ros_abi,
        hardware,webphone,dev_tools,ubuntu_full_deps,precommit_hook,voice_tts,
        perception,fullsubnet,grasp,graspgen_abi_pin,gitlint_hook,kinematics
inference | narrow:src/ibrobot_msgs,src/tensormsg,src/inference_manifest,
        src/torch_models,src/inference_service,src/model_utils,src/dataset_tools
        | lerobot | inference | ros_runtime,colcon,ros_compat_pins,ros_abi,onnx_tooling
training | skip:- | lerobot | training | training_requirements
'

_SETUP_PROFILE_PROFILE=""
_SETUP_PROFILE_ROSDEP_CELL=""
_SETUP_PROFILE_ROSDEP=""
_SETUP_PROFILE_ROSDEP_PATHS=""
_SETUP_PROFILE_SUBMODULES=""
_SETUP_PROFILE_VERIFY=""
_SETUP_PROFILE_FEATURES=""

# Print one logical table row per line (indented continuation lines glued onto
# the row above, comments and blank lines dropped).
_setup_profile_rows() {
    local line row=""
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%%#*}"
        [[ "${line}" == *[![:space:]]* ]] || continue
        if [[ -n "${row}" && "${line}" == [[:space:]]* ]]; then
            row+="${line}"
            continue
        fi
        [[ -z "${row}" ]] || printf '%s\n' "${row}"
        row="${line}"
    done <<<"${_SETUP_PROFILE_TABLE}"
    [[ -z "${row}" ]] || printf '%s\n' "${row}"
}

# NOTE: the profile machinery deliberately avoids bash process substitution
# (`done < <(cmd)`). openEuler Embedded's chroot rootfs has no /dev/fd, so that
# redirection aborts with "/dev/fd/63: No such file or directory" before a
# single row is read (observed under the openEuler Docker verification).
# Command substitution plus a here-string works on every supported platform.

# Print the configured profile names, in table order.
setup_profile_names() {
    local rows row profile
    rows="$(_setup_profile_rows)"
    while IFS= read -r row; do
        IFS='|' read -r profile _ <<<"${row}"
        profile="${profile//[[:space:]]/}"
        [[ -n "${profile}" ]] && printf '%s\n' "${profile}"
    done <<<"${rows}"
}

# Print every capability known to the table (union over all profiles), sorted.
setup_profile_vocabulary() {
    local rows row _ _ _ _ features
    rows="$(_setup_profile_rows)"
    while IFS= read -r row; do
        IFS='|' read -r _ _ _ _ features <<<"${row}"
        features="${features//[[:space:]]/}"
        [[ -n "${features}" ]] || continue
        tr ',' '\n' <<<"${features}"
    done <<<"${rows}" | sort -u
}

# Load the active profile row into the _SETUP_PROFILE_* variables. Idempotent;
# an unknown SETUP_PROFILE is a hard error (the whitelist IS the table).
setup_profile_load() {
    local rows row pipes
    rows="$(_setup_profile_rows)"
    while IFS= read -r row; do
        IFS='|' read -r _SETUP_PROFILE_PROFILE _SETUP_PROFILE_ROSDEP_CELL \
            _SETUP_PROFILE_SUBMODULES _SETUP_PROFILE_VERIFY _SETUP_PROFILE_FEATURES <<<"${row}"
        _SETUP_PROFILE_PROFILE="${_SETUP_PROFILE_PROFILE//[[:space:]]/}"
        [[ "${_SETUP_PROFILE_PROFILE}" == "${SETUP_PROFILE:-full}" ]] || continue

        pipes="${row//[^|]/}"
        if [[ "${#pipes}" -ne 4 ]]; then
            log_error "Malformed profile row for '${_SETUP_PROFILE_PROFILE}' " \
                "(expected 4 '|' separators, found ${#pipes})"
            return 1
        fi
        _SETUP_PROFILE_ROSDEP_CELL="${_SETUP_PROFILE_ROSDEP_CELL//[[:space:]]/}"
        _SETUP_PROFILE_SUBMODULES="${_SETUP_PROFILE_SUBMODULES//[[:space:]]/}"
        _SETUP_PROFILE_VERIFY="${_SETUP_PROFILE_VERIFY//[[:space:]]/}"
        _SETUP_PROFILE_FEATURES="${_SETUP_PROFILE_FEATURES//[[:space:]]/}"
        if [[ -z "${_SETUP_PROFILE_ROSDEP_CELL}" || -z "${_SETUP_PROFILE_SUBMODULES}" \
            || -z "${_SETUP_PROFILE_VERIFY}" || -z "${_SETUP_PROFILE_FEATURES}" ]]; then
            log_error "Malformed profile row for '${_SETUP_PROFILE_PROFILE}' (a cell is empty)"
            return 1
        fi
        if [[ "${_SETUP_PROFILE_ROSDEP_CELL}" != *:* ]]; then
            log_error "Malformed rosdep cell for profile '${_SETUP_PROFILE_PROFILE}': " \
                "'${_SETUP_PROFILE_ROSDEP_CELL}' (expected <scope>:<paths>)"
            return 1
        fi

        # Bracket the feature list so the membership test stays exact.
        _SETUP_PROFILE_FEATURES=",${_SETUP_PROFILE_FEATURES},"
        # Split "<scope>:<paths>"; "-" means "this profile runs no rosdep".
        _SETUP_PROFILE_ROSDEP="${_SETUP_PROFILE_ROSDEP_CELL%%:*}"
        _SETUP_PROFILE_ROSDEP_PATHS="${_SETUP_PROFILE_ROSDEP_CELL#*:}"
        [[ "${_SETUP_PROFILE_ROSDEP_PATHS}" == "-" ]] && _SETUP_PROFILE_ROSDEP_PATHS=""
        _SETUP_PROFILE_ROSDEP_PATHS="${_SETUP_PROFILE_ROSDEP_PATHS//,/ }"
        return 0
    done <<<"${rows}"

    log_error "Unknown setup profile: '${SETUP_PROFILE:-full}' " \
        "(supported: $(setup_profile_names | tr '\n' ' '))"
    return 1
}

_setup_profile_loaded() {
    # Keyed on the profile name, not a one-shot flag: a consumer that switches
    # SETUP_PROFILE (the fixture harness does) must see the matching row.
    [[ "${_SETUP_PROFILE_PROFILE}" == "${SETUP_PROFILE:-full}" ]] || setup_profile_load
}

# Is <feature> enabled for the active profile?
setup_profile_enables() {
    _setup_profile_loaded || exit 1
    [[ "${_SETUP_PROFILE_FEATURES}" == *",$1,"* ]]
}

# Print a profile attribute: rosdep | rosdep_paths | submodules | verify.
setup_profile_get() {
    _setup_profile_loaded || exit 1
    case "$1" in
        rosdep)       printf '%s\n' "${_SETUP_PROFILE_ROSDEP}" ;;
        rosdep_paths) printf '%s\n' "${_SETUP_PROFILE_ROSDEP_PATHS}" ;;
        submodules)   printf '%s\n' "${_SETUP_PROFILE_SUBMODULES}" ;;
        verify)       printf '%s\n' "${_SETUP_PROFILE_VERIFY}" ;;
        *)
            log_error "Unknown setup profile attribute: '$1' " \
                "(expected rosdep, rosdep_paths, submodules, or verify)"
            exit 1
            ;;
    esac
}
