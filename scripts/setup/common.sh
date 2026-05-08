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
    shift
    log_info "${msg}"
    if [[ "${VERBOSE:-0}" == "1" ]]; then
        run_sudo "$@"
    else
        # We still show progress, but normally apt output is quieted
        run_sudo "$@" >/dev/null 2>&1
    fi
}

run_with_live_output() {
    local msg="$1"
    shift
    log_info "${msg}"
    if [[ "${VERBOSE:-0}" == "1" ]]; then
        "$@"
    else
        "$@" >/dev/null 2>&1
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
