#!/bin/bash

python_version_of() {
    local python_bin="${1:-}"
    [[ -z "${python_bin}" || ! -x "${python_bin}" ]] && return 1
    "${python_bin}" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1
}

detect_libc() {
    if command -v ldd >/dev/null 2>&1; then
        if ldd --version 2>&1 | grep -qi musl; then
            echo "musl"
            return 0
        fi
    fi
    echo "glibc"
}

detect_host_metadata() {
    SETUP_ARCH="$(uname -m)"
    SETUP_KERNEL="$(uname -sr)"
    SETUP_ACTIVE_VENV="${VIRTUAL_ENV:-}"
    SETUP_PYTHONPATH="${PYTHONPATH:-}"
    SETUP_OS_ID="unknown"
    SETUP_OS_VERSION="unknown"
    SETUP_OS_PRETTY_NAME="unknown"
    SETUP_PACKAGE_MANAGER="unknown"

    if [[ -r /etc/os-release ]]; then
        SETUP_OS_ID="$(awk -F= '$1=="ID"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release)"
        SETUP_OS_VERSION="$(awk -F= '$1=="VERSION_ID"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release)"
        SETUP_OS_PRETTY_NAME="$(awk -F= '$1=="PRETTY_NAME"{gsub(/"/,"",$2); print $2}' /etc/os-release)"
    fi

    if command -v apt-get >/dev/null 2>&1; then
        SETUP_PACKAGE_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then
        SETUP_PACKAGE_MANAGER="dnf"
    fi
}

detect_platform_id() {
    if [[ -n "${PLATFORM_OVERRIDE:-}" ]]; then
        echo "${PLATFORM_OVERRIDE}"
        return 0
    fi

    local libc
    libc="$(detect_libc)"

    if [[ "${SETUP_OS_ID}" == "ubuntu" && "${SETUP_OS_VERSION}" == "22.04" ]]; then
        echo "ubuntu-22.04"
        return 0
    fi

    if [[ "${SETUP_OS_ID}" == "openeuler" || "${SETUP_OS_PRETTY_NAME,,}" == *"openeuler"* || "$(uname -r)" == *"openeuler"* ]]; then
        echo "openeuler-embedded-24.03"
        return 0
    fi

    if [[ "${libc}" == "musl" ]] && \
        { [[ "${SETUP_OS_ID}" == "openharmony" || "${SETUP_OS_ID}" == "ohos" ]] || [[ "${SETUP_OS_PRETTY_NAME,,}" == *"openharmony"* ]]; }; then
        echo "openharmony-5.1.0-musl"
        return 0
    fi

    echo "unknown"
}

load_platform_impl() {
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            if [[ -f "${SCRIPT_DIR}/setup/platforms/ubuntu-22.04.sh" ]]; then
                # shellcheck disable=SC1090
                source "${SCRIPT_DIR}/setup/platforms/ubuntu-22.04.sh"
            fi
            ;;
        openeuler-embedded-24.03)
            if [[ -f "${SCRIPT_DIR}/setup/platforms/openeuler-embedded-24.03.sh" ]]; then
                # shellcheck disable=SC1090
                source "${SCRIPT_DIR}/setup/platforms/openeuler-embedded-24.03.sh"
            fi
            ;;
        openharmony-5.1.0-musl)
            if [[ -f "${SCRIPT_DIR}/setup/platforms/openharmony-5.1.0-musl.sh" ]]; then
                # shellcheck disable=SC1090
                source "${SCRIPT_DIR}/setup/platforms/openharmony-5.1.0-musl.sh"
            fi
            ;;
        *)
            log_error "Unsupported platform '${SETUP_PLATFORM_ID}'."
            log_error "Expected one of: ubuntu-22.04, openeuler-embedded-24.03, openharmony-5.1.0-musl."
            exit 1
            ;;
    esac
}

initialize_platform() {
    detect_host_metadata
    SETUP_LIBC="$(detect_libc)"
    SETUP_PLATFORM_ID="$(detect_platform_id)"
    load_platform_impl
    detect_python_runtimes
}

detect_python_runtimes() {
    SETUP_SHELL_PYTHON_BIN="$(command -v python3 || true)"
    SETUP_SHELL_PYTHON_VERSION="$(python_version_of "${SETUP_SHELL_PYTHON_BIN}" || true)"

    local candidates=()
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            candidates=(/usr/bin/python3.10 /usr/bin/python3 python3.10 python3)
            ;;
        openeuler-embedded-24.03|openharmony-5.1.0-musl)
            candidates=(/usr/bin/python3.11 /usr/bin/python3 python3.11 python3)
            ;;
        *)
            candidates=(/usr/bin/python3 /usr/local/bin/python3 python3)
            ;;
    esac

    SETUP_BOOTSTRAP_PYTHON_BIN=""
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            SETUP_BOOTSTRAP_PYTHON_BIN="$(command -v "${candidate}")"
            break
        elif [[ -x "${candidate}" ]]; then
            SETUP_BOOTSTRAP_PYTHON_BIN="${candidate}"
            break
        fi
    done

    if [[ -z "${SETUP_BOOTSTRAP_PYTHON_BIN}" ]]; then
        SETUP_BOOTSTRAP_PYTHON_BIN="${SETUP_SHELL_PYTHON_BIN}"
    fi

    SETUP_BOOTSTRAP_PYTHON_VERSION="$(python_version_of "${SETUP_BOOTSTRAP_PYTHON_BIN}" || true)"
    SETUP_ROS_SETUP_PATH="$(platform_ros_setup_path)"
}

print_platform_summary() {
    echo ""
    echo -e "${YELLOW}--- Platform Detection ---${NC}"
    log_info "Platform: ${SETUP_PLATFORM_ID}"
    log_info "OS: ${SETUP_OS_PRETTY_NAME} (${SETUP_ARCH})"
    log_info "Kernel: ${SETUP_KERNEL}"
    log_info "libc: ${SETUP_LIBC}"
    log_info "Package manager: ${SETUP_PACKAGE_MANAGER}"
    log_info "Shell python: ${SETUP_SHELL_PYTHON_BIN:-not found} ${SETUP_SHELL_PYTHON_VERSION:+(${SETUP_SHELL_PYTHON_VERSION})}"
    log_info "Bootstrap python: ${SETUP_BOOTSTRAP_PYTHON_BIN:-not found} ${SETUP_BOOTSTRAP_PYTHON_VERSION:+(${SETUP_BOOTSTRAP_PYTHON_VERSION})}"
    log_info "ROS setup: ${SETUP_ROS_SETUP_PATH:-not configured}"
    log_info "Workspace: ${WORKSPACE}"

    if [[ -n "${SETUP_ACTIVE_VENV}" ]]; then
        log_warn "Active virtualenv detected: ${SETUP_ACTIVE_VENV}"
        if [[ "${SETUP_ACTIVE_VENV}" != "${WORKSPACE}/venv" ]]; then
            log_warn "Setup will bootstrap with ${SETUP_BOOTSTRAP_PYTHON_BIN:-system python} instead of the shell virtualenv."
        fi
    fi

    if [[ -n "${SETUP_PYTHONPATH}" ]]; then
        log_warn "PYTHONPATH is set: ${SETUP_PYTHONPATH}"
        log_warn "Imported modules may resolve outside this workspace until PYTHONPATH is cleared."
    fi
}
