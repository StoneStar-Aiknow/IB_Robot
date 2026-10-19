#!/bin/bash

python_version_of() {
    local python_bin="${1:-}"
    # Use -e (existence) rather than -x (executable bit) so foreign-arch
    # ELF binaries handled via binfmt_misc / qemu-user under chroot are
    # still considered candidates. The actual --version invocation will
    # surface a real failure if the binary is genuinely broken.
    [[ -z "${python_bin}" || ! -e "${python_bin}" ]] && return 1
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

format_kib_as_gb() {
    local kib="${1:-0}"
    awk -v kib="${kib}" 'BEGIN {
        if (kib <= 0) {
            print "unknown"
            exit
        }
        printf "%.0f GB", kib / 1024 / 1024
    }'
}

detect_gpu_summary() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name cuda_version
        gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
        cuda_version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]\+\).*/\1/p' | head -n1)"

        if [[ -n "${gpu_name}" ]]; then
            if [[ -n "${cuda_version}" ]]; then
                echo "${gpu_name} (CUDA ${cuda_version})"
            else
                echo "${gpu_name}"
            fi
            return 0
        fi
    fi

    if command -v lspci >/dev/null 2>&1; then
        local gpu_name
        gpu_name="$(lspci | awk -F': ' '/VGA compatible controller|3D controller/{print $2; exit}')"
        if [[ -n "${gpu_name}" ]]; then
            echo "${gpu_name}"
            return 0
        fi
    fi

    echo "not detected"
}

_cann_version_from_file() {
    local version_file="$1"
    local line version

    [[ -r "${version_file}" ]] || return 1
    while IFS= read -r line || [[ -n "${line}" ]]; do
        case "${line}" in
            version=*)
                version="${line#version=}"
                ;;
            version_dir=*)
                version="${line#version_dir=}"
                ;;
            *_running_version=\[*:*\])
                version="${line#*:}"
                version="${version%%]*}"
                ;;
            *)
                continue
                ;;
        esac
        version="${version%$'\r'}"
        if [[ "${version}" =~ ^[0-9]+\.[0-9]+([.][A-Za-z0-9]+)*$ ]]; then
            printf '%s\n' "${version}"
            return 0
        fi
    done < "${version_file}"
    return 1
}

detect_cann_version() {
    # CANN's "latest" path can be a merged directory of component symlinks,
    # not a symlink to the release directory. Prefer its public version.cfg,
    # then cover versioned toolkit roots and component metadata.
    local toolkit_root="${IBR_CANN_TOOLKIT_ROOT:-${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"
    local version_file version resolved_root
    local version_files=(
        "${toolkit_root}/version.cfg"
        "${toolkit_root}/ascend_toolkit_install.info"
        "${toolkit_root}/aarch64-linux/ascend_toolkit_install.info"
        "${toolkit_root}/arm64-linux/ascend_toolkit_install.info"
        "${toolkit_root}/x86_64-linux/ascend_toolkit_install.info"
        "${toolkit_root}/runtime/version.info"
        "${toolkit_root}/toolkit/version.info"
    )

    for version_file in "${version_files[@]}"; do
        if version="$(_cann_version_from_file "${version_file}")"; then
            printf '%s\n' "${version}"
            return 0
        fi
    done

    resolved_root="$(readlink -f "${toolkit_root}" 2>/dev/null || true)"
    version="${resolved_root##*/}"
    if [[ "${version}" =~ ^[0-9]+\.[0-9]+([.][A-Za-z0-9]+)*$ ]]; then
        printf '%s\n' "${version}"
        return 0
    fi
    return 1
}

# The two accelerator probes are separate functions so tests can stub them as
# shell functions. A fake nvidia-smi on PATH is not an option: the board
# targets mount /tmp noexec, so an executable fixture there cannot run.
_accel_has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

_accel_has_ascend_npu() {
    # A CANN toolkit alone is NOT an Ascend NPU: hosts that only convert OM
    # model bundles install the toolkit as a cross-compilation toolchain, and
    # treating that as an NPU made such a host look like it had two
    # accelerators. Probe the device instead, mirroring the nvidia-smi -L
    # device check. The davinci device nodes are the driver's own view of the
    # NPUs and are present even where npu-smi is not on PATH, which is the case
    # on the Ascend 310P openEuler board.
    local device_glob="${IBR_ASCEND_DEVICE_GLOB:-/dev/davinci[0-9]*}"

    if compgen -G "${device_glob}" >/dev/null 2>&1; then
        return 0
    fi
    command -v npu-smi >/dev/null 2>&1 && npu-smi info >/dev/null 2>&1
}

# The accelerator axis is independent of the OS axis: Ascend is deployed on
# both openEuler and Ubuntu, and an Ubuntu host may carry either an NVIDIA GPU
# or an Ascend NPU. Probe the toolkits and drivers on disk, never an installed
# Torch: which Torch build to install is exactly what this axis decides.
detect_accel_id() {
    if [[ -n "${ACCEL_OVERRIDE:-}" ]]; then
        echo "${ACCEL_OVERRIDE}"
        return 0
    fi

    local has_cuda=false has_cann=false
    if _accel_has_nvidia_gpu; then
        has_cuda=true
    fi
    if _accel_has_ascend_npu; then
        has_cann=true
    fi

    if [[ "${has_cuda}" == true && "${has_cann}" == true ]]; then
        # Guessing here silently installs one vendor's Torch over the other's.
        log_error "Detected both an NVIDIA GPU and an Ascend NPU device on this host."
        log_error "Set ACCEL_OVERRIDE=cuda or ACCEL_OVERRIDE=cann to select the runtime to install."
        return 1
    fi
    if [[ "${has_cuda}" == true ]]; then
        echo "cuda"
        return 0
    fi
    if [[ "${has_cann}" == true ]]; then
        echo "cann"
        return 0
    fi
    # No accelerator is a supported configuration: the workspace still builds
    # and its CPU-only tests still run.
    echo "cpu"
}

detect_accel_summary() {
    local detail=""

    if [[ "${SETUP_ACCEL_ID}" == "cann" ]]; then
        local cann_version
        cann_version="$(detect_cann_version 2>/dev/null || true)"
        if [[ -n "${cann_version}" ]]; then
            detail=" (CANN ${cann_version})"
        fi
    fi
    if [[ -n "${ACCEL_OVERRIDE:-}" ]]; then
        detail="${detail} (ACCEL_OVERRIDE)"
    fi
    printf '%s%s\n' "${SETUP_ACCEL_ID}" "${detail}"
}

load_accel_impl() {
    case "${SETUP_ACCEL_ID}" in
        cuda | cann | cpu)
            # shellcheck disable=SC1090
            source "${SCRIPT_DIR}/setup/accels/${SETUP_ACCEL_ID}.sh"
            ;;
        *)
            log_error "Unsupported accelerator '${SETUP_ACCEL_ID}'."
            log_error "Expected one of: cuda, cann, cpu (set via ACCEL_OVERRIDE)."
            exit 1
            ;;
    esac
}

detect_ram_summary() {
    if [[ -e /proc/meminfo ]]; then
        local total_kib
        total_kib="$(awk '/MemTotal/{print $2; exit}' /proc/meminfo)"
        format_kib_as_gb "${total_kib}"
        return 0
    fi

    echo "unknown"
}

detect_disk_free_summary() {
    local free_kib
    free_kib="$(df -Pk "${WORKSPACE}" 2>/dev/null | awk 'NR==2 {print $4}')"

    if [[ -n "${free_kib}" ]]; then
        echo "$(format_kib_as_gb "${free_kib}") free"
        return 0
    fi

    echo "unknown"
}

detect_ros_summary() {
    if [[ -n "${SETUP_ROS_SETUP_PATH}" && -f "${SETUP_ROS_SETUP_PATH}" ]]; then
        local ros_distro=""

        ros_distro="$(
            bash -lc '
                source "$1" >/dev/null 2>&1
                printf "%s" "${ROS_DISTRO:-}"
            ' _ "${SETUP_ROS_SETUP_PATH}" 2>/dev/null
        )"

        if [[ -z "${ros_distro}" ]] && [[ "${SETUP_ROS_SETUP_PATH}" =~ /opt/ros/([^/]+)/ ]]; then
            ros_distro="${BASH_REMATCH[1]}"
        fi

        if [[ -n "${ros_distro}" ]]; then
            echo "${ros_distro}"
        else
            echo "installed"
        fi
        return 0
    fi

    if [[ "${SETUP_PLATFORM_ID}" == "openharmony-5.1.0-musl" ]]; then
        echo "not configured"
        return 0
    fi

    echo "not installed"
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

    if [[ -e /etc/os-release ]]; then
        SETUP_OS_ID="$(awk -F= '$1=="ID"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release)"
        SETUP_OS_VERSION="$(awk -F= '$1=="VERSION_ID"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release)"
        SETUP_OS_PRETTY_NAME="$(awk -F= '$1=="PRETTY_NAME"{gsub(/"/,"",$2); print $2}' /etc/os-release)"
    fi

    if command -v apt-get >/dev/null 2>&1; then
        SETUP_PACKAGE_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then
        SETUP_PACKAGE_MANAGER="dnf"
    fi

    SETUP_GPU_SUMMARY="$(detect_gpu_summary)"
    SETUP_RAM_SUMMARY="$(detect_ram_summary)"
    SETUP_DISK_FREE_SUMMARY="$(detect_disk_free_summary)"
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
    SETUP_PLATFORM_ID="$(detect_platform_id)"
    load_platform_impl
    if ! SETUP_ACCEL_ID="$(detect_accel_id)"; then
        exit 1
    fi
    load_accel_impl
    detect_python_runtimes
    export_lerobot_host_facts
}

detect_python_runtimes() {
    SETUP_SHELL_PYTHON_BIN="$(command -v python3 || true)"
    SETUP_SHELL_PYTHON_VERSION="$(python_version_of "${SETUP_SHELL_PYTHON_BIN}" || true)"

    local candidates=()
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            candidates=(/usr/bin/python3.10 /usr/bin/python3 python3.10 python3)
            ;;
        openeuler-embedded-24.03)
            candidates=(/usr/bin/python3.11 /usr/bin/python3 python3.11 python3)
            ;;
        openharmony-5.1.0-musl)
            candidates=(/data/out/bin/python3.12 /usr/bin/python3.12 /usr/bin/python3 python3.12 python3)
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
    SETUP_ROS_SUMMARY="$(detect_ros_summary)"
}

print_platform_summary() {
    local python_summary="${SETUP_BOOTSTRAP_PYTHON_VERSION:-not found}"
    local summary_block=""

    if [[ -n "${SETUP_SHELL_PYTHON_VERSION}" && -n "${SETUP_BOOTSTRAP_PYTHON_VERSION}" && "${SETUP_SHELL_PYTHON_VERSION}" != "${SETUP_BOOTSTRAP_PYTHON_VERSION}" ]]; then
        python_summary="${SETUP_BOOTSTRAP_PYTHON_VERSION} (shell: ${SETUP_SHELL_PYTHON_VERSION})"
    fi

    summary_block="$(printf "OS:       %s\nPython:   %s\nGPU:      %s\nAccel:    %s\nROS:      %s\nRAM:      %s\nDisk:     %s" \
        "${SETUP_OS_PRETTY_NAME} (${SETUP_ARCH})" \
        "${python_summary}" \
        "${SETUP_GPU_SUMMARY}" \
        "$(detect_accel_summary)" \
        "${SETUP_ROS_SUMMARY}" \
        "${SETUP_RAM_SUMMARY}" \
        "${SETUP_DISK_FREE_SUMMARY}")"

    ui_render_block "${summary_block}"

    if [[ "${SETUP_PACKAGE_MANAGER}" != "unknown" ]]; then
        echo -e "${GREEN}✓${NC} will use ${SETUP_PACKAGE_MANAGER} system package manager"
    else
        log_warn "No supported system package manager detected; system dependencies may need manual provisioning."
    fi

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

# -----------------------------------------------------------------------------
# Host facts for lerobot patch-series dispatch
# -----------------------------------------------------------------------------
# These helpers expose the minimal, stable surface consumed by
# scripts/setup/lerobot_filter_series.py:
#   IBR_HOST_PYTHON_VERSION  - "X.Y" major.minor of the bootstrap interpreter
#   IBR_LEROBOT_PROFILES     - comma-separated profile identifiers
#
# The applier reads these via os.environ. They are deliberately decoupled
# from SETUP_* vars so future callers (CI, tooling) can set them directly
# without sourcing detect.sh.

# Print the major.minor (X.Y) of the most representative host Python.
# Preference order:
#   1. SETUP_BOOTSTRAP_PYTHON_BIN  (the interpreter the venv will be built from)
#   2. SETUP_SHELL_PYTHON_BIN      (whatever python3 is on PATH right now)
# Emits an empty line and a warning if neither produces a usable version.
#
# Note: we deliberately avoid `[ -x ]` for the existence gate because
# qemu-user emulation under chroot returns false for foreign-arch ELF
# binaries (binfmt_misc handles exec(2), but access(X_OK) does not).
# Instead we accept any path that exists and rely on the actual
# `--version` invocation to surface a real failure.
detect_host_python_version() {
    local raw=""
    local candidate

    candidate="${SETUP_BOOTSTRAP_PYTHON_BIN:-}"
    if [[ -n "${candidate}" && -e "${candidate}" ]]; then
        raw="$(python_version_of "${candidate}" 2>/dev/null || true)"
    fi
    if [[ -z "${raw}" ]]; then
        candidate="${SETUP_SHELL_PYTHON_BIN:-}"
        if [[ -n "${candidate}" && -e "${candidate}" ]]; then
            raw="$(python_version_of "${candidate}" 2>/dev/null || true)"
        fi
    fi

    if [[ -z "${raw}" ]]; then
        log_warn "detect_host_python_version: no usable Python interpreter found; lerobot filter will treat python_min/python_max as unsatisfiable."
        echo ""
        return 0
    fi

    # Reduce X.Y.Z -> X.Y; tolerate already-X.Y inputs.
    echo "${raw}" | awk -F. '{ if (NF >= 2) printf "%s.%s", $1, $2; else printf "%s", $0 }'
}

# Resolve the active lerobot profile list using the documented precedence:
#   1. IBR_LEROBOT_PROFILES_CLI   (set by --lerobot-profiles)
#   2. IBR_LEROBOT_PROFILES       (operator override via env)
#   3. platform_lerobot_profiles  (per-platform default, if defined)
#   4. "core,ros,hardware,dev"    (legacy fallback == Ubuntu 22.04 baseline)
# Whitespace around tokens is stripped; empty tokens are dropped.
resolve_lerobot_profiles() {
    local raw=""

    if [[ -n "${IBR_LEROBOT_PROFILES_CLI:-}" ]]; then
        raw="${IBR_LEROBOT_PROFILES_CLI}"
    elif [[ -n "${IBR_LEROBOT_PROFILES:-}" ]]; then
        raw="${IBR_LEROBOT_PROFILES}"
    elif declare -F platform_lerobot_profiles >/dev/null 2>&1; then
        raw="$(platform_lerobot_profiles 2>/dev/null || true)"
    fi

    if [[ -z "${raw}" ]]; then
        raw="core,ros,hardware,dev"
    fi

    # Normalise: strip whitespace, drop empty fields, preserve order.
    echo "${raw}" | awk -F, '{
        out = ""
        for (i = 1; i <= NF; i++) {
            tok = $i
            gsub(/^[ \t]+|[ \t]+$/, "", tok)
            if (tok == "") continue
            if (out == "") out = tok
            else out = out "," tok
        }
        printf "%s", out
    }'
}

# Compute and export the host-fact env vars consumed by the applier.
# Idempotent; safe to call after detect_python_runtimes and after
# load_platform_impl (so platform_lerobot_profiles is in scope).
export_lerobot_host_facts() {
    IBR_HOST_PYTHON_VERSION="$(detect_host_python_version)"
    IBR_LEROBOT_PROFILES="$(resolve_lerobot_profiles)"
    export IBR_HOST_PYTHON_VERSION IBR_LEROBOT_PROFILES
}
