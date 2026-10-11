#!/bin/bash
# setup.sh - Workspace setup script for ROS 2 Humble
# Handles repository import, dependency installation, and environment setup
#
# Usage:
#   ./scripts/setup.sh                               # Interactive mode
#   ./scripts/setup.sh --yes                         # Auto-yes mode
#   ./scripts/setup.sh --with-diagnostics            # Install speech_direction report deps
#   ./scripts/setup.sh --skip-verify                 # Skip final ROS/Python verification
#   ./scripts/setup.sh --only-patch                  # Only apply LeRobot patches
#   ./scripts/setup.sh --platform <id>               # Override detected platform
#   ./scripts/setup.sh --profile <name>              # Dependency scope: full (default),
#                                                    # inference (edge-cloud inference
#                                                    # verification runtime), or
#                                                    # training (LeRobot-only training host)
#   ./scripts/setup.sh --help                        # Show help
#
# Auto-yes defaults:
#   - Missing submodules:       initialized
#   - Existing submodule update: skipped
#   - Fork setup:               skipped
#   - Other prompts:            confirmed automatically
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(pwd)}"
AUTO_YES=false
VERBOSE=${VERBOSE:-false}
DRY_RUN=${DRY_RUN:-false}
USE_SUDO=true
SUMMARY=()
# Set by install_accel_requirements when the cann axis installs no NPU backend
# (no CANN toolkit detected, or a release without a validated Torch ABI here).
# The hook itself runs in a subshell and cannot set it; read by the summary.
ACCEL_NPU_PENDING_MANUAL_CONFIG=false
SETUP_PLATFORM_ID="unknown"
SETUP_OS_ID="unknown"
SETUP_OS_VERSION="unknown"
SETUP_OS_PRETTY_NAME="unknown"
SETUP_ARCH="unknown"
SETUP_KERNEL="unknown"
SETUP_PACKAGE_MANAGER="unknown"
SETUP_ACTIVE_VENV=""
SETUP_PYTHONPATH=""
SETUP_SHELL_PYTHON_BIN=""
SETUP_SHELL_PYTHON_VERSION=""
SETUP_BOOTSTRAP_PYTHON_BIN=""
SETUP_BOOTSTRAP_PYTHON_VERSION=""
SETUP_ROS_SETUP_PATH=""
SETUP_GPU_SUMMARY="unknown"
SETUP_RAM_SUMMARY="unknown"
SETUP_DISK_FREE_SUMMARY="unknown"
SETUP_ROS_SUMMARY="unknown"
SUDO_AUTH_READY=false
PLATFORM_OVERRIDE=""
# Accelerator axis, independent of the OS axis: Ascend ships on both openEuler
# and Ubuntu, so the Torch build to install cannot be derived from the OS.
SETUP_ACCEL_ID="unknown"
# Honor an inherited value: the ambiguous-host error tells the user to set this
# variable, so clearing it unconditionally would make that advice a dead end.
ACCEL_OVERRIDE="${ACCEL_OVERRIDE:-}"
# Dependency-scope profile: "full" (default, complete development workspace) or
# Dependency-scope profile: "full" (default, complete development workspace),
# "inference" (minimal runtime to run policy inference only, used for
# edge-cloud collaborative inference verification hosts), or "training"
# (LeRobot training host: venv + LeRobot patch stack only, without ROS 2,
# rosdep, colcon, or the ROS third-party patch stacks). Resolution order:
# --profile CLI argument > IBR_SETUP_PROFILE env > "full".
#
# NOTE: this "training" value selects the install scope. It is unrelated to the
# lerobot patch-series profile of the same name (IBR_LEROBOT_PROFILES /
# --lerobot-profiles), which keeps its platform default so that the
# training-relevant LeRobot patches are still applied.
SETUP_PROFILE="${IBR_SETUP_PROFILE:-full}"

SKIP_VERIFY=false
ONLY_PATCH=${ONLY_PATCH:-false}
INSTALL_DIAGNOSTICS_DEPS="${IBR_SETUP_WITH_DIAGNOSTICS:-false}"
INSTALL_BENCHMARK_DEPS="${IBR_SETUP_WITH_BENCHMARK:-false}"
CURRENT_STAGE="initializing"
SYSTEM_DEPS_STATUS="pending"
PYTHON_ENV_STATUS="pending"
SETUP_ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml}"
SETUP_ROSDEP_DEFAULT_SOURCES_URL="${SETUP_ROSDEP_DEFAULT_SOURCES_URL:-https://mirrors.tuna.tsinghua.edu.cn/github-raw/ros/rosdistro/master/rosdep/sources.list.d/20-default.list}"
SETUP_ROSDEP_DEFAULT_SOURCES_FILE="${SETUP_ROSDEP_DEFAULT_SOURCES_FILE:-/etc/ros/rosdep/sources.list.d/20-default.list}"
SETUP_PIP_INDEX_URL="${SETUP_PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple/}"
SETUP_PIP_TRUSTED_HOST="${SETUP_PIP_TRUSTED_HOST:-repo.huaweicloud.com}"

# Mirror build.sh / .shrc_local: ignore ~/.local site-packages so that the
# install-time view of Python packages matches what the build/runtime sees.
# Otherwise tools installed via `pip install --user` (e.g. legacy colcon in
# ~/.local) can make setup verification pass while build.sh fails with
# "No module named colcon" because PYTHONNOUSERSITE=1 is set there.
export PYTHONNOUSERSITE=1

# NOTE on NumPy / OpenCV pinning strategy
# ---------------------------------------
# We deliberately do NOT pass `-c numpy==1.26.4` to the lerobot install.
# lerobot's transitive dependency graph (rerun-sdk, opencv, datasets, ...)
# is solved against numpy>=2.x, and forcing numpy==1.26.4 as a constraint
# during the lerobot resolution explodes pip backtracking into
# `resolution-too-deep`. Instead, we let lerobot install whatever NumPy/
# OpenCV it wants, and AFTERWARDS force-reinstall numpy==1.26.4 +
# opencv-python-headless<4.12 to restore ROS 2 Humble ABI compatibility.
# Optional dependency installs use constraints that also cap opencv-python<4.12
# if a third-party package pulls the GUI wheel transitively.
# This produces a few cosmetic dependency-resolver warnings during install,
# which are harmless because we never call the numpy-2-only APIs in the
# affected packages from the ROS pipeline.

# Detect if running as root
if [[ $EUID -eq 0 ]]; then
    USE_SUDO=false
fi

# Load shared infrastructure
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/submodules.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/ros_third_party.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/rosdep.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/python_venv.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/benchmark_profile.sh"

# Overrides for setup.sh specific logging behavior
log_done()    { SUMMARY+=("${GREEN}✓${NC} $*"); }
log_skipped() { SUMMARY+=("${YELLOW}⊘${NC} $* (skipped by --yes)"); }

# Override common.sh default (which hides output unless VERBOSE=1)
# to pass-through output for long-running operations in setup.sh
run_with_live_output() {
    local title="$1"
    shift
    log_info "${title}"
    "$@"
    return $?
}

print_banner() {
    local cols banner
    cols=$(stty size 2>/dev/null </dev/tty | awk '{print $2}') \
        || cols=$(tput cols 2>/dev/null) \
        || cols=80

    if [[ ${cols} -ge 90 ]]; then
        banner="$(cat <<'EOF'
   ██╗██████╗     ██████╗  ██████╗ ██████╗  ██████╗ ████████╗
   ██║██╔══██╗    ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝
   ██║██████╔╝    ██████╔╝██║   ██║██████╔╝██║   ██║   ██║
   ██║██╔══██╗    ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║
   ██║██████╔╝    ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║
   ╚═╝╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝
EOF
)"
    elif [[ ${cols} -ge 56 ]]; then
        banner="$(cat <<'EOF'
   ██╗██████╗   ██████╗  ██████╗
   ██║██╔══██╗  ██╔══██╗██╔═══██╗
   ██║██████╔╝  ██████╔╝██║   ██║
   ██║██╔══██╗  ██╔══██╗██║   ██║
   ██║██████╔╝  ██║  ██║╚██████╔╝
   ╚═╝╚═════╝   ╚═╝  ╚═╝ ╚═════╝
EOF
)"
    else
        banner="IB Robot"
    fi

    echo ""
    echo "${banner}"
    echo ""
    echo -e "\033[38;5;245m   Intelligent BooM Robot · workspace installer${NC}"
    echo ""
}

set_stage() {
    CURRENT_STAGE="$1"
    echo ""
    echo -e "${YELLOW}▸ ${CURRENT_STAGE}...${NC}"
}

on_setup_failure() {
    local exit_code="$1"
    local line_no="$2"
    trap - ERR
    echo ""
    log_error "Setup failed during stage: ${CURRENT_STAGE} (line ${line_no})"
    if [[ ${#SUMMARY[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Partial summary:${NC}"
        for entry in "${SUMMARY[@]}"; do
            echo -e "  ${entry}"
        done
    fi
    exit "${exit_code}"
}

on_setup_interrupt() {
    trap - INT
    echo ""
    log_error "Setup interrupted during stage: ${CURRENT_STAGE}"
    exit 130
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/detect.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/lerobot_patches.sh"

show_help() {
    cat <<'EOF'
Workspace setup script for IB_Robot

Usage:
  ./scripts/setup.sh [OPTIONS]

Options:
  -y, --yes              Auto-confirm prompts using defaults
      --sudo             Force sudo for privileged operations
      --no-sudo          Never use sudo
      --with-diagnostics Install optional Plotly/Matplotlib dependencies for
                          the speech_direction_report offline CLI only;
                          the speech_direction_node runtime does not need them.
      --with-benchmark   Install the Ubuntu 22.04 x86_64 GPU LIBERO profile.
                          Pins the validated PyTorch/TorchCodec CUDA ABI, needs
                          NVIDIA driver >=560.28.03, does not need local nvcc,
                          and skips unrelated GraspGen dependencies.

      --skip-verify      Skip final ROS/Python verification
      --only-patch       Skip dependency/venv setup and verification; only
                         apply the managed LeRobot patch stack. Use
                         IBR_LEROBOT_FORCE_REBUILD=1 to rebuild from the
                         recorded upstream base.
      --profile NAME     Dependency-scope profile (default: full; env:
                         IBR_SETUP_PROFILE).
                           full      complete development workspace
                           inference minimal policy-inference runtime for
                                      edge-cloud collaborative inference
                                      verification: skips hardware, teleop,
                                      perception, grasp, voice, sim, and
                                      dev tooling; narrows rosdep and
                                      submodules to the inference packages
                           training  LeRobot training host: creates the venv,
                                      applies the LeRobot patch stack, and
                                      installs LeRobot plus the training Python
                                      dependencies only. Installs no ROS 2 (no
                                      install_ros.sh), no rosdep, no colcon, and
                                      no ROS third-party patch stacks; the
                                      submodule scope is libs/lerobot only.
      --platform ID      Override platform (OS) detection
      --accel ID         Override accelerator detection (env: ACCEL_OVERRIDE).
                         Required when a host carries both an NVIDIA GPU and
                         an Ascend NPU, since the two need different Torch
                         builds and setup refuses to guess.
      --lerobot-profiles CSV
                         Override lerobot patch profile selection
                         (e.g. core,ros,hardware,ascend). Highest
                         precedence; overrides IBR_LEROBOT_PROFILES env.
  -h, --help             Show this help

Known platform IDs:
  ubuntu-22.04
  openeuler-embedded-24.03
  openharmony-5.1.0-musl

Known accelerator IDs:
  cuda   NVIDIA GPU (detected via nvidia-smi -L)
  cann   Ascend NPU (detected via the /dev/davinci* devices)
  cpu    no accelerator; the workspace still builds and CPU tests still run

Detection reads devices, not toolkits: a CUDA or CANN toolkit installed only to
cross-compile or convert models does not select that accelerator. Use --accel to
install a runtime on a host whose device is absent or not passed through.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y) AUTO_YES=true ;;
            --no-sudo) USE_SUDO=false ;;
            --sudo) USE_SUDO=true ;;
            --with-diagnostics|--with-diagnostics-deps) INSTALL_DIAGNOSTICS_DEPS=true ;;
            --with-benchmark|--with-benchmark-deps) INSTALL_BENCHMARK_DEPS=true ;;

            --skip-verify) SKIP_VERIFY=true ;;
            --only-patch) ONLY_PATCH=true ;;
            --profile)
                shift
                if [[ $# -eq 0 ]]; then
                    log_error "--profile requires a profile name (full|inference|training)."
                    exit 1
                fi
                SETUP_PROFILE="$1"
                ;;
            --platform)
                shift
                if [[ $# -eq 0 ]]; then
                    log_error "--platform requires a platform ID."
                    exit 1
                fi
                PLATFORM_OVERRIDE="$1"
                ;;
            --accel)
                shift
                if [[ $# -eq 0 ]]; then
                    log_error "--accel requires an accelerator ID."
                    exit 1
                fi
                ACCEL_OVERRIDE="$1"
                ;;
            --lerobot-profiles)
                shift
                if [[ $# -eq 0 ]]; then
                    log_error "--lerobot-profiles requires a comma-separated profile list."
                    exit 1
                fi
                # Exported here (not just assigned) so detect.sh's
                # resolve_lerobot_profiles can consume it later.
                export IBR_LEROBOT_PROFILES_CLI="$1"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

validate_setup_profile() {
    case "${SETUP_PROFILE}" in
        full|inference|training) ;;
        *)
            log_error "Unknown setup profile: '${SETUP_PROFILE}' (expected 'full', 'inference', or 'training')."
            exit 1
            ;;
    esac
    # Exported so verify_env.sh (sourced later) and child processes follow the
    # same dependency scope.
    export SETUP_PROFILE
}

# Rosdep resolution scope per profile. The full profile resolves keys for every
# workspace package. The inference profile limits rosdep to the packages
# required to build and run inference_service, so bringup-only dependencies
# (nav2, slam_toolbox, gz_ros2_control, tracing, cameras, LiDARs, MoveIt) are
# not pulled onto edge-cloud inference verification hosts. robot_config is
# intentionally excluded here: it must still be BUILT (it carries the SSOT
# YAML), but its package.xml exec_depends describe full-robot bringup, not the
# inference runtime; its Python-level needs are covered by the venv install.
setup_profile_rosdep_paths() {
    # Only reachable for profiles that install rosdep; the training profile
    # skips rosdep entirely (see install_system_deps).
    if [[ "${SETUP_PROFILE}" == "inference" ]]; then
        echo "src/ibrobot_msgs src/tensormsg src/inference_manifest src/torch_models src/inference_service src/model_utils src/dataset_tools"
    else
        echo "src"
    fi
}

system_package_installed() {
    local package_name="$1"

    if command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -W -f='${Status}' "${package_name}" 2>/dev/null | grep -q "install ok installed"
        return $?
    fi

    if command -v rpm >/dev/null 2>&1; then
        rpm -q "${package_name}" >/dev/null 2>&1
        return $?
    fi

    return 1
}

preview_system_packages() {
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            if setup_profile_is_training; then
                echo "python3-venv python3-pip ffmpeg"
                return 0
            fi
            local pkgs="python3-colcon-common-extensions python3-venv python3-pip"
            if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
                pkgs="${pkgs} libosmesa6-dev ffmpeg"
            fi
            echo "${pkgs}"
            ;;
        openeuler-embedded-24.03)
            echo "gcc-c++ vim-enhanced ffmpeg-devel libvpx libvpx-devel nlohmann-json-devel yaml-cpp yaml-cpp-devel python3-virtualenv python3-pip python3-devel"
            ;;
        *)
            echo ""
            ;;
    esac
}

print_dependency_preview() {
    local preview_message=""

    if [[ "${SETUP_PLATFORM_ID}" == "openharmony-5.1.0-musl" || "${SETUP_PACKAGE_MANAGER}" == "unknown" ]]; then
        preview_message="manual system dependency provisioning required on this platform"
        ui_render_block "${preview_message}"
        return 0
    fi

    local package_name
    local missing_packages=()
    for package_name in $(preview_system_packages); do
        if ! system_package_installed "${package_name}"; then
            missing_packages+=("${package_name}")
        fi
    done

    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        preview_message="need to install: ${missing_packages[*]}"
    else
        preview_message="system bootstrap packages already installed"
    fi

    ui_render_block "${preview_message}"
}

# ============================================================================
# Environment Checks
# ============================================================================
check_foreign_env() {
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        log_error "Active Conda environment detected at: ${CONDA_PREFIX}"
        log_warn "Conda environments are known to conflict with ROS 2 dependencies (especially Python libraries)."
        log_warn "Please deactivate the Conda environment before running this script:"
        echo -e "    ${YELLOW}conda deactivate${NC}"
        exit 1
    fi

    if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${WORKSPACE}/venv" ]]; then
        log_error "Active virtual environment detected at: ${VIRTUAL_ENV}"
        log_error "This is not the workspace venv (${WORKSPACE}/venv)."
        log_error "Please deactivate it before running this script:"
        echo -e "    ${YELLOW}deactivate${NC}"
        exit 1
    fi
}

# --- Platform Template Method Defaults ---
# These functions act as base template methods or hooks.
# They are expected to be overridden by the platform implementation scripts
# loaded from scripts/setup/platforms/*.sh.

platform_ros_setup_path() {
    if [[ -n "${ROS_HUMBLE_SETUP_PATH:-}" ]]; then
        echo "${ROS_HUMBLE_SETUP_PATH}"
    elif [[ -f /opt/ros/humble/setup.sh ]]; then
        echo "/opt/ros/humble/setup.sh"
    else
        echo "/opt/ros/humble/setup.bash"
    fi
}

platform_handle_missing_ros() {
    log_info "Running ROS 2 installation script..."

    local install_args=()
    if [[ "${AUTO_YES}" == true ]]; then
        install_args+=("--yes")
    fi
    if [[ "${USE_SUDO}" == false ]]; then
        install_args+=("--no-sudo")
    fi

    if "${WORKSPACE}/scripts/install_ros.sh" "${install_args[@]}"; then
        log_done "ROS 2 Humble installed"
    else
        log_error "ROS 2 installation failed"
        log_error "Please run ${WORKSPACE}/scripts/install_ros.sh manually to diagnose the issue"
        exit 1
    fi
}

platform_prepare_host() { return 0; }
platform_supports_local_workspace_build() { return 0; }
platform_install_colcon() {
    log_error "colcon installation hook not implemented for this platform."
    exit 1
}
platform_install_python_bootstrap() {
    log_error "Python bootstrap hook not implemented for this platform."
    exit 1
}
platform_install_benchmark_system_deps() { return 0; }

# --- Training-profile host prerequisites ---
# The training profile installs no ROS 2 runtime; its only system-level needs
# are the Python bootstrap (python3-venv / python3-pip) and ffmpeg, which
# backs torchcodec / dataset video decoding. Platforms override
# platform_install_training_system_deps() when they need more.
platform_training_system_deps_satisfied() {
    # Pure install-time shortcut: this only decides whether apt-get would add
    # anything, so every probe must test the capability the install path is
    # about to rely on.
    command -v ffmpeg >/dev/null 2>&1 || return 1
    if [[ ! -d "${WORKSPACE}/venv" ]]; then
        # Creating the venv goes through ensurepip, which Debian/Ubuntu ship in
        # python3-venv. The venv module itself lives in libpython3-stdlib, so its
        # --help output succeeds on hosts where creation would fail.
        python3 -c 'import ensurepip' >/dev/null 2>&1 || return 1
    fi
    # A missing system python3-pip is not a reason to run apt here: the venv's
    # pip comes from ensurepip's bundled wheel.
    return 0
}
platform_install_training_system_deps() { return 0; }

# --- Template Method Hooks for platform_install_rosdeps ---
platform_skip_rosdep_install() { return 1; }  # Return 0 to skip rosdep install
platform_pre_install_rosdeps() {
    if command -v apt-get &>/dev/null; then
        run_privileged_with_live_output "Updating apt package lists..." apt-get update
    fi
}
platform_get_extra_skip_keys() { echo ""; }
platform_post_install_rosdeps() { return 0; }
# ---------------------------------------------------------

platform_install_rosdeps() {
    if platform_skip_rosdep_install; then
        return 0
    fi

    platform_pre_install_rosdeps

    local rosdep_cmd="${ROSDEP_BIN:-rosdep}"
    local rosdep_install_extra_args=()

    if [[ "${USE_SUDO}" != true ]]; then
        rosdep_install_extra_args+=(--as-root apt:false --as-root dnf:false --as-root pip:false)
    fi

    if ! run_with_live_output "Updating rosdep database..." env ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL}" "${rosdep_cmd}" update --rosdistro=humble; then
        log_error "rosdep update failed. This is usually due to network issues."
        log_error "Please check your network connection and re-run ./scripts/setup.sh"
        exit 1
    fi

    # Base skip keys common to all platforms
    local base_skip_keys=(
        catkin
        roscpp
        lerobot
        trimesh
        "trimesh[easy]"
        simple-parsing
        cupy-cuda12x
        ctl_system_interface
        numpy_lessthan_2
        ament_python
        feetech-servo-sdk
        pyserial
        sherpa-onnx
    )

    # Append platform-specific skip keys
    local extra_keys_str
    extra_keys_str="$(platform_get_extra_skip_keys)"
    local all_skip_keys=("${base_skip_keys[@]}")
    if [[ -n "${extra_keys_str}" ]]; then
        read -ra extra_keys_arr <<< "${extra_keys_str}"
        all_skip_keys+=("${extra_keys_arr[@]}")
    fi

    local skip_args=()
    for key in "${all_skip_keys[@]}"; do
        skip_args+=("--skip-keys=${key}")
    done

    local rosdep_paths
    read -ra rosdep_paths <<< "$(setup_profile_rosdep_paths)"
    if [[ "${SETUP_PROFILE}" == "inference" ]]; then
        log_info "Inference profile: rosdep scope limited to: ${rosdep_paths[*]}"
    fi

    if ! run_with_live_output "Installing ROS dependencies..." env ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL}" "${rosdep_cmd}" install \
        "${rosdep_install_extra_args[@]}" \
        --from-paths "${rosdep_paths[@]}" \
        --ignore-src \
        --rosdistro=humble \
        -y -r \
        "${skip_args[@]}"; then
        log_error "rosdep install encountered errors. Some packages may be missing or require manual installation."
        log_error "Re-run with VERBOSE=1 for full output, or install missing packages manually."
        log_error "Fix the reported dependency issues and re-run ./scripts/setup.sh."
        SYSTEM_DEPS_STATUS="rosdep-errors"
    fi

    maybe_platform_post_install_rosdeps
}

# Tracing tools (lttng-tools, python3-lttngust, babeltrace2) are full-workspace
# diagnostics installed unconditionally by the platform post-install hook; the
# inference profile skips them together with verify_tracing.
maybe_platform_post_install_rosdeps() {
    if [[ "${SETUP_PROFILE}" == "inference" ]]; then
        log_info "Skipping tracing post-install hook (inference profile)."
        return 0
    fi
    platform_post_install_rosdeps
}

# ============================================================================
# Dependency Management
# ============================================================================
check_ros_installation() {
    local ros_setup_path
    ros_setup_path="$(platform_ros_setup_path)"

    # Check if ROS 2 Humble is installed
    if [[ -n "${ros_setup_path}" && -f "${ros_setup_path}" ]]; then
        log_info "ROS 2 Humble is already installed"
    else
        log_warn "ROS 2 Humble setup script not found${ros_setup_path:+ at ${ros_setup_path}}"
        platform_handle_missing_ros
    fi
}

ensure_colcon() {
    # Note: this only checks whether a colcon CLI exists somewhere on PATH.
    # The authoritative installation that build.sh consumes happens later in
    # setup_python_venv via `${venv_python} -m pip install colcon-common-extensions`,
    # which guarantees the module is importable from the workspace venv even
    # when PYTHONNOUSERSITE=1 is set (which is the case in build.sh and
    # .shrc_local). Installing colcon system-wide here is only a convenience
    # for users invoking `colcon` directly outside of build.sh.
    if command -v colcon &>/dev/null; then
        log_info "colcon CLI is already available on PATH (will also be installed into venv later)"
        return 0
    fi

    log_info "Installing colcon build tool..."
    platform_install_colcon
    log_done "colcon installed"
}

install_system_deps() {
    if setup_profile_is_training; then
        log_info "Training profile: skipping ROS 2, rosdep, and colcon installation."
    else
        # Check for ROS 2 installation first
        check_ros_installation
    fi

    platform_prepare_host

    if ! platform_supports_local_workspace_build; then
        log_info "Skipping local workspace build prerequisites on ${SETUP_PLATFORM_ID}."
        log_info "This platform consumes a prebuilt ROS runtime and host cross-built deployables."
        SYSTEM_DEPS_STATUS="skipped"
        log_skipped "Local workspace build prerequisites"
        return 0
    fi

    if setup_profile_is_training; then
        # A ROS-less host only needs the Python bootstrap and ffmpeg, and the
        # system package manager is touched only when one of them is actually
        # missing. The preflight exists purely to skip an unnecessary install.
        if platform_training_system_deps_satisfied; then
            log_info "Python bootstrap tooling and ffmpeg are already present; skipping system package installation."
            log_skipped "System package installation (training prerequisites already present)"
        else
            ensure_sudo_session
            platform_install_python_bootstrap
            platform_install_training_system_deps
        fi
        # Create the workspace venv early so VENV_PYTHON and the pip mirror are
        # configured before setup_python_venv runs. rosdep is deliberately not
        # installed (ensure_rosdep is never called on this path).
        ensure_workspace_venv
        SYSTEM_DEPS_STATUS="done"
        log_skipped "ROS 2 / rosdep / colcon (training profile)"
        return 0
    fi

    ensure_sudo_session
    platform_install_python_bootstrap
    platform_install_benchmark_system_deps
    ensure_colcon
    ensure_rosdep
    platform_install_rosdeps
    if [[ "${SYSTEM_DEPS_STATUS}" != "rosdep-errors" ]]; then
        SYSTEM_DEPS_STATUS="done"
    fi
}

verify_setup() {
    if [[ "${SKIP_VERIFY}" == true ]]; then
        log_info "Skipping ROS/Python verification (--skip-verify)."
        log_skipped "ROS/Python verification"
        return 0
    fi

    export SETUP_ROS_SETUP_PATH="$(platform_ros_setup_path)"
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/setup/verify_env.sh"
    if ! verify_env; then
        PYTHON_ENV_STATUS="failed"
        exit 1
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    trap 'on_setup_failure $? ${BASH_LINENO[0]}' ERR
    trap 'on_setup_interrupt' INT
    parse_args "$@"
    validate_setup_profile
    check_foreign_env

    cd "${WORKSPACE}"
    print_banner
    set_stage "detecting system"
    initialize_platform
    print_platform_summary
    if [[ "${INSTALL_BENCHMARK_DEPS}" == true ]]; then
        benchmark_preflight_host || exit 1
    fi
    set_stage "checking system dependencies"
    print_dependency_preview
    
    log_info "Setting up workspace at ${WORKSPACE} (profile: ${SETUP_PROFILE})"
    
    # Update submodules
    set_stage "syncing submodules"
    update_submodules
    if setup_profile_skips_ros_third_party; then
        # LiDAR/ROS third-party patch stacks operate on submodules that stay
        # uninitialized in the inference/training profiles (see submodules.sh).
        log_info "Skipping ROS third-party patch stacks (${SETUP_PROFILE} profile)."
        log_skipped "ROS third-party patch stacks (${SETUP_PROFILE} profile)"
    else
        ensure_ros_third_party_patch_stacks
    fi

    if [[ "${ONLY_PATCH}" != true ]]; then
        # Install dependencies
        set_stage "installing system dependencies"
        install_system_deps
        if [[ "${SYSTEM_DEPS_STATUS}" == "done" ]]; then
            if setup_profile_is_training; then
                log_done "Training host dependencies installed (venv bootstrap + ffmpeg)"
            else
                log_done "System ROS dependencies installed"
            fi
        elif [[ "${SYSTEM_DEPS_STATUS}" == "rosdep-errors" ]]; then
            log_error "System dependency installation had errors — see messages above."
            log_error "Fix the reported issues and re-run ./scripts/setup.sh."
            exit 1
        fi
        set_stage "configuring python environment"
        setup_python_venv
        if [[ "${PYTHON_ENV_STATUS}" == "done" ]]; then
            log_done "Python environment configured"
        fi
    else
        log_info "Only applying LeRobot patch stack (--only-patch); skipping dependency and venv setup."
    fi

    if platform_supports_local_workspace_build; then
        # LeRobot patch stack is normally applied inline by setup_python_venv
        # (before install_lerobot_editable so check_lerobot_python_compat sees
        # the patched pyproject.toml). The call below is an idempotent safety
        # net for edge cases where the workspace patch branch was lost. It is
        # a no-op when the patch stack is already applied.
        set_stage "verifying lerobot patch stack"
        ensure_lerobot_patch_stack_applied
    else
        log_info "Skipping LeRobot patch stack checks on runtime-only platform ${SETUP_PLATFORM_ID}."
    fi

    if [[ "${ONLY_PATCH}" != true ]]; then
        set_stage "verifying environment"
        verify_setup
    else
        log_skipped "ROS/Python verification"
    fi

    echo ""
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW} Setup Summary${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    if [[ "${ACCEL_NPU_PENDING_MANUAL_CONFIG:-false}" == true ]]; then
        SUMMARY+=("${YELLOW}!${NC} Ascend NPU backend not installed: NPU runtime pending manual configuration")
    fi
    for entry in "${SUMMARY[@]}"; do
        echo -e "  ${entry}"
    done
    echo ""
    local completion_message="Setup complete! Run ./scripts/build.sh to build the workspace."
    if setup_profile_is_training; then
        completion_message="Setup complete (training profile)! Start training with: source .shrc_local && lerobot-train --dataset.repo_id=<repo_id> --policy.type=act"
    elif [[ "${SETUP_PROFILE}" == "inference" ]]; then
        completion_message="Setup complete (inference profile)! Build the inference subset with: ./scripts/build.sh --packages-up-to inference_service"
    fi
    if [[ "${ONLY_PATCH}" == true ]]; then
        completion_message="LeRobot patch stack check complete."
    fi
    if [[ "${ONLY_PATCH}" != true ]] && ! platform_supports_local_workspace_build; then
        completion_message="Setup complete! OpenHarmony runtime verified; cross-build deployables on the host."
    fi
    echo -e "\033[1;32m[INFO] ${completion_message}${NC}"

    # A cann host whose NPU backend was not installed must not read "Setup
    # complete!" as "everything works": NPU inference is unavailable until the
    # runtime is configured by hand. The state is recorded by the caller of
    # accel_requirements (the hook runs in a subshell); see python_venv.sh.
    if [[ "${ACCEL_NPU_PENDING_MANUAL_CONFIG:-false}" == true ]]; then
        echo ""
        echo -e "${YELLOW}------------------------------------------------------------${NC}"
        echo -e "${YELLOW} Ascend NPU runtime is NOT configured${NC}"
        echo -e "${YELLOW} -${NC} No torch_npu is installed; NPU inference is unavailable until it is."
        echo -e "${YELLOW} -${NC} Check the CANN/Torch/torch_npu/torchvision pairing for your CANN release:"
        echo -e "${YELLOW}   https://www.hiascend.com/developer/download/compatibility${NC}"
        echo -e "${YELLOW} -${NC} Then install the matched versions with the workspace venv, not the system pip:"
        echo -e "${YELLOW}   venv/bin/python -m pip install torch==<v> torch_npu==<v> torchvision==<v>${NC}"
        echo -e "${YELLOW}------------------------------------------------------------${NC}"
    fi
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
