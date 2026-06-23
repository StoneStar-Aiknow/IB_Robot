#!/bin/bash
# Apply only the managed LeRobot patch stack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

AUTO_YES=true
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"
USE_SUDO=false
SUDO_AUTH_READY=true
PLATFORM_OVERRIDE=""
FORCE_REBUILD=false

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

show_help() {
    cat <<'EOF'
Apply only the IB_Robot-managed LeRobot patch stack.

Usage:
  scripts/apply_lerobot_patches.sh [OPTIONS]

Options:
      --force             Rebuild libs/lerobot from the recorded upstream base.
      --platform ID       Override platform detection.
      --lerobot-profiles CSV
                          Override patch profile selection.
  -h, --help              Show this help.

Platform IDs:
  ubuntu-22.04                  default profiles: core,ros,hardware,dev,training,distillation
  openeuler-embedded-24.03      default profiles: core,ros,hardware,openeuler
  openharmony-5.1.0-musl        default profiles: core,openharmony

Profile tags:
  core, ros, hardware, dev, openeuler, openharmony, ascend, om, 3403,
  inference, training, distillation, models, mt-act, tooling,
  master-parity-candidates

Examples:
  scripts/apply_lerobot_patches.sh --force
  scripts/apply_lerobot_patches.sh --platform ubuntu-22.04 --lerobot-profiles core,ros,hardware,dev
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE_REBUILD=true
            ;;
        --platform)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--platform requires a platform ID" >&2
                exit 1
            fi
            PLATFORM_OVERRIDE="$1"
            ;;
        --lerobot-profiles)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--lerobot-profiles requires a comma-separated profile list" >&2
                exit 1
            fi
            export IBR_LEROBOT_PROFILES_CLI="$1"
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
    shift
done

# shellcheck disable=SC1091
source "${WORKSPACE}/scripts/setup/common.sh"
# shellcheck disable=SC1091
source "${WORKSPACE}/scripts/setup/detect.sh"
# shellcheck disable=SC1091
source "${WORKSPACE}/scripts/setup/lerobot_patches.sh"

# Minimal platform hook defaults copied from setup.sh. Platform scripts loaded
# by detect.sh may override these, but detect_python_runtimes needs
# platform_ros_setup_path to exist before that happens on generic hosts.
platform_ros_setup_path() {
    if [[ -n "${ROS_HUMBLE_SETUP_PATH:-}" ]]; then
        echo "${ROS_HUMBLE_SETUP_PATH}"
    elif [[ -f /opt/ros/humble/setup.sh ]]; then
        echo "/opt/ros/humble/setup.sh"
    else
        echo "/opt/ros/humble/setup.bash"
    fi
}

platform_supports_local_workspace_build() { return 0; }

cd "${WORKSPACE}"

if [[ ! -d "libs/lerobot/.git" && ! -f "libs/lerobot/.git" ]]; then
    log_info "Initializing libs/lerobot submodule..."
    git submodule update --init libs/lerobot
fi

if [[ -x "${WORKSPACE}/venv/bin/python" ]]; then
    VENV_PYTHON="${WORKSPACE}/venv/bin/python"
    export VENV_PYTHON
fi

initialize_platform

if [[ "${FORCE_REBUILD}" == true ]]; then
    export IBR_LEROBOT_FORCE_REBUILD=1
fi

ensure_lerobot_patch_stack_applied
