#!/bin/bash
# verify_env.sh — Independent environment verification for IB_Robot workspace
#
# This file is designed to be sourced (not executed directly).
# It exposes individual verify_* functions and a top-level verify_env entry point.
#
# Required variables (must be set by the caller before sourcing):
#   WORKSPACE          — absolute path to the IB_Robot workspace root
#   VENV_PYTHON        — absolute path to the venv python interpreter
#   SETUP_PLATFORM_ID  — one of: ubuntu-22.04, openeuler-embedded-24.03, openharmony-5.1.0-musl
#   ROSDEP_BIN         — absolute path to the rosdep binary in the venv
#   USE_SUDO           — "true" or "false"
#   SETUP_ROS_SETUP_PATH — path to ROS 2 setup script (e.g. /opt/ros/humble/setup.bash)
#
# Colors (imported from setup.sh context):
#   RED, GREEN, YELLOW, NC, log_info, log_warn, log_error, log_done
#
# Fail-fast: if any required variable is missing, abort immediately.

: "${WORKSPACE:?Variable WORKSPACE is not set}"
: "${VENV_PYTHON:?Variable VENV_PYTHON is not set}"
: "${SETUP_PLATFORM_ID:?Variable SETUP_PLATFORM_ID is not set}"
: "${ROSDEP_BIN:?Variable ROSDEP_BIN is not set}"
: "${USE_SUDO:?Variable USE_SUDO is not set}"
: "${SETUP_ROS_SETUP_PATH:?Variable SETUP_ROS_SETUP_PATH is not set}"

verify_ros() {
    local ros_setup="${SETUP_ROS_SETUP_PATH}"
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying ROS 2 connection..."

    if [[ -z "${ros_setup}" || ! -f "${ros_setup}" ]]; then
        log_error "ROS 2 setup script not found at: ${ros_setup:-<empty>}"
        return 1
    fi

    if (set +u; source "${ros_setup}" >/dev/null 2>&1 && set -u && "${venv_python}" -c 'import rclpy; print("ROS 2 Humble connection successful")' >/dev/null 2>&1); then
        log_info "ROS 2 verification: venv can access ROS 2 packages."
        return 0
    fi

    log_error "Verification failed: rclpy not found. Ensure ROS 2 is installed and --system-site-packages was used."
    log_error "If ${WORKSPACE}/venv was created without --system-site-packages, remove it and rerun ./scripts/setup.sh."
    return 1
}

verify_colcon() {
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying colcon..."

    if ! PYTHONNOUSERSITE=1 "${venv_python}" -m colcon --help >/dev/null 2>&1; then
        log_error "Verification failed: 'python3 -m colcon --help' does not work inside the venv."
        log_error "build.sh runs colcon this exact way; please reinstall colcon into the venv:"
        log_error "  source venv/bin/activate"
        log_error "  PYTHONNOUSERSITE=1 python3 -m pip install --upgrade colcon-common-extensions colcon-mixin"
        return 1
    fi

    if ! command -v colcon &>/dev/null; then
        log_warn "colcon is importable from the venv but no 'colcon' CLI is on PATH."
        log_warn "Activate the venv (source venv/bin/activate) before running colcon directly."
    fi

    return 0
}

verify_rosdep() {
    local rosdep_bin="${ROSDEP_BIN}"

    log_info "Verifying rosdep..."

    if ! "${rosdep_bin}" --help >/dev/null 2>&1; then
        log_error "Verification failed: ${rosdep_bin} did not respond to --help."
        log_error "  - Check that rosdep was installed into the workspace venv:"
        log_error "      ${VENV_PYTHON} -m pip show rosdep"
        log_error "  - Re-run setup with VERBOSE=1 to see the install transcript."
        return 1
    fi

    return 0
}

verify_numpy_compat() {
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying NumPy + Empy compatibility..."

    if ! PYTHONNOUSERSITE=1 "${venv_python}" - <<'PY'
import em

raise SystemExit(0 if hasattr(em, "BUFFERED_OPT") else 1)
PY
    then
        log_error "Verification failed: Empy is not ROS 2 Humble compatible."
        log_error "rosidl_adapter requires Empy 3.x (em.BUFFERED_OPT)."
        log_error "Re-run ./scripts/setup.sh to restore empy==3.3.4 in the venv."
        return 1
    fi

    if ! "${venv_python}" -c "import numpy; assert numpy.__version__.startswith('1.26.')" >/dev/null 2>&1; then
        log_error "Verification failed: NumPy is not pinned to the expected ROS-compatible 1.26.x series."
        return 1
    fi

    return 0
}

verify_lerobot() {
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying lerobot..."

    if ! "${venv_python}" -c "import lerobot" >/dev/null 2>&1; then
        log_error "Verification failed: lerobot import failed."
        return 1
    fi

    return 0
}

verify_pygraphviz() {
    local venv_python="${VENV_PYTHON}"

    case "${SETUP_PLATFORM_ID}" in
        openeuler-embedded-24.03)
            log_info "Verifying pygraphviz..."

            if ! PYTHONNOUSERSITE=1 "${venv_python}" -c "import pygraphviz" >/dev/null 2>&1; then
                log_error "Verification failed: pygraphviz is not importable from the workspace venv."
                log_error "Re-run ./scripts/setup.sh after ensuring graphviz and graphviz-devel are installed."
                return 1
            fi
            ;;
    esac

    return 0
}

verify_tracing() {
    local venv_python="${VENV_PYTHON}"
    local ros_setup="${SETUP_ROS_SETUP_PATH}"

    log_info "Verifying tracing tools..."

    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            if ! command -v lttng &>/dev/null; then
                log_error "Verification failed: lttng CLI is not available on PATH."
                return 1
            fi

            if ! command -v babeltrace2 &>/dev/null; then
                log_error "Verification failed: babeltrace2 CLI is not available on PATH."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && ros2 trace --help >/dev/null 2>&1); then
                log_error "Verification failed: ros2 trace CLI is not available from the ROS 2 environment."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import lttngust" >/dev/null 2>&1); then
                log_error "Verification failed: python3-lttngust is not importable from the workspace venv."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import tracetools_analysis" >/dev/null 2>&1); then
                log_error "Verification failed: tracetools-analysis is not importable from the workspace venv."
                return 1
            fi
            ;;
        openeuler-embedded-24.03)
            if ! (set +u; source "${ros_setup}" && set -u && ros2 trace --help >/dev/null 2>&1); then
                log_error "Verification failed: ros2 trace CLI is not available from the ROS 2 environment."
                return 1
            fi

            if ! command -v babeltrace &>/dev/null && ! command -v babeltrace2 &>/dev/null; then
                log_error "Verification failed: neither babeltrace nor babeltrace2 is available on PATH."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import babeltrace" >/dev/null 2>&1); then
                log_error "Verification failed: python3-babeltrace is not importable from the workspace venv."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import tracetools_analysis" >/dev/null 2>&1); then
                log_error "Verification failed: tracetools-analysis is not importable from the workspace venv."
                return 1
            fi

            if ! rpm -q lttng-ust >/dev/null 2>&1; then
                log_error "Verification failed: lttng-ust is not installed on openEuler."
                return 1
            fi

            if ! "${venv_python}" -c "import lttngust" >/dev/null 2>&1; then
                log_warn "python3-lttngust is not packaged in the current openEuler repos."
                log_warn "ROS trace CLI is available, but Python-domain ib_trace.* logging remains disabled."
            fi
            ;;
    esac

    return 0
}

verify_runtime_ros_python_bridge() {
    local ros_setup="${SETUP_ROS_SETUP_PATH}"
    local python_bin="$(command -v python3 || true)"
    
    if [[ -z "${ros_setup}" || ! -f "${ros_setup}" || -z "${python_bin}" ]]; then
        return 1
    fi

    (
        set +u
        set +e
        source "${ros_setup}" >/dev/null 2>&1
        "${python_bin}" -c 'import rclpy; print("ROS 2 Humble connection successful")'
    ) 2>/dev/null
}

verify_env() {
    if ! platform_supports_local_workspace_build 2>/dev/null; then
        log_info "Verifying OpenHarmony ROS runtime..."
        if verify_runtime_ros_python_bridge >/dev/null 2>&1; then
            log_done "Verified OpenHarmony ROS runtime"
            return 0
        fi

        log_error "Verification failed: could not import rclpy from the OpenHarmony runtime."
        log_error "Source /data/ros2ohos.env and ensure /data/out/bin/python3.12 is available."
        return 1
    fi

    verify_ros || return 1
    verify_rosdep || return 1
    verify_colcon || return 1
    verify_numpy_compat || return 1
    verify_lerobot || return 1
    verify_pygraphviz || return 1
    verify_tracing || return 1

    log_done "Verified ROS, rosdep, colcon, lerobot, and NumPy compatibility"
}
