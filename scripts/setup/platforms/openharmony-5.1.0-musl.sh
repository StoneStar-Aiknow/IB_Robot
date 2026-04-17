#!/bin/bash

platform_ros_setup_path() {
    if [[ -n "${ROS_HUMBLE_SETUP_PATH:-}" ]]; then
        echo "${ROS_HUMBLE_SETUP_PATH}"
    else
        echo ""
    fi
}

platform_handle_missing_ros() {
    log_error "ROS 2 Humble setup script is not configured for OpenHarmony."
    log_error "Set ROS_HUMBLE_SETUP_PATH to your Python 3.11 ROS Humble environment before running setup."
    exit 1
}

platform_prepare_host() {
    log_warn "OpenHarmony musl platform detected."
    log_warn "System dependency bootstrap is not yet fully automated for this platform."
    log_warn "Proceeding with shared setup steps only; platform-specific package installation must be supplied separately."
}

platform_install_colcon() {
    if command -v pip3 &> /dev/null; then
        pip3 install colcon-common-extensions --quiet
    else
        log_error "pip3 not found, cannot install colcon."
        exit 1
    fi
}

platform_install_python_bootstrap() {
    if ! python3 -m venv --help >/dev/null 2>&1; then
        log_error "python3 venv module is unavailable on this OpenHarmony environment."
        log_error "Install the Python venv package or provide a pre-created virtual environment."
        exit 1
    fi
}

platform_install_rosdeps() {
    log_warn "Skipping automated rosdepc installation on OpenHarmony musl."
    log_warn "Provide ROS/system dependencies externally or rerun with --skip-system-deps if this is intentional."
}

platform_verify_ros_python_bridge() {
    local ros_setup
    ros_setup="$(platform_ros_setup_path)"
    [[ -z "${ros_setup}" || ! -f "${ros_setup}" ]] && return 1
    (source "${ros_setup}" && python3 -c "import rclpy; print('ROS 2 Humble connection successful')") 2>/dev/null
}
