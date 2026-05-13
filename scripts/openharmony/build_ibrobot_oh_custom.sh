#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IB_ROBOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

ensure_dir() {
    mkdir -p "$1"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log_error "Missing required command: $1"
        exit 1
    fi
}

OH_ROOT="${OH_ROOT:-}"
OH_DOWNLOAD_ROOT="${OH_DOWNLOAD_ROOT:-}"
OH_CUSTOM_ROOT="${OH_CUSTOM_ROOT:-}"
OH_CUSTOM_WS="${OH_CUSTOM_WS:-}"
OH_CUSTOM_SRC=""
OH_CUSTOM_TOOLCHAIN_ROOT="${OH_CUSTOM_TOOLCHAIN_ROOT:-}"
OH_CUSTOM_SDK_TAR_GLOB="${OH_CUSTOM_SDK_TAR_GLOB:-}"
OH_CUSTOM_IMAGE="${OH_CUSTOM_IMAGE:-voxelsky/ohos-ros-humble-builder:v0.1.5}"
OH_CUSTOM_CONTAINER_NAME="${OH_CUSTOM_CONTAINER_NAME:-ibrobot-oh-build}"
OH_CUSTOM_PREFIX="${OH_CUSTOM_PREFIX:-/data/ibrobot/install}"
OH_BOARD_ROS_PREFIX="${OH_BOARD_ROS_PREFIX:-/data/install}"
OH_BOARD_TORCH_RUNTIME_ROOT="${OH_BOARD_TORCH_RUNTIME_ROOT:-/data/local/skh-run}"
OH_CUSTOM_CPU="${OH_CUSTOM_CPU:-aarch64}"
OH_CUSTOM_SYSDEPS_TAR_GLOB="${OH_CUSTOM_SYSDEPS_TAR_GLOB:-}"
OH_CUSTOM_ROS2_BASE_REPO="${OH_CUSTOM_ROS2_BASE_REPO:-}"
OH_CUSTOM_VERSION_REPO="${OH_CUSTOM_VERSION_REPO:-}"
OH_CUSTOM_HUMBLE_TAR_GLOB="${OH_CUSTOM_HUMBLE_TAR_GLOB:-}"

USE_SUDO=0
DRY_RUN=0
PULL_IMAGE=1
declare -a PACKAGES=("ibrobot_msgs" "tensormsg" "robot_config" "inference_service")
declare -a COLCON_ARGS=()
declare -a CMAKE_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/openharmony/build_ibrobot_oh_custom.sh [options]

Prepare and run the official OpenHarmony ROS custom-package cross-build for the
minimum IB_Robot inference workspace.

Options:
  --oh-root <dir>          Unified external OpenHarmony root (recommended)
                           Defaults to deriving downloads/ and custom_build_root/ from it
  --root <dir>             Custom build root (default: <OH_ROOT>/custom_build_root)
  --workspace <dir>        Custom ROS workspace root (default: <root>/ibrobot_oh_ws)
  --toolchain-root <dir>   OHOS SDK root containing 18/native
  --sdk-tar <path>         Official OH ROS SDK tarball providing the 18/ tree
                           (default: <OH_ROOT>/downloads/sdk/ohos-sdk-18-linux-aarch64-*.tar.gz)
  --sysdeps-tar <path>     OH sysdeps tarball used to augment the SDK sysroot
                           (default: <OH_ROOT>/downloads/sysdeps/ohos-*-sysdeps-*.tar.gz)
  --humble-tar <path>      OH ROS 2 Humble runtime tarball
                           (default: <OH_ROOT>/downloads/runtime/ohos-humble-build-aarch64-*.tar.gz)
  --custom-prefix <path>   Final on-device install prefix (default: /data/ibrobot/install)
  --cpu <arch>             Target OHOS CPU (default: aarch64)
  --image <image>          Builder image (default: voxelsky/ohos-ros-humble-builder:v0.1.5)
  --container-name <name>  Container name (default: ibrobot-oh-build)
  --packages <csv>         Comma-separated package list
  --colcon-args <...>      Extra arguments passed through to build-ros-humble
  --cmk-args <...>         Extra CMake arguments passed through to build-ros-humble
  --sudo                   Run docker via sudo
  --no-pull                Skip docker pull if image is missing locally
  --dry-run                Print the docker command without running it
  -h, --help               Show this help
EOF
}

docker_cmd() {
    if [[ "${USE_SUDO}" -eq 1 ]]; then
        sudo docker "$@"
    else
        docker "$@"
    fi
}

abspath() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${PWD}" "${path}"
    fi
}

apply_layout_defaults() {
    if [[ -n "${OH_ROOT}" ]]; then
        [[ -z "${OH_DOWNLOAD_ROOT}" ]] && OH_DOWNLOAD_ROOT="${OH_ROOT}/downloads"
        [[ -z "${OH_CUSTOM_ROOT}" ]] && OH_CUSTOM_ROOT="${OH_ROOT}/custom_build_root"
    fi

    if [[ -z "${OH_CUSTOM_ROOT}" ]]; then
        log_error "Missing OpenHarmony build root."
        log_error "Pass --oh-root <dir> (recommended), or set OH_CUSTOM_ROOT / --root explicitly."
        exit 1
    fi

    [[ -z "${OH_CUSTOM_WS}" ]] && OH_CUSTOM_WS="${OH_CUSTOM_ROOT}/ibrobot_oh_ws"
    [[ -z "${OH_CUSTOM_TOOLCHAIN_ROOT}" ]] && OH_CUSTOM_TOOLCHAIN_ROOT="${OH_CUSTOM_ROOT}/ohos-robot-toolchain"
    [[ -z "${OH_CUSTOM_ROS2_BASE_REPO}" ]] && OH_CUSTOM_ROS2_BASE_REPO="${OH_CUSTOM_ROOT}/ros_ros2_base"
    [[ -z "${OH_CUSTOM_VERSION_REPO}" ]] && OH_CUSTOM_VERSION_REPO="${OH_CUSTOM_ROOT}/version"

    if [[ -n "${OH_DOWNLOAD_ROOT}" ]]; then
        [[ -z "${OH_CUSTOM_SDK_TAR_GLOB}" ]] && OH_CUSTOM_SDK_TAR_GLOB="${OH_DOWNLOAD_ROOT}/sdk/ohos-sdk-18-linux-aarch64-*.tar.gz"
        [[ -z "${OH_CUSTOM_SYSDEPS_TAR_GLOB}" ]] && OH_CUSTOM_SYSDEPS_TAR_GLOB="${OH_DOWNLOAD_ROOT}/sysdeps/ohos-*-sysdeps-*.tar.gz"
        [[ -z "${OH_CUSTOM_HUMBLE_TAR_GLOB}" ]] && OH_CUSTOM_HUMBLE_TAR_GLOB="${OH_DOWNLOAD_ROOT}/runtime/ohos-humble-build-aarch64-*.tar.gz"
    fi

    OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
}

normalize_paths() {
    [[ -n "${OH_ROOT}" ]] && OH_ROOT="$(abspath "${OH_ROOT}")"
    [[ -n "${OH_DOWNLOAD_ROOT}" ]] && OH_DOWNLOAD_ROOT="$(abspath "${OH_DOWNLOAD_ROOT}")"
    OH_CUSTOM_ROOT="$(abspath "${OH_CUSTOM_ROOT}")"
    OH_CUSTOM_WS="$(abspath "${OH_CUSTOM_WS}")"
    OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
    OH_CUSTOM_TOOLCHAIN_ROOT="$(abspath "${OH_CUSTOM_TOOLCHAIN_ROOT}")"
    OH_CUSTOM_ROS2_BASE_REPO="$(abspath "${OH_CUSTOM_ROS2_BASE_REPO}")"
    OH_CUSTOM_VERSION_REPO="$(abspath "${OH_CUSTOM_VERSION_REPO}")"
}

ensure_repo_checkout() {
    local repo_url="$1"
    local dest_dir="$2"

    if [[ -d "${dest_dir}/.git" ]]; then
        return
    fi

    log_info "Cloning $(basename "${dest_dir}") into ${dest_dir}..."
    git clone --depth 1 "${repo_url}" "${dest_dir}"
}

ensure_humble_install() {
    local tar_path

    if [[ -d "${OH_CUSTOM_ROOT}/install" ]]; then
        return
    fi

    tar_path="$(compgen -G "${OH_CUSTOM_HUMBLE_TAR_GLOB}" | sort | tail -n 1 || true)"
    if [[ -z "${tar_path}" ]]; then
        log_error "Cannot find a host ohos-humble-build tarball matching:"
        log_error "  ${OH_CUSTOM_HUMBLE_TAR_GLOB}"
        exit 1
    fi

    log_info "Extracting $(basename "${tar_path}") into ${OH_CUSTOM_ROOT}..."
    tar -zxpf "${tar_path}" -C "${OH_CUSTOM_ROOT}"
}

ensure_workspace_links() {
    ensure_dir "${OH_CUSTOM_SRC}"

    for pkg in "${PACKAGES[@]}"; do
        local src_pkg="${IB_ROBOT_ROOT}/src/${pkg}"
        local dst_pkg="${OH_CUSTOM_SRC}/${pkg}"
        if [[ ! -e "${src_pkg}" ]]; then
            log_error "Source package not found: ${src_pkg}"
            exit 1
        fi
        rm -rf "${dst_pkg}"
        cp -a "${src_pkg}" "${dst_pkg}"
    done
}

ensure_lerobot_submodule() {
    local lerobot_dir="${IB_ROBOT_ROOT}/libs/lerobot"

    if [[ -d "${lerobot_dir}/.git" || -f "${lerobot_dir}/.git" ]] && [[ -d "${lerobot_dir}/src" ]]; then
        return
    fi

    log_info "Initializing libs/lerobot submodule for OpenHarmony runtime staging..."
    git -C "${IB_ROBOT_ROOT}" submodule update --init --recursive libs/lerobot

    if [[ ! -d "${lerobot_dir}/src" ]]; then
        log_error "LeRobot source tree still missing after submodule init: ${lerobot_dir}/src"
        exit 1
    fi
}

resolve_openharmony_lerobot_patch_stack() {
    local index_file="${IB_ROBOT_ROOT}/third_party/patches/lerobot/INDEX.yaml"
    local active_tag=""

    if [[ ! -f "${index_file}" ]]; then
        log_error "LeRobot patch index not found: ${index_file}"
        exit 1
    fi

    active_tag="$(awk -F': *' '/^active_tag:/ { print $2; exit }' "${index_file}")"
    if [[ -z "${active_tag}" ]]; then
        log_error "Could not resolve active_tag from ${index_file}"
        exit 1
    fi

    LEROBOT_OH_PATCH_DIR="${IB_ROBOT_ROOT}/third_party/patches/lerobot/${active_tag}"
    LEROBOT_OH_PATCH_SERIES="${LEROBOT_OH_PATCH_DIR}/series.openharmony-5.1.0-musl.txt"
    LEROBOT_OH_PATCH_MANIFEST="${LEROBOT_OH_PATCH_DIR}/manifest.yaml"
    LEROBOT_OH_BASE_COMMIT="$(awk '
        /^lerobot_commit_range:/ { in_range=1; next }
        in_range && /^[^[:space:]]/ { in_range=0 }
        in_range && $1 == "min:" { print $2; exit }
    ' "${LEROBOT_OH_PATCH_MANIFEST}")"

    if [[ ! -d "${LEROBOT_OH_PATCH_DIR}" || ! -f "${LEROBOT_OH_PATCH_SERIES}" || ! -f "${LEROBOT_OH_PATCH_MANIFEST}" ]]; then
        log_error "OpenHarmony lerobot patch stack is incomplete under ${LEROBOT_OH_PATCH_DIR}"
        exit 1
    fi
    if [[ -z "${LEROBOT_OH_BASE_COMMIT}" ]]; then
        log_error "Could not resolve lerobot base commit from ${LEROBOT_OH_PATCH_MANIFEST}"
        exit 1
    fi
}

verify_openharmony_lerobot_runtime_patch() {
    local src_root="$1"
    local init_file="${src_root}/lerobot/policies/__init__.py"
    local factory_file="${src_root}/lerobot/policies/factory.py"
    local optim_file="${src_root}/lerobot/optim/optimizers.py"

    grep -q "_LAZY_EXPORTS" "${init_file}" || {
        log_error "OpenHarmony lerobot runtime staging is missing lazy exports in ${init_file}"
        exit 1
    }
    grep -q "__getattr__" "${init_file}" || {
        log_error "OpenHarmony lerobot runtime staging is missing __getattr__ lazy loading in ${init_file}"
        exit 1
    }
    grep -q "_get_builtin_policy_config_class" "${factory_file}" || {
        log_error "OpenHarmony lerobot runtime staging is missing lazy policy factory logic in ${factory_file}"
        exit 1
    }
    if grep -q '^from lerobot\.datasets' "${factory_file}"; then
        log_error "OpenHarmony lerobot runtime staging still has top-level dataset imports in ${factory_file}"
        exit 1
    fi
    if grep -q '^from lerobot\.datasets' "${optim_file}"; then
        log_error "OpenHarmony lerobot runtime staging still has top-level dataset imports in ${optim_file}"
        exit 1
    fi
}

prepare_openharmony_lerobot_runtime_src() {
    local stage_root="${OH_CUSTOM_ROOT}/.lerobot_openharmony_runtime"
    local repo_dir="${stage_root}/repo"
    local lerobot_dir="${IB_ROBOT_ROOT}/libs/lerobot"
    local git_user_name="${IBR_LEROBOT_GIT_USER_NAME:-IB Robot Setup}"
    local git_user_email="${IBR_LEROBOT_GIT_USER_EMAIL:-ibrobot@example.invalid}"
    local patch_file=""

    ensure_lerobot_submodule
    resolve_openharmony_lerobot_patch_stack

    rm -rf "${stage_root}"
    ensure_dir "${stage_root}"

    log_info "Preparing OpenHarmony-patched LeRobot runtime staging tree..." >&2
    git clone --local --no-checkout "${lerobot_dir}" "${repo_dir}" >/dev/null
    git -C "${repo_dir}" checkout --detach "${LEROBOT_OH_BASE_COMMIT}" >/dev/null

    while IFS= read -r patch_file; do
        [[ -z "${patch_file}" || "${patch_file}" == \#* ]] && continue
        log_info "Applying OpenHarmony lerobot runtime patch ${patch_file}..." >&2
        git -C "${repo_dir}" \
            -c "user.name=${git_user_name}" \
            -c "user.email=${git_user_email}" \
            am "${LEROBOT_OH_PATCH_DIR}/${patch_file}" >/dev/null
    done < "${LEROBOT_OH_PATCH_SERIES}"

    verify_openharmony_lerobot_runtime_patch "${repo_dir}/src"
    printf '%s\n' "${repo_dir}/src"
}

stage_lerobot_runtime() {
    local install_root="$1"
    local lerobot_dst="${install_root}/lerobot/src"
    local lerobot_src=""

    lerobot_src="$(prepare_openharmony_lerobot_runtime_src)"

    rm -rf "${install_root}/lerobot"
    ensure_dir "${lerobot_dst}"
    cp -a "${lerobot_src}/." "${lerobot_dst}/"
}

rewrite_runtime_prefix_chain() {
    local install_root="$1"
    local file
    local package_setup

    for file in \
        "${install_root}/setup.sh" \
        "${install_root}/setup.bash" \
        "${install_root}/setup.zsh" \
        "${install_root}/setup.ps1"; do
        [[ -f "${file}" ]] || continue
        sed -i "s|/mnt/ohos/tmp/install|${OH_BOARD_ROS_PREFIX}|g" "${file}"
    done

    while IFS= read -r -d '' package_setup; do
        sed -i "s|/mnt/ohos/tmp/install|${OH_BOARD_ROS_PREFIX}|g" "${package_setup}"
    done < <(find "${install_root}" \( -path '*/local_setup.sh' -o -path '*/local_setup.bash' -o -path '*/local_setup.zsh' -o -path '*/package.sh' \) -type f -print0)

    while IFS= read -r -d '' file; do
        sed -i "s|/mnt/ohos/tmp/install|${OH_BOARD_ROS_PREFIX}|g" "${file}"
    done < <(find "${install_root}" -path '*/share/ament_index/resource_index/parent_prefix_path/*' -type f -print0)
}

append_lerobot_runtime_hook() {
    local install_root="$1"
    local file
    local marker="# ibrobot openharmony lerobot runtime path"

    for file in \
        "${install_root}/setup.sh" \
        "${install_root}/setup.bash" \
        "${install_root}/setup.zsh"; do
        [[ -f "${file}" ]] || continue
        if grep -qF "${marker}" "${file}"; then
            continue
        fi
        cat <<EOF >> "${file}"

${marker}
_ibrobot_lerobot_src="${OH_CUSTOM_PREFIX}/lerobot/src"
if [ -d "\$_ibrobot_lerobot_src" ]; then
  case ":\${PYTHONPATH:-}:" in
    *":\$_ibrobot_lerobot_src:"*) ;;
    *) export PYTHONPATH="\$_ibrobot_lerobot_src\${PYTHONPATH:+:\$PYTHONPATH}" ;;
  esac
fi
unset _ibrobot_lerobot_src
EOF
    done

    while IFS= read -r -d '' file; do
        [[ -f "${file}" ]] || continue
        if grep -qF "${marker}" "${file}"; then
            continue
        fi
        cat <<EOF >> "${file}"

${marker}
_ibrobot_lerobot_src="${OH_CUSTOM_PREFIX}/lerobot/src"
if [ -d "\$_ibrobot_lerobot_src" ]; then
  case ":\${PYTHONPATH:-}:" in
    *":\$_ibrobot_lerobot_src:"*) ;;
    *) export PYTHONPATH="\$_ibrobot_lerobot_src\${PYTHONPATH:+:\$PYTHONPATH}" ;;
  esac
fi
unset _ibrobot_lerobot_src
EOF
    done < <(find "${install_root}" -path '*/local_setup.sh' -type f -print0)
}

rewrite_inference_entrypoints_for_board_runtime() {
    local install_root="$1"
    local rel_path=""
    local module_name=""
    local script_path=""

    while IFS='|' read -r rel_path module_name; do
        for script_path in "${install_root}/${rel_path}" "${install_root}"/*/${rel_path}; do
            [[ -f "${script_path}" ]] || continue

            cat <<EOF > "${script_path}"
#!/system/bin/sh
SKH_ROOT="${OH_BOARD_TORCH_RUNTIME_ROOT}"
ROS_HOME_ROOT="/data/local/tmp/ros_home"
ROS_LOG_ROOT="/data/local/tmp/ros_logs"

if [ ! -x "\${SKH_ROOT}/bin/python3" ]; then
  echo "[IB_Robot] Missing OpenHarmony PyTorch runtime at \${SKH_ROOT}" >&2
  echo "[IB_Robot] Deploy thirdparty_pytorch test/skh-run.tar.gz to /data/local first." >&2
  exit 1
fi

mkdir -p "\${ROS_HOME_ROOT}" "\${ROS_LOG_ROOT}" >/dev/null 2>&1 || true

export PYTHONHOME="\${SKH_ROOT}"
export PATH="\${SKH_ROOT}/bin:\$PATH"
export HOME="\${ROS_HOME_ROOT}"
export ROS_LOG_DIR="\${ROS_LOG_ROOT}"
export PYTHONPATH="\${SKH_ROOT}/lib/python3.12/site-packages:\${SKH_ROOT}/usr/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/lerobot/src:${OH_CUSTOM_PREFIX}/inference_service/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/robot_config/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/tensormsg/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/ibrobot_msgs/lib/python3.12/site-packages:${OH_BOARD_ROS_PREFIX}/lib/python3.12/site-packages:/data/out/lib/python3.12/site-packages:/data/out/lib\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="\${SKH_ROOT}/lib:\${SKH_ROOT}/lib/python3.12/site-packages/torchaudio/lib:${OH_CUSTOM_PREFIX}/inference_service/lib:${OH_CUSTOM_PREFIX}/robot_config/lib:${OH_CUSTOM_PREFIX}/tensormsg/lib:${OH_CUSTOM_PREFIX}/ibrobot_msgs/lib:${OH_BOARD_ROS_PREFIX}/lib:/data/out/lib:/usr/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LD_PRELOAD="\${SKH_ROOT}/lib/libpython3.12.so.1.0:\${SKH_ROOT}/lib/libomp.so\${LD_PRELOAD:+:\$LD_PRELOAD}"

exec "\${SKH_ROOT}/bin/python3" -m ${module_name} "\$@"
EOF
            chmod +x "${script_path}"
        done
    done <<'EOF'
lib/inference_service/lerobot_policy_node|inference_service.lerobot_policy_node
lib/inference_service/pure_inference_node|inference_service.pure_inference_node
EOF
}

postprocess_runtime_bundle() {
    local install_root="${OH_CUSTOM_WS}/install"

    if [[ ! -d "${install_root}" ]]; then
        log_error "Missing install tree after build: ${install_root}"
        exit 1
    fi

    log_info "Post-processing OpenHarmony runtime bundle..."
    stage_lerobot_runtime "${install_root}"
    rewrite_runtime_prefix_chain "${install_root}"
    append_lerobot_runtime_hook "${install_root}"
    rewrite_inference_entrypoints_for_board_runtime "${install_root}"
}

normalize_runtime_bundle_ownership() {
    local host_uid
    local host_gid

    host_uid="$(id -u)"
    host_gid="$(id -g)"

    docker_cmd run --rm \
        -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
        "${OH_CUSTOM_IMAGE}" \
        sh -c "chown -R ${host_uid}:${host_gid} /mnt/ohos/ibrobot_oh_ws/install /mnt/ohos/ibrobot_oh_ws/build /mnt/ohos/ibrobot_oh_ws/log || true"
}

ensure_toolchain_root() {
    local sdk_tar=""

    ensure_dir "${OH_CUSTOM_TOOLCHAIN_ROOT}"

    if [[ ! -d "${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native" ]]; then
        sdk_tar="$(compgen -G "${OH_CUSTOM_SDK_TAR_GLOB}" | sort | tail -n 1 || true)"
        if [[ -n "${sdk_tar}" ]]; then
            log_info "Extracting official OH ROS SDK $(basename "${sdk_tar}") into ${OH_CUSTOM_TOOLCHAIN_ROOT}..."
            tar -zxpf "${sdk_tar}" -C "${OH_CUSTOM_TOOLCHAIN_ROOT}"
        fi
    fi

    if [[ ! -d "${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native" ]]; then
        log_error "Missing OHOS SDK under ${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native"
        log_error "Tried SDK archive glob:"
        log_error "  ${OH_CUSTOM_SDK_TAR_GLOB}"
        log_error "Place the downloaded OHOS ROS SDK there, set OH_CUSTOM_SDK_TAR_GLOB, or pass --sdk-tar."
        exit 1
    fi
}

ensure_sysdeps_overlay() {
    local sysroot_usr="${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native/sysroot/usr"
    local sysdeps_tar=""
    local stage_dir="${OH_CUSTOM_ROOT}/.sysdeps_overlay"

    if [[ -f "${sysroot_usr}/include/python3.12/Python.h" && \
          -f "${sysroot_usr}/lib/libpython3.12.so" && \
          -f "${sysroot_usr}/lib/libsframe.a" ]]; then
        return
    fi

    sysdeps_tar="$(compgen -G "${OH_CUSTOM_SYSDEPS_TAR_GLOB}" | sort | tail -n 1 || true)"
    if [[ -z "${sysdeps_tar}" ]]; then
        log_error "Cannot find an OH sysdeps tarball matching:"
        log_error "  ${OH_CUSTOM_SYSDEPS_TAR_GLOB}"
        log_error "Set OH_CUSTOM_SYSDEPS_TAR_GLOB or pass --sysdeps-tar to point at ohos-*-sysdeps-*.tar.gz."
        exit 1
    fi

    log_info "Overlaying Python/sframe sysdeps from $(basename "${sysdeps_tar}") into the SDK sysroot..."
    rm -rf "${stage_dir}"
    mkdir -p "${stage_dir}" "${sysroot_usr}/include" "${sysroot_usr}/lib"
    tar -xzf "${sysdeps_tar}" -C "${stage_dir}" \
        out/include/python3.12 \
        out/include/sframe.h \
        out/include/sframe-api.h \
        out/lib/libpython3.12.so \
        out/lib/libpython3.12.so.1.0 \
        out/lib/libsframe.a \
        out/lib/libsframe.la
    cp -a "${stage_dir}/out/include/python3.12" "${sysroot_usr}/include/"
    cp -a "${stage_dir}/out/include/sframe.h" "${stage_dir}/out/include/sframe-api.h" "${sysroot_usr}/include/"
    cp -a "${stage_dir}/out/lib/libpython3.12.so" \
          "${stage_dir}/out/lib/libpython3.12.so.1.0" \
          "${stage_dir}/out/lib/libsframe.a" \
          "${stage_dir}/out/lib/libsframe.la" \
          "${sysroot_usr}/lib/"
    rm -rf "${stage_dir}"
}

prepare_root_layout() {
    ensure_dir "${OH_CUSTOM_ROOT}"
    ensure_dir "${OH_CUSTOM_TOOLCHAIN_ROOT}"
    ensure_repo_checkout "https://gitcode.com/openharmony-robot/ros_ros2_base.git" "${OH_CUSTOM_ROS2_BASE_REPO}"
    ensure_repo_checkout "https://gitcode.com/openharmony-robot/version.git" "${OH_CUSTOM_VERSION_REPO}"
    ensure_humble_install
    ensure_workspace_links
    ensure_toolchain_root
    ensure_sysdeps_overlay
}

ensure_builder_image() {
    if docker_cmd image inspect "${OH_CUSTOM_IMAGE}" >/dev/null 2>&1; then
        return
    fi

    if [[ "${PULL_IMAGE}" -eq 0 ]]; then
        log_error "Docker image not found locally: ${OH_CUSTOM_IMAGE}"
        exit 1
    fi

    log_info "Pulling builder image ${OH_CUSTOM_IMAGE}..."
    docker_cmd pull "${OH_CUSTOM_IMAGE}"
}

build_command_string() {
    local package_args
    local colcon_str=""
    local cmake_str=""

    package_args="$(IFS=,; echo "${PACKAGES[*]}")"

    if [[ "${#COLCON_ARGS[@]}" -gt 0 ]]; then
        # shellcheck disable=SC2206
        colcon_str=" --colcon-args ${COLCON_ARGS[*]}"
    fi
    if [[ "${#CMAKE_ARGS[@]}" -gt 0 ]]; then
        # shellcheck disable=SC2206
        cmake_str=" --cmk-args ${CMAKE_ARGS[*]}"
    fi

    cat <<EOF
set -euo pipefail
export OHOS_CPU=${OH_CUSTOM_CPU}
export OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
build-ros-humble --custom \
  --wd /mnt/ohos/tmp/ibrobot_oh_ws \
  --custom-prefix ${OH_CUSTOM_PREFIX} \
  --colcon-args --packages-select ${package_args//,/ }${colcon_str}${cmake_str}
EOF
}

run_builder() {
    local inner_cmd
    inner_cmd="$(build_command_string)"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "docker run --rm -it -e WS_ROOT=/mnt/ohos/tmp -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 --name ${OH_CUSTOM_CONTAINER_NAME} -v ${OH_CUSTOM_ROOT}:/mnt/ohos -v ${OH_CUSTOM_ROOT}:/mnt/ohos/tmp ${OH_CUSTOM_IMAGE} bash -lc '<build command>'"
        echo ""
        echo "${inner_cmd}"
        return
    fi

    docker_cmd run --rm -i \
        -e WS_ROOT=/mnt/ohos/tmp \
        -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 \
        --name "${OH_CUSTOM_CONTAINER_NAME}" \
        -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
        -v "${OH_CUSTOM_ROOT}:/mnt/ohos/tmp" \
        "${OH_CUSTOM_IMAGE}" \
        bash -lc "${inner_cmd}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --oh-root)
            shift
            OH_ROOT="$1"
            ;;
        --root)
            shift
            OH_CUSTOM_ROOT="$1"
            OH_CUSTOM_WS="${OH_CUSTOM_ROOT}/ibrobot_oh_ws"
            OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
            OH_CUSTOM_TOOLCHAIN_ROOT="${OH_CUSTOM_ROOT}/ohos-robot-toolchain"
            OH_CUSTOM_ROS2_BASE_REPO="${OH_CUSTOM_ROOT}/ros_ros2_base"
            OH_CUSTOM_VERSION_REPO="${OH_CUSTOM_ROOT}/version"
            ;;
        --workspace)
            shift
            OH_CUSTOM_WS="$1"
            OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
            ;;
        --toolchain-root)
            shift
            OH_CUSTOM_TOOLCHAIN_ROOT="$1"
            ;;
        --sdk-tar)
            shift
            OH_CUSTOM_SDK_TAR_GLOB="$1"
            ;;
        --sysdeps-tar)
            shift
            OH_CUSTOM_SYSDEPS_TAR_GLOB="$1"
            ;;
        --humble-tar)
            shift
            OH_CUSTOM_HUMBLE_TAR_GLOB="$1"
            ;;
        --custom-prefix)
            shift
            OH_CUSTOM_PREFIX="$1"
            ;;
        --cpu)
            shift
            OH_CUSTOM_CPU="$1"
            ;;
        --image)
            shift
            OH_CUSTOM_IMAGE="$1"
            ;;
        --container-name)
            shift
            OH_CUSTOM_CONTAINER_NAME="$1"
            ;;
        --packages)
            shift
            IFS=',' read -r -a PACKAGES <<<"$1"
            ;;
        --colcon-args)
            shift
            while [[ $# -gt 0 && "$1" != --cmk-args && "$1" != --oh-root && "$1" != --root && "$1" != --workspace && "$1" != --toolchain-root && "$1" != --sdk-tar && "$1" != --sysdeps-tar && "$1" != --humble-tar && "$1" != --custom-prefix && "$1" != --cpu && "$1" != --image && "$1" != --container-name && "$1" != --packages && "$1" != --sudo && "$1" != --no-pull && "$1" != --dry-run && "$1" != -h && "$1" != --help ]]; do
                COLCON_ARGS+=("$1")
                shift
            done
            continue
            ;;
        --cmk-args)
            shift
            while [[ $# -gt 0 && "$1" != --colcon-args && "$1" != --oh-root && "$1" != --root && "$1" != --workspace && "$1" != --toolchain-root && "$1" != --sdk-tar && "$1" != --sysdeps-tar && "$1" != --humble-tar && "$1" != --custom-prefix && "$1" != --cpu && "$1" != --image && "$1" != --container-name && "$1" != --packages && "$1" != --sudo && "$1" != --no-pull && "$1" != --dry-run && "$1" != -h && "$1" != --help ]]; do
                CMAKE_ARGS+=("$1")
                shift
            done
            continue
            ;;
        --sudo)
            USE_SUDO=1
            ;;
        --no-pull)
            PULL_IMAGE=0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

require_cmd git
require_cmd awk
require_cmd tar
require_cmd docker

apply_layout_defaults
normalize_paths
prepare_root_layout
ensure_builder_image
run_builder
normalize_runtime_bundle_ownership
postprocess_runtime_bundle
