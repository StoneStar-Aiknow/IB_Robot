#!/usr/bin/env bash
# Validate and patch the in-tree ROS LiDAR submodules.
# Compilation belongs to scripts/build.sh and must not be performed here.

_require_ros_submodule_commit() {
    local source_dir="$1"
    local expected_commit="$2"
    local label="$3"

    if [[ ! -e "${source_dir}/.git" ]]; then
        log_error "Missing ${label} submodule: ${source_dir}"
        return 1
    fi

    local actual_commit
    actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${expected_commit}" ]]; then
        log_error "${label} commit mismatch: expected ${expected_commit}, got ${actual_commit}"
        return 1
    fi
}

_apply_ros_patch_once() {
    local source_dir="$1"
    local patch_file="$2"

    if git -C "${source_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        return 0
    fi
    if ! git -C "${source_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
        log_error "Patch does not match pinned source: ${patch_file}"
        return 1
    fi
    log_info "Applying $(basename "${patch_file}")"
    git -C "${source_dir}" apply "${patch_file}"
}

_migrate_fast_lio_build_patch() {
    local source_dir="$1"
    local patch_file="$2"
    local cmake_file="${source_dir}/CMakeLists.txt"

    if git -C "${source_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1 || \
        git -C "${source_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
        return 0
    fi

    local cmake_blob
    cmake_blob="$(git -C "${source_dir}" hash-object "${cmake_file}")"
    case "${cmake_blob}" in
        ce8b1233b110e0893f9c1d1b461a7a7a425415dd|653e7261996c1abcbc184d4def33771842cbeaa7|baa6ded8206965d89b48bb2ece8c3a2ad97fd06d)
            git -C "${source_dir}" checkout HEAD -- CMakeLists.txt
            ;;
        *)
            return 0
            ;;
    esac
}

_ensure_fast_calib_patch_stack() {
    local source_dir="$1"
    local patch_dir="$2"
    local expected_diff_sha256="e5e807e08fa82fefc257ad5c01ca894143d06a210bb9fb83fe65c6956ad71331"
    local actual_diff_sha256

    actual_diff_sha256="$(git -C "${source_dir}" diff --cached --binary | sha256sum | cut -d' ' -f1)"
    if [[ "${actual_diff_sha256}" == "${expected_diff_sha256}" ]]; then
        if ! git -C "${source_dir}" diff --quiet || \
            [[ -n "$(git -C "${source_dir}" ls-files --others --exclude-standard)" ]]; then
            log_error "FAST-Calib has changes outside the verified patch stack"
            return 1
        fi
        return 0
    fi
    if [[ -n "$(git -C "${source_dir}" status --porcelain)" ]]; then
        log_error "FAST-Calib has a partial or unexpected local patch state"
        log_error "Expected complete diff SHA-256: ${expected_diff_sha256}"
        return 1
    fi

    local patch
    while IFS= read -r patch; do
        [[ -n "${patch}" ]] || continue
        log_info "Applying ${patch}"
        if ! git -C "${source_dir}" apply --index "${patch_dir}/${patch}"; then
            log_error "FAST-Calib patch does not match the pinned source: ${patch}"
            return 1
        fi
    done < "${patch_dir}/series.txt"

    actual_diff_sha256="$(git -C "${source_dir}" diff --cached --binary | sha256sum | cut -d' ' -f1)"
    if [[ "${actual_diff_sha256}" != "${expected_diff_sha256}" ]]; then
        log_error "FAST-Calib patch-stack provenance mismatch"
        log_error "Expected ${expected_diff_sha256}, got ${actual_diff_sha256}"
        return 1
    fi
}

ensure_ros_third_party_patch_stacks() {
    local livox_sdk_src="${WORKSPACE}/libs/Livox-SDK2"
    local livox_driver_src="${WORKSPACE}/src/livox_ros_driver2"
    local fast_lio_src="${WORKSPACE}/src/fast_lio"
    local fast_calib_src="${WORKSPACE}/src/fast_calib"
    local ikd_tree_src="${fast_lio_src}/include/ikd-Tree"
    local fast_lio_patch_dir="${WORKSPACE}/third_party/patches/fast_lio"
    local fast_calib_patch_dir="${WORKSPACE}/third_party/patches/fast_calib/7747dfc"
    local livox_sdk_patch_dir="${WORKSPACE}/third_party/patches/livox_sdk2"
    local livox_driver_patch_dir="${WORKSPACE}/third_party/patches/livox_ros_driver2"

    _require_ros_submodule_commit \
        "${livox_sdk_src}" "f5d9375f84efe2b15bc0a052d3e18482ed13adf4" "Livox-SDK2" || return 1
    _require_ros_submodule_commit \
        "${livox_driver_src}" "13eb05e4e6dd7a765b934d0c5fd6236676a57b49" "livox_ros_driver2" || return 1
    _require_ros_submodule_commit \
        "${fast_lio_src}" "a4743b095409588842a5b30ddfa27e29d2f99164" "FAST-LIO" || return 1
    _require_ros_submodule_commit \
        "${fast_calib_src}" "7747dfc6109c04b4bf81d2e3661e41626c8392e1" "FAST-Calib" || return 1
    _require_ros_submodule_commit \
        "${ikd_tree_src}" "e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4" "ikd-Tree" || return 1

    _apply_ros_patch_once \
        "${livox_sdk_src}" \
        "${livox_sdk_patch_dir}/f5d9375-warning-cleanup.patch" || return 1
    _apply_ros_patch_once \
        "${livox_driver_src}" \
        "${livox_driver_patch_dir}/13eb05e-ros2-workspace-sdk.patch" || return 1
    _apply_ros_patch_once \
        "${livox_driver_src}" \
        "${livox_driver_patch_dir}/13eb05e-quiet-cmake.patch" || return 1
    _migrate_fast_lio_build_patch \
        "${fast_lio_src}" "${fast_lio_patch_dir}/a4743b-build.patch" || return 1
    _apply_ros_patch_once \
        "${fast_lio_src}" "${fast_lio_patch_dir}/a4743b-build.patch" || return 1
    _apply_ros_patch_once \
        "${fast_lio_src}" "${fast_lio_patch_dir}/a4743b-runtime.patch" || return 1
    _ensure_fast_calib_patch_stack "${fast_calib_src}" "${fast_calib_patch_dir}" || return 1
    log_done "ROS LiDAR third-party sources are ready"
}
