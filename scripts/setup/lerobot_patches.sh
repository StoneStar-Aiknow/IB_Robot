#!/bin/bash

lerobot_patch_dir() {
    echo "${WORKSPACE}/third_party/patches/lerobot/v0.5.1"
}

lerobot_patch_base_commit() {
    local manifest
    manifest="$(lerobot_patch_dir)/manifest.yaml"
    grep -E '^  commit:' "${manifest}" | head -n1 | awk '{print $2}'
}

lerobot_apply_patch_series() {
    local submodule_dir="$1"
    local patch_dir="$2"
    local series_file="$3"

    while IFS= read -r patch_file; do
        [[ -z "${patch_file}" ]] && continue
        log_info "Applying ${patch_file}..."
        git -C "${submodule_dir}" am "${patch_dir}/${patch_file}" >/dev/null
    done < "${series_file}"
}

lerobot_rebuild_patch_branch() {
    local submodule_dir="$1"
    local patch_dir="$2"
    local series_file="$3"
    local base_commit="$4"
    local branch_name="$5"

    log_warn "Rebuilding ${branch_name} to match the in-repo patch stack."
    git -C "${submodule_dir}" checkout --detach "${base_commit}" >/dev/null
    git -C "${submodule_dir}" branch -D "${branch_name}" >/dev/null
    git -C "${submodule_dir}" checkout -b "${branch_name}" >/dev/null
    lerobot_apply_patch_series "${submodule_dir}" "${patch_dir}" "${series_file}"
    log_done "LeRobot patch stack rebuilt"
}

ensure_lerobot_patch_stack_applied() {
    local submodule_dir="${WORKSPACE}/libs/lerobot"
    local patch_dir
    local series_file
    local base_commit
    local branch_name="ibrobot/lerobot-v0.5.1-patched"
    local expected_patch_count
    local applied_patch_count

    [[ ! -d "${submodule_dir}" ]] && return 0
    [[ ! -d "${submodule_dir}/.git" && ! -f "${submodule_dir}/.git" ]] && return 0

    patch_dir="$(lerobot_patch_dir)"
    series_file="${patch_dir}/series.txt"
    base_commit="$(lerobot_patch_base_commit)"

    if [[ ! -f "${series_file}" || -z "${base_commit}" ]]; then
        log_warn "LeRobot patch stack metadata is missing. Skipping automatic patch application."
        return 0
    fi

    expected_patch_count="$(grep -cv '^[[:space:]]*$' "${series_file}")"
    log_info "Checking IB_Robot lerobot patch stack (${expected_patch_count} patches)..."

    if git -C "${submodule_dir}" show-ref --verify --quiet "refs/heads/${branch_name}"; then
        if [[ "$(git -C "${submodule_dir}" branch --show-current)" != "${branch_name}" ]]; then
            log_info "Switching libs/lerobot to existing patched branch ${branch_name}..."
            git -C "${submodule_dir}" checkout "${branch_name}" >/dev/null
        fi

        applied_patch_count="$(git -C "${submodule_dir}" rev-list --count "${base_commit}..HEAD")"

        if [[ "${applied_patch_count}" -eq "${expected_patch_count}" ]]; then
            log_done "LeRobot patch stack already applied"
            return 0
        fi

        if ! git -C "${submodule_dir}" diff --quiet || ! git -C "${submodule_dir}" diff --cached --quiet; then
            log_error "libs/lerobot has local changes; refusing to update the IB_Robot patch stack automatically."
            exit 1
        fi

        if [[ "${applied_patch_count}" -gt "${expected_patch_count}" ]]; then
            log_warn "Existing patched branch contains ${applied_patch_count} commits after ${base_commit}, expected ${expected_patch_count}."
            lerobot_rebuild_patch_branch "${submodule_dir}" "${patch_dir}" "${series_file}" "${base_commit}" "${branch_name}"
            return 0
        fi

        log_info "Applying ${expected_patch_count}-${applied_patch_count} new LeRobot compatibility patch(es) on ${branch_name}..."
        tail -n +"$((applied_patch_count + 1))" "${series_file}" | while IFS= read -r patch_file; do
            [[ -z "${patch_file}" ]] && continue
            log_info "Applying ${patch_file}..."
            git -C "${submodule_dir}" am "${patch_dir}/${patch_file}" >/dev/null
        done
        log_done "LeRobot patch stack updated"
        return 0
    fi

    if [[ "$(git -C "${submodule_dir}" rev-parse HEAD)" != "${base_commit}" ]]; then
        log_warn "libs/lerobot is not at the expected upstream base commit ${base_commit}; skipping automatic patch application."
        return 0
    fi

    if ! git -C "${submodule_dir}" diff --quiet || ! git -C "${submodule_dir}" diff --cached --quiet; then
        log_error "libs/lerobot has local changes; refusing to apply the IB_Robot patch stack automatically."
        exit 1
    fi

    log_info "Applying IB_Robot lerobot patch stack on top of upstream v0.5.1..."
    git -C "${submodule_dir}" checkout -b "${branch_name}" >/dev/null
    lerobot_apply_patch_series "${submodule_dir}" "${patch_dir}" "${series_file}"
    log_done "LeRobot patch stack applied"
}
