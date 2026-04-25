#!/bin/bash

# ----------------------------------------------------------------------------
# LeRobot patch stack — platform-aware applier
# ----------------------------------------------------------------------------
# This module resolves the active lerobot patch tag through
# third_party/patches/lerobot/INDEX.yaml (single source of truth for
# multi-tag support) and applies only the subset of patches that match
# the current host's Python version and active profile list.
# Filtering is delegated to scripts/setup/lerobot_filter_series.py
# and tag resolution to scripts/setup/lerobot_resolve_active.py; both run
# under ${VENV_PYTHON} so PyYAML is guaranteed available.
#
# Tag binding contract (fail-closed):
#   * INDEX.yaml.active_tag MUST match <tag-dir>/manifest.yaml.lerobot_tag.
#   * INDEX entry's upstream_commit MUST equal manifest.lerobot_commit_range.min.
#   * libs/lerobot HEAD prior to patch application MUST fall inside
#     manifest.lerobot_commit_range. Mismatches abort setup with a hint to
#     update INDEX.yaml or add a new tag directory.
#
# Escape hatches (env vars):
#   IBR_LEROBOT_FORCE_UNFILTERED=1   bypass the platform filter and apply the
#                                    raw series.txt verbatim. Tag resolution
#                                    via INDEX.yaml is still enforced.
#   IBR_LEROBOT_FORCE_REBUILD=1      reset a dirty libs/lerobot worktree
#                                    before rebuilding the patched branch
#                                    (default: refuse and exit 1).
#
# The host facts IBR_HOST_PYTHON_VERSION / IBR_LEROBOT_PROFILES
# are populated by scripts/setup/detect.sh::export_lerobot_host_facts.
# ----------------------------------------------------------------------------

_lerobot_index_path() {
    echo "${WORKSPACE}/third_party/patches/lerobot/INDEX.yaml"
}

# Pick the most appropriate Python interpreter to drive the helpers.
# The helpers only need PyYAML, which the venv always provides.
# Bootstrap python on stripped openEuler/OpenHarmony may lack PyYAML;
# in that case the user must either install it or set
# IBR_LEROBOT_FORCE_UNFILTERED=1 (note: this still requires INDEX.yaml
# resolution to succeed).
_lerobot_filter_python() {
    if [[ -n "${VENV_PYTHON:-}" && -x "${VENV_PYTHON}" ]]; then
        echo "${VENV_PYTHON}"
        return 0
    fi
    if [[ -n "${SETUP_BOOTSTRAP_PYTHON_BIN:-}" && -x "${SETUP_BOOTSTRAP_PYTHON_BIN}" ]]; then
        echo "${SETUP_BOOTSTRAP_PYTHON_BIN}"
        return 0
    fi
    command -v python3 || true
}

# Resolve INDEX.yaml's active tag into shell variables. On success exports:
#   LEROBOT_TAG, LEROBOT_DIR, LEROBOT_BASE_COMMIT, LEROBOT_BASE_COMMIT_MIN,
#   LEROBOT_BASE_COMMIT_MAX, LEROBOT_BRANCH_NAME, LEROBOT_MANIFEST,
#   LEROBOT_SERIES.
# Returns 1 on any resolver failure (mismatch, archived tag, missing dir).
_lerobot_resolve_active() {
    local resolver="${WORKSPACE}/scripts/setup/lerobot_resolve_active.py"
    local index_file
    index_file="$(_lerobot_index_path)"

    if [[ ! -f "${index_file}" ]]; then
        log_error "LeRobot patch INDEX.yaml not found at ${index_file}"
        return 1
    fi
    if [[ ! -f "${resolver}" ]]; then
        log_error "LeRobot tag resolver not found: ${resolver}"
        return 1
    fi

    local py_bin
    py_bin="$(_lerobot_filter_python)"
    if [[ -z "${py_bin}" ]]; then
        log_error "No usable Python interpreter to run lerobot_resolve_active.py."
        return 1
    fi

    local resolved
    if ! resolved="$("${py_bin}" "${resolver}" --index "${index_file}" 2>&1 1>/dev/stdout)"; then
        log_error "lerobot_resolve_active.py failed:"
        sed 's/^/    /' <<<"${resolved}" >&2
        return 1
    fi

    # The resolver prints `KEY=value` lines on stdout. We deliberately do
    # NOT eval the output (defence in depth — the values come from a YAML
    # file checked into the repo, but we still parse strictly).
    local line key value
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        case "${key}" in
            LEROBOT_TAG|LEROBOT_DIR|LEROBOT_BASE_COMMIT|LEROBOT_BASE_COMMIT_MIN|\
            LEROBOT_BASE_COMMIT_MAX|LEROBOT_BRANCH_NAME|LEROBOT_MANIFEST|LEROBOT_SERIES)
                printf -v "${key}" '%s' "${value}"
                export "${key?}"
                ;;
            *)
                log_warn "lerobot_resolve_active emitted unknown key '${key}'; ignoring."
                ;;
        esac
    done <<<"${resolved}"
    return 0
}

# Compute the filtered patch series and write one filename per line to
# the path given by $2. Audit output (KEEP/SKIP reasons) is mirrored to
# the user via log_info. Returns 0 on success, 1 on filter failure
# (caller should abort unless IBR_LEROBOT_FORCE_UNFILTERED=1).
lerobot_compute_filtered_series() {
    local manifest="$1"
    local raw_series="$2"
    local out_file="$3"
    local head_commit="${4:-}"

    local filter_script="${WORKSPACE}/scripts/setup/lerobot_filter_series.py"

    if [[ "${IBR_LEROBOT_FORCE_UNFILTERED:-0}" == "1" ]]; then
        log_warn "IBR_LEROBOT_FORCE_UNFILTERED=1 set; bypassing platform filter and applying full series.txt verbatim."
        # Drop blank lines so downstream count is consistent with filtered path.
        grep -v '^[[:space:]]*$' "${raw_series}" > "${out_file}"
        return 0
    fi

    if [[ ! -f "${filter_script}" ]]; then
        log_error "Filter helper not found: ${filter_script}"
        return 1
    fi

    local py_bin
    py_bin="$(_lerobot_filter_python)"
    if [[ -z "${py_bin}" ]]; then
        log_error "No usable Python interpreter to run lerobot_filter_series.py."
        log_error "Hint: ensure setup_python_venv ran before ensure_lerobot_patch_stack_applied,"
        log_error "      or set IBR_LEROBOT_FORCE_UNFILTERED=1 to skip filtering."
        return 1
    fi

    local audit_file="${out_file}.audit"
    if ! "${py_bin}" "${filter_script}" \
            --manifest "${manifest}" \
            --series "${raw_series}" \
            --lerobot-head-commit "${head_commit}" \
            >"${out_file}" 2>"${audit_file}"; then
        local rc=$?
        log_error "lerobot_filter_series.py failed with exit ${rc}."
        if [[ -s "${audit_file}" ]]; then
            log_error "Filter stderr:"
            sed 's/^/    /' "${audit_file}" >&2
        fi
        log_error "Set IBR_LEROBOT_FORCE_UNFILTERED=1 to bypass the platform filter (tag binding still enforced)."
        rm -f "${out_file}" "${audit_file}"
        return 1
    fi

    # Surface KEEP/SKIP audit lines to the user (informational).
    if [[ -s "${audit_file}" ]]; then
        while IFS= read -r line; do
            [[ -z "${line}" ]] && continue
            log_info "  ${line}"
        done < "${audit_file}"
    fi
    rm -f "${audit_file}"
    return 0
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

# Reset libs/lerobot to a clean state at base_commit. Used by the
# IBR_LEROBOT_FORCE_REBUILD escape hatch when a dirty worktree would
# otherwise abort the rebuild.
_lerobot_reset_dirty_worktree() {
    local submodule_dir="$1"
    local base_commit="$2"

    log_warn "IBR_LEROBOT_FORCE_REBUILD=1: discarding local changes in libs/lerobot."
    git -C "${submodule_dir}" reset --hard >/dev/null
    git -C "${submodule_dir}" clean -fdx >/dev/null
    git -C "${submodule_dir}" checkout --detach "${base_commit}" >/dev/null
}

ensure_lerobot_patch_stack_applied() {
    local submodule_dir="${WORKSPACE}/libs/lerobot"
    local expected_patch_count
    local applied_patch_count

    [[ ! -d "${submodule_dir}" ]] && return 0
    [[ ! -d "${submodule_dir}/.git" && ! -f "${submodule_dir}/.git" ]] && return 0

    # Resolve the active tag's directory + base commit + branch name from
    # INDEX.yaml. On failure (mismatch, archived tag, missing manifest)
    # we abort: tag binding is part of the contract.
    if ! _lerobot_resolve_active; then
        log_error "Cannot resolve active lerobot patch tag; aborting setup."
        exit 1
    fi
    local patch_dir="${LEROBOT_DIR}"
    local base_commit="${LEROBOT_BASE_COMMIT}"
    local branch_name="${LEROBOT_BRANCH_NAME}"
    local manifest_file="${LEROBOT_MANIFEST}"
    local raw_series="${LEROBOT_SERIES}"

    if [[ ! -f "${raw_series}" || -z "${base_commit}" ]]; then
        log_warn "LeRobot patch stack metadata is missing for tag ${LEROBOT_TAG}. Skipping automatic patch application."
        return 0
    fi

    # Capture the current libs/lerobot HEAD so the filter can validate it
    # against manifest.lerobot_commit_range. Empty string when rev-parse
    # fails (e.g. detached without HEAD); the filter treats empty as
    # "skip the predicate" rather than fail-closed.
    local head_commit=""
    head_commit="$(git -C "${submodule_dir}" rev-parse HEAD 2>/dev/null || true)"

    # Compute the platform-filtered series into a temp file. All downstream
    # logic (count comparisons, am, rebuild) consumes this filtered file
    # rather than the raw series.txt, so per-platform skips are reflected
    # in the "applied vs expected" count.
    local series_file
    series_file="$(mktemp -t lerobot-series-XXXXXX.txt)"
    # shellcheck disable=SC2064
    trap "rm -f '${series_file}' '${series_file}.audit'" RETURN

    if ! lerobot_compute_filtered_series "${manifest_file}" "${raw_series}" "${series_file}" "${head_commit}"; then
        log_error "Cannot compute filtered lerobot patch series; aborting setup."
        exit 1
    fi

    expected_patch_count="$(grep -cv '^[[:space:]]*$' "${series_file}")"
    log_info "Checking IB_Robot lerobot patch stack for tag ${LEROBOT_TAG} (${expected_patch_count} patches after platform filter; profiles=${IBR_LEROBOT_PROFILES:-unknown})..."

    if [[ "${expected_patch_count}" -eq 0 ]]; then
        log_done "No lerobot patches apply to this platform; nothing to do."
        return 0
    fi

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

        local has_dirty=0
        if ! git -C "${submodule_dir}" diff --quiet || ! git -C "${submodule_dir}" diff --cached --quiet; then
            has_dirty=1
        fi

        if [[ "${has_dirty}" -eq 1 ]]; then
            if [[ "${IBR_LEROBOT_FORCE_REBUILD:-0}" == "1" ]]; then
                _lerobot_reset_dirty_worktree "${submodule_dir}" "${base_commit}"
            else
                log_error "libs/lerobot has local changes; refusing to update the IB_Robot patch stack automatically."
                log_error "Hint: commit/stash the changes, or set IBR_LEROBOT_FORCE_REBUILD=1 to discard them."
                exit 1
            fi
        fi

        if [[ "${applied_patch_count}" -gt "${expected_patch_count}" ]] || [[ "${IBR_LEROBOT_FORCE_REBUILD:-0}" == "1" ]]; then
            log_warn "Existing patched branch contains ${applied_patch_count} commits after ${base_commit}, expected ${expected_patch_count}."
            lerobot_rebuild_patch_branch "${submodule_dir}" "${patch_dir}" "${series_file}" "${base_commit}" "${branch_name}"
            return 0
        fi

        log_info "Applying $((expected_patch_count - applied_patch_count)) new LeRobot compatibility patch(es) on ${branch_name}..."
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
        if [[ "${IBR_LEROBOT_FORCE_REBUILD:-0}" == "1" ]]; then
            _lerobot_reset_dirty_worktree "${submodule_dir}" "${base_commit}"
        else
            log_error "libs/lerobot has local changes; refusing to apply the IB_Robot patch stack automatically."
            log_error "Hint: commit/stash the changes, or set IBR_LEROBOT_FORCE_REBUILD=1 to discard them."
            exit 1
        fi
    fi

    log_info "Applying IB_Robot lerobot patch stack on top of upstream ${LEROBOT_TAG}..."
    git -C "${submodule_dir}" checkout -b "${branch_name}" >/dev/null
    lerobot_apply_patch_series "${submodule_dir}" "${patch_dir}" "${series_file}"
    log_done "LeRobot patch stack applied"
}
