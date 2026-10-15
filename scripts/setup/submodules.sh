#!/bin/bash
# submodules.sh - Submodule initialization and update routines

submodule_is_initialized() {
    local path="$1"
    [[ -e "${path}/.git" ]] || return 1
    git -C "${path}" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

submodule_update_block_reason() {
    local path="$1"
    
    if ! submodule_is_initialized "${path}"; then
        echo "not initialized"
        return 0
    fi

    if [[ -n "$(git -C "${path}" status --porcelain)" ]]; then
        echo "has uncommitted changes"
        return 0
    fi

    local head_commit
    head_commit="$(git -C "${path}" rev-parse HEAD 2>/dev/null || echo "")"
    local expected_commit
    expected_commit="$(git ls-tree HEAD "${path}" | awk '{print $3}')"

    if [[ "${head_commit}" != "${expected_commit}" ]]; then
        echo "currently checked out to ${head_commit:0:7}, expected ${expected_commit:0:7}"
        return 0
    fi

    return 1
}

update_submodules() {
    log_info "Synchronizing submodule URLs..."
    git submodule sync --recursive >/dev/null

    local submodules=()
    local sm_path
    while IFS= read -r sm_path; do
        [[ -n "${sm_path}" ]] && submodules+=("${sm_path}")
    done <<< "$(git config --file .gitmodules --get-regexp path | awk '{ print $2 }')"

    if [[ "${SETUP_PROFILE:-full}" == "inference" ]]; then
        # Inference-only hosts (edge-cloud collaborative inference
        # verification) need just the lerobot policy sources; bringup, teleop,
        # and LiDAR submodules stay uninitialized to keep the verification
        # environment lean and avoid unrelated patch-stack requirements.
        log_info "Inference profile: initializing only the libs/lerobot submodule."
        submodules=("libs/lerobot")
    fi

    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" != true ]]; then
        # RoboTwin is opt-in benchmark tooling, gated by --with-benchmark like
        # its runtime wheels: its 4.6 GB clone and its nested GitHub-hosted
        # XPolicyLab submodule must not gate the default setup path.
        local default_path_submodules=()
        for sm in "${submodules[@]}"; do
            [[ "${sm}" == "libs/robotwin" ]] && continue
            default_path_submodules+=("${sm}")
        done
        submodules=("${default_path_submodules[@]}")
    fi

    local to_update=()
    local skipped=()

    for sm in "${submodules[@]}"; do
        local reason
        if reason="$(submodule_update_block_reason "${sm}")"; then
            if [[ "${reason}" == "not initialized" ]]; then
                to_update+=("${sm}")
            else
                skipped+=("${sm} (${reason})")
            fi
        else
            to_update+=("${sm}")
        fi
    done

    if [[ ${#skipped[@]} -gt 0 ]]; then
        log_warn "The following submodules will NOT be updated automatically:"
        for sm in "${skipped[@]}"; do
            echo "  - ${sm}"
        done
        echo ""
        if ! ask_yn "Continue anyway?" "y"; then
            log_error "Submodule update aborted by user."
            exit 1
        fi
    fi

    if [[ ${#to_update[@]} -gt 0 ]]; then
        log_info "Updating submodules..."
        for sm in "${to_update[@]}"; do
            log_info "  -> ${sm}"
            if [[ "${sm}" == "libs/robotwin" ]]; then
                # XPolicyLab is RoboTwin's optional policy-serving stack on
                # GitHub; the benchmark adapter drives Base_Task directly and
                # never imports it, so the opt-in sync stays non-recursive.
                run_cmd git submodule update --init "${sm}"
            else
                run_cmd git submodule update --init --recursive "${sm}"
            fi
        done
        log_done "Submodules updated"
    else
        log_info "All submodules are up to date."
        log_done "Submodules checked"
    fi
}
