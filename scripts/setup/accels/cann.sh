#!/bin/bash

# Ascend CANN accelerator profile. Sourced by detect.sh::load_accel_impl when
# SETUP_ACCEL_ID is "cann". Every hook below is part of the accelerator
# contract consumed by python_venv.sh; keep the three profiles in sync.

# Whether the Torch GraspGen stack (grasp_gen plus the pointnet2_ops CUDA
# extension) can be installed on this accelerator.
accel_supports_torch_grasp() {
    # pointnet2_ops is a CUDA extension with no Ascend build. The Ascend grasp
    # path is the OM runtime in perception_service, which does not go through
    # Torch GraspGen.
    return 1
}

# Whether GraspGenX can be installed on this accelerator. It is a separate
# question from accel_supports_torch_grasp: GraspGenX is a pure-Python wheel
# whose encoder is PTV3, so it never touches pointnet2_ops, and its 310P
# deployment runs on torch_npu with the diffusion loop as an offline model.
# Answering it with the GraspGen predicate is what kept GraspGenX off an
# Ascend host for a reason that does not apply to it.
accel_supports_torch_graspgenx() {
    return 0
}

# CANN releases with a supported Torch ABI in this tree, keyed off the
# detected release. Every hook below keys off this one selector so a release
# cannot be half adopted. The family tests are anchored so 8.1 does not match
# 8.10 and 8.3 does not match 8.30.
#
# 8.1's triple is hardware-validated on Ascend310P1. 8.3's comes from Ascend's
# compatibility table with aarch64 wheel availability verified (cp310/cp311);
# it has not run on hardware yet — see accel-cann-8.3.txt.
_cann_supported_abi_release() {
    local cann_version
    cann_version="$(detect_cann_version 2>/dev/null || true)"
    # Anchored family tests: 8.1|8.1.* matches 8.1 and 8.1.RC1 but not 8.10,
    # just as 8.3|8.3.* excludes 8.30.
    case "${cann_version}" in
        8.1|8.1.*) printf '8.1\n' ;;
        8.3|8.3.*) printf '8.3\n' ;;
        *) return 1 ;;
    esac
}

# torch_npu package versions Ascend documents for a CANN release, from the
# upstream compatibility table (Ascend/pytorch COMPATIBILITY.en.md). Used only
# to tell an operator what an unsupported release would need; the supported
# releases above each carry their own verified triple, constraints file and
# end-of-setup check in this tree.
#
# Each torch_npu release requires one exact torch version, so a single pin can
# never fit several CANN releases. Note that LeRobot declares torch>=2.7,<2.12:
# 8.3.RC1 and newer have a documented torch_npu inside that range, while 8.1
# and 8.2 top out at torch 2.6.0, below LeRobot's floor — which is why 8.1
# needs the documented exception in accel-cann-8.1.txt while 8.3's 2.8.0 does
# not.
_cann_vendor_torch_npu() {
    case "${1}" in
        8.1*) printf '%s\n' "2.5.1, 2.4.0.post4, 2.3.1.post6, 2.1.0.post12" ;;
        8.2*) printf '%s\n' "2.6.0, 2.5.1.post1, 2.1.0.post13" ;;
        8.3*) printf '%s\n' "2.8.0, 2.7.1, 2.6.0.post3, 2.1.0.post17" ;;
        8.5*) printf '%s\n' "2.9.0, 2.8.0.post2, 2.7.1.post2, 2.6.0.post5" ;;
        9.0*) printf '%s\n' "2.10.0, 2.9.0.post2, 2.8.0.post4, 2.7.1.post4" ;;
        9.1*) printf '%s\n' "2.12.0, 2.11.0, 2.10.0.post4, 2.9.0.post6, 2.7.1.post8" ;;
        *) printf '%s\n' "" ;;
    esac
}

# Workspace pip constraint file that protects this accelerator's Torch ABI
# during every dependency install, or no output when none is needed.
accel_pip_constraints() {
    local release
    release="$(_cann_supported_abi_release)" || return 0
    printf '%s\n' "${WORKSPACE}/requirements/constraints-cann-${release}.txt"
}

# Workspace requirements file holding this accelerator's own Python
# dependencies, or no output when it has none. Installed on every OS: an
# Ascend NPU needs torch_npu whether the host runs openEuler or Ubuntu.
#
# A detected release without a supported ABI here also yields no output, but
# returns 0: installing some other release's torch_npu would be worse than
# installing none (it resolves, setup reports success, and the mismatch only
# surfaces at runtime), yet aborting would strand an already half-installed
# workspace. Setup continues on LeRobot's plain Torch and the caller records
# the pending state for the end-of-setup manual-configuration notice.
accel_requirements() {
    local release
    if release="$(_cann_supported_abi_release)"; then
        printf '%s\n' "${WORKSPACE}/requirements/accel-cann-${release}.txt"
        return 0
    fi

    local cann_version vendor
    cann_version="$(detect_cann_version 2>/dev/null || true)"

    if [[ -z "${cann_version}" ]]; then
        # Nothing to key off: the CANN release is what selects torch_npu, and
        # without a toolkit it cannot be known. Install no NPU backend rather
        # than guess one. This is the state inside the openEuler verification
        # container, which has no toolkit.
        log_warn "No CANN toolkit detected; installing no torch_npu." >&2
        log_warn "An Ascend host needs a CANN release whose supported Torch ABI exists here (currently 8.1 and 8.3)." >&2
        return 0
    fi

    vendor="$(_cann_vendor_torch_npu "${cann_version}")"
    log_warn "CANN ${cann_version} has no supported Torch ABI in this tree; only 8.1 and 8.3 do." >&2
    if [[ -n "${vendor}" ]]; then
        log_warn "Ascend documents torch_npu ${vendor} for this release (each requires that exact torch)." >&2
    else
        log_warn "This release is absent from the recorded Ascend compatibility table." >&2
    fi
    log_warn "Installing no NPU backend; LeRobot's Torch stays in place and setup continues." >&2
    log_warn "To configure the NPU runtime by hand, check the CANN/Torch/torch_npu/torchvision pairing at" >&2
    log_warn "https://www.hiascend.com/developer/download/compatibility, then install the matched versions" >&2
    log_warn "with the workspace venv (not the system pip): venv/bin/python -m pip install torch==<v> torch_npu==<v> torchvision==<v>." >&2
    return 0
}

# Runs right after accel_requirements, with the pip runner as "$@". Use it to
# remove packages LeRobot's own resolution installs that the accelerator's
# Torch ABI cannot support.
accel_post_requirements() {
    local pip_runner=("$@")

    _cann_supported_abi_release >/dev/null || return 0

    # torchcodec is in LeRobot's base dependencies on aarch64 (>=0.11.0), but
    # its metadata declares no torch requirement at all, so pip happily leaves
    # it in place after accel-cann-<release>.txt pins an older torch — and
    # then its compiled extension cannot load. Nothing in the resolver catches
    # this; `pip check` stays quiet. Remove it so LeRobot falls back to the
    # PyAV decoder that the accel file installs.
    log_info "Removing torchcodec: its aarch64 wheel is ABI-bound to a Torch newer than the supported CANN set."
    # Swap only the trailing `install` verb. "${pip_runner[@]/install/uninstall}"
    # would rewrite the first `install` substring of EVERY element, mangling the
    # interpreter path on any workspace whose path contains "install" — a colcon
    # install tree, /opt/install/..., and so on.
    local -a uninstall_cmd
    if [[ "${pip_runner[-1]}" != "install" ]]; then
        log_error "accel_post_requirements expects a pip runner ending in 'install', got: ${pip_runner[*]}"
        return 1
    fi
    uninstall_cmd=("${pip_runner[@]:0:$((${#pip_runner[@]} - 1))}" uninstall)
    run_cmd "${uninstall_cmd[@]}" -y torchcodec --quiet
}

# Post-install validation of the accelerator runtime. $1 is the venv Python.
accel_verify_runtime() {
    verify_cann_runtime_versions "$1"
}
