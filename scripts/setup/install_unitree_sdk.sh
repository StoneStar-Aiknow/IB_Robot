#!/bin/bash
# install_unitree_sdk.sh - Unitree G1 SDK (unitree_sdk2_python) installer.
#
# unitree_sdk2py is NOT published on PyPI; the vendor's own README installs it
# from a git checkout with `pip install -e .`. This helper reproduces that flow
# reproducibly for the IB_Robot workspace venv:
#
#   1. gate on architecture (initial scope: x86_64 only, see below),
#   2. acquire the upstream source at an immutable pinned revision into a
#      per-host cache outside the ROS tree,
#   3. install its pinned CycloneDDS Python dependency, then the SDK itself
#      with --no-deps (its full dependency closure is managed here),
#   4. verify the install by importing the modules the G1 adapter uses and
#      constructing the IDL types it builds -- never by touching DDS or a robot.
#
# Editable install is REQUIRED, not a style choice: upstream setup.py declares
# no package_data, so a regular (non-editable) wheel build drops the bundled
# unitree_sdk2py/utils/lib/crc_*.so native libraries and the low-level CRC
# utility would only fail at first use. The editable install keeps the vendored
# resources reachable through the source tree.
#
# Initial scope (AtomGit Issue #171): cyclonedds==0.10.2 -- the exact version
# upstream pins -- publishes CPython Linux wheels for x86_64 only, so this
# helper installs on x86_64 and reports every other architecture as explicitly
# unsupported and skipped. The G1 runtime fails closed when the SDK is absent;
# a skipped install must never be reported as a usable vendor tier.

# Upstream source and revision. The revision is the immutable pin; bump it
# deliberately (re-run verification on the robot afterwards) and record it in
# src/robots/unitree/unitree_g1_adapter/README.md.
# Default to the gitcode mirror of the GitHub origin -- the same convention as
# feetech_sdk's pinned FTServo fetch -- because the mirror is reachable from
# the environments this repo builds on while github.com is not guaranteed to
# be. The mirror HEAD must carry the pinned revision; override with the GitHub
# URL when fetching straight from upstream.
UNITREE_SDK2_PYTHON_REPO="${UNITREE_SDK2_PYTHON_REPO:-https://gitcode.com/unitreerobotics/unitree_sdk2_python.git}"
UNITREE_SDK2_PYTHON_REVISION="${UNITREE_SDK2_PYTHON_REVISION:-814556d15970dd2ecf1c9984e845ca02ab07e206}"
# Per-host cache for the pinned checkout. Default follows the pointnet2_ops
# precedent (the closest sibling: a pip-installed, non-PyPI Python package
# installed --no-deps into the workspace venv) and lives under the venv's own
# .cache, so the cache shares the venv's lifecycle instead of outliving it
# with a dangling editable pointer after a venv recreation. Worktrees that
# share one venv therefore also share one checkout. Overridable; the
# graspgenx_ascend_ops-style ${HOME}/.cache/ibrobot location remains valid via
# the override for hosts that want the source to survive venv rebuilds.
unitree_sdk_default_home() {
    if [[ -n "${VENV_PYTHON:-}" ]]; then
        echo "$(cd "$(dirname "${VENV_PYTHON}")/.." && pwd)/.cache/unitree_sdk2_python"
    else
        # No venv in sight (direct helper invocation): keep the graspgenx-style
        # per-host location rather than guessing at a workspace.
        echo "${HOME}/.cache/ibrobot/unitree_sdk2_python"
    fi
}
# Upstream setup.py pins cyclonedds==0.10.2 exactly; do not relax silently.
UNITREE_SDK2_PYTHON_CYCLONEDDS_PIN="${UNITREE_SDK2_PYTHON_CYCLONEDDS_PIN:-0.10.2}"

unitree_sdk_log() { printf '[unitree-sdk] %s\n' "$*"; }

# "install" on hosts this initial scope supports, "skip" elsewhere.
unitree_sdk_arch_action() {
    case "$(uname -m)" in
        x86_64) echo install ;;
        *) echo skip ;;
    esac
}

unitree_sdk_revision_is_valid() {
    [[ "${UNITREE_SDK2_PYTHON_REVISION}" =~ ^[0-9a-f]{40}$ ]]
}

# Prints "missing" (no checkout), "dirty" (tracked changes -- never discarded),
# "ready" (clean checkout at the pinned revision) or "mismatched" (clean
# checkout of a different revision). Untracked files are ignored on purpose:
# `pip install -e` drops unitree_sdk2_python.egg-info/ and __pycache__ into the
# source tree, and neither can conflict with a fetch/checkout.
unitree_sdk_source_state() {
    local dir="${UNITREE_SDK2_PYTHON_HOME}" current
    if [[ ! -d "${dir}/.git" ]]; then
        echo missing
        return
    fi
    if ! git -C "${dir}" diff --quiet 2>/dev/null || ! git -C "${dir}" diff --cached --quiet 2>/dev/null; then
        echo dirty
        return
    fi
    current="$(git -C "${dir}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${current}" == "${UNITREE_SDK2_PYTHON_REVISION}" ]]; then
        echo ready
    else
        echo mismatched
    fi
}

_unitree_sdk_fetch_impl() {
    local dir="${UNITREE_SDK2_PYTHON_HOME}"
    if [[ ! -d "${dir}/.git" ]]; then
        mkdir -p "$(dirname "${dir}")"
        git clone "${UNITREE_SDK2_PYTHON_REPO}" "${dir}" || return 1
    fi
    if ! git -C "${dir}" cat-file -e "${UNITREE_SDK2_PYTHON_REVISION}^{commit}" 2>/dev/null; then
        git -C "${dir}" fetch --depth 1 origin "${UNITREE_SDK2_PYTHON_REVISION}" || return 1
    fi
    git -C "${dir}" checkout --detach "${UNITREE_SDK2_PYTHON_REVISION}" || return 1
    unitree_sdk_log "source at ${dir} ($(git -C "${dir}" rev-parse --short HEAD))"
}

# Idempotent: reuses a clean pinned checkout, repairs a mismatched one, and
# refuses to touch one with tracked local changes.
_unitree_sdk_ensure_source() {
    local state
    state="$(unitree_sdk_source_state)"
    case "${state}" in
        ready)
            unitree_sdk_log "pinned source already present at ${UNITREE_SDK2_PYTHON_HOME}"
            return 0
            ;;
        dirty)
            log_error "unitree_sdk2_python checkout at ${UNITREE_SDK2_PYTHON_HOME} has tracked local changes."
            log_error "Resolve or remove them manually; this installer never discards them."
            return 1
            ;;
    esac
    run_cmd _unitree_sdk_fetch_impl
}

_unitree_sdk_install_deps() {
    local pip_runner=("$@")
    # Upstream pins cyclonedds==0.10.2 exactly. The binding is imported (and its
    # native library loaded) in the same processes as ROS 2; keep the pin until
    # a newer CycloneDDS has been verified against the robot tier.
    run_cmd "${pip_runner[@]}" "cyclonedds==${UNITREE_SDK2_PYTHON_CYCLONEDDS_PIN}" --quiet
}

_unitree_sdk_install_editable() {
    local pip_runner=("$@")
    # --no-deps: the closure (cyclonedds above; numpy and opencv via the
    # workspace's own pinned requirements) is managed by this repo, so pip must
    # not re-resolve upstream's unpinned extras on top of it.
    run_cmd "${pip_runner[@]}" --no-deps -e "${UNITREE_SDK2_PYTHON_HOME}" --quiet
}

# Import-only verification: no DDS participant, no network, no device. This is
# what "the SDK is installed" means here -- a version match alone is not proof.
unitree_sdk_verify() {
    PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import importlib.metadata
import os

expected_cyclonedds = os.environ.get("UNITREE_SDK2_PYTHON_CYCLONEDDS_PIN", "0.10.2")

import numpy  # upstream dependency; version is owned by the workspace pins

import cyclonedds  # noqa: F401  (loads the native binding, creates no participant)

installed_cyclonedds = importlib.metadata.version("cyclonedds")
if installed_cyclonedds != expected_cyclonedds:
    raise SystemExit(
        f"cyclonedds {installed_cyclonedds} installed; unitree_sdk2py pins =={expected_cyclonedds}"
    )

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: F401
from unitree_sdk2py.core.channel import (  # noqa: F401
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient  # noqa: F401
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient  # noqa: F401
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient  # noqa: F401
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__HandState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__MotorCmd_,
)

# The G1 bridge builds these frames through the IDL factories; a factory that
# constructs must expose the documented fixed slot count.
low_cmd = unitree_hg_msg_dds__LowCmd_()
if len(low_cmd.motor_cmd) != 35:
    raise SystemExit(f"unitree_hg LowCmd_ exposes {len(low_cmd.motor_cmd)} motor slots, expected 35")
unitree_hg_msg_dds__HandCmd_()
unitree_hg_msg_dds__HandState_()
unitree_hg_msg_dds__LowState_()
unitree_hg_msg_dds__MotorCmd_()

# Editable-install resource check: upstream ships no package_data, so the
# bundled CRC native libraries only exist through the source tree. The expert
# low-level tier needs them; fail here rather than at first robot use.
import unitree_sdk2py.utils.crc as crc_module

crc_lib_dir = os.path.join(os.path.dirname(os.path.abspath(crc_module.__file__)), "lib")
if not any(
    os.path.isfile(os.path.join(crc_lib_dir, name)) for name in ("crc_amd64.so", "crc_aarch64.so")
):
    raise SystemExit(
        f"unitree_sdk2py bundled CRC libraries missing under {crc_lib_dir}; "
        "the install must stay editable against the pinned source checkout"
    )

print(
    f"unitree_sdk2py {importlib.metadata.version('unitree_sdk2py')} verified "
    f"(cyclonedds=={installed_cyclonedds}, numpy {numpy.__version__})"
)
PY
}

# Entry point, mirroring install_graspgen_pip.sh: receives the workspace pip
# command array ("${VENV_PYTHON}" -m pip install ...).
install_unitree_sdk() {
    local pip_runner=("$@")

    # Resolved at call time, not source time: the venv-scoped default needs the
    # caller's VENV_PYTHON, which is only known inside the hardware branch.
    UNITREE_SDK2_PYTHON_HOME="${UNITREE_SDK2_PYTHON_HOME:-$(unitree_sdk_default_home)}"

    if [[ "$(unitree_sdk_arch_action)" == "skip" ]]; then
        log_warn "unitree_sdk2py install skipped: $(uname -m) is unsupported in the initial G1 scope."
        log_warn "The vendor SDK's pinned cyclonedds==${UNITREE_SDK2_PYTHON_CYCLONEDDS_PIN} has no wheel for this architecture."
        log_warn "The G1 runtime will refuse to start on this host until the SDK tier is ported (AtomGit Issue #171)."
        return 0
    fi

    if ! unitree_sdk_revision_is_valid; then
        log_error "UNITREE_SDK2_PYTHON_REVISION='${UNITREE_SDK2_PYTHON_REVISION}' is not a full 40-hex commit."
        return 1
    fi

    unitree_sdk_log "installing unitree_sdk2py at pinned revision ${UNITREE_SDK2_PYTHON_REVISION}"
    _unitree_sdk_ensure_source || return 1
    _unitree_sdk_install_deps "${pip_runner[@]}" || return 1
    _unitree_sdk_install_editable "${pip_runner[@]}" || return 1

    if [[ "${DRY_RUN}" == true ]]; then
        unitree_sdk_log "dry-run: skipping import verification."
        return 0
    fi
    if ! unitree_sdk_verify; then
        log_error "unitree_sdk2py verification failed; see the output above."
        log_error "The G1 runtime requires this SDK; fix the install and re-run setup."
        return 1
    fi
}
