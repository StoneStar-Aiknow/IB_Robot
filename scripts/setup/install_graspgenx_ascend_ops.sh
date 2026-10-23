#!/usr/bin/env bash
# Build the custom Ascend C operators GraspGenX's PTV3 encoder calls.
#
# Two products come out of this, and neither can be shipped as a downloadable
# artifact (see src/torch_models/torch_models/graspgenx_ascend_310p/custom_ops/kernels/manifest.yaml):
#
#   * a CANN vendor package, compiled for one CANN release and the ascend310p
#     compute unit;
#   * a Torch bridge, linked against the venv's libtorch and libtorch_npu with
#     the ABI flag read from the installed Torch.
#
# The layout written here is the one
# src/torch_models/torch_models/graspgenx_ascend_310p/custom_ops/__init__.py
# looks for:
#
#   <root>/<vendor_name>/torch_bridge.so
#   <root>/vendors/<vendor_name>/...        (for ASCEND_CUSTOM_OPP_PATH)
#
# Usage: install_graspgenx_ascend_ops.sh [--root <dir>] [--force]
#
# CANN's setenv.bash runs `[ -d ... ]` probes that fail normally, so errexit is
# enabled only once it has been sourced; reversing that aborts the script with
# no output. Everything after graspgenx_ops_prepare_env therefore runs under
# `set -e`, so a failed stage copy, vendor-package install or bridge link
# cannot fall through to the "built into ..." log and a zero exit.

# Overridable like every other setting here: on the board the operator sources
# are often all that was copied over, with no workspace tree around them.
GRASPGENX_OPS_SOURCE_ROOT="${GRASPGENX_OPS_SOURCE_ROOT:-${WORKSPACE:-$(pwd)}/src/torch_models/torch_models/graspgenx_ascend_310p/custom_ops/kernels}"
# The products are built for one CANN release, the ascend310p compute unit and
# the installed Torch's ABI, so they are a per-host cache rather than part of
# the tree. torch_models.graspgenx_ascend_310p.custom_ops reads the same
# location, and IBROBOT_GRASPGENX_ASCEND_OPS overrides both.
GRASPGENX_OPS_DEFAULT_ROOT="${IBROBOT_GRASPGENX_ASCEND_OPS:-${HOME}/.cache/ibrobot/graspgenx_ascend_ops}"
# The release the operators are verified on. This is pinned rather than taken
# from `latest`, because `latest` is a directory of component symlinks that a
# partial install leaves inconsistent: the verification host has 8.3's toolkit
# unpacked over 8.1's, so `latest/bin/atc` and `latest/lib64/libascendcl.so`
# resolve to 8.3 while `latest/opp_kernel` and `latest/version.cfg` are still
# 8.1's. An operator package built against one release and run against another
# is untested, so the release is named here and recorded in the output.
GRASPGENX_OPS_CANN_HOME="${GRASPGENX_OPS_CANN_HOME:-/usr/local/Ascend/ascend-toolkit/8.1.RC1}"
GRASPGENX_OPS_COMPUTE_UNIT="ascend310p"
# Resolve the workspace from this script, never from the operator cache.
GRASPGENX_OPS_WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

graspgenx_ops_log() { printf '[ascend-ops] %s\n' "$*"; }
graspgenx_ops_die() { printf '[ascend-ops] ERROR: %s\n' "$*" >&2; exit 1; }

graspgenx_ops_prepare_env() {
    [[ -d "${GRASPGENX_OPS_CANN_HOME}" ]] ||
        graspgenx_ops_die "CANN toolkit not found at ${GRASPGENX_OPS_CANN_HOME}. Set GRASPGENX_OPS_CANN_HOME."
    # The release named above has to be the one actually unpacked there. On the
    # verification host 8.3's toolkit is unpacked over 8.1's, which is exactly
    # the mismatch the pinned path exists to avoid -- but a `[[ -d ]]` test
    # cannot see it. version.cfg's layout is not contractual across releases,
    # so a disagreement warns and prints the file rather than failing closed.
    local expected_release version_file
    expected_release="$(basename "${GRASPGENX_OPS_CANN_HOME}")"
    version_file="${GRASPGENX_OPS_CANN_HOME}/version.cfg"
    if [[ -f "${version_file}" ]] && ! grep -qiF "${expected_release}" "${version_file}"; then
        graspgenx_ops_log "WARNING: ${version_file} does not name ${expected_release}; it reads:"
        sed -e 's/^/[ascend-ops]   /' "${version_file}" >&2
        graspgenx_ops_log "WARNING: the operators are verified on ${expected_release} only"
    fi
    export ASCEND_HOME_PATH="${GRASPGENX_OPS_CANN_HOME}"
    local setenv="${ASCEND_HOME_PATH}/$(uname -m)-linux/bin/setenv.bash"
    [[ -f "${setenv}" ]] || graspgenx_ops_die "missing ${setenv}; is this the toolkit rather than the runtime?"
    # shellcheck disable=SC1090
    source "${setenv}"
    export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH}"
    export TOOLCHAIN_HOME="${ASCEND_HOME_PATH}/toolkit"
    # CANN 8.1 warns on an unset switch although it already behaves like 0.
    export TILINGKEY_PAR_COMPILE="${TILINGKEY_PAR_COMPILE:-0}"
    export PYTHONPATH="${ASCEND_HOME_PATH}/toolkit/python/site-packages:${ASCEND_HOME_PATH}/compiler/python/site-packages:${PYTHONPATH:-}"
    export PATH="${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin:${ASCEND_HOME_PATH}/compiler/bin:${PATH}"

    # ccec does not discover the host libstdc++ headers on its own.
    local gcc_version gcc_target gcc_config
    gcc_version="$(g++ -dumpversion)"
    gcc_target="$(g++ -dumpmachine)"
    gcc_config="/usr/include/c++/${gcc_version}/${gcc_target}"
    [[ -d "${gcc_config}" ]] || gcc_config="/usr/include/${gcc_target}/c++/${gcc_version}"
    export CPLUS_INCLUDE_PATH="/usr/include/c++/${gcc_version}:${gcc_config}:/usr/include/c++/${gcc_version}/backward:$(g++ -print-file-name=include)${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
}

# Run one build stage, keeping its output and showing it only on failure: a
# kernel compile is verbose, and a silent failure here is what made the first
# version of this script report a missing .run instead of the real cause.
# ${vendor} and ${step_log} are the caller's locals.
graspgenx_ops_step() {
    local label="$1"; shift
    if ! "$@" >>"${step_log}" 2>&1; then
        graspgenx_ops_log "${vendor}: ${label} failed; last lines:"
        tail -30 "${step_log}" >&2
        graspgenx_ops_die "${vendor}: ${label} failed, full log at ${step_log}"
    fi
}

# $1 vendor name, $2.. the operator names in ops.json order.
graspgenx_ops_build_one() {
    local vendor="$1"; shift
    local source_dir="${GRASPGENX_OPS_SOURCE_ROOT}/${vendor}"
    local work="${GRASPGENX_OPS_ROOT}/_build/${vendor}"
    local project="${work}/opp_project"
    local generator="${ASCEND_HOME_PATH}/toolkit/python/site-packages/op_gen/msopgen.py"

    [[ -f "${source_dir}/ops.json" ]] || graspgenx_ops_die "no operator sources at ${source_dir}"
    [[ -f "${generator}" ]] || graspgenx_ops_die "msopgen not found at ${generator}; the CANN toolkit is required, not only the runtime"

    rm -rf "${work}"
    # msopgen creates the project directory itself and behaves differently when
    # one is already there, so only its parent is made.
    mkdir -p "${work}/runtime" "${work}/src"
    export ASCEND_WORK_PATH="${work}/runtime"

    # msopgen refuses to read an input that is group- or other-writable
    # ("should not be written by user group or others"), and a checkout's modes
    # depend on the umask, on git and on however the tree was copied here. The
    # sources are therefore staged at 0644 and the build only ever reads the
    # staged copy.
    cp -r "${source_dir}/." "${work}/src/"
    chmod -R u=rwX,go=rX "${work}/src"
    local staged="${work}/src"

    local merge=""
    for operator in "$@"; do
        graspgenx_ops_log "generating ${vendor}::${operator}"
        # shellcheck disable=SC2086
        "${GRASPGENX_OPS_CANN_PYTHON}" "${generator}" gen -i "${staged}/ops.json" \
            -op "${operator}" -f pytorch -c "ai_core-${GRASPGENX_OPS_COMPUTE_UNIT}" \
            -out "${project}" -lan cpp ${merge} ||
            graspgenx_ops_die "${vendor}: msopgen failed for ${operator}"
        # Every operator after the first merges into the same project.
        merge="-m 1"
    done
    [[ -f "${project}/CMakeLists.txt" ]] ||
        graspgenx_ops_die "${vendor}: msopgen wrote no CMakeLists.txt into ${project}"

    cp "${staged}"/op_host/* "${project}/op_host/"
    cp "${staged}"/op_kernel/* "${project}/op_kernel/"

    graspgenx_ops_log "compiling ${vendor} kernels for ${GRASPGENX_OPS_COMPUTE_UNIT}"
    local step_log="${work}/build.log"
    graspgenx_ops_step configure cmake -G "Unix Makefiles" -S "${project}" -B "${work}/cmake" \
        -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
        -DASCEND_COMPUTE_UNIT="${GRASPGENX_OPS_COMPUTE_UNIT}" \
        -Dvendor_name="${vendor}" \
        -DASCEND_PYTHON_EXECUTABLE="${GRASPGENX_OPS_CANN_PYTHON}" \
        -DENABLE_SOURCE_PACKAGE=ON -DENABLE_BINARY_PACKAGE=ON -DENABLE_TEST=OFF
    # CANN 8.1 leaves per-operator directories behind after a failed compile,
    # and a later run then reuses them instead of regenerating.
    for operator in "$@"; do
        rm -rf "${work}/cmake/op_kernel/${operator}_${GRASPGENX_OPS_COMPUTE_UNIT}"
    done
    graspgenx_ops_step "kernel build" cmake --build "${work}/cmake" --target binary -j2
    graspgenx_ops_step packaging cmake --build "${work}/cmake" --target package -j2

    # An unmatched glob leaves the pattern itself as the single element, so the
    # file test is what distinguishes "none built" from "one built".
    local packages=("${project}"/build_out/*.run)
    [[ -f "${packages[0]}" ]] ||
        graspgenx_ops_die "${vendor}: the package target produced no .run in ${project}/build_out"
    [[ ${#packages[@]} == 1 ]] ||
        graspgenx_ops_die "${vendor}: expected one .run, found ${#packages[@]}: ${packages[*]}"
    graspgenx_ops_step "vendor package install" \
        bash "${packages[0]}" --install-path="${GRASPGENX_OPS_ROOT}" --quiet

    graspgenx_ops_log "linking ${vendor} Torch bridge"
    local torch_dir npu_dir python_include abi
    torch_dir="$("${GRASPGENX_OPS_VENV_PYTHON}" -c 'import torch; print(torch.__path__[0])')"
    npu_dir="$("${GRASPGENX_OPS_VENV_PYTHON}" -c 'import torch_npu; print(torch_npu.__path__[0])')"
    python_include="$("${GRASPGENX_OPS_VENV_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
    abi="$("${GRASPGENX_OPS_VENV_PYTHON}" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
    mkdir -p "${GRASPGENX_OPS_ROOT}/${vendor}"
    graspgenx_ops_step "Torch bridge link" \
        ${CXX:-c++} -O2 -std=c++17 -fPIC -shared "${staged}/torch_bridge.cpp" \
        -o "${GRASPGENX_OPS_ROOT}/${vendor}/torch_bridge.so" \
        -D_GLIBCXX_USE_CXX11_ABI="${abi}" \
        -DTORCH_EXTENSION_NAME="${vendor}_bridge" \
        -I"${torch_dir}/include" -I"${torch_dir}/include/torch/csrc/api/include" \
        -I"${npu_dir}/include" -I"${python_include}" \
        -L"${torch_dir}/lib" -L"${npu_dir}/lib" \
        -Wl,-rpath,"${torch_dir}/lib" -Wl,-rpath,"${npu_dir}/lib" \
        -ltorch -ltorch_cpu -ltorch_python -lc10 -ltorch_npu
}

# The identity the products are only valid for. Mixing any of these produces a
# vendor package or a bridge that loads and then fails, so a build whose stamp
# disagrees with the host is rebuilt rather than reused.
graspgenx_ops_stamp() {
    printf 'cann_release=%s\ncann_home=%s\ncompute_unit=%s\ntorch_version=%s\ntorch_cxx11_abi=%s\n' \
        "$(basename "${GRASPGENX_OPS_CANN_HOME}")" \
        "${GRASPGENX_OPS_CANN_HOME}" \
        "${GRASPGENX_OPS_COMPUTE_UNIT}" \
        "$("${GRASPGENX_OPS_VENV_PYTHON}" -c 'import torch; print(torch.__version__)')" \
        "$("${GRASPGENX_OPS_VENV_PYTHON}" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
}

install_graspgenx_ascend_ops() {
    local root="${GRASPGENX_OPS_DEFAULT_ROOT}" force=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --root)
                # Without this, a missing operand leaves root empty and the
                # per-vendor `rm -rf "${root}/_build/..."` runs against /.
                [[ $# -ge 2 && -n "$2" ]] || graspgenx_ops_die "--root needs a directory"
                root="$2"; shift 2 ;;
            --force) force=true; shift ;;
            *) graspgenx_ops_die "unknown argument: $1" ;;
        esac
    done
    GRASPGENX_OPS_ROOT="${root}"
    GRASPGENX_OPS_CANN_PYTHON="${GRASPGENX_OPS_CANN_PYTHON:-$(command -v python3)}"
    GRASPGENX_OPS_VENV_PYTHON="${GRASPGENX_OPS_VENV_PYTHON:-${VENV_PYTHON:-${VIRTUAL_ENV:-${GRASPGENX_OPS_WORKSPACE}/venv}/bin/python}}"
    [[ -x "${GRASPGENX_OPS_VENV_PYTHON}" ]] ||
        graspgenx_ops_die "workspace venv Python not found at ${GRASPGENX_OPS_VENV_PYTHON}; set GRASPGENX_OPS_VENV_PYTHON."

    local stamp_path="${root}/build-stamp.txt" want
    want="$(graspgenx_ops_stamp)"
    if [[ "${force}" != true ]] &&
        [[ -f "${root}/graspgenx_subm/torch_bridge.so" ]] &&
        [[ -f "${root}/graspgenx_grid/torch_bridge.so" ]]; then
        if [[ -f "${stamp_path}" ]] && [[ "$(cat "${stamp_path}")" == "${want}" ]]; then
            graspgenx_ops_log "already built in ${root} for this host; pass --force to rebuild"
            return 0
        fi
        # Testing only that the .so files exist would reuse a build made for
        # another CANN release, compute unit or Torch ABI, which is what
        # GraspGenX-ascend's own notes warn against.
        if [[ -f "${stamp_path}" ]]; then
            graspgenx_ops_log "rebuilding: ${stamp_path} does not describe this host"
            # No process substitution: this script has to run under the board's
            # minimal shell too (scripts/setup/tests/test_accel_axis.sh guards
            # that). Report the two stamps instead of diffing them.
            graspgenx_ops_log "  recorded:"
            sed 's/^/[ascend-ops]     /' "${stamp_path}" >&2
            graspgenx_ops_log "  this host:"
            printf '%s\n' "${want}" | sed 's/^/[ascend-ops]     /' >&2
        else
            graspgenx_ops_log "rebuilding: ${root} has no build stamp to check"
        fi
    fi

    graspgenx_ops_prepare_env
    # Past this point every stage is load-bearing, and `set -e` is what stops a
    # failed copy, vendor-package install or link from reaching the success log
    # below. It cannot be enabled earlier: CANN's setenv.bash fails its own
    # probes. Stages that produce diagnostics of their own still go through
    # graspgenx_ops_step, which reports before it dies.
    set -e
    mkdir -p "${root}"
    graspgenx_ops_build_one graspgenx_subm BuildSubmMap SubmConv3d
    graspgenx_ops_build_one graspgenx_grid GridEncode

    # Written from the same function the guard above compares against, so the
    # two cannot drift into disagreeing about what identity means.
    graspgenx_ops_stamp > "${stamp_path}"

    graspgenx_ops_log "built into ${root}"
    graspgenx_ops_log "the runtime needs these exported before the process starts:"
    graspgenx_ops_log "  export IBROBOT_GRASPGENX_ASCEND_OPS=${root}"
    graspgenx_ops_log "  export ASCEND_CUSTOM_OPP_PATH=${root}/vendors/graspgenx_subm:${root}/vendors/graspgenx_grid"
}

# Allow both `source` for the setup script and direct execution.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -o pipefail
    install_graspgenx_ascend_ops "$@"
fi
