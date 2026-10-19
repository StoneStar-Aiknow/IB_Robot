#!/bin/bash
# install_graspgen_pip.sh - GraspGen installer: audited grasp_gen and pointnet2_ops wheels.

install_graspgen_pointnet2ops_wheel() {
    local pip_runner=("$@")
    local wheel_root="${WORKSPACE}/third_party/wheels/pointnet2_ops/a56d518"
    local audited_wheel="${wheel_root}/pointnet2_ops-3.0.0+ibrobot.1-cp310-cp310-manylinux_2_17_x86_64.whl"
    # Empty for the audited install; graspgen_pointnet2ops_build exports it to a
    # locally compiled wheel for GPUs the audited sm_86 wheel cannot run on.
    local wheel="${GRASPGEN_POINTNET2OPS_WHEEL:-${audited_wheel}}"
    local python_version torch_version torch_cuda_version

    python_version="$(${VENV_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    torch_version="$(${VENV_PYTHON} -c 'import torch; print(torch.__version__)')"
    torch_cuda_version="$(${VENV_PYTHON} -c 'import torch; print(torch.version.cuda or "")')"
    if [[ "${python_version}" != "3.10" ]]; then
        log_error "Precompiled pointnet2_ops requires Python 3.10, got ${python_version}."
        return 1
    fi
    if [[ "${torch_version}" != "2.7.1+cu126" || "${torch_cuda_version}" != "12.6" ]]; then
        log_error "Precompiled pointnet2_ops requires torch==2.7.1+cu126 (CUDA 12.6)."
        log_error "Found torch=${torch_version}, torch CUDA=${torch_cuda_version}."
        log_error "Install the matching Torch build, or rebuild the wheel with scripts/build_pointnet2ops_wheel.sh."
        return 1
    fi
    if [[ -z "${GRASPGEN_POINTNET2OPS_WHEEL:-}" ]]; then
        # Audited artifact: checksum gate plus the architecture guard as the
        # last line of defense (graspgen_pointnet2ops_plan already decided, so
        # this only fires if the visible GPUs changed mid-setup).
        if [[ ! -f "${audited_wheel}" || ! -f "${wheel_root}/SHA256SUMS" ]]; then
            log_error "Precompiled pointnet2_ops wheel is missing from ${wheel_root}."
            return 1
        fi
        if ! (cd "${wheel_root}" && sha256sum --check SHA256SUMS); then
            log_error "pointnet2_ops wheel checksum verification failed."
            return 1
        fi
        check_pointnet2ops_gpu_arch || return 1
    fi
    # Audited and local architecture builds share a version. Pip otherwise
    # keeps the installed binary even when this run selected a different one.
    run_cmd "${pip_runner[@]}" --force-reinstall --no-deps "${wheel}" --quiet
}

# The audited wheel carries sm_86 cubins and no PTX (cuobjdump --list-elf shows
# sm_86.cubin only, --list-ptx nothing), so there is nothing for the driver to
# JIT from. A cubin runs on a later minor revision of the same major, which
# covers sm_86/87/89, but not on sm_80 (earlier minor) or sm_90 (other major).
# graspgen_pointnet2ops_plan makes the install/skip decision up front; this
# check remains as defense in depth for direct callers of the audited wheel
# path. A host with no visible GPU cannot be judged (a container without
# --gpus is the normal verification case), so the check passes there.
check_pointnet2ops_gpu_arch() {
    local caps cap major minor
    caps="$(PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY' 2>/dev/null || true
import torch

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print("%d.%d" % torch.cuda.get_device_capability(index))
PY
)"
    if [[ -z "${caps}" ]]; then
        log_info "No CUDA device visible; skipping the pointnet2_ops GPU architecture check."
        return 0
    fi
    while read -r cap; do
        [[ -n "${cap}" ]] || continue
        major="${cap%%.*}"
        minor="${cap##*.}"
        if [[ "${major}" -ne 8 || "${minor}" -lt 6 ]]; then
            log_error "The audited pointnet2_ops wheel is built for sm_86 and carries no PTX."
            log_error "This host reports compute capability ${cap}, which cannot run those cubins."
            log_error "Install a CUDA 12.x toolkit and re-run setup to compile a local wheel for this GPU."
            return 1
        fi
    done <<< "${caps}"
    log_info "GPU architecture check passed for the sm_86 pointnet2_ops wheel (compute capability: ${caps//$'\n'/, })."
}

# The build script resolves its toolkit from CUDA_HOME (defaulting to
# /usr/local/cuda-12.6); prefer that, then any nvcc on PATH.
graspgen_find_nvcc() {
    local candidate
    for candidate in "${CUDA_HOME:-/usr/local/cuda-12.6}/bin/nvcc" "$(command -v nvcc 2>/dev/null || true)"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            readlink -f "${candidate}"
            return 0
        fi
    done
    return 0
}

# Decide how the pointnet2_ops CUDA extension can be provided on this host.
# Echoes exactly one line of the form:
#   ok:<summary>    install the audited sm_86 wheel (compatible, or no visible
#                   GPU to judge, which is the normal verification container)
#   build:<arches>  compile an unaudited local wheel for every reported GPU
#                   capability, separated by ';', with a CUDA 12.x nvcc
#   skip:<reason>   install nothing; the caller skips grasp without failing
# Must stay silent on stdout apart from that line: the caller captures it in a
# command substitution, and log_info/log_warn write to stdout.
graspgen_pointnet2ops_plan() {
    local python_version caps="" cap major minor incompatible=false arches=""
    if [[ "$(uname -m)" != "x86_64" ]]; then
        echo "skip:the pointnet2_ops wheel and its build script are x86_64-only"
        return 0
    fi
    python_version="$("${VENV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${python_version}" != "3.10" ]]; then
        echo "skip:pointnet2_ops requires Python 3.10, got ${python_version}"
        return 0
    fi
    # Decide before installing Torch: its pinned CUDA build cannot run on
    # Blackwell, and a fresh venv does not have Torch to probe devices with.
    if command -v nvidia-smi >/dev/null 2>&1; then
        if ! caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null)"; then
            if [[ -n "$(nvidia-smi -L 2>/dev/null || true)" ]]; then
                echo "skip:could not query NVIDIA GPU compute capabilities; leaving the Torch runtime selected by LeRobot"
                return 0
            fi
            caps=""
        fi
    fi
    if [[ -z "${caps}" ]]; then
        echo "ok:no visible CUDA device, so the sm_86 architecture cannot be judged"
        return 0
    fi
    while read -r cap; do
        cap="${cap//[[:space:]]/}"
        [[ -n "${cap}" ]] || continue
        if [[ ! "${cap}" =~ ^[0-9]+\.[0-9]+$ ]]; then
            echo "skip:could not interpret NVIDIA GPU compute capability ${cap}"
            return 0
        fi
        major="${cap%%.*}"
        minor="${cap##*.}"
        if [[ "${major}" -ge 10 ]]; then
            echo "skip:compute capability ${cap} is beyond the pinned Torch 2.7.1+cu126; leaving the Torch runtime selected by LeRobot"
            return 0
        fi
        if [[ "${major}" -ne 8 || "${minor}" -lt 6 ]]; then
            incompatible=true
        fi
        arches+="${cap}"$'\n'
    done <<< "${caps}"
    if [[ "${incompatible}" == false ]]; then
        echo "ok:compute capability ${caps//$'\n'/, } runs the audited sm_86 wheel"
        return 0
    fi
    arches="$(printf '%s' "${arches}" | sort -Vu)"
    arches="${arches//$'\n'/;}"
    arches="${arches%;}"
    local nvcc_bin nvcc_major
    nvcc_bin="$(graspgen_find_nvcc)"
    if [[ -z "${nvcc_bin}" ]]; then
        echo "skip:this host's GPUs (compute capabilities ${arches}) cannot all run the audited sm_86 wheel; install a CUDA 12.x toolkit and re-run setup to compile pointnet2_ops from source"
        return 0
    fi
    nvcc_major="$("${nvcc_bin}" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
    # Only the major version must match: torch.utils.cpp_extension warns on a
    # minor mismatch but refuses a major one, so any CUDA 12.x nvcc can build
    # against the cu126 Torch that will be installed after this preflight.
    if [[ "${nvcc_major}" != "12" ]]; then
        echo "skip:nvcc reports CUDA ${nvcc_major:-unknown}, which does not match the target Torch's CUDA 12.6; install a CUDA 12.x toolkit and re-run setup to compile pointnet2_ops from source"
        return 0
    fi
    echo "build:${arches}"
}

# Compile an UNAUDITED local pointnet2_ops wheel for a GPU the audited sm_86
# wheel cannot run on (sm_75/80/90 with a major-version-matching nvcc). This
# goes through scripts/build_pointnet2ops_wheel.sh — same pinned upstream
# commit and patch stack as the audited build — but it clones from GitHub and
# compiles locally, so it cannot serve an offline or GitHub-unreachable host;
# the caller degrades to a skipped grasp when it fails. Wheels are cached per
# architecture set under venv/.cache and reused on re-run. The resulting wheel path
# is exported in GRASPGEN_POINTNET2OPS_WHEEL, which
# install_graspgen_pointnet2ops_wheel installs instead of the audited wheel.
graspgen_pointnet2ops_build() {
    local arches="$1"
    local nvcc_bin cuda_home cache_dir cache_key wheel_name
    nvcc_bin="$(graspgen_find_nvcc)"
    if [[ -z "${nvcc_bin}" ]]; then
        log_error "Cannot build pointnet2_ops without a CUDA 12.x nvcc."
        return 1
    fi
    cuda_home="$(cd "$(dirname "${nvcc_bin}")/.." && pwd)" || return 1
    cache_key="${arches//./}"
    cache_key="${cache_key//;/_}"
    cache_dir="${WORKSPACE}/venv/.cache/pointnet2_ops/sm_${cache_key}"
    wheel_name="pointnet2_ops-3.0.0+ibrobot.1-cp310-cp310-manylinux_2_17_x86_64.whl"
    if [[ -f "${cache_dir}/${wheel_name}" && -f "${cache_dir}/SHA256SUMS" ]] &&
        (cd "${cache_dir}" && sha256sum --check --status SHA256SUMS); then
        log_info "Reusing the locally built pointnet2_ops wheel for ${arches} from ${cache_dir}."
        export GRASPGEN_POINTNET2OPS_WHEEL="${cache_dir}/${wheel_name}"
        return 0
    fi
    log_warn "The audited pointnet2_ops wheel cannot run on every GPU (compute capabilities ${arches})."
    log_warn "Building an UNAUDITED local wheel from the pinned upstream source; this step clones from GitHub and needs the CUDA toolkit found at ${cuda_home}."
    run_cmd env \
        POINTNET2OPS_PYTHON="${VENV_PYTHON}" \
        TORCH_CUDA_ARCH_LIST="${arches}" \
        POINTNET2OPS_WHEEL_OUT_DIR="${cache_dir}" \
        CUDA_HOME="${cuda_home}" \
        bash "${WORKSPACE}/scripts/build_pointnet2ops_wheel.sh" || return 1
    if [[ ! -f "${cache_dir}/${wheel_name}" ]]; then
        log_error "The local pointnet2_ops build did not produce ${cache_dir}/${wheel_name}."
        return 1
    fi
    export GRASPGEN_POINTNET2OPS_WHEEL="${cache_dir}/${wheel_name}"
}

install_graspgen_pip() {
    local pip_runner=("$@")
    local graspgen_wheel_root="${WORKSPACE}/third_party/wheels/graspgen/a56d518"
    local graspgen_wheel="${graspgen_wheel_root}/grasp_gen-1.0.0+ibrobot.1-py3-none-any.whl"
    local constraint_args=()
    local pyg_find_links

    if [[ -n "${ROS_ABI_CONSTRAINTS:-}" && -f "${ROS_ABI_CONSTRAINTS}" ]]; then
        constraint_args=(--constraint "${ROS_ABI_CONSTRAINTS}")
    fi

    log_info "Installing GraspGen runtime dependencies..."
    pyg_find_links="${GRASPGEN_PYG_FIND_LINKS:-$("${VENV_PYTHON}" - <<'PY' 2>/dev/null || true
import torch

version = torch.__version__.split("+", 1)[0].split(".")
torch_tag = f"{version[0]}.{version[1]}.0"
cuda_version = torch.version.cuda
if cuda_version:
    cuda_tag = "cu" + cuda_version.replace(".", "")
else:
    cuda_tag = "cpu"
print(f"https://data.pyg.org/whl/torch-{torch_tag}+{cuda_tag}.html")
PY
)}"
    if [[ -n "${pyg_find_links}" ]]; then
        run_cmd "${pip_runner[@]}" "${constraint_args[@]}" --find-links "${pyg_find_links}" \
            -r "${WORKSPACE}/requirements/manipulation.txt" --quiet
    else
        run_cmd "${pip_runner[@]}" "${constraint_args[@]}" \
            -r "${WORKSPACE}/requirements/manipulation.txt" --quiet
    fi

    # GraspGen ships as an audited pure-Python wheel built from its pinned
    # upstream commit (scripts/build_graspgen_wheel.sh), with its gripper
    # definitions and meshes packaged inside it. Only a local PointNet2 build
    # needs GitHub access.
    log_info "Installing GraspGen from the audited wheel..."
    if ! (cd "${graspgen_wheel_root}" && sha256sum --check SHA256SUMS); then
        log_error "GraspGen wheel checksum verification failed."
        return 1
    fi
    remove_legacy_graspgen_checkout "${pip_runner[@]}"
    run_cmd "${pip_runner[@]}" --no-deps "${graspgen_wheel}" --quiet

    # The CUDA PointNet2 extension also ships as an audited wheel, ABI-bound to
    # Python 3.10 and Torch 2.7.1+cu126.
    log_info "Installing GraspGen PointNet2 CUDA extension..."
    install_graspgen_pointnet2ops_wheel "${pip_runner[@]}"
}

# Earlier setups installed grasp_gen editable from a clone in venv/src/grasp-gen.
# Uninstall it so the wheel is the only grasp_gen on the path; the clone itself is
# left in place because it may hold local work.
remove_legacy_graspgen_checkout() {
    local pip_runner=("$@")
    local direct_url
    direct_url="$("${VENV_PYTHON}" - <<'PY' 2>/dev/null || true
from importlib.metadata import PackageNotFoundError, distribution

try:
    print(distribution("grasp_gen").read_text("direct_url.json") or "")
except PackageNotFoundError:
    print("")
PY
)"
    if [[ "${direct_url}" == *'"editable": true'* ]]; then
        log_info "Replacing the editable GraspGen checkout install with the audited wheel..."
        # Spelled out rather than substituting "install" inside pip_runner: that
        # substitution also rewrites the interpreter path when the workspace
        # itself lives under a directory whose name contains "install".
        run_cmd "${VENV_PYTHON}" -m pip uninstall -y grasp_gen --quiet
    fi
}
