#!/bin/bash
# install_graspgen_pip.sh - Optional GraspGen CUDA dependency installer.

check_graspgen_cuda_toolkit() {
    local torch_cuda_version cuda_home_candidate cuda_home="${CUDA_HOME:-}" nvcc_path

    torch_cuda_version="$("${VENV_PYTHON}" - <<'PY' 2>/dev/null || true
import torch
print(torch.version.cuda or "")
PY
)"

    if [[ -n "${cuda_home}" && -x "${cuda_home}/bin/nvcc" ]]; then
        return 0
    fi

    if [[ -n "${torch_cuda_version}" ]]; then
        cuda_home_candidate="/usr/local/cuda-${torch_cuda_version}"
        if [[ -x "${cuda_home_candidate}/bin/nvcc" ]]; then
            return 0
        fi
    fi

    nvcc_path="$(command -v nvcc 2>/dev/null || true)"
    if [[ -n "${nvcc_path}" ]]; then
        export CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
        return 0
    fi

    log_error "GraspGen pointnet2_ops requires a CUDA toolkit before installing."
    log_error "Set CUDA_HOME to a CUDA toolkit root with bin/nvcc, or install a matching toolkit"
    log_error "such as /usr/local/cuda-${torch_cuda_version:-<torch-cuda-version>}."
    log_error "GraspGen is now installed by default on Ubuntu; CPU-only hosts cannot build pointnet2_ops."
    return 1
}

install_graspgen_pip() {
    local pip_runner=("$@")
    local graspgen_ref="${GRASPGEN_GIT_REF:-a56d518f3b76ea2a432b5b838b3c68027d29be49}"
    local graspgen_repo="${GRASPGEN_GIT_URL:-https://github.com/NVlabs/GraspGen.git}"
    local graspgen_src="${WORKSPACE}/venv/src"
    local pointnet_url="git+${graspgen_repo}@${graspgen_ref}#subdirectory=pointnet2_ops"
    local torch_cuda_version cuda_home_candidate cuda_env=()
    local constraint_args=()
    local pyg_find_links

    check_graspgen_cuda_toolkit || exit 1

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

    # Upstream GraspGen expects top-level config/ and assets/ to be present next
    # to grasp_gen/. Editable VCS install keeps that source tree under venv/src
    # while avoiding a vendored copy under IB-Robot libs/.
    log_info "Installing GraspGen from pinned upstream source (${graspgen_ref})..."
    run_cmd "${pip_runner[@]}" --src "${graspgen_src}" --no-build-isolation --no-deps \
        -e "git+${graspgen_repo}@${graspgen_ref}#egg=grasp_gen" --quiet

    # GraspGen's CUDA PointNet2 extension is a sibling package in upstream.
    # Keep it as a pip-installed package as well, rather than importing from libs/.
    log_info "Installing GraspGen PointNet2 CUDA extension..."
    torch_cuda_version="$("${VENV_PYTHON}" - <<'PY' 2>/dev/null || true
import torch
print(torch.version.cuda or "")
PY
)"
    cuda_home_candidate="/usr/local/cuda-${torch_cuda_version}"
    if [[ -n "${torch_cuda_version}" && -x "${cuda_home_candidate}/bin/nvcc" ]]; then
        cuda_env=(CUDA_HOME="${cuda_home_candidate}" PATH="${cuda_home_candidate}/bin:${PATH}")
        log_info "Using CUDA toolkit ${cuda_home_candidate} for pointnet2_ops build."
    fi
    run_cmd env "${cuda_env[@]}" TORCH_CUDA_ARCH_LIST="${GRASPGEN_TORCH_CUDA_ARCH_LIST:-8.6}" \
        "${pip_runner[@]}" --no-build-isolation --no-deps "${pointnet_url}" --quiet
}
