#!/bin/bash
# Rebuild the audited GraspGen pointnet2_ops CUDA wheel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${WORKSPACE}/third_party/patches/pointnet2_ops/a56d518"
WHEEL_ROOT="${WORKSPACE}/third_party/wheels/pointnet2_ops/a56d518"
UPSTREAM_URL="https://github.com/NVlabs/GraspGen.git"
UPSTREAM_COMMIT="a56d518f3b76ea2a432b5b838b3c68027d29be49"
EXPECTED_WHEEL="pointnet2_ops-3.0.0+ibrobot.1-cp310-cp310-manylinux_2_17_x86_64.whl"
PYTHON_BIN="${POINTNET2OPS_PYTHON:-python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1739846159}"

if [[ "${TORCH_CUDA_ARCH_LIST}" != "8.6" ]]; then
    echo "This audited wheel is fixed to TORCH_CUDA_ARCH_LIST=8.6, got ${TORCH_CUDA_ARCH_LIST}" >&2
    exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA toolkit not found: ${CUDA_HOME}/bin/nvcc" >&2
    exit 1
fi

BUILD_ROOT="${POINTNET2OPS_WHEEL_BUILD_ROOT:-$(mktemp -d)}"
cleanup() {
    if [[ -z "${POINTNET2OPS_WHEEL_BUILD_ROOT:-}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}
trap cleanup EXIT
if [[ -z "${BUILD_ROOT}" || "${BUILD_ROOT}" == "/" ]]; then
    echo "Invalid pointnet2_ops wheel build root: ${BUILD_ROOT}" >&2
    exit 1
fi
mkdir -p "${BUILD_ROOT}"
rm -rf "${BUILD_ROOT}/source" "${BUILD_ROOT}/dist"
git clone --quiet "${UPSTREAM_URL}" "${BUILD_ROOT}/source"
git -C "${BUILD_ROOT}/source" checkout --quiet "${UPSTREAM_COMMIT}"
while IFS= read -r patch_file; do
    [[ -z "${patch_file}" ]] && continue
    git -C "${BUILD_ROOT}/source" apply "${PATCH_ROOT}/${patch_file}"
done < "${PATCH_ROOT}/series.txt"

TORCH_VERSION="$(${PYTHON_BIN} -c 'import torch; print(torch.__version__)')"
TORCH_CUDA_VERSION="$(${PYTHON_BIN} -c 'import torch; print(torch.version.cuda or "")')"
[[ "${TORCH_VERSION}" == "2.7.1+cu126" ]] || { echo "Expected torch==2.7.1+cu126, got ${TORCH_VERSION}" >&2; exit 1; }
[[ "${TORCH_CUDA_VERSION}" == "12.6" ]] || { echo "Expected torch CUDA 12.6, got ${TORCH_CUDA_VERSION}" >&2; exit 1; }
rm -rf "${BUILD_ROOT}/dist"
(
    cd "${BUILD_ROOT}/source/pointnet2_ops"
    "${PYTHON_BIN}" setup.py bdist_wheel --plat-name manylinux_2_17_x86_64 \
        --dist-dir "${BUILD_ROOT}/dist"
)
built_wheel="${BUILD_ROOT}/dist/${EXPECTED_WHEEL}"
"${PYTHON_BIN}" - "${built_wheel}" <<'PY'
import sys
import zipfile
wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    metadata = archive.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode()
    wheel_metadata = archive.read(next(n for n in names if n.endswith('.dist-info/WHEEL'))).decode()
    assert 'Name: pointnet2_ops' in metadata
    assert 'Version: 3.0.0+ibrobot.1' in metadata
    assert 'Tag: cp310-cp310-manylinux_2_17_x86_64' in wheel_metadata
    assert any(n.endswith('_ext.cpython-310-x86_64-linux-gnu.so') for n in names)
    assert 'pointnet2_ops/LICENSE' in names
print(f'Wheel content verified: {wheel}')
PY
mkdir -p "${WHEEL_ROOT}"
cp "${built_wheel}" "${WHEEL_ROOT}/${EXPECTED_WHEEL}"
(
    cd "${WHEEL_ROOT}"
    sha256sum "${EXPECTED_WHEEL}" > SHA256SUMS
)
echo "Built ${WHEEL_ROOT}/${EXPECTED_WHEEL}"
