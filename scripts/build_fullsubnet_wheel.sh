#!/bin/bash
# Rebuild the audited FullSubNet wheel from its pinned upstream revision.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${WORKSPACE}/third_party/patches/fullsubnet/e97448375"
WHEEL_ROOT="${WORKSPACE}/third_party/wheels/fullsubnet/e97448375"
UPSTREAM_URL="https://github.com/Audio-WestlakeU/FullSubNet.git"
UPSTREAM_COMMIT="e97448375cd1e883276ad583317b1828318910dc"
EXPECTED_WHEEL="ibrobot_fullsubnet-0.0.1+ibrobot.1-py3-none-any.whl"
export SOURCE_DATE_EPOCH=1739846159

BUILD_ROOT="${FULLSUBNET_WHEEL_BUILD_ROOT:-$(mktemp -d)}"
cleanup() {
    if [[ -z "${FULLSUBNET_WHEEL_BUILD_ROOT:-}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}
trap cleanup EXIT

if [[ -z "${BUILD_ROOT}" || "${BUILD_ROOT}" == "/" ]]; then
    echo "Invalid FullSubNet wheel build root: ${BUILD_ROOT}" >&2
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

rm -rf "${BUILD_ROOT}/dist"
python3 -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "${BUILD_ROOT}/dist" "${BUILD_ROOT}/source"

mkdir -p "${WHEEL_ROOT}"
cp "${BUILD_ROOT}/dist/${EXPECTED_WHEEL}" "${WHEEL_ROOT}/${EXPECTED_WHEEL}"
(
    cd "${WHEEL_ROOT}"
    sha256sum "${EXPECTED_WHEEL}" > SHA256SUMS
)
echo "Built ${WHEEL_ROOT}/${EXPECTED_WHEEL}"
