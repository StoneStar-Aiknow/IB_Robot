#!/bin/bash
# Rebuild the audited RAM++ wheel from its pinned upstream revision.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${WORKSPACE}/third_party/patches/recognize-anything/7cb804a"
WHEEL_ROOT="${WORKSPACE}/third_party/wheels/recognize-anything/7cb804a"
UPSTREAM_URL="https://github.com/xinyu1205/recognize-anything.git"
UPSTREAM_COMMIT="7cb804a8609e9f4b1a50b7f31436d2df40bb9481"
EXPECTED_WHEEL="ibrobot_ram-0.0.1+ibrobot.1-py3-none-any.whl"
export SOURCE_DATE_EPOCH=1739846159

BUILD_ROOT="${RAM_WHEEL_BUILD_ROOT:-$(mktemp -d)}"
cleanup() {
    if [[ -z "${RAM_WHEEL_BUILD_ROOT:-}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}
trap cleanup EXIT

if [[ -z "${BUILD_ROOT}" || "${BUILD_ROOT}" == "/" ]]; then
    echo "Invalid RAM++ wheel build root: ${BUILD_ROOT}" >&2
    exit 1
fi

mkdir -p "${BUILD_ROOT}"
# source/ and dist/ are script-private scratch directories. Recreate them on
# every run so a retained RAM_WHEEL_BUILD_ROOT stays reproducible and does
# not fail on the second `git clone` because of a stale source tree.
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
