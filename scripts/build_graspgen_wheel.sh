#!/bin/bash
# Rebuild the audited GraspGen (grasp_gen) wheel from its pinned upstream revision.
#
# Produces a pure-Python wheel (py3-none-any) of grasp_gen with its gripper
# definitions and meshes packaged inside it. Upstream resolves config/grippers
# and assets/ next to the package directory, which only works from a source
# checkout; the patch moves them into the package so an installed wheel works.
# The CUDA pointnet2_ops extension is built separately by
# scripts/build_pointnet2ops_wheel.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${WORKSPACE}/third_party/patches/graspgen/a56d518"
WHEEL_ROOT="${WORKSPACE}/third_party/wheels/graspgen/a56d518"
UPSTREAM_URL="${GRASPGEN_GIT_URL:-https://github.com/NVlabs/GraspGen.git}"
UPSTREAM_COMMIT="a56d518f3b76ea2a432b5b838b3c68027d29be49"
EXPECTED_WHEEL="grasp_gen-1.0.0+ibrobot.1-py3-none-any.whl"
export SOURCE_DATE_EPOCH=1739846159
# The upstream build requires setuptools-scm; the patched version is static.
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0+ibrobot.1

BUILD_ROOT="${GRASPGEN_WHEEL_BUILD_ROOT:-$(mktemp -d)}"
cleanup() {
    if [[ -z "${GRASPGEN_WHEEL_BUILD_ROOT:-}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}
trap cleanup EXIT

if [[ -z "${BUILD_ROOT}" || "${BUILD_ROOT}" == "/" ]]; then
    echo "Invalid GraspGen wheel build root: ${BUILD_ROOT}" >&2
    exit 1
fi

mkdir -p "${BUILD_ROOT}"
rm -rf "${BUILD_ROOT}/source" "${BUILD_ROOT}/dist"

git clone --quiet --no-checkout "${UPSTREAM_URL}" "${BUILD_ROOT}/source"
git -C "${BUILD_ROOT}/source" -c advice.detachedHead=false checkout --quiet "${UPSTREAM_COMMIT}"
while IFS= read -r patch_file; do
    [[ -z "${patch_file}" ]] && continue
    git -C "${BUILD_ROOT}/source" apply "${PATCH_ROOT}/${patch_file}"
done < "${PATCH_ROOT}/series.txt"
# Build without .git so setuptools-scm cannot pull unrelated tracked files in.
rm -rf "${BUILD_ROOT}/source/.git"

python3 -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "${BUILD_ROOT}/dist" "${BUILD_ROOT}/source"

# Verify the audited wheel identity before overwriting the in-tree artifact.
built_wheel="${BUILD_ROOT}/dist/${EXPECTED_WHEEL}"
python3 - "${built_wheel}" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    metadata = archive.read(next(name for name in names if name.endswith(".dist-info/METADATA"))).decode()
    wheel_metadata = archive.read(next(name for name in names if name.endswith(".dist-info/WHEEL"))).decode()
    assert "Name: grasp_gen" in metadata, "METADATA Name mismatch"
    assert "Version: 1.0.0+ibrobot.1" in metadata, "METADATA Version mismatch"
    assert "Tag: py3-none-any" in wheel_metadata, "WHEEL platform tag is not py3-none-any"
    for required in (
        "grasp_gen/LICENSE",
        "grasp_gen/LICENSE_ASSETS.md",
        "grasp_gen/config/grippers/robotiq_2f_140.yaml",
        "grasp_gen/config/grippers/robotiq_2f_140.py",
        "grasp_gen/assets/robotiq/robotiq_140_collision.obj",
        "grasp_gen/assets/panda_gripper/hand.stl",
        "grasp_gen/assets/suction/suction_cup.obj",
    ):
        assert required in names, f"{required} missing"
    outside = sorted(name for name in names if not name.startswith(("grasp_gen/", "grasp_gen-")))
    assert not outside, f"wheel ships files outside the grasp_gen package: {outside[:5]}"
    assert not any(name.endswith((".so", ".pyd", ".pth")) for name in names), "wheel ships binaries or weights"
print(f"Wheel content verified: {wheel}")
PY

mkdir -p "${WHEEL_ROOT}"
cp "${built_wheel}" "${WHEEL_ROOT}/${EXPECTED_WHEEL}"
cp "${PATCH_ROOT}/LICENSE" "${PATCH_ROOT}/LICENSE_ASSETS.md" "${WHEEL_ROOT}/"
(
    cd "${WHEEL_ROOT}"
    sha256sum "${EXPECTED_WHEEL}" > SHA256SUMS
)
echo "Built ${WHEEL_ROOT}/${EXPECTED_WHEEL}"
