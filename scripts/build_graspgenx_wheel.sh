#!/bin/bash
# Rebuild the audited GraspGenX wheel from its pinned upstream revision.
#
# Produces a pure-Python wheel (py3-none-any) of the graspgenx package only.
# The upstream repository tracks demo meshes and renders in Git LFS; none of
# them belong to the package, so LFS is kept out of every git call and the
# wheel is built from a checkout without .git (setuptools-scm would otherwise
# run `git archive` over the whole repository and fetch every LFS object).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${WORKSPACE}/third_party/patches/graspgenx/b942909"
WHEEL_ROOT="${WORKSPACE}/third_party/wheels/graspgenx/b942909"
UPSTREAM_URL="${GRASPGENX_GIT_URL:-https://github.com/NVlabs/GraspGenX.git}"
UPSTREAM_COMMIT="b9429097728cb1c430dd78b92edf17ba318aad03"
EXPECTED_WHEEL="graspgenx-1.0.0+ibrobot.2-py3-none-any.whl"
export SOURCE_DATE_EPOCH=1739846159

BUILD_ROOT="${GRASPGENX_WHEEL_BUILD_ROOT:-$(mktemp -d)}"
cleanup() {
    if [[ -z "${GRASPGENX_WHEEL_BUILD_ROOT:-}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}
trap cleanup EXIT

if [[ -z "${BUILD_ROOT}" || "${BUILD_ROOT}" == "/" ]]; then
    echo "Invalid GraspGenX wheel build root: ${BUILD_ROOT}" >&2
    exit 1
fi

mkdir -p "${BUILD_ROOT}"
rm -rf "${BUILD_ROOT}/git" "${BUILD_ROOT}/source" "${BUILD_ROOT}/dist"
mkdir -p "${BUILD_ROOT}/source"

git_no_lfs=(git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false)
git init --quiet "${BUILD_ROOT}/git"
GIT_LFS_SKIP_SMUDGE=1 "${git_no_lfs[@]}" -C "${BUILD_ROOT}/git" fetch --quiet --depth 1 \
    "${UPSTREAM_URL}" "${UPSTREAM_COMMIT}"
GIT_LFS_SKIP_SMUDGE=1 "${git_no_lfs[@]}" -C "${BUILD_ROOT}/git" --work-tree="${BUILD_ROOT}/source" \
    checkout --quiet FETCH_HEAD -- graspgenx pyproject.toml README.md LICENSE
while IFS= read -r patch_file; do
    [[ -z "${patch_file}" ]] && continue
    git -C "${BUILD_ROOT}/source" apply "${PATCH_ROOT}/${patch_file}"
done < "${PATCH_ROOT}/series.txt"

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
    assert "Name: graspgenx" in metadata, "METADATA Name mismatch"
    assert "Version: 1.0.0+ibrobot.2" in metadata, "METADATA Version mismatch"
    assert "Tag: py3-none-any" in wheel_metadata, "WHEEL platform tag is not py3-none-any"
    assert any(name.endswith(".dist-info/LICENSE") or "/licenses/LICENSE" in name for name in names), \
        "Apache-2.0 LICENSE missing from the wheel"
    assert "graspgenx/grasp_server.py" in names, "graspgenx package missing"
    outside = sorted(name for name in names if not name.startswith(("graspgenx/", "graspgenx-")))
    assert not outside, f"wheel ships files outside the graspgenx package: {outside[:5]}"
    assert not any(name.endswith((".so", ".pyd", ".obj", ".npy", ".pth")) for name in names), \
        "wheel unexpectedly ships binaries or assets"
print(f"Wheel content verified: {wheel}")
PY

mkdir -p "${WHEEL_ROOT}"
cp "${built_wheel}" "${WHEEL_ROOT}/${EXPECTED_WHEEL}"
cp "${PATCH_ROOT}/LICENSE" "${WHEEL_ROOT}/LICENSE"
(
    cd "${WHEEL_ROOT}"
    sha256sum "${EXPECTED_WHEEL}" > SHA256SUMS
)
echo "Built ${WHEEL_ROOT}/${EXPECTED_WHEEL}"
