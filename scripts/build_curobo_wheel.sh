#!/bin/bash
# Rebuild the audited cuRobo wheel from its pinned upstream revision.
#
# Produces a pure-Python wheel (py3-none-any): cuRobo 0.8 compiles its CUDA
# kernels at run time through cuda.core NVRTC, so no extension is built and the
# same wheel installs on the x86_64 workstation and the aarch64 X2 Orin. The
# wheel is built from a checkout without .git, so only the files the patched
# package-data names are packaged: the code, the kernel sources and the solver
# configurations, not the tests, examples or the example robots' meshes.
#
# CUROBO_GIT_URL may name a local clone holding the pinned commit when GitHub
# is unreachable; the commit is fetched by hash either way.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${WORKSPACE}/third_party/patches/curobo/4ea7736"
WHEEL_ROOT="${WORKSPACE}/third_party/wheels/curobo/4ea7736"
UPSTREAM_URL="${CUROBO_GIT_URL:-https://github.com/NVlabs/curobo.git}"
UPSTREAM_COMMIT="4ea77366ca48ee453e7df139e39fa6532af49f3b"
EXPECTED_WHEEL="nvidia_curobo-0.8.0+ibrobot.1-py3-none-any.whl"
export SOURCE_DATE_EPOCH=1776540438

BUILD_ROOT="${CUROBO_WHEEL_BUILD_ROOT:-$(mktemp -d)}"
cleanup() {
    if [[ -z "${CUROBO_WHEEL_BUILD_ROOT:-}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}
trap cleanup EXIT

if [[ -z "${BUILD_ROOT}" || "${BUILD_ROOT}" == "/" ]]; then
    echo "Invalid cuRobo wheel build root: ${BUILD_ROOT}" >&2
    exit 1
fi

mkdir -p "${BUILD_ROOT}"
rm -rf "${BUILD_ROOT}/git" "${BUILD_ROOT}/source" "${BUILD_ROOT}/dist"
mkdir -p "${BUILD_ROOT}/source"

git init --quiet "${BUILD_ROOT}/git"
git -C "${BUILD_ROOT}/git" fetch --quiet --depth 1 "${UPSTREAM_URL}" "${UPSTREAM_COMMIT}"
git -C "${BUILD_ROOT}/git" --work-tree="${BUILD_ROOT}/source" \
    checkout --quiet FETCH_HEAD -- curobo pyproject.toml setup.py README.md LICENSE
while IFS= read -r patch_file; do
    [[ -z "${patch_file}" ]] && continue
    git -C "${BUILD_ROOT}/source" apply "${PATCH_ROOT}/${patch_file}"
done < "${PATCH_ROOT}/series.txt"

# setup.py builds the deprecated pybind extensions only when asked to.
CUROBO_USE_PYBIND=0 python3 -m pip wheel --no-deps --no-build-isolation \
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
    assert "Name: nvidia-curobo" in metadata, "METADATA Name mismatch"
    assert "Version: 0.8.0+ibrobot.1" in metadata, "METADATA Version mismatch"
    assert "Tag: py3-none-any" in wheel_metadata, "WHEEL platform tag is not py3-none-any"
    assert any(name.endswith(".dist-info/LICENSE") or "/licenses/LICENSE" in name for name in names), \
        "Apache-2.0 LICENSE missing from the wheel"
    for required in ("curobo/motion_planner.py", "curobo/content/configs/task/README.md"):
        assert required in names, f"{required} missing"
    assert any(name.startswith("curobo/_src/curobolib/kernels/") and name.endswith(".cuh") for name in names), \
        "NVRTC kernel sources missing"
    assert any(name.startswith("curobo/content/configs/task/") and name.endswith(".yml") for name in names), \
        "solver task configurations missing"
    outside = sorted(name for name in names if not name.startswith(("curobo/", "nvidia_curobo-")))
    assert not outside, f"wheel ships files outside the curobo package: {outside[:5]}"
    excluded = sorted(
        name for name in names
        if name.startswith(("curobo/tests/", "curobo/examples/", "curobo/content/assets/"))
    )
    assert not excluded, f"wheel ships tests, examples or example-robot assets: {excluded[:5]}"
    assert not any(name.endswith((".so", ".pyd", ".obj", ".stl", ".dae", ".usd")) for name in names), \
        "wheel unexpectedly ships binaries or meshes"
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
