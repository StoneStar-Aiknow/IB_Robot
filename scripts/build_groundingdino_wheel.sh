#!/bin/bash
# Rebuild the audited GroundingDINO wheel from its pinned upstream revision.
#
# Produces a portable pure-Python wheel (py3-none-any) by setting
# GROUNDINGDINO_SKIP_CUDA=1 so the optional CUDA op is not compiled. The
# MultiScaleDeformableAttention forward falls back to the pure-PyTorch path
# when groundingdino._C is unavailable. Users requiring the optimized CUDA
# op may build it from the upstream source tree separately.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_ROOT="${WORKSPACE}/third_party/patches/groundingdino/313392a"
WHEEL_ROOT="${WORKSPACE}/third_party/wheels/groundingdino/313392a"
UPSTREAM_URL="https://github.com/IDEA-Research/GroundingDINO.git"
UPSTREAM_COMMIT="313392a91d987567198fea1279488370fe49d5d8"
EXPECTED_WHEEL="ibrobot_groundingdino-0.1.0+ibrobot.1-py3-none-any.whl"
export SOURCE_DATE_EPOCH=1739846159
export GROUNDINGDINO_SKIP_CUDA=1
export GROUNDINGDINO_PACKAGE_NAME=ibrobot-groundingdino

BUILD_ROOT="${GROUNDINGDINO_WHEEL_BUILD_ROOT:-$(mktemp -d)}"
cleanup() {
    if [[ -z "${GROUNDINGDINO_WHEEL_BUILD_ROOT:-}" ]]; then
        rm -rf "${BUILD_ROOT}"
    fi
}
trap cleanup EXIT

if [[ -z "${BUILD_ROOT}" || "${BUILD_ROOT}" == "/" ]]; then
    echo "Invalid GroundingDINO wheel build root: ${BUILD_ROOT}" >&2
    exit 1
fi

mkdir -p "${BUILD_ROOT}"
# source/ and dist/ are script-private scratch directories. Recreate them on
# every run so a retained GROUNDINGDINO_WHEEL_BUILD_ROOT stays reproducible and
# does not fail on the second `git clone` because of a stale source tree.
rm -rf "${BUILD_ROOT}/source" "${BUILD_ROOT}/dist"

git clone --quiet "${UPSTREAM_URL}" "${BUILD_ROOT}/source"
git -C "${BUILD_ROOT}/source" fetch --quiet origin "${UPSTREAM_COMMIT}"
git -C "${BUILD_ROOT}/source" checkout --quiet "${UPSTREAM_COMMIT}"
while IFS= read -r patch_file; do
    [[ -z "${patch_file}" ]] && continue
    git -C "${BUILD_ROOT}/source" apply "${PATCH_ROOT}/${patch_file}"
done < "${PATCH_ROOT}/series.txt"

rm -rf "${BUILD_ROOT}/dist"
python3 -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "${BUILD_ROOT}/dist" "${BUILD_ROOT}/source"

# Verify the audited wheel identity before overwriting the in-tree artifact.
# File name and SHA256 only prove that some binary was saved; the wheel must
# also carry the expected METADATA, platform tag, patched source files, and
# absence of any compiled _C extension so a mis-built artifact fails fast.
built_wheel="${BUILD_ROOT}/dist/${EXPECTED_WHEEL}"
python3 - "${built_wheel}" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
    wheel_name = next(name for name in names if name.endswith(".dist-info/WHEEL"))
    metadata = archive.read(metadata_name).decode()
    wheel_metadata = archive.read(wheel_name).decode()
    assert "Name: ibrobot-groundingdino" in metadata, "METADATA Name mismatch"
    assert "Version: 0.1.0+ibrobot.1" in metadata, "METADATA Version mismatch"
    assert "Tag: py3-none-any" in wheel_metadata, "WHEEL platform tag is not py3-none-any"
    assert "groundingdino/models/GroundingDINO/bertwarper.py" in names, "patched bertwarper.py missing"
    assert "groundingdino/models/GroundingDINO/ms_deform_attn.py" in names, "patched ms_deform_attn.py missing"
    assert not any(name.endswith((".so", ".pyd")) or "groundingdino/_C" in name for name in names), \
        "wheel unexpectedly ships a compiled _C extension"
    bertwarper_src = archive.read("groundingdino/models/GroundingDINO/bertwarper.py")
    assert b"def get_head_mask" in bertwarper_src, "patched get_head_mask helper missing"
    assert b"_bert_model_ref" in bertwarper_src, "patched BertModel reference rename missing"
print(f"Wheel content verified: {wheel}")
PY

mkdir -p "${WHEEL_ROOT}"
cp "${built_wheel}" "${WHEEL_ROOT}/${EXPECTED_WHEEL}"
(
    cd "${WHEEL_ROOT}"
    sha256sum "${EXPECTED_WHEEL}" > SHA256SUMS
)
echo "Built ${WHEEL_ROOT}/${EXPECTED_WHEEL}"
