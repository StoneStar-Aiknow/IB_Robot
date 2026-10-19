#!/usr/bin/env bash
# Regression harness for requirements/graspgenx.txt.
#
# GraspGenX is installed from an audited wheel with --no-deps, behind its own
# accelerator hook (accel_supports_torch_graspgenx: cuda and cann). The cann
# accelerator never runs install_graspgen_pip, so requirements/manipulation.txt
# is not there to supply GraspGenX's imports. That made the install succeed and
# the smoke test's `import graspgenx.grasp_server` abort setup on exactly the
# Ascend host the hook exists to serve.
#
# This suite walks the shipped wheel's module-level import closure and asserts
# requirements/graspgenx.txt still covers it, so a wheel bump that reaches for
# a new module cannot quietly reintroduce that abort. It reads the wheel with
# zipfile and parses it with ast; no wheel code is imported or executed, and
# nothing here needs a GPU, an Ascend device or a populated venv.
#
# What it cannot see: an import inside a function or a branch. The PTV3
# backbone is reached that way (generator.py imports ptv3_vanilla inside its
# `elif object_backbone == "ptv3vanilla"` branch), so addict and timm never
# appear in this closure even though serving a request needs them. They come
# from requirements/perception.txt, which the full profile installs on every
# platform. A deferred import that is NOT covered elsewhere has to be named in
# requirements/graspgenx.txt by hand.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
REQUIREMENTS="${REPO_ROOT}/requirements/graspgenx.txt"
BASE_REQUIREMENTS="${REPO_ROOT}/requirements/base.txt"

PASS=0
FAIL=0

assert_eq() {
    local name="$1"
    local expected="$2"
    local actual="$3"

    if [[ "${actual}" == "${expected}" ]]; then
        printf '  PASS  %s\n' "${name}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected: %s\n' "${expected}" >&2
        printf '    actual:   %s\n' "${actual}" >&2
        FAIL=$((FAIL + 1))
    fi
}

wheels=("${REPO_ROOT}"/third_party/wheels/graspgenx/*/graspgenx-*.whl)
if [[ ! -f "${wheels[0]}" ]]; then
    echo "[ERROR] no graspgenx wheel under third_party/wheels/graspgenx/" >&2
    exit 1
fi
if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "[ERROR] expected one graspgenx wheel, found ${#wheels[@]}: ${wheels[*]}" >&2
    exit 1
fi
WHEEL="${wheels[0]}"

# numpy, tqdm, PyYAML and torch are core dependencies of LeRobot and the colcon
# toolchain, which every profile installs before this one; scipy comes from
# requirements/base.txt. Anything else the closure reaches has to be named in
# requirements/graspgenx.txt itself.
GUARANTEED="numpy tqdm pyyaml torch"

echo "== requirements/graspgenx.txt covers the wheel's import closure =="

missing="$(
    python3 -I - "${WHEEL}" "${REQUIREMENTS}" "${BASE_REQUIREMENTS}" "${GUARANTEED}" <<'PY'
import ast
import re
import sys
import zipfile

wheel, requirements, base, guaranteed = sys.argv[1:5]

ENTRY = "graspgenx.grasp_server"
# Import name -> distribution name, where they differ.
ALIAS = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python"}

with zipfile.ZipFile(wheel) as archive:
    sources = {name for name in archive.namelist() if name.endswith(".py")}

    def source_for(module):
        """Return the archive member holding ``module``, or None."""
        flat = module.replace(".", "/") + ".py"
        package = module.replace(".", "/") + "/__init__.py"
        if flat in sources:
            return flat
        return package if package in sources else None

    external, walked, pending = set(), set(), [ENTRY]
    while pending:
        module = pending.pop()
        if module in walked:
            continue
        walked.add(module)
        member = source_for(module)
        if member is None:
            continue
        tree = ast.parse(archive.read(member).decode("utf-8"), member)
        # Only module-level imports: anything inside a try/except or a function
        # is guarded or deferred, and does not decide whether the import of
        # graspgenx.grasp_server succeeds.
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".")[0] == "graspgenx":
                    pending.append(name)
                    # `from graspgenx.pkg import submodule` reaches pkg too.
                    if source_for(name) is None and "." in name:
                        pending.append(name.rsplit(".", 1)[0])
                else:
                    external.add(name.split(".")[0])


def declared(path):
    names = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            entry = line.split("#")[0].strip()
            if entry and not entry.startswith("-"):
                names.add(re.split(r"[\[<>=!~;]", entry)[0].strip().lower())
    return names


available = declared(requirements) | declared(base) | set(guaranteed.split())
needed = {ALIAS.get(name, name).lower() for name in external if name not in sys.stdlib_module_names}
print(",".join(sorted(needed - available)))
PY
)"

assert_eq "every module-level import of grasp_server is installable on cann" "" "${missing}"

echo "== the accelerator axis stays out of this file =="

# Torch belongs to the accelerator axis (requirements/accel-*.txt and
# install_graspgen_torch_abi), never to a feature's requirements file.
assert_eq "no Torch build is pinned here" \
    "0" "$(grep -cE '^(torch|torch_npu|torchvision|torchaudio)([[<>=!~]|$)' "${REQUIREMENTS}" || true)"

echo
if [[ "${FAIL}" -eq 0 ]]; then
    echo "All ${PASS} assertions passed."
    exit 0
fi
echo "${FAIL} of $((PASS + FAIL)) assertions failed."
exit 1
