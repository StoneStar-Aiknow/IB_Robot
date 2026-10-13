#!/usr/bin/env bash
# Regression harness for CANN detection and openEuler Torch ABI selection.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SETUP_LIB="${REPO_ROOT}/scripts/setup/python_venv.sh"
# detect_cann_version feeds the accelerator axis, so it lives in detect.sh.
DETECT_LIB="${REPO_ROOT}/scripts/setup/detect.sh"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT
WORKSPACE="${REPO_ROOT}"

# python_venv.sh resolves its dependency groups through the profile table and
# helpers in common.sh; setup.sh always sources common.sh first.
# shellcheck disable=SC1090
source "${REPO_ROOT}/scripts/setup/common.sh"
# shellcheck disable=SC1090
source "${DETECT_LIB}"
# shellcheck disable=SC1090
source "${SETUP_LIB}"

# The hooks log through these; the suites below capture stdout and stderr
# separately, so keep the loggers real but deterministic. log_info is quiet:
# its messages would pollute the captured stdout of no-op installs.
log_info() { :; }
log_warn() { echo "[WARN] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

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

detect_or_empty() {
    IBR_CANN_TOOLKIT_ROOT="$1" detect_cann_version 2>/dev/null || true
}

LATEST_ROOT="${TMP_ROOT}/ascend-toolkit/latest"
mkdir -p "${LATEST_ROOT}"
cat > "${LATEST_ROOT}/version.cfg" <<'EOF'
# CANN 8.1.RC1 layout observed on the Ascend310P1 openEuler host.
runtime_running_version=[7.7.0.1.238:8.1.RC1]
compiler_running_version=[7.7.0.1.238:8.1.RC1]
runtime_installed_version=[7.7.0.1.238:8.1.RC1]
EOF
assert_eq "merged latest/version.cfg reports the public CANN version" \
    "8.1.RC1" "$(detect_or_empty "${LATEST_ROOT}")"

RELEASE_ROOT="${TMP_ROOT}/ascend-toolkit/8.1.RC1"
mkdir -p "${RELEASE_ROOT}/aarch64-linux"
cat > "${RELEASE_ROOT}/aarch64-linux/ascend_toolkit_install.info" <<'EOF'
package_name=Ascend-cann-toolkit
version=8.1.RC1
innerversion=V100R001C21SPC001B238
arch=aarch64
os=linux
path=/usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux
EOF
assert_eq "versioned aarch64 install metadata reports CANN 8.1" \
    "8.1.RC1" "$(detect_or_empty "${RELEASE_ROOT}")"

COMPONENT_ROOT="${TMP_ROOT}/component-layout"
mkdir -p "${COMPONENT_ROOT}/runtime"
cat > "${COMPONENT_ROOT}/runtime/version.info" <<'EOF'
Version=7.7.0.1.238
version_dir=8.1.RC1
timestamp=20250428_203017761
EOF
assert_eq "component metadata ignores the internal runtime version" \
    "8.1.RC1" "$(detect_or_empty "${COMPONENT_ROOT}")"

# The CANN-dictated Torch ABI is an accelerator-axis choice now, so the
# requirements variant is selected by accels/cann.sh rather than by an
# openEuler-named helper. load_accel_impl resolves impls under SCRIPT_DIR/setup.
SCRIPT_DIR="${REPO_ROOT}/scripts"
SETUP_ACCEL_ID="cann"
load_accel_impl

accel_requirements_for() {
    IBR_CANN_TOOLKIT_ROOT="$1" accel_requirements
}
fake_cann_root() {
    local root="${TMP_ROOT}/cann-$1"
    mkdir -p "${root}"
    printf 'version=%s\n' "$1" > "${root}/version.cfg"
    printf '%s\n' "${root}"
}
assert_eq "CANN 8.1 selects its own accelerator requirements" \
    "${REPO_ROOT}/requirements/accel-cann-8.1.txt" \
    "$(accel_requirements_for "${LATEST_ROOT}")"
assert_eq "CANN 8.3 selects its own accelerator requirements" \
    "${REPO_ROOT}/requirements/accel-cann-8.3.txt" \
    "$(accel_requirements_for "$(fake_cann_root 8.3.RC1)")"
accel_requirements_rc_for() {
    local rc=0 out
    out="$(IBR_CANN_TOOLKIT_ROOT="$1" accel_requirements 2>/dev/null)" || rc=$?
    printf 'rc=%s out=%s\n' "${rc}" "${out##*/}"
}
# 8.10 must not be read as the 8.1 family and 8.30 not as 8.3; an unsupported
# release must install nothing rather than some other release's torch_npu —
# without aborting setup, which by now has LeRobot already installed.
assert_eq "CANN 8.10 does not match the CANN 8.1 family" \
    "rc=0 out=" "$(accel_requirements_rc_for "$(fake_cann_root 8.10.RC1)")"
assert_eq "CANN 8.30 does not match the CANN 8.3 family" \
    "rc=0 out=" "$(accel_requirements_rc_for "$(fake_cann_root 8.30.RC1)")"
assert_eq "CANN 8.2 installs nothing, not a neighbour's pin" \
    "rc=0 out=" "$(accel_requirements_rc_for "$(fake_cann_root 8.2.RC1)")"
assert_eq "CANN 8.5 installs nothing and lets setup continue" \
    "rc=0 out=" "$(accel_requirements_rc_for "$(fake_cann_root 8.5.RC1)")"
assert_eq "a release absent from the compatibility table lets setup continue" \
    "rc=0 out=" "$(accel_requirements_rc_for "$(fake_cann_root 7.0.0)")"
assert_eq "missing CANN installs nothing instead of guessing a release" \
    "rc=0 out=" "$(accel_requirements_rc_for "${TMP_ROOT}/absent")"

# The deferral must carry operator instructions, not just the refusal text.
cann_guidance_on_stderr() {
    IBR_CANN_TOOLKIT_ROOT="$1" accel_requirements 2>&1 >/dev/null \
        | grep -c "hiascend.com/developer/download/compatibility"
}
assert_eq "CANN 8.5 points to the Ascend compatibility table" \
    "1" "$(cann_guidance_on_stderr "$(fake_cann_root 8.5.RC1)")"
assert_eq "CANN 8.1 needs no manual guidance" \
    "0" "$(cann_guidance_on_stderr "${LATEST_ROOT}")"
assert_eq "CANN 8.3 needs no manual guidance" \
    "0" "$(cann_guidance_on_stderr "$(fake_cann_root 8.3.RC1)")"

# The same version family rule gates the global Torch ABI constraint.
assert_eq "CANN 8.1 pins its Torch ABI constraint" \
    "${REPO_ROOT}/requirements/constraints-cann-8.1.txt" \
    "$(IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" accel_pip_constraints 2>/dev/null)"
assert_eq "CANN 8.3 pins its Torch ABI constraint" \
    "${REPO_ROOT}/requirements/constraints-cann-8.3.txt" \
    "$(IBR_CANN_TOOLKIT_ROOT="$(fake_cann_root 8.3.RC1)" accel_pip_constraints 2>/dev/null)"
assert_eq "CANN 8.10 pins no Torch ABI constraint" \
    "" \
    "$(IBR_CANN_TOOLKIT_ROOT="$(fake_cann_root 8.10.RC1)" accel_pip_constraints 2>/dev/null)"

requirements_packages() {
    local line
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" || "${line}" == \#* || "${line}" == -c\ * ]] || printf '%s\n' "${line}"
    done < "$1"
}
# Every `==` pin in the constraint file must also be INSTALLED by the accel
# file. The constraint activates only after LeRobot resolves, and a constraint
# never downgrades what is already installed, so anything merely constrained
# keeps whatever LeRobot chose. This drifted once: transformers stayed at 5.5.4
# on real 310P hardware and the end-of-setup check caught it.
constraint_pins="$(grep -oE '^[A-Za-z0-9_.-]+==[^ ]+' "${REPO_ROOT}/requirements/constraints-cann-8.1.txt" | sort)"
accel_installs="$(grep -oE '^[A-Za-z0-9_.-]+==[^ ]+' "${REPO_ROOT}/requirements/accel-cann-8.1.txt" | sort)"
assert_eq "CANN 8.1 installs every version its constraint file pins" \
    "" "$(comm -23 <(printf '%s\n' "${constraint_pins}") <(printf '%s\n' "${accel_installs}") | paste -sd, -)"
assert_eq "CANN 8.1 accelerator requirements pin the complete Torch ABI" \
    "torch==2.5.1,torch_npu==2.5.1,torchvision==0.20.1,transformers==5.3.0,tokenizers==0.22.2,safetensors==0.8.0,pillow==10.3.0,opencv-python-headless==4.11.0.86,decorator,av>=15,<16" \
    "$(requirements_packages "${REPO_ROOT}/requirements/accel-cann-8.1.txt" | paste -sd, -)"
# The 8.3 pair must obey the same install-vs-contract discipline; its triple is
# from Ascend's table, so a drift here cannot be caught on hardware like 8.1's.
constraint_pins_83="$(grep -oE '^[A-Za-z0-9_.-]+==[^ ]+' "${REPO_ROOT}/requirements/constraints-cann-8.3.txt" | sort)"
accel_installs_83="$(grep -oE '^[A-Za-z0-9_.-]+==[^ ]+' "${REPO_ROOT}/requirements/accel-cann-8.3.txt" | sort)"
assert_eq "CANN 8.3 installs every version its constraint file pins" \
    "" "$(comm -23 <(printf '%s\n' "${constraint_pins_83}") <(printf '%s\n' "${accel_installs_83}") | paste -sd, -)"
assert_eq "CANN 8.3 accelerator requirements pin the complete Torch ABI" \
    "torch==2.8.0,torch_npu==2.8.0,torchvision==0.23.0,transformers==5.3.0,tokenizers==0.22.2,safetensors==0.8.0,pillow==10.3.0,opencv-python-headless==4.11.0.86,decorator,av>=15,<16" \
    "$(requirements_packages "${REPO_ROOT}/requirements/accel-cann-8.3.txt" | paste -sd, -)"
# The OS list must not carry accelerator packages: torch_npu living here is
# exactly what made an Ubuntu host with an Ascend NPU unservable.
assert_eq "the openEuler OS requirements carry no Torch or NPU packages" \
    "aiortc,onnx,onnxruntime,pygraphviz" \
    "$(requirements_packages "${REPO_ROOT}/requirements/openeuler-24.03.txt" | paste -sd, -)"

# Exercise the setup integration without invoking pip. The fake CANN root is
# selected through the same environment variable honored by a sourced CANN
# environment, and run_cmd captures the exact package arguments.
CAPTURED_COMMAND=""
run_cmd() { CAPTURED_COMMAND="$(printf '%q ' "$@")"; }
# The OS list no longer varies with CANN: the openEuler branch installs the
# same file either way, and the accelerator axis adds the Torch ABI on top.
EMPTY_ROOT="${TMP_ROOT}/no-cann"
mkdir -p "${EMPTY_ROOT}"
for cann_root in "${LATEST_ROOT}" "${EMPTY_ROOT}"; do
    ASCEND_TOOLKIT_HOME="${cann_root}" \
        install_openeuler_python_dependencies fake-python -m pip install
    assert_eq "openEuler OS dependencies are independent of the CANN release" \
        "fake-python -m pip install -r ${REPO_ROOT}/requirements/openeuler-24.03.txt --quiet " \
        "${CAPTURED_COMMAND}"
done

# install_accel_requirements is what now carries torch_npu, on every OS.
# It must also record the pending-manual-config state on the empty result:
# the hook itself runs inside a $( ) subshell and cannot set variables, and
# only cann can defer an NPU backend (cuda/cpu have nothing to defer).
ACCEL_NPU_PENDING_MANUAL_CONFIG=false
IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" \
    install_accel_requirements fake-python -m pip install
assert_eq "CANN 8.1 accelerator dependencies are passed to pip" \
    "fake-python -m pip install -r ${REPO_ROOT}/requirements/accel-cann-8.1.txt --quiet " \
    "${CAPTURED_COMMAND}"
assert_eq "CANN 8.1 does not defer the NPU runtime" \
    "false" "${ACCEL_NPU_PENDING_MANUAL_CONFIG}"

CAPTURED_COMMAND=""
ACCEL_NPU_PENDING_MANUAL_CONFIG=false
IBR_CANN_TOOLKIT_ROOT="${EMPTY_ROOT}" \
    install_accel_requirements fake-python -m pip install
assert_eq "no detected CANN toolkit invokes no pip" "" "${CAPTURED_COMMAND}"
assert_eq "no detected CANN toolkit defers the NPU runtime to manual config" \
    "true" "${ACCEL_NPU_PENDING_MANUAL_CONFIG}"

CAPTURED_COMMAND=""
ACCEL_NPU_PENDING_MANUAL_CONFIG=false
IBR_CANN_TOOLKIT_ROOT="$(fake_cann_root 8.3.RC1)" \
    install_accel_requirements fake-python -m pip install
assert_eq "CANN 8.3 accelerator dependencies are passed to pip" \
    "fake-python -m pip install -r ${REPO_ROOT}/requirements/accel-cann-8.3.txt --quiet " \
    "${CAPTURED_COMMAND}"
assert_eq "CANN 8.3 does not defer the NPU runtime" \
    "false" "${ACCEL_NPU_PENDING_MANUAL_CONFIG}"

CAPTURED_COMMAND=""
ACCEL_NPU_PENDING_MANUAL_CONFIG=false
IBR_CANN_TOOLKIT_ROOT="$(fake_cann_root 8.5.RC1)" \
    install_accel_requirements fake-python -m pip install
assert_eq "an unsupported CANN release invokes no pip and does not abort" \
    "" "${CAPTURED_COMMAND}"
assert_eq "an unsupported CANN release defers the NPU runtime to manual config" \
    "true" "${ACCEL_NPU_PENDING_MANUAL_CONFIG}"

# An accelerator with no requirements of its own must install nothing at all,
# not an empty -r invocation.
CAPTURED_COMMAND=""
( SETUP_ACCEL_ID="cpu"; load_accel_impl
  install_accel_requirements fake-python -m pip install
  [[ -z "${CAPTURED_COMMAND}" ]] ) && cpu_accel_rc=0 || cpu_accel_rc=1
assert_eq "an accelerator without its own requirements invokes no pip" "0" "${cpu_accel_rc}"

ABI_STUBS="${TMP_ROOT}/abi-stubs"
mkdir -p "${ABI_STUBS}/torch" "${ABI_STUBS}/torch_npu" "${ABI_STUBS}/torchvision"
printf '__version__ = "2.5.1"\n' > "${ABI_STUBS}/torch/__init__.py"
printf '__version__ = "2.5.1"\n' > "${ABI_STUBS}/torch_npu/__init__.py"
printf '__version__ = "0.20.1"\n' > "${ABI_STUBS}/torchvision/__init__.py"
if IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" verify_cann_torch_abi python3 >/dev/null; then
    PASS=$((PASS + 1))
    printf '  PASS  CANN 8.1 accepts the exact Torch ABI\n'
else
    FAIL=$((FAIL + 1))
    printf '  FAIL  CANN 8.1 accepts the exact Torch ABI\n' >&2
fi

printf '__version__ = "2.10.0"\n' > "${ABI_STUBS}/torch/__init__.py"
if IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" verify_cann_torch_abi python3 >/dev/null 2>&1; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  CANN 8.1 rejects a mixed Torch ABI\n' >&2
else
    PASS=$((PASS + 1))
    printf '  PASS  CANN 8.1 rejects a mixed Torch ABI\n'
fi

printf '__version__ = "2.8.0"\n' > "${ABI_STUBS}/torch/__init__.py"
printf '__version__ = "2.8.0"\n' > "${ABI_STUBS}/torch_npu/__init__.py"
printf '__version__ = "0.23.0"\n' > "${ABI_STUBS}/torchvision/__init__.py"
# No bytecode cache: a pyc written in the same second as a stub rewrite is
# still considered fresh by mtime, and the check would read the stale version.
if IBR_CANN_TOOLKIT_ROOT="$(fake_cann_root 8.3.RC1)" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" verify_cann_torch_abi python3 >/dev/null; then
    PASS=$((PASS + 1))
    printf '  PASS  CANN 8.3 accepts the exact Torch ABI\n'
else
    FAIL=$((FAIL + 1))
    printf '  FAIL  CANN 8.3 accepts the exact Torch ABI\n' >&2
fi

printf '__version__ = "2.5.1"\n' > "${ABI_STUBS}/torch/__init__.py"
if IBR_CANN_TOOLKIT_ROOT="$(fake_cann_root 8.3.RC1)" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" verify_cann_torch_abi python3 >/dev/null 2>&1; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  CANN 8.3 rejects the 8.1 Torch ABI\n' >&2
else
    PASS=$((PASS + 1))
    printf '  PASS  CANN 8.3 rejects the 8.1 Torch ABI\n'
fi

check_lerobot_python_compat() { return 0; }
check_lerobot_ros_numpy_compat() { return 0; }
CAPTURED_COMMANDS=""
run_cmd() { CAPTURED_COMMANDS+="$(printf '%q ' "$@")"$'\n'; }
# LeRobot now resolves its own dependencies on every accelerator, including
# CANN 8.1: accel_requirements overrides Torch afterwards and
# accel_pip_constraints protects it. Hand-maintained substitute lists and
# --no-deps are gone, so nothing here may install LeRobot with --no-deps.
IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" SETUP_PROFILE=full \
    install_lerobot_editable fake-python -m pip
assert_eq "CANN 8.1 installs LeRobot with its own extras, not --no-deps" \
    "0" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c -- '--no-deps')"
assert_eq "CANN 8.1 keeps the kinematics extra in the full profile" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'lerobot.*kinematics')"

# torchcodec is in LeRobot's aarch64 base deps but declares no torch
# requirement, so after a supported override its extension is ABI-broken while
# pip stays quiet. accel_post_requirements must remove it — on both releases.
CAPTURED_COMMANDS=""
IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" accel_post_requirements fake-python -m pip install
assert_eq "CANN 8.1 removes the ABI-incompatible torchcodec" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'uninstall -y torchcodec')"
CAPTURED_COMMANDS=""
IBR_CANN_TOOLKIT_ROOT="$(fake_cann_root 8.3.RC1)" accel_post_requirements fake-python -m pip install
assert_eq "CANN 8.3 removes the ABI-incompatible torchcodec" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c 'uninstall -y torchcodec')"
CAPTURED_COMMANDS=""
IBR_CANN_TOOLKIT_ROOT="${EMPTY_ROOT}" accel_post_requirements fake-python -m pip install
assert_eq "an unsupported CANN release removes nothing" "" "${CAPTURED_COMMANDS}"

# Only the trailing `install` verb may be swapped. "${arr[@]/install/uninstall}"
# rewrites the first `install` substring of every element, so an interpreter
# path containing "install" — a colcon install tree, /opt/install/... — would
# be mangled into a nonexistent binary.
CAPTURED_COMMANDS=""
IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" \
    accel_post_requirements /opt/install/ws/venv/bin/python3 -m pip install
assert_eq "the interpreter path survives the install-to-uninstall swap" \
    "1" "$(printf '%s' "${CAPTURED_COMMANDS}" | grep -c '^/opt/install/ws/venv/bin/python3 -m pip uninstall ')"

# A runner that does not end in `install` is a contract violation, not something
# to silently rewrite.
post_rc=0
( IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}"; accel_post_requirements fake-python -m pip download ) >/dev/null 2>&1 || post_rc=$?
assert_eq "a pip runner not ending in install is rejected" "1" "${post_rc}"

# Simulate transitive version drift after optional full-profile installs. The
# final check must catch it even if all three Torch imports still match.
printf '__version__ = "2.5.1"\n' > "${ABI_STUBS}/torch/__init__.py"
printf '__version__ = "2.5.1"\n' > "${ABI_STUBS}/torch_npu/__init__.py"
printf '__version__ = "0.20.1"\n' > "${ABI_STUBS}/torchvision/__init__.py"
cat > "${ABI_STUBS}/sitecustomize.py" <<'PY'
import importlib.metadata
import os

versions = {
    "torch": "2.5.1", "torch_npu": "2.5.1", "torchvision": "0.20.1",
    "transformers": os.environ.get("TEST_TRANSFORMERS_VERSION", "5.3.0"),
    "numpy": os.environ.get("TEST_NUMPY_VERSION", "1.26.4"),
    "opencv-python-headless": "4.11.0.86", "opencv-python": "4.11.0.86",
    "tokenizers": "0.22.2", "safetensors": "0.8.0", "pillow": "10.3.0",
}
importlib.metadata.version = versions.__getitem__
PY
if IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" \
    verify_cann_runtime_versions python3 >/dev/null; then
    PASS=$((PASS + 1))
    printf '  PASS  final runtime check accepts the preserved baseline\n'
else
    FAIL=$((FAIL + 1))
    printf '  FAIL  final runtime check accepts the preserved baseline\n' >&2
fi
if TEST_TRANSFORMERS_VERSION=5.5.4 IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" \
    verify_cann_runtime_versions python3 >/dev/null 2>&1; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  final runtime check rejects Transformers drift with matching Torch ABI\n' >&2
else
    PASS=$((PASS + 1))
    printf '  PASS  final runtime check rejects Transformers drift with matching Torch ABI\n'
fi
if TEST_NUMPY_VERSION=2.3.0 IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" \
    verify_cann_runtime_versions python3 >/dev/null 2>&1; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  final runtime check rejects an incomplete ROS NumPy restoration\n' >&2
else
    PASS=$((PASS + 1))
    printf '  PASS  final runtime check rejects an incomplete ROS NumPy restoration\n'
fi

echo
# A training environment deliberately keeps NumPy 2.x. Only ROS profiles
# require the restored NumPy 1.26 ABI; CANN package constraints still apply.
if SETUP_PROFILE=training TEST_NUMPY_VERSION=2.3.0 IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" \
    verify_cann_runtime_versions python3 >/dev/null; then
    assert_eq "training CANN accepts NumPy without the ROS ABI pin" "0" "0"
else
    assert_eq "training CANN accepts NumPy without the ROS ABI pin" "0" "1"
fi
if SETUP_PROFILE=inference TEST_NUMPY_VERSION=2.3.0 IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" \
    verify_cann_runtime_versions python3 >/dev/null 2>&1; then
    assert_eq "inference CANN still rejects a broken ROS NumPy ABI" "1" "0"
else
    assert_eq "inference CANN still rejects a broken ROS NumPy ABI" "1" "1"
fi
if SETUP_PROFILE=training TEST_TRANSFORMERS_VERSION=5.5.4 IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ABI_STUBS}" \
    verify_cann_runtime_versions python3 >/dev/null 2>&1; then
    assert_eq "training CANN still enforces non-ROS runtime constraints" "1" "0"
else
    assert_eq "training CANN still enforces non-ROS runtime constraints" "1" "1"
fi

echo "== summary: ${PASS} passed, ${FAIL} failed =="
[[ ${FAIL} -eq 0 ]]
