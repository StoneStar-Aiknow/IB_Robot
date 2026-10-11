#!/usr/bin/env bash
# Regression harness for the setup.sh "training" dependency-scope profile.
#
# Covers issue #146: a training host must install the venv + LeRobot only, with
# no ROS 2 / rosdep / colcon / third-party patch stacks, while the existing
# full / inference profiles keep their behaviour.
#
# Everything here is hermetic: no network, no apt, no writes outside a temp
# directory. The gates are exercised by sourcing setup.sh (its main() only runs
# when executed directly) and stubbing the side-effecting helpers.
#
# Usage:
#   scripts/setup/tests/test_setup_profile_training.sh
#
# Exit status:
#   0  — all checks pass.
#   1  — at least one check failed.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

PASS=0
FAIL=0

ok() { PASS=$((PASS + 1)); printf '  PASS  %s\n' "$1"; }

ko() {
    FAIL=$((FAIL + 1))
    printf '  FAIL  %s\n' "$1" >&2
    if [[ $# -gt 1 ]]; then
        printf '        %s\n' "$2" >&2
    fi
}

assert_contains() {
    if grep -qF -- "$2" <<<"$3"; then
        ok "$1"
    else
        ko "$1" "missing: $2"
        printf '        ---- actual ----\n%s\n' "$3" >&2
    fi
}

assert_not_contains() {
    if grep -qF -- "$2" <<<"$3"; then
        ko "$1" "unexpected: $2"
        printf '        ---- actual ----\n%s\n' "$3" >&2
    else
        ok "$1"
    fi
}

assert_eq() {
    if [[ "$2" == "$3" ]]; then ok "$1"; else ko "$1" "expected '$2', got '$3'"; fi
}

# line_for <tag> <haystack>: the single tagged output line, so negative
# assertions never accidentally match another profile's line.
line_for() { grep -F -- "$1" <<<"$2" | head -1; }

probe() { ( cd "${REPO_ROOT}" && bash -c "$1" 2>&1 ); }

# ---------------------------------------------------------------------------
# 1. help text and profile whitelist
# ---------------------------------------------------------------------------
HELP_OUT="$("${REPO_ROOT}/scripts/setup.sh" --help 2>&1)"
assert_contains "help documents the training profile" "training  LeRobot training host" "${HELP_OUT}"
assert_contains "help documents the env-var override" "IBR_SETUP_PROFILE" "${HELP_OUT}"

BAD_OUT="$("${REPO_ROOT}/scripts/setup.sh" --profile bogus 2>&1)"
assert_contains "unknown profile is rejected" "Unknown setup profile: 'bogus'" "${BAD_OUT}"
assert_contains "rejection lists all three profiles" "full', 'inference', or 'training" "${BAD_OUT}"

# ---------------------------------------------------------------------------
# 2. profile predicates and validate_setup_profile()
# ---------------------------------------------------------------------------
PROBE_PREDICATES="$(probe '
set +u
source scripts/setup.sh
for p in full inference training; do
  SETUP_PROFILE=$p
  if setup_profile_is_training; then t=training; else t=other; fi
  if setup_profile_limits_submodules; then s=narrow; else s=wide; fi
  if setup_profile_skips_ros_third_party; then tp=skipped; else tp=installed; fi
  echo "PRED $p is_training=$t submodules=$s third_party=$tp"
done
SETUP_PROFILE=training
validate_setup_profile && echo "VALIDATE_TRAINING_OK"
')"
assert_contains "training is recognised" "PRED training is_training=training submodules=narrow third_party=skipped" "${PROBE_PREDICATES}"
assert_contains "inference keeps narrow submodules" "PRED inference is_training=other submodules=narrow third_party=skipped" "${PROBE_PREDICATES}"
assert_contains "full keeps the wide submodule scope" "PRED full is_training=other submodules=wide third_party=installed" "${PROBE_PREDICATES}"
assert_contains "validate_setup_profile accepts training" "VALIDATE_TRAINING_OK" "${PROBE_PREDICATES}"

PROBE_BAD="$(probe '
set +u
source scripts/setup.sh
SETUP_PROFILE=bogus
validate_setup_profile
echo "SHOULD_NOT_REACH"
')"
assert_not_contains "validate_setup_profile rejects unknown profiles" "SHOULD_NOT_REACH" "${PROBE_BAD}"

# ---------------------------------------------------------------------------
# 3. install_system_deps(): training must not touch ROS 2 / rosdep / colcon
# ---------------------------------------------------------------------------
PROBE_SYSDEPS="$(probe '
set +u
source scripts/setup.sh
USE_SUDO=false
SETUP_PLATFORM_ID=ubuntu-22.04
CALLS=""
check_ros_installation() { CALLS+="check_ros "; }
ensure_colcon() { CALLS+="ensure_colcon "; }
ensure_rosdep() { CALLS+="ensure_rosdep "; }
platform_install_rosdeps() { CALLS+="rosdeps "; }
platform_prepare_host() { CALLS+="prepare_host "; }
platform_install_python_bootstrap() { CALLS+="python_bootstrap "; }
platform_install_training_system_deps() { CALLS+="training_deps "; }
platform_training_system_deps_satisfied() { return 0; }
platform_install_benchmark_system_deps() { CALLS+="benchmark_deps "; }
platform_supports_local_workspace_build() { return 0; }
ensure_workspace_venv() { CALLS+="workspace_venv "; }
for p in training full inference; do
  SETUP_PROFILE=$p
  CALLS=""
  SYSTEM_DEPS_STATUS=pending
  install_system_deps >/dev/null 2>&1
  echo "SYSDEPS $p calls=[$CALLS] status=$SYSTEM_DEPS_STATUS"
done
')"
TRAINING_SYSDEPS="$(line_for "SYSDEPS training" "${PROBE_SYSDEPS}")"
FULL_SYSDEPS="$(line_for "SYSDEPS full" "${PROBE_SYSDEPS}")"
INFERENCE_SYSDEPS="$(line_for "SYSDEPS inference" "${PROBE_SYSDEPS}")"

assert_eq "training only prepares the host and the workspace venv" \
    "SYSDEPS training calls=[prepare_host workspace_venv ] status=done" "${TRAINING_SYSDEPS}"
assert_not_contains "training never runs check_ros_installation" "check_ros" "${TRAINING_SYSDEPS}"
assert_not_contains "training never runs ensure_colcon" "ensure_colcon" "${TRAINING_SYSDEPS}"
assert_not_contains "training never runs ensure_rosdep" "ensure_rosdep" "${TRAINING_SYSDEPS}"
assert_not_contains "training never runs platform_install_rosdeps" "rosdeps" "${TRAINING_SYSDEPS}"
assert_not_contains "training never installs benchmark system deps" "benchmark_deps" "${TRAINING_SYSDEPS}"
assert_eq "full keeps the ROS 2 + rosdep + colcon path" \
    "SYSDEPS full calls=[check_ros prepare_host python_bootstrap benchmark_deps ensure_colcon ensure_rosdep rosdeps ] status=done" "${FULL_SYSDEPS}"
assert_eq "inference keeps the ROS 2 + rosdep + colcon path" \
    "SYSDEPS inference calls=[check_ros prepare_host python_bootstrap benchmark_deps ensure_colcon ensure_rosdep rosdeps ] status=done" "${INFERENCE_SYSDEPS}"

# ---------------------------------------------------------------------------
# 4. dependency preview
# ---------------------------------------------------------------------------
PROBE_PREVIEW="$(probe '
set +u
source scripts/setup.sh
SETUP_PLATFORM_ID=ubuntu-22.04
for p in training full; do
  SETUP_PROFILE=$p
  echo "PREVIEW $p [$(preview_system_packages)]"
done
')"
TRAINING_PREVIEW="$(line_for "PREVIEW training" "${PROBE_PREVIEW}")"
FULL_PREVIEW="$(line_for "PREVIEW full" "${PROBE_PREVIEW}")"
assert_eq "training preview lists only venv/pip/ffmpeg" \
    "PREVIEW training [python3-venv python3-pip ffmpeg]" "${TRAINING_PREVIEW}"
assert_not_contains "training preview omits colcon" "colcon" "${TRAINING_PREVIEW}"
assert_contains "full preview still lists colcon" "python3-colcon-common-extensions" "${FULL_PREVIEW}"

# ---------------------------------------------------------------------------
# 5. LeRobot extras: training keeps dataset/policy deps, drops teleop kinematics
# ---------------------------------------------------------------------------
PROBE_EXTRAS="$(probe '
set +u
source scripts/setup.sh
IBR_CANN_TOOLKIT_ROOT=/nonexistent-cann-root
INSTALL_BENCHMARK_DEPS=false
check_lerobot_python_compat() { return 0; }
check_lerobot_ros_numpy_compat() { return 0; }
CAPTURED=""
run_cmd() { CAPTURED+="$(printf "%q " "$@") ; "; }
for p in training full inference; do
  SETUP_PROFILE=$p
  CAPTURED=""
  install_lerobot_editable fake-python -m pip >/dev/null 2>&1
  echo "EXTRAS $p [$CAPTURED]"
done
')"
TRAINING_EXTRAS="$(line_for "EXTRAS training" "${PROBE_EXTRAS}")"
FULL_EXTRAS="$(line_for "EXTRAS full" "${PROBE_EXTRAS}")"
INFERENCE_EXTRAS="$(line_for "EXTRAS inference" "${PROBE_EXTRAS}")"

assert_contains "training installs LeRobot editable with policy/dataset extras" \
    "libs/lerobot\[smolvla" "${TRAINING_EXTRAS}"
assert_contains "training keeps the dataset extra" "deepdiff-dep\]" "${TRAINING_EXTRAS}"
assert_not_contains "training drops the teleop kinematics extra" "kinematics" "${TRAINING_EXTRAS}"
assert_contains "full keeps the kinematics extra" "kinematics" "${FULL_EXTRAS}"
assert_not_contains "inference drops the kinematics extra" "kinematics" "${INFERENCE_EXTRAS}"

# ---------------------------------------------------------------------------
# 6. submodule scope
# ---------------------------------------------------------------------------
PROBE_SUBMODULES="$(probe '
set +u
source scripts/setup.sh
log_info() { :; }
log_warn() { :; }
log_done() { :; }
ask_yn() { return 0; }
# Force the uninitialized state so the probe does not depend on whether
# libs/lerobot happens to be checked out in this workspace.
submodule_update_block_reason() { echo "not initialized"; }
CAPTURED=""
run_cmd() { CAPTURED+="$(printf "%q " "$@") ; "; }
for p in training full; do
  SETUP_PROFILE=$p
  CAPTURED=""
  update_submodules >/dev/null 2>&1
  echo "SUBMODULES $p [$CAPTURED]"
done
')"
TRAINING_SUBMODULES="$(line_for "SUBMODULES training" "${PROBE_SUBMODULES}")"
FULL_SUBMODULES="$(line_for "SUBMODULES full" "${PROBE_SUBMODULES}")"
assert_contains "training initialises libs/lerobot" \
    "git submodule update --init --recursive libs/lerobot" "${TRAINING_SUBMODULES}"
assert_not_contains "training never initialises the lidar submodule" "livox_ros_driver2" "${TRAINING_SUBMODULES}"
assert_contains "full still initialises every submodule" "src/livox_ros_driver2" "${FULL_SUBMODULES}"

# ---------------------------------------------------------------------------
# 7. verify_env.sh preconditions
# ---------------------------------------------------------------------------
PROBE_VERIFY_TRAINING="$(probe '
set +u
SETUP_PROFILE=training
WORKSPACE="$(pwd)"
VENV_PYTHON=/usr/bin/python3
SETUP_PLATFORM_ID=ubuntu-22.04
USE_SUDO=false
unset ROSDEP_BIN
unset SETUP_ROS_SETUP_PATH
source scripts/setup/verify_env.sh
declare -F verify_training_env >/dev/null && echo "HAS_VERIFY_TRAINING_ENV"
')"
assert_contains "verify_env.sh sources for training without rosdep/ROS paths" \
    "HAS_VERIFY_TRAINING_ENV" "${PROBE_VERIFY_TRAINING}"

PROBE_VERIFY_FULL="$(probe '
set +u
SETUP_PROFILE=full
WORKSPACE="$(pwd)"
VENV_PYTHON=/usr/bin/python3
SETUP_PLATFORM_ID=ubuntu-22.04
USE_SUDO=false
unset ROSDEP_BIN
unset SETUP_ROS_SETUP_PATH
source scripts/setup/verify_env.sh
echo "SHOULD_NOT_REACH"
')"
assert_not_contains "full keeps the fail-closed ROSDEP_BIN precondition" \
    "SHOULD_NOT_REACH" "${PROBE_VERIFY_FULL}"
assert_contains "full reports the missing ROSDEP_BIN" "ROSDEP_BIN" "${PROBE_VERIFY_FULL}"

# ---------------------------------------------------------------------------
# 8. training requirements file
# ---------------------------------------------------------------------------
assert_contains "requirements/training.txt exists with TensorBoard" \
    "tensorboard" "$(cat "${REPO_ROOT}/requirements/training.txt" 2>&1)"

# ---------------------------------------------------------------------------
# 9. .shrc_local degradation on a ROS-less training host
# ---------------------------------------------------------------------------
STUB_WS="${TMP_ROOT}/ws"
mkdir -p "${STUB_WS}/venv/bin" "${STUB_WS}/venv/lib/python3.10/site-packages"
printf 'training\n' >"${STUB_WS}/venv/.ibrobot_setup_profile"
printf '# stub activation\n' >"${STUB_WS}/venv/bin/activate"
cp "${REPO_ROOT}/.shrc_local" "${STUB_WS}/.shrc_local"

cat >"${TMP_ROOT}/shrc_probe.sh" <<PROBE
source "${STUB_WS}/.shrc_local"
echo "SRC_EXIT=\$?"
PROBE
cat >"${TMP_ROOT}/shrc_probe_interactive.sh" <<PROBE
source "${STUB_WS}/.shrc_local"
echo "INTERACTIVE_DONE"
PROBE

SHRL_OUT="$(cd "${STUB_WS}" && bash --noprofile --norc "${TMP_ROOT}/shrc_probe.sh" 2>&1)" || true
assert_contains ".shrc_local sources cleanly without ROS 2 / install / CANN" "SRC_EXIT=0" "${SHRL_OUT}"
assert_not_contains ".shrc_local prints no errors for a training profile" "[ERROR]" "${SHRL_OUT}"

SHRL_IT="$(cd "${STUB_WS}" && bash --noprofile --norc -i "${TMP_ROOT}/shrc_probe_interactive.sh" 2>&1)" || true
assert_contains "interactive banner labels the training profile" "(profile: training)" "${SHRL_IT}"
assert_contains "interactive banner records the setup profile step" "setup profile: training" "${SHRL_IT}"
assert_contains "missing install overlay is expected for training" "training profile 下按预期缺失" "${SHRL_IT}"
assert_contains "training hint mentions lerobot-train" "lerobot-train" "${SHRL_IT}"
assert_not_contains "training hint does not advertise colcon aliases" "常用指令: cb" "${SHRL_IT}"

if [[ -f /opt/ros/humble/setup.sh ]]; then
    assert_contains "host-installed ROS 2 is still picked up" "ROS 2 Humble" "${SHRL_IT}"
else
    assert_contains "missing ROS 2 is reported as expected for training" "– ROS 2 Humble" "${SHRL_IT}"
fi

# The preflight is exercised for real here (the probes above stub it): it must
# test the capability the install path relies on, so "skip apt" can never be a
# false positive. Regression: `python3 -m venv --help` succeeds on Debian and
# Ubuntu even when python3-venv -- and therefore ensurepip -- is missing.
PF_ROOT="${TMP_ROOT}/preflight"
mkdir -p "${PF_ROOT}/bin-ok" "${PF_ROOT}/bin-noensurepip" "${PF_ROOT}/bin-nopip"
mkdir -p "${PF_ROOT}/ws-empty" "${PF_ROOT}/ws-with-venv/venv"
for d in bin-ok bin-noensurepip bin-nopip; do
    printf '#!/bin/sh\nexit 0\n' >"${PF_ROOT}/${d}/ffmpeg"
    chmod +x "${PF_ROOT}/${d}/ffmpeg"
done
cat >"${PF_ROOT}/bin-noensurepip/python3" <<'STUB'
#!/bin/sh
case "$*" in
    *ensurepip*) exit 1 ;;
    *) exit 0 ;;
esac
STUB
cat >"${PF_ROOT}/bin-nopip/python3" <<'STUB'
#!/bin/sh
case "$*" in
    "-m pip"*) exit 1 ;;   # no system pip (must not match the ensurepip probe)
    *) exit 0 ;;
esac
STUB
chmod +x "${PF_ROOT}/bin-noensurepip/python3" "${PF_ROOT}/bin-nopip/python3"

preflight_probe() {  # $1 = stub bin dir, $2 = workspace
    PATH="$1:${PATH}" probe '
set +u
source scripts/setup.sh
WORKSPACE="'"$2"'"
if python3 -m venv --help >/dev/null 2>&1; then echo "OLD_PROBE=passes"; else echo "OLD_PROBE=fails"; fi
if python3 -m pip --version >/dev/null 2>&1; then echo "OLD_PIP_PROBE=passes"; else echo "OLD_PIP_PROBE=fails"; fi
platform_training_system_deps_satisfied && echo "PREDICATE=skip-apt" || echo "PREDICATE=run-apt"
'
}
PF_OK="$(preflight_probe "${PF_ROOT}/bin-ok" "${PF_ROOT}/ws-empty")"
assert_contains "a ready host skips the install" "PREDICATE=skip-apt" "${PF_OK}"
PF_NOENSUREPIP="$(preflight_probe "${PF_ROOT}/bin-noensurepip" "${PF_ROOT}/ws-empty")"
assert_contains "host without ensurepip runs the install" "PREDICATE=run-apt" "${PF_NOENSUREPIP}"
assert_contains "the old venv --help probe would have passed (regression)" \
    "OLD_PROBE=passes" "${PF_NOENSUREPIP}"
PF_EXISTING="$(preflight_probe "${PF_ROOT}/bin-noensurepip" "${PF_ROOT}/ws-with-venv")"
assert_contains "an existing venv needs no installer" "PREDICATE=skip-apt" "${PF_EXISTING}"
PF_NOPIP="$(preflight_probe "${PF_ROOT}/bin-nopip" "${PF_ROOT}/ws-empty")"
assert_contains "a missing system pip does not force an install" "PREDICATE=skip-apt" "${PF_NOPIP}"
assert_contains "the preflight no longer probes system pip" "OLD_PIP_PROBE=fails" "${PF_NOPIP}"

# ---------------------------------------------------------------------------
printf '\n%s passed, %s failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]
