#!/usr/bin/env bash
# verify_runtime_independence.sh — the two-direction independence gate
# (robot-runtime-packaging spec, design D9; so101-runtime-migration task 6.4,
# lekiwi-runtime-migration task 7.5).
#
# Usage:
#   scripts/verify_runtime_independence.sh runtime so101_robot [--profile so101_single_arm]
#   scripts/verify_runtime_independence.sh runtime lekiwi_robot [--profile lekiwi_lidar]
#   scripts/verify_runtime_independence.sh runtime aimdk_robot [--profile x2_ultra]
#   scripts/verify_runtime_independence.sh runtime unitree_g1_robot [--profile g1_29dof_dex3]
#   scripts/verify_runtime_independence.sh core
#
# runtime mode: build <robot>_robot in an ISOLATED workspace containing only its
#   declared dependency closure. Three closures are distinguished
#   (sim-platform-registry D10, computed by scripts/runtime_closure.py from the
#   workspace package.xml files with catkin_pkg):
#   - the EXECUTION closure (exec_depend/depend/run_depend/build_export_depend,
#     transitive) is what the FIXED allow-list audits (runtime members + the
#     contract layer + vendored third-party); membership is NOT derived from the
#     closure itself, and it must exclude sim_models and robot_runtime_sim;
#   - the BUILD closure (execution edges + build_depend/buildtool_depend)
#     populates the isolated workspace, so no simulation package is built or
#     installed unless a runtime edge pulls it in;
#   - the TEST closure (build + test_depend) would be used where the gate runs
#     package tests; this gate runs none (the conformance suite is driven from
#     outside the isolated workspace), so it is only logged for diagnosis.
#   The isolated build uses the workspace venv and the project's colcon flags
#   (merge-install, symlink-install, the repo's mixin file) so it matches
#   ./scripts/build.sh. Inside the isolated run the gate asserts, BEFORE any
#   conformance case, that robot_runtime_sim and sim_models are invisible
#   (no importable module, no ibrobot.sim_platforms entry point, no ament share
#   directory), so a leak from the full worktree install fails the gate instead
#   of passing silently. The runtime is then launched through its own entry in
#   simulated transport and the capability-scoped conformance suite runs
#   against it.
#
# core mode: build the generic IB-Robot packages with every robot runtime,
#   SDK, driver adapter, motion package, robot suite, robot description and
#   planning-framework package excluded, then actually launch
#   robot_config/robot.launch.py with the mock runtime provider and verify the
#   contract chain: RuntimeStatus reconciliation, a mock motion service call,
#   and the stop service.
#
# Both modes run in a temporary workspace (package dirs are symlinked; C++
# requires real copies, so colcon copies via symlink-install only for python).
set -eo pipefail

MODE="${1:-}"
shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_WORKSPACE="${MAIN_WORKSPACE:-$(dirname "$SCRIPT_DIR")}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
# The workspace venv, or an explicit one. A worktree that shares the main
# repository's venv (the documented worktree flow) has no venv/ of its own, so
# RUNTIME_GATE_PYTHON points the gate at the interpreter it should use.
GATE_PYTHON="${RUNTIME_GATE_PYTHON:-$MAIN_WORKSPACE/venv/bin/python3}"
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Fixed allow-lists (so101-runtime-migration): membership is declared here,
# never derived from the closure under audit.
# ---------------------------------------------------------------------------
CONTRACT_PACKAGES=(ibrobot_msgs robot_runtime)
# Packages the contract packages themselves depend on. The simulation code
# moved to robot_runtime_sim (sim-platform-registry #191): the contract layer
# depends on no simulator package, so a sim package in an execution closure is
# a dependency leak and fails the gate (see SIM_EXCLUDED below).
CONTRACT_DEPENDENCIES=()
SO101_RUNTIME_MEMBERS=(
  so101_robot so101_sdk so101_hardware so101_description so101_motion so101_suite
  feetech_sdk
)
LEKIWI_RUNTIME_MEMBERS=(
  lekiwi_robot lekiwi_sdk lekiwi_hardware lekiwi_description lekiwi_calibration
  feetech_sdk
  # The LeKiwi arm IS an SO-101 arm, so its runtime reuses these rather than
  # forking them. Declared by name, never by an so101_* prefix rule, so an
  # unrelated SO-101 package appearing in the closure still fails the gate.
  so101_description so101_sdk so101_hardware so101_motion
  # The SO-101 grasp providers declared by lekiwi_nav_grasp's grasp_execution
  # (openspec cleanup-manipulation-so101-providers D4): a dependency leaf, so
  # it joins the closure without the manipulation/inference packages.
  so101_suite
)
# The X2 runtime is a bridge onto the vendor MC tier: no SDK, hardware adapter,
# description or motion package of its own. Its in-tree C++ data plane
# (aimdk_dataplane, launched by runtime.launch.py and exec_depended by
# aimdk_robot) and its X2-owned interface package (aimdk_interfaces) are
# runtime members. aimdk_msgs is supplied by the developer's AimDK overlay
# and is never vendored into this repository.
AIMDK_RUNTIME_MEMBERS=(aimdk_robot aimdk_dataplane aimdk_interfaces)
AIMDK_EXTERNAL_PACKAGES=(aimdk_msgs)
# The G1 runtime is a bridge onto the vendor DDS tier: it owns the official
# vendor SDK adapter (unitree_g1_adapter, the single unitree_sdk2py boundary)
# and the vendored robot description, all in-tree — no external overlay,
# unlike aimdk. The vendored message packages (unitree_hg/unitree_go/
# unitree_api) were removed in Issue #171 PR 2: the vendor frames come from
# the official unitree_sdk2py through the adapter.
UNITREE_RUNTIME_MEMBERS=(
  unitree_g1_robot unitree_g1_description unitree_g1_adapter
)
# Third-party/vendored packages that may appear in a runtime closure without
# being IB-Robot generic packages (they are not listed by `colcon list` in the
# main workspace src/ tree anyway; this list documents intent).
# usb_cam joined with so101_robot's exec_depend on the vendored camera driver
# (same vendored-third-party class as pymoveit2/livox).
THIRD_PARTY_ALLOWED=(pymoveit2 aimdk_msgs livox_ros_driver2 fast_lio fast_calib usb_cam)

# Generic packages that must build WITHOUT any robot package (core gate).
CORE_EXCLUDED_SUFFIXES=("_robot" "_sdk" "_hardware" "_motion" "_suite")
CORE_EXCLUDED_PACKAGES=(
  so101_description moveit lekiwi_description lekiwi_calibration
  livox_ros_driver2 fast_lio fast_calib rosclaw pymoveit2
)
# The contract layer is part of the generic core, not a robot package:
# ibrobot_msgs + robot_runtime (the mock runtime provider IS robot_runtime's
# reference implementation) and, since #191 moved the sim implementations out
# of the contract, robot_runtime_sim + sim_models (robot_config's launch
# builders import robot_runtime_sim at module level, and robot_runtime_sim
# exec_depends sim_models for its scenes). Without them the core launch only
# ever worked through the full-worktree install leaking into the run
# environment — exactly what the scrubbed isolated run exists to catch.
CORE_BUILD_PACKAGES=(
  ibrobot_msgs robot_runtime robot_runtime_sim sim_models
  robot_config embodied_bringup manipulation_execution skill_library skill_catalog
  task_dispatch action_dispatch inference_service dataset_tools robot_navigation
  robot_teleop safety_guard embodied_agent embodied_common
)

list_workspace_packages() {
  (cd "$MAIN_WORKSPACE" && colcon list --base-paths src --names-only 2>/dev/null)
}

package_dir_for() {
  local pkg="$1" dir
  dir="$(cd "$MAIN_WORKSPACE" && colcon list --base-paths src 2>/dev/null | awk -F'\t' -v p="$pkg" '$1==p {print $2}')"
  [[ -n "$dir" ]] || return 1
  echo "$MAIN_WORKSPACE/$dir"
}

# ---------------------------------------------------------------------------
# Closures (sim-platform-registry D10): scripts/runtime_closure.py follows one
# dependency-tag class per kind through the workspace package.xml files
# (catkin_pkg, the same parser colcon uses). Each kind has exactly one use:
# execution -> the allow-list audit; build -> the isolated workspace contents;
# test -> package tests, which this gate does not run (logged for diagnosis).
# ---------------------------------------------------------------------------
SIM_EXCLUDED=(sim_models robot_runtime_sim)

runtime_closure() {
  local kind="$1"; shift
  local out
  if ! out="$("$GATE_PYTHON" "$SCRIPT_DIR/runtime_closure.py" --kind "$kind" --base-path "$MAIN_WORKSPACE/src" "$@")"; then
    log_error "cannot compute the ${kind} closure of $* (runtime_closure.py failed)"
    exit 1
  fi
  printf '%s\n' "$out"
}

# ---------------------------------------------------------------------------
# Scrubbed isolated environment: a fresh subshell whose PYTHONPATH /
# AMENT_PREFIX_PATH / COLCON_PREFIX_PATH / LD_LIBRARY_PATH hold ONLY the gate
# interpreter's own site-packages (the venv layer the interpreter would see
# anyway — /opt/ros' ros2 entry points use #!/usr/bin/python3, which does NOT
# inherit the venv, so the venv's jsonschema must stay reachable through
# PYTHONPATH exactly as it is in a normal workspace run), /opt/ros, the
# isolated install prefix and (aimdk) the vendor overlay — never the main
# worktree's own install/build trees. Without this, the full-worktree paths
# the gate itself runs under would satisfy every import and the isolation
# assertions below would test nothing.
# ---------------------------------------------------------------------------
GATE_PYTHON_SITE=""
compute_gate_python_site() {
  if [[ -z "$GATE_PYTHON_SITE" ]]; then
    GATE_PYTHON_SITE="$("$GATE_PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
  fi
  return 0
}

scrub_environment() {
  unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH
  compute_gate_python_site
  export PYTHONPATH="$GATE_PYTHON_SITE"
  source /opt/ros/humble/setup.sh
  # The AimDK vendor overlay (aimdk mode only): the C++ data plane's CMake and
  # the runtime both need aimdk_msgs from it.
  if [[ -n "${VENDOR_OVERLAY_SETUP:-}" ]]; then
    source "$VENDOR_OVERLAY_SETUP"
  fi
  return 0
}

run_in_isolated_env() {
  local ws="$1"; shift
  (
    set +u
    scrub_environment
    source "$ws/install/setup.sh"
    set -u
    "$@"
  )
}

# The in-run isolation assertions (D10): before any conformance case runs, the
# isolated environment must not see the simulation packages at all — not as an
# importable module, not as a registered sim platform, not as an ament
# package. A leak here means the "isolated" run is silently using the full
# worktree's install and every later pass is void.
assert_simulation_invisible() {
  local ws="$1"
  run_in_isolated_env "$ws" "$GATE_PYTHON" - "$ws" <<'PY'
import importlib.metadata
import importlib.util
import sys

ws = sys.argv[1]
failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


for module in ("robot_runtime_sim", "sim_models"):
    check(importlib.util.find_spec(module) is None, f"module {module!r} is importable in the isolated run")

try:
    entry_points = [ep.name for ep in importlib.metadata.entry_points(group="ibrobot.sim_platforms")]
except Exception as exc:  # any discovery error is itself a gate failure, reported verbatim
    entry_points = []
    failures.append(f"entry point discovery raised {exc!r}")
check(not entry_points, f"ibrobot.sim_platforms entry points are registered: {entry_points}")

from ament_index_python import get_package_share_directory

for package in ("robot_runtime_sim", "sim_models"):
    try:
        share = get_package_share_directory(package)
    except Exception:
        continue
    check(False, f"ament share of {package!r} resolves in the isolated run: {share}")

if failures:
    for line in failures:
        print(f"ISOLATION FAILURE: {line}")
    sys.exit(1)
print(f"isolation OK: no sim module, no sim platform entry point, no sim share dir visible from {ws}")
PY
}

# ---------------------------------------------------------------------------
# Isolated build environment: /opt/ros + the workspace venv, project colcon
# flags (mirrors scripts/build.sh: merge-install, symlink-install, repo mixin).
# ---------------------------------------------------------------------------
isolated_colcon_build() {
  local ws="$1"; shift
  local venv_python="$GATE_PYTHON"
  [[ -x "$venv_python" ]] || {
    log_error "gate interpreter not found at $venv_python"
    log_error "set RUNTIME_GATE_PYTHON to the venv python when the worktree shares another venv"
    exit 1
  }
  (
    cd "$ws"
    set +u
    # Scrubbed build environment, same as the run below: colcon records every
    # prefix visible at build time into install/setup.sh's prefix chain, so a
    # worktree install inherited from the gate's own environment would be
    # re-sourced by every later "isolated" run and defeat the gate.
    scrub_environment
    set -u
    # Same flags as scripts/build.sh (dev mixin): merge + symlink install.
    PYTHONNOUSERSITE=1 "$venv_python" -m colcon build \
      --merge-install --symlink-install \
      --parallel-workers "$(nproc)" \
      --cmake-args -Wno-dev \
      "$@" >"$ws/build.log" 2>&1
  ) || { log_error "isolated build failed (see $ws/build.log)"; tail -40 "$ws/build.log"; exit 1; }
}

# ---------------------------------------------------------------------------
# runtime mode
# ---------------------------------------------------------------------------
runtime_gate() {
  local robot_pkg="$1"; shift
  local profile=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile) profile="$2"; shift 2;;
      *) log_error "unknown option: $1"; exit 2;;
    esac
  done

  # Per-robot members and profile location. Membership is declared here, never
  # derived from the closure under audit.
  local members=() profile_dir="" default_profile=""
  case "$robot_pkg" in
    so101_robot)
      members=("${SO101_RUNTIME_MEMBERS[@]}")
      profile_dir="$MAIN_WORKSPACE/src/robots/so101/so101_robot/profiles"
      default_profile="so101_single_arm"
      ;;
    lekiwi_robot)
      members=("${LEKIWI_RUNTIME_MEMBERS[@]}")
      profile_dir="$MAIN_WORKSPACE/src/robots/lekiwi/lekiwi_robot/profiles"
      default_profile="lekiwi_lidar"
      ;;
    aimdk_robot)
      members=("${AIMDK_RUNTIME_MEMBERS[@]}")
      profile_dir="$MAIN_WORKSPACE/src/robots/aimdk/aimdk_robot/profiles"
      default_profile="x2_ultra"
      # The vendor overlay is a developer-supplied dependency. Without it this
      # gate cannot run; say so by name instead of failing or silently passing.
      # AIMDK_OVERLAY_SETUP may point at the overlay's local_setup.sh so the
      # isolated workspace can see aimdk_msgs too.
      if [[ -n "${AIMDK_OVERLAY_SETUP:-}" && ! -f "${AIMDK_OVERLAY_SETUP}" ]]; then
        log_error "AIMDK_OVERLAY_SETUP is set but not a file: ${AIMDK_OVERLAY_SETUP}"
        exit 1
      fi
      if ! python3 -c "import aimdk_msgs" >/dev/null 2>&1 && [[ -z "${AIMDK_OVERLAY_SETUP:-}" ]]; then
        log_warn "SKIPPED: the AimDK overlay is not on the ROS 2 path (aimdk_msgs missing)."
        log_warn "Build and source the vendor workspace, then re-run:"
        log_warn "  cd ~/aimdk && colcon build && source install/local_setup.bash"
        log_warn "  (or set AIMDK_OVERLAY_SETUP=<aimdk>/install/local_setup.sh)"
        exit 0
      fi
      VENDOR_OVERLAY_SETUP="${AIMDK_OVERLAY_SETUP:-}"
      if [[ -z "$VENDOR_OVERLAY_SETUP" ]]; then
        # aimdk_msgs is importable through the environment this gate runs in,
        # but the isolated build/run below is scrubbed of that environment, so
        # the overlay must be re-sourced by name. Derive its local_setup.sh
        # from the ament prefix aimdk_msgs resolves through right now.
        VENDOR_OVERLAY_SETUP="$(python3 - <<'PY' 2>/dev/null || true
import os

prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
for prefix in prefixes:
    if prefix.endswith("/aimdk_msgs") and os.path.isfile(os.path.join(prefix, "..", "local_setup.sh")):
        print(os.path.abspath(os.path.join(prefix, "..", "local_setup.sh")))
        break
PY
)"
        if [[ -n "$VENDOR_OVERLAY_SETUP" ]]; then
          log_info "aimdk overlay derived from the environment: ${VENDOR_OVERLAY_SETUP}"
        else
          log_error "aimdk_msgs is importable but no overlay local_setup.sh was found;"
          log_error "set AIMDK_OVERLAY_SETUP=<aimdk>/install/local_setup.sh and re-run"
          exit 1
        fi
      fi
      ;;
    unitree_g1_robot)
      members=("${UNITREE_RUNTIME_MEMBERS[@]}")
      profile_dir="$MAIN_WORKSPACE/src/robots/unitree/unitree_g1_robot/profiles"
      default_profile="g1_29dof_dex3"
      ;;
    *)
      log_error "no runtime allow-list declared for '${robot_pkg}'"
      log_error "add one to this script rather than relaxing the audit."
      exit 2
      ;;
  esac
  profile="${profile:-$default_profile}"

  local allowlist=("${CONTRACT_PACKAGES[@]}" "${CONTRACT_DEPENDENCIES[@]}" "${members[@]}" "${THIRD_PARTY_ALLOWED[@]}")

  log_info "runtime gate: computing dependency closure of ${robot_pkg}"
  local closure
  closure="$(runtime_closure execution "$robot_pkg")"
  [[ -n "$closure" ]] || { log_error "cannot resolve closure for ${robot_pkg}"; exit 1; }
  grep -Fx "$robot_pkg" <<<"$closure" >/dev/null || { log_error "${robot_pkg} not in its own closure"; exit 1; }

  # The execution closure is a runtime-dependency statement: the simulation
  # packages must not be reachable through any runtime edge (D10). This is a
  # separate, harder assertion than the allow-list below — a leak here is a
  # package.xml bug even when the package would sit in some allow-list.
  local excluded
  for excluded in "${SIM_EXCLUDED[@]}"; do
    if grep -Fx "$excluded" <<<"$closure" >/dev/null; then
      log_error "execution closure of ${robot_pkg} contains ${excluded}:"
      log_error "  a runtime edge (exec_depend/depend/run_depend/build_export_depend)"
      log_error "  reaches a simulation package; that is a dependency leak, not an allow-list matter"
      exit 1
    fi
  done

  # Audit against the FIXED allow-list (not the closure itself).
  local offenders=()
  while IFS= read -r pkg; do
    local allowed=false
    for entry in "${allowlist[@]}"; do
      [[ "$pkg" == "$entry" ]] && allowed=true
    done
    # Packages not listed by colcon (third-party system deps) are fine.
    if [[ "$allowed" == false ]] && list_workspace_packages | grep -Fx "$pkg" >/dev/null; then
      offenders+=("$pkg")
    fi
  done <<<"$closure"
  if [[ ${#offenders[@]} -gt 0 ]]; then
    log_error "execution closure of ${robot_pkg} contains non-member workspace packages:"
    printf '  - %s\n' "${offenders[@]}"
    log_error "allowed members: ${allowlist[*]}"
    exit 1
  fi
  log_info "closure audit OK: $(wc -l <<<"$closure") packages, fixed allow-list honored"

  # Isolated workspace: the BUILD closure (execution edges + build edges, no
  # test edges), so a test_depend on a simulation package never puts it into
  # the built install. The test closure is logged for diagnosis; the gate runs
  # no package tests inside this workspace (the conformance suite is driven
  # from outside, against the launched runtime).
  local build_closure test_closure
  build_closure="$(runtime_closure build "$robot_pkg")"
  test_closure="$(runtime_closure test "$robot_pkg")"
  log_info "build closure: $(wc -l <<<"$build_closure") packages; test closure (unused, no package tests run here): $(wc -l <<<"$test_closure")"

  local tmp
  tmp="$(mktemp -d /tmp/runtime_independence_XXXXXX)"
  mkdir -p "$tmp/src"
  trap 'rm -rf "$tmp"' EXIT
  while IFS= read -r pkg; do
    local dir
    dir="$(package_dir_for "$pkg")" || { log_error "no source dir for $pkg"; exit 1; }
    ln -s "$dir" "$tmp/src/$pkg"
  done <<<"$build_closure"
  # Pinned repo-level libraries under libs/ (Livox-SDK2 for livox).
  [[ -d "$MAIN_WORKSPACE/libs" ]] && ln -s "$MAIN_WORKSPACE/libs" "$tmp/libs"

  log_info "building isolated workspace: $tmp (${robot_pkg} build closure)"
  isolated_colcon_build "$tmp"

  # Before any conformance case: the isolated run must not see the simulation
  # packages at all (import, entry point, ament share). This is the check that
  # catches a leak from the full worktree install or a sim package sneaking
  # into the built workspace.
  log_info "asserting simulation packages are invisible in the isolated run"
  assert_simulation_invisible "$tmp"

  # Launch the runtime through its own entry and run conformance.
  local profile_path="${profile_dir}/${profile}.yaml"
  [[ -f "$profile_path" ]] || { log_error "profile not found: $profile_path"; exit 1; }
  export CONFORMANCE_LAUNCH="ros2 launch ${robot_pkg} runtime.launch.py profile:=${profile} simulated:=true"
  export CONFORMANCE_PROFILE="$profile_path"
  export CONFORMANCE_STARTUP_TIMEOUT="${CONFORMANCE_STARTUP_TIMEOUT:-120}"
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-110}" ROS_LOCALHOST_ONLY=1

  log_info "running conformance against ${robot_pkg} (profile=${profile}, simulated transport)"
  (
    cd "$tmp"
    set +u
    scrub_environment
    source "$tmp/install/setup.sh"
    set -u
    cd "$MAIN_WORKSPACE"
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$GATE_PYTHON" -m pytest \
      "$MAIN_WORKSPACE/src/robot_runtime/test/test_conformance.py" -q -p no:cacheprovider
  )
  log_info "runtime gate PASSED for ${robot_pkg} (fixed allow-list, isolated build, simulated conformance)"
}

# ---------------------------------------------------------------------------
# core mode
# ---------------------------------------------------------------------------
core_gate() {
  local tmp
  tmp="$(mktemp -d /tmp/core_independence_XXXXXX)"
  mkdir -p "$tmp/src"
  trap 'rm -rf "$tmp"' EXIT

  log_info "core gate: building the generic set with robot packages excluded"
  local ignore_count=0
  local pkg name rel_dir
  # colcon list prints name <TAB> path <TAB> build_type: read the path through
  # awk -F'\t' (like package_dir_for), because a plain two-variable read lets
  # rel_dir swallow "path<TAB>build_type" and every symlink below is broken —
  # colcon then builds zero packages and "succeeds".
  while IFS= read -r line; do
    pkg="$(awk -F'\t' '{print $1}' <<<"$line")"
    rel_dir="$(awk -F'\t' '{print $2}' <<<"$line")"
    name="$pkg"
    local skip=false
    for suffix in "${CORE_EXCLUDED_SUFFIXES[@]}"; do
      [[ "$name" == *"$suffix" ]] && skip=true
    done
    for pkg_name in "${CORE_EXCLUDED_PACKAGES[@]}"; do
      [[ "$name" == "$pkg_name" ]] && skip=true
    done
    if [[ "$skip" == "true" ]]; then
      ignore_count=$((ignore_count + 1))
      continue
    fi
    ln -s "$MAIN_WORKSPACE/$rel_dir" "$tmp/src/$name"
  done < <(cd "$MAIN_WORKSPACE" && colcon list --base-paths src 2>/dev/null)
  log_info "excluded $ignore_count robot-specific packages from the temp workspace"

  # Build selection = the core roots' build closure, minus the excluded set:
  # a scrubbed build chain-sources every dependency's package.sh, so a
  # dependency left unbuilt fails the build (or, before the scrub, silently
  # resolved through the full worktree install — the leak this gate exists to
  # catch). Derived mechanically from CORE_BUILD_PACKAGES so it cannot drift.
  local core_select=() core_closure=""
  core_closure="$(runtime_closure build "${CORE_BUILD_PACKAGES[@]}")"
  while IFS= read -r pkg; do
    local skip=false
    for suffix in "${CORE_EXCLUDED_SUFFIXES[@]}"; do
      [[ "$pkg" == *"$suffix" ]] && skip=true
    done
    for pkg_name in "${CORE_EXCLUDED_PACKAGES[@]}"; do
      [[ "$pkg" == "$pkg_name" ]] && skip=true
    done
    [[ "$skip" == "true" ]] && continue
    core_select+=("$pkg")
  done <<<"$core_closure"
  log_info "core build set: ${core_select[*]}"

  isolated_colcon_build "$tmp" --packages-select "${core_select[@]}"
  log_info "generic build OK without robot packages"

  # Contract-chain smoke: mock-runtime provider launch + reconciliation + a
  # mock motion service call + the stop service. The run environment is
  # scrubbed to /opt/ros + this isolated install only, so the contract chain
  # (which includes the simulation-layer packages robot_config legitimately
  # exec_depends) is proven from the isolated build, never from the full
  # worktree install the gate itself runs under.
  log_info "core gate: launching robot.launch.py with the mock runtime provider"
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-111}" ROS_LOCALHOST_ONLY=1
  (
    cd "$tmp"
    set +u
    scrub_environment
    source install/setup.sh
    set -u
    python3 - "$MAIN_WORKSPACE" "$tmp" <<'PY'
import os
import subprocess
import sys
import time

import rclpy
from ibrobot_msgs.srv import ComputeIk, GetRuntimeStatus, StopRuntime
from rclpy.node import Node

main_ws = sys.argv[1]
tmp_dir = sys.argv[2]


def await_future(node, future, what, timeout_s=30.0):
    """Spin until the future resolves or the deadline passes; never spin forever."""
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    assert future.done(), f"{what} did not answer within {timeout_s:.0f}s"
    return future.result()


# 1) The generic launch must compose and reconcile against the mock runtime.
#    Launch output goes to a file: a PIPE that is never read deadlocks the
#    launch process once the kernel pipe buffer fills (same pattern as
#    isolated_colcon_build's build.log).
launch_log_path = os.path.join(tmp_dir, "core_gate_launch.log")
launch_log = open(launch_log_path, "w", encoding="utf-8")
launch = subprocess.Popen(
    ["ros2", "launch", "robot_config", "robot.launch.py",
     "robot_config:=so101_mock_core_gate", "use_sim:=false",
     "control_mode:=teleop"],
    stdout=launch_log, stderr=subprocess.STDOUT, text=True)
try:
    rclpy.init()
    node = Node("core_gate_probe")
    deadline = time.monotonic() + 120
    status = None
    client = node.create_client(GetRuntimeStatus, "/runtime/get_status")
    while time.monotonic() < deadline and status is None:
        if client.service_is_ready():
            future = client.call_async(GetRuntimeStatus.Request())
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.2)
            if future.done() and future.result() is not None:
                status = future.result().status
                break
        rclpy.spin_once(node, timeout_sec=0.5)
    assert status is not None, "mock runtime never reached reconciliation"
    assert status.lifecycle == "ACTIVE", f"lifecycle {status.lifecycle}"
    assert status.runtime_name == "mock_runtime", status.runtime_name

    # 2) One mock motion call: ComputeFk then ComputeIk through the contract.
    from ibrobot_msgs.srv import ComputeFk
    from sensor_msgs.msg import JointState
    fk = node.create_client(ComputeFk, "/motion/compute_fk")
    assert fk.wait_for_service(timeout_sec=30), "mock /motion/compute_fk unavailable"
    fkreq = ComputeFk.Request()
    fkreq.joint_state = JointState()
    fkreq.joint_state.name = [f"joint_{i}" for i in range(1, 6)]
    fkreq.joint_state.position = [0.1, 0.2, 0.3, 0.2, 0.1]
    fkreq.link_names = ["ee"]
    fkres = await_future(node, fk.call_async(fkreq), "mock /motion/compute_fk")
    assert fkres is not None and fkres.success and fkres.poses, "mock FK call failed"

    ik = node.create_client(ComputeIk, "/motion/compute_ik")
    assert ik.wait_for_service(timeout_sec=30), "mock /motion/compute_ik unavailable"
    req = ComputeIk.Request()
    req.target = fkres.poses[0]
    req.seed = JointState()
    req.seed.name = [f"joint_{i}" for i in range(1, 6)]
    req.seed.position = [0.1, 0.2, 0.3, 0.2, 0.1]
    ikres = await_future(node, ik.call_async(req), "mock /motion/compute_ik")
    assert ikres is not None and ikres.success, "mock IK call failed"

    # 3) StopRuntime(HOLD) actually engages through the contract: the call
    #    must succeed, report its latencies, and the post-stop status must
    #    show the latched stop, the STOPPED lifecycle and the idle mode.
    stop = node.create_client(StopRuntime, "/runtime/stop")
    assert stop.wait_for_service(timeout_sec=10), "mock /runtime/stop unavailable"
    stop_req = StopRuntime.Request()
    stop_req.policy = "HOLD"
    stop_res = await_future(node, stop.call_async(stop_req), "mock /runtime/stop")
    assert stop_res is not None and stop_res.success, f"mock stop failed: {stop_res}"
    assert stop_res.cancel_latency_s >= 0.0 and stop_res.idle_latency_s >= 0.0, (
        f"stop latencies not reported: {stop_res}"
    )
    after = await_future(
        node, client.call_async(GetRuntimeStatus.Request()), "mock /runtime/get_status after stop"
    ).status
    assert after.stop_latched, "stop did not latch"
    assert after.lifecycle == "STOPPED", f"lifecycle {after.lifecycle} after stop"
    assert after.active_mode == "idle", f"active_mode {after.active_mode} after stop"

    node.destroy_node()
    rclpy.shutdown()
    launch_log.flush()
    print("core gate: reconciliation ACTIVE + mock ComputeFk/ComputeIk + StopRuntime(HOLD) confirmed")
finally:
    launch.terminate()
    try:
        launch.wait(timeout=15)
    except subprocess.TimeoutExpired:
        launch.kill()
    # The launch output is the only diagnosis when reconciliation never
    # happens; without this the gate fails with a bare assertion and no
    # indication of which node refused to come up.
    if sys.exc_info()[0] is not None:
        output = launch.stdout.read() if launch.stdout else ""
        tail = output.splitlines()[-60:]
        print("--- launch output (tail) ---", file=sys.stderr)
        print("\n".join(tail), file=sys.stderr)
PY
  ) || {
    log_error "core gate contract-chain check failed; tail of the launch log follows"
    tail -60 "$tmp/core_gate_launch.log" 2>/dev/null || true
    exit 1
  }
  log_info "core gate PASSED (generic core independent of robot packages)"
}

# ---------------------------------------------------------------------------
case "$MODE" in
  runtime)
    [[ $# -ge 1 ]] || { log_error "usage: $0 runtime <robot>_robot [--profile NAME]"; exit 2; }
    runtime_gate "$@";;
  core)
    core_gate "$@";;
  *)
    log_error "usage: $0 runtime <robot>_robot | core"
    exit 2;;
esac
