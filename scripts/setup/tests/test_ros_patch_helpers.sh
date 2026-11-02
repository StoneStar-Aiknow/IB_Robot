#!/usr/bin/env bash
# Exercise the existing ROS patch helpers without network or ROS dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/setup/ros_third_party.sh"
log_info() { printf '%s\n' "$*"; }
log_error() { printf '%s\n' "$*" >&2; }
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
git init -q "${TMP}/source"
git -C "${TMP}/source" config user.name 'Patch test'
git -C "${TMP}/source" config user.email 'patch-test@localhost'
git -C "${TMP}/source" config core.hooksPath /dev/null
git -C "${TMP}/source" config commit.gpgsign false
printf 'base\n' > "${TMP}/source/file"
git -C "${TMP}/source" add file
git -C "${TMP}/source" commit -qm base
BASE="$(git -C "${TMP}/source" rev-parse HEAD)"
printf 'patched\n' > "${TMP}/source/file"
git -C "${TMP}/source" add file
git -C "${TMP}/source" commit -qm patch
git -C "${TMP}/source" format-patch -1 --stdout > "${TMP}/change.patch"
git -C "${TMP}/source" checkout -q --detach "${BASE}"

_require_ros_submodule_commit "${TMP}/source" "${BASE}" fixture
_apply_ros_patch_once "${TMP}/source" "${TMP}/change.patch"
test "$(cat "${TMP}/source/file")" = patched
test "$(git -C "${TMP}/source" rev-parse HEAD)" = "${BASE}"
git -C "${TMP}/source" diff --binary > "${TMP}/before.diff"
_apply_ros_patch_once "${TMP}/source" "${TMP}/change.patch"
git -C "${TMP}/source" diff --binary > "${TMP}/after.diff"
cmp "${TMP}/before.diff" "${TMP}/after.diff"
echo 'PASS: mailbox patch applied idempotently without moving HEAD'

printf 'developer work\n' > "${TMP}/source/untracked"
_apply_ros_patch_once "${TMP}/source" "${TMP}/change.patch"
test "$(cat "${TMP}/source/untracked")" = 'developer work'
printf 'conflicting developer work\n' > "${TMP}/source/file"
if _apply_ros_patch_once "${TMP}/source" "${TMP}/change.patch"; then
    echo 'FAIL: conflicting edit accepted' >&2; exit 1
fi
test "$(cat "${TMP}/source/file")" = 'conflicting developer work'
echo 'PASS: unrelated untracked file and conflicting edit preserved'

if _require_ros_submodule_commit "${TMP}/source" "0000000000000000000000000000000000000000" fixture; then
    echo 'FAIL: wrong base accepted' >&2; exit 1
fi
if _require_ros_submodule_commit "${TMP}/missing" "${BASE}" fixture; then
    echo 'FAIL: missing source accepted' >&2; exit 1
fi
echo 'PASS: wrong base and missing source rejected'

# Exercise the actual dispatcher with harmless stand-ins for unrelated packages.
WORKSPACE="${TMP}/workspace"
mkdir -p "${WORKSPACE}/third_party/patches/usb_cam/0.8.1"
_require_ros_submodule_commit() { printf 'base %s %s\n' "$1" "$2" >> "${TMP}/calls"; }
_apply_ros_patch_once() { printf 'patch %s %s\n' "$1" "$2" >> "${TMP}/calls"; }
_migrate_fast_lio_build_patch() { :; }
_ensure_fast_calib_patch_stack() { :; }
log_done() { :; }
ensure_ros_third_party_patch_stacks
grep -Fx "base ${WORKSPACE}/src/usb_cam cb2ae6bc0a312de629f84fef933fb3e07dac1ef8" "${TMP}/calls"
grep -Fx "patch ${WORKSPACE}/src/usb_cam ${WORKSPACE}/third_party/patches/usb_cam/0.8.1/0001-upstream-pr378-raw-mjpeg-routing.patch" "${TMP}/calls"
echo 'PASS: ROS dispatcher wires the single usb_cam patch directly'
