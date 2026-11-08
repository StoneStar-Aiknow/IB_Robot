#!/bin/bash
# Copyright 2026 IB-Robot contributors
# SPDX-License-Identifier: Apache-2.0
# Run only on .5 after the unmodified feature deployment succeeds. No motion.
set -euo pipefail
[ "$#" = 1 ] || { echo "usage: $0 /root/x2-feat-cbf-safety-pkg-SHA"; exit 2; }
PKG=$(realpath "$1")
WT=/root/x2-feat-cbf-safety-wt
RUN=/root/x2-feat-cbf-safety-run
case "$PKG" in /root/x2-feat-cbf-safety-pkg-*) ;; *) echo "wrong feature package"; exit 2;; esac
[ "$(id -u)" = 0 ] || { echo "run on the board as root"; exit 2; }
[ -f "$PKG/kit/build-env.sh" ] && [ -f "$PKG/code/HEAD" ] || exit 2
EXPECTED=$(cat "$PKG/code/HEAD")
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || exit 2
[ "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED" ] || { echo "package/tree SHA mismatch"; exit 3; }
[ "$(git -C "$WT" branch --show-current)" = feat/x2-cbf-safety ] || exit 3
exec 9>/root/x2-voice-run/x2-voice.lock
flock -n 9 || { echo "board build/test lock is occupied"; exit 4; }
if pgrep -af '[r]os2 launch'; then echo "robot launch is running; do not test"; exit 4; fi
OUT=$RUN/test/$(date +%Y%m%d-%H%M%S)-cbf-formal
mkdir -p "$OUT"
echo "$EXPECTED" > "$OUT/HEAD"
echo "CBF formal tests: $EXPECTED -> $OUT"
set +e
env -i HOME=/root USER=root TERM=xterm PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash --noprofile --norc -c '
set -o pipefail
pkg=$1 out=$2
source "$pkg/kit/build-env.sh" || exit 90
set -euo pipefail
[ "$ROS_DOMAIN_ID" = 81 ] && [ "$ROS_LOCALHOST_ONLY" = 1 ]
cd /root/x2-feat-cbf-safety-wt
export ROS_LOG_DIR=$out/ros_log
# Match the default openEuler package discovery in scripts/build.sh. These
# unavailable simulation/calibration packages are not CBF test dependencies.
platform_test_args=(--packages-ignore fast_calib sim_models)
# deploy.sh intentionally uses the unmodified dev/no-test workspace mixin.
# Enable the C++ tests of this package through the normal build entry point.
export MAKEFLAGS=-j4
nice -n 10 ./scripts/build.sh --mixin dev test -- --packages-select aimdk_dataplane --parallel-workers 1
colcon test --merge-install "${platform_test_args[@]}" --packages-select aimdk_dataplane --event-handlers console_direct+
colcon test-result --test-result-base build/aimdk_dataplane --verbose
test -n "$(find build/aimdk_dataplane -name 'test_cbf_kernel*.xml' -print -quit)"
test -n "$(find build/aimdk_dataplane -name 'test_cbf_safety*.xml' -print -quit)"
cp -a build/aimdk_dataplane/test_results "$out/cpp-test-results"
build/aimdk_dataplane/cbf_pose_probe > "$out/neutral-model-candidate.json"
build/aimdk_dataplane/cbf_pose_probe --replay 1000 > "$out/adapter-replay.json"
build/aimdk_dataplane/cbf_pose_probe --left-hand-only > "$out/neutral-left-only-candidate.json"
build/aimdk_dataplane/cbf_pose_probe --replay 1000 --left-hand-only > "$out/adapter-replay-left-only.json"
# This file also contains unrelated Gazebo scene tests. The board gate covers
# its CBF configuration cases, including exact disabled-path equivalence.
colcon test --merge-install "${platform_test_args[@]}" --packages-select robot_config --pytest-args "$PWD/src/robot_config/test/test_runtime_provider_platform.py" " -k" cbf --event-handlers console_direct+
colcon test-result --test-result-base build/robot_config --verbose
test -s build/robot_config/pytest.xml
cp build/robot_config/pytest.xml "$out/robot_config-pytest.xml"
colcon test --merge-install "${platform_test_args[@]}" --packages-select aimdk_robot --pytest-args "$PWD/src/robots/aimdk/aimdk_robot/test/test_runtime_audio_launch.py" --event-handlers console_direct+
colcon test-result --test-result-base build/aimdk_robot --verbose
test -s build/aimdk_robot/pytest.xml
cp build/aimdk_robot/pytest.xml "$out/aimdk_robot-pytest.xml"
' _ "$PKG" "$OUT" 2>&1 | tee "$OUT/colcon.log"
rc=${PIPESTATUS[0]}
set -e
echo "$rc" > "$OUT/exit_code"
echo "CBF_COLCON: rc=$rc log=$OUT/colcon.log"
exit "$rc"
