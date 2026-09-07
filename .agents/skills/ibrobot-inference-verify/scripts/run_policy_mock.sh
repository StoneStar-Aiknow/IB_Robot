#!/bin/zsh
# Launch a policy mock stack and wait for the first inference.
#
# MUST run under zsh: sourcing .shrc_local breaks under bash + `set -u`
# (ROS setup scripts are not nounset-safe).
#
# Usage: run_policy_mock.sh <robot.yaml> <ros_domain_id> <logfile> [timeout_s]
# Pass criteria: "First inference received" appears in <logfile>.
set -e
YAML="$1"; DOMAIN="$2"; LOG="$3"; TIMEOUT="${4:-420}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
source .shrc_local >/dev/null 2>&1
export ROS_DOMAIN_ID="$DOMAIN"
ros2 launch robot_config robot.launch.py \
  config_path:="$YAML" \
  control_mode:=model_inference use_sim:=true >"$LOG" 2>&1 &
LPID=$!
trap 'pkill -TERM -P $LPID 2>/dev/null; kill -TERM $LPID 2>/dev/null' EXIT
START=$(date +%s)
RESULT="TIMEOUT"
while true; do
  sleep 5
  if (( $(date +%s) - START > TIMEOUT )); then break; fi
  if grep -q "First inference received" "$LOG"; then RESULT="PASS"; break; fi
  if grep -q "MODEL_INIT_FAILED" "$LOG"; then RESULT="FAIL"; break; fi
done
echo "RESULT=$RESULT ELAPSED=$(( $(date +%s) - START ))s"
grep -E "Unified pipeline started|First inference received" "$LOG" | tail -2 || true
