#!/bin/sh
# Tier 6 board ROS mock verification (single session: launch + calls + teardown)
cd /IB_Robot || exit 1
. ./.shrc_local >/dev/null 2>&1
. /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export ROS_DOMAIN_ID=52
export PYTHONUNBUFFERED=1
LOG=/data/local/tmp/board_tier6.log
nohup ros2 launch robot_config robot.launch.py \
  config_path:=/IB_Robot/board_mock_verify.yaml \
  control_mode:=model_inference use_sim:=true >"$LOG" 2>&1 &
LPID=$!

START=$(date +%s)
STATE="LAUNCH_TIMEOUT"
while true; do
  sleep 5
  NOW=$(date +%s)
  [ $((NOW - START)) -gt 420 ] && break
  READY=$(ros2 service list 2>/dev/null | grep -cE "/perception/(siglip2/encode_embeddings|ram_plus/recognize_tags|sam2/generate_masks)")
  [ "$READY" -ge 3 ] && STATE="SERVICES_READY" && break
done
echo "LAUNCH_STATE=$STATE ELAPSED=$(( $(date +%s) - START ))s"

if [ "$STATE" = "SERVICES_READY" ]; then
  sleep 8
  echo "=== service calls ==="
  python3 /data/local/tmp/call_perception_services.py
  # policy check AFTER service calls: torch_npu import + OM load delay the
  # first dispatch; grepping too early misses "First inference received"
  echo "=== policy pipeline ==="
  grep -aE "contract_mock active|Unified pipeline started|First inference received" "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | head -5
fi

kill "$LPID" 2>/dev/null
sleep 3
kill -9 "$LPID" 2>/dev/null
pkill -f "robot.launch.py" 2>/dev/null
pkill -f "model_service_node" 2>/dev/null
pkill -f "pipeline_policy_node" 2>/dev/null
pkill -f "action_dispatcher" 2>/dev/null
pkill -f "contract_mock" 2>/dev/null
echo "DONE"
