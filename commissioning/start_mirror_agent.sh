#!/usr/bin/env bash
# Start the full voice-agent + mirror pipeline on the 310P and the grasp head on Orin.
# Run this script on the 310P checkout that contains commissioning/site_runner.sh.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN="${HZ_ROS_DOMAIN_ID:-0}"
ORIN_SSH="${HZ_ORIN_SSH:-agi@10.0.1.41}"
ORIN_ROOT="${HZ_ORIN_WORKSPACE:-/home/agi/IB_Robot_hz_20261104}"
LOG_DIR="$ROOT/commissioning/logs"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$DOMAIN" =~ ^[0-9]+$ ]] && (( DOMAIN <= 101 && DOMAIN != 96 && DOMAIN != 97 )) || \
    die "Set HZ_ROS_DOMAIN_ID to the robot domain (0..101, excluding 96/97)."
[[ -f "$ROOT/commissioning/robots/hz_site_mirror_live.yaml" ]] || die "Full mirror YAML is missing."
[[ -f "$ROOT/commissioning/site_runner.sh" ]] || die "commissioning/site_runner.sh is missing."
[[ -f "$ROOT/commissioning/env.sh" ]] || die "commissioning/env.sh is missing."
[[ -t 0 ]] || die "Run from an interactive terminal; hardware startup requires a human confirmation."

echo "This starts the full 9-skill voice Agent and mirror session on the 310P, plus the grasp head on Orin."
echo "Domain: $DOMAIN | 310P: $ROOT | Orin: $ORIN_SSH:$ORIN_ROOT"
echo "The live pipeline authorizes robot motion. Make sure the robot area is clear."
read -r -p 'If the robot is ready, type START to launch: ' answer
[[ "$answer" == START ]] || die "Startup cancelled."

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"

# Preflight both hosts before starting either process.
ssh -o BatchMode=yes "$ORIN_SSH" "test -f '$ORIN_ROOT/commissioning/robots/hz_site_live.yaml' && test -f '$ORIN_ROOT/commissioning/site_runner.sh' && test -f '$ORIN_ROOT/commissioning/env_orin.sh'" || \
    die "Orin checkout/config is unavailable; nothing started."
local_conflicts="$(ps -eo args | grep -E '[e]mbodied_pipeline.launch.py|[x]2_grasp_head.launch.py' || true)"
[[ -z "$local_conflicts" ]] || die "A robot pipeline is already running on 310P:\n$local_conflicts"
remote_conflicts="$(ssh -o BatchMode=yes "$ORIN_SSH" "ps -eo args | grep -E '[e]mbodied_pipeline.launch.py|[x]2_grasp_head.launch.py' || true")"
[[ -z "$remote_conflicts" ]] || die "A robot pipeline is already running on Orin:\n$remote_conflicts"

# Start the head process first; its runner loads the Orin ROS environment.
orin_log="$ORIN_ROOT/commissioning/logs/site_orin_mirror_live_${timestamp}.log"
ssh -o BatchMode=yes "$ORIN_SSH" "mkdir -p '$ORIN_ROOT/commissioning/logs'; cd '$ORIN_ROOT'; \
  export HZ_TARGET=robot HZ_ROS_DOMAIN_ID='$DOMAIN'; \
  nohup setsid bash commissioning/site_runner.sh orin \
    python3 /opt/ros/humble/bin/ros2 launch aimdk_grasp x2_grasp_head.launch.py \
    config_path:='$ORIN_ROOT/commissioning/robots/hz_site_live.yaml' runtime_timeout:=180 \
    > '$orin_log' 2>&1 </dev/null & \
  pid=\$!; sleep 1; kill -0 \$pid 2>/dev/null || exit 1; \
  ticks=\$(awk '{print \$22}' /proc/\$pid/stat); \
  printf '%s %s\\n' \$pid \$ticks > commissioning/site_process_orin; \
  echo \"Orin started PID=\$pid log=$orin_log\"" || die "Orin launch failed. Check the Orin log."

# Start full Agent/mirror stack on 310P. This is the live YAML and allows real execution.
export HZ_TARGET=robot HZ_ROS_DOMAIN_ID="$DOMAIN"
local_log="$LOG_DIR/site_310p_mirror_live_${timestamp}.log"
nohup setsid bash "$ROOT/commissioning/site_runner.sh" 310p \
    python3 /opt/ros/humble/bin/ros2 launch embodied_bringup embodied_pipeline.launch.py \
    "config_path:=$ROOT/commissioning/robots/hz_site_mirror_live.yaml" \
    control_mode:=model_inference with_moveit:=false use_sim:=false authorize_motion:=true \
    >"$local_log" 2>&1 </dev/null &
pid=$!
sleep 1
kill -0 "$pid" 2>/dev/null || die "310P launch exited during startup. Check $local_log"
ticks="$(awk '{print $22}' "/proc/$pid/stat")"
printf '%s %s\n' "$pid" "$ticks" > "$ROOT/commissioning/site_process_310p"

echo "Started: 310P PID=$pid, log=$local_log"
echo "Started: Orin head, log=$orin_log"
echo "Startup does not mean readiness. Check Agent/runtime status with commissioning/site_ready.py --mirror."
