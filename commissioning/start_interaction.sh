#!/bin/bash
# Start from this checkout and reuse commissioning/env.sh (env_orin.sh on the head),
# falling back to .shrc_local. IBROBOT_INTERACTION_ENV selects another site setup.
# The same script must be present in the Orin checkout.
set -eo pipefail
INTERACTION_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROLE=pipeline
case "${1:-}" in
    --head) ROLE=head ;;
    -h|--help)
        cat <<'HELP'
Usage: commissioning/start_interaction.sh [--head]
Starts the canonical voice_mirror_grasp entry; --head starts only the Orin head.
IBROBOT_INTERACTION_ENV: site environment script (otherwise commissioning/env*.sh or .shrc_local).
X2_PLANNER_API_KEY or IBROBOT_PLANNER_KEY_FILE: planner credential (pipeline only).
IBROBOT_MANAGE_GRASP_HEAD=0: leave the remote head lifecycle to the operator.
IBROBOT_GRASP_HEAD_HOST: SSH destination (default agi@10.0.1.41).
IBROBOT_GRASP_HEAD_WORKSPACE: remote checkout, absolute or relative to the SSH home.
IBROBOT_INTERACTION_STATE: override the local PID/log directory.
HELP
        exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
cd "$INTERACTION_ROOT"
if [[ "$ROLE" == head ]]; then
    export HZ_TARGET=robot
    SITE_ENV="$INTERACTION_ROOT/commissioning/env_orin.sh"
    STATE="$INTERACTION_ROOT/log/grasp_head"
    NAME=head
    LAUNCH="ros2 launch aimdk_grasp x2_grasp_head.launch.py"
else
    export VOICE_POSE_TARGET=robot
    SITE_ENV="$INTERACTION_ROOT/commissioning/env.sh"
    STATE="$INTERACTION_ROOT/log/agent_interaction"
    NAME=pipeline
    LAUNCH="ros2 launch embodied_bringup embodied_pipeline.launch.py"
fi
[[ -f "$SITE_ENV" ]] || SITE_ENV="$INTERACTION_ROOT/.shrc_local"
source "${IBROBOT_INTERACTION_ENV:-$SITE_ENV}" >/dev/null
[[ "${WORKSPACE:-}" -ef "$INTERACTION_ROOT" ]] || { echo "Environment points at another checkout" >&2; exit 1; }
cd "$INTERACTION_ROOT"
STATE="${IBROBOT_INTERACTION_STATE:-$STATE}"
CONFIG="$INTERACTION_ROOT/src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"
if [[ "$ROLE" == pipeline ]]; then
    python3 - <<'PY'
import json
import time

import rclpy
from aimdk_robot.operator_microphone import RosBackend, select_microphone

rclpy.init()
backend = RosBackend()
try:
    # Give the vendor time to discover the service client before its first RPC.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        rclpy.spin_once(backend.node, timeout_sec=0.1)
    print(json.dumps(select_microphone(backend, "external")), flush=True)
    print(json.dumps({"audio": backend.verify_audio("external")}), flush=True)
finally:
    backend.node.destroy_node()
    rclpy.try_shutdown()
PY
    if [[ -z "${X2_PLANNER_API_KEY:-}" ]]; then
        KEY_FILE="${IBROBOT_PLANNER_KEY_FILE:-$HOME/.config/ibrobot/planner_qwen233_interaction.key}"
        export X2_PLANNER_API_KEY="$(cat "$KEY_FILE")"
    fi
    [[ -n "$X2_PLANNER_API_KEY" ]] || { echo "Planner credential is empty" >&2; exit 1; }
fi
mkdir -p "$STATE"
exec 9>"$STATE/lifecycle.lock"
flock -x 9
if [[ -f "$STATE/$NAME.pgid" ]]; then
    GROUP=$(cat "$STATE/$NAME.pgid")
    [[ "$GROUP" =~ ^[0-9]+$ && "$GROUP" -gt 1 ]] || { echo "Invalid process group: $GROUP" >&2; exit 1; }
fi
if [[ -f "$STATE/$NAME.pgid" ]] && pgrep -g "$(cat "$STATE/$NAME.pgid")" >/dev/null 2>&1; then
    GROUP=$(cat "$STATE/$NAME.pgid")
    OWNER=$(ps -o args= -p "$GROUP")
    [[ "$OWNER" == *"$LAUNCH"* && "$OWNER" == *"config_path:=$CONFIG"* ]] || { echo "PID ownership mismatch: $GROUP" >&2; exit 1; }
    echo "$NAME already running: pgid $GROUP"
else
    if ! python3 "$INTERACTION_ROOT/commissioning/interaction_processes.py" check --root "$INTERACTION_ROOT" --launch "${LAUNCH##* }"; then
        echo "Checkout-owned nodes remain; run stop_prepared_pipeline.sh before starting" >&2
        exit 1
    fi
    LOG="$STATE/${NAME}_$(date +%Y%m%d_%H%M%S).log"
    if [[ "$ROLE" == head ]]; then
        IBROBOT_INTERACTION_OWNER="$INTERACTION_ROOT" setsid ros2 launch aimdk_grasp x2_grasp_head.launch.py \
            config_path:="$CONFIG" control_mode:=model_inference runtime_timeout:=60 \
            >"$LOG" 2>&1 </dev/null 9>&- &
    else
        IBROBOT_INTERACTION_OWNER="$INTERACTION_ROOT" setsid ros2 launch embodied_bringup embodied_pipeline.launch.py \
            config_path:="$CONFIG" control_mode:=model_inference with_moveit:=false \
            use_sim:=false authorize_motion:=true >"$LOG" 2>&1 </dev/null 9>&- &
    fi
    echo $! > "$STATE/$NAME.pgid"
    ln -sfn "$LOG" "$STATE/${NAME}_latest.log"
    echo "$NAME started: pgid $(cat "$STATE/$NAME.pgid")"
    echo "log: $LOG"
fi
if [[ "$ROLE" == pipeline && "${IBROBOT_MANAGE_GRASP_HEAD:-1}" == 1 ]]; then
    HEAD_WORKSPACE="${IBROBOT_GRASP_HEAD_WORKSPACE:-ibrobot_x2_skill_pause_8b68e9b9}"
    printf -v REMOTE_COMMAND '%q --head' "$HEAD_WORKSPACE/commissioning/start_interaction.sh"
    ssh -nT -o BatchMode=yes -o ConnectTimeout=5 \
        "${IBROBOT_GRASP_HEAD_HOST:-agi@10.0.1.41}" "$REMOTE_COMMAND"
fi
