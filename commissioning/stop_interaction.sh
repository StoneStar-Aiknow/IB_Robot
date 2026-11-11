#!/bin/bash
# Stop only the launch group recorded by this checkout, then its remote head.
# Uses the same IBROBOT_* lifecycle settings as start_interaction.sh.
set -u
INTERACTION_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROLE=pipeline
case "${1:-}" in
    --head) ROLE=head ;;
    -h|--help) echo "Usage: commissioning/stop_interaction.sh [--head]"; exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if [[ "$ROLE" == head ]]; then
    STATE="$INTERACTION_ROOT/log/grasp_head"
    NAME=head
    LAUNCH="ros2 launch aimdk_grasp x2_grasp_head.launch.py"
else
    STATE="$INTERACTION_ROOT/log/agent_interaction"
    NAME=pipeline
    LAUNCH="ros2 launch embodied_bringup embodied_pipeline.launch.py"
fi
STATE="${IBROBOT_INTERACTION_STATE:-$STATE}"
FILE="$STATE/$NAME.pgid"
CONFIG="$INTERACTION_ROOT/src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"
if [[ -f "$FILE" ]]; then
    GROUP=$(cat "$FILE")
    [[ "$GROUP" =~ ^[0-9]+$ && "$GROUP" -gt 1 ]] || { echo "Invalid process group: $GROUP" >&2; exit 1; }
    if pgrep -g "$GROUP" >/dev/null 2>&1; then
        OWNER=$(ps -o args= -p "$GROUP")
        [[ "$OWNER" == *"$LAUNCH"* && "$OWNER" == *"config_path:=$CONFIG"* ]] || { echo "PID ownership mismatch; refusing to stop pgid $GROUP" >&2; exit 1; }
        kill -INT -- -"$GROUP"
        for _ in $(seq 1 100); do pgrep -g "$GROUP" >/dev/null || break; sleep 0.5; done
        pgrep -g "$GROUP" >/dev/null && kill -TERM -- -"$GROUP"
        for _ in $(seq 1 40); do pgrep -g "$GROUP" >/dev/null || break; sleep 0.5; done
        if pgrep -g "$GROUP" >/dev/null; then echo "Processes still running in pgid $GROUP" >&2; exit 1; fi
    fi
    rm -f "$FILE"
    echo "$NAME stopped: pgid $GROUP"
else
    echo "$NAME is not running"
fi
if [[ "$ROLE" == pipeline && "${IBROBOT_MANAGE_GRASP_HEAD:-1}" == 1 ]]; then
    HEAD_WORKSPACE="${IBROBOT_GRASP_HEAD_WORKSPACE:-ibrobot_x2_skill_pause_8b68e9b9}"
    printf -v REMOTE_COMMAND '%q --head' "$HEAD_WORKSPACE/commissioning/stop_interaction.sh"
    ssh -nT -o BatchMode=yes -o ConnectTimeout=5 \
        "${IBROBOT_GRASP_HEAD_HOST:-agi@10.0.1.41}" "$REMOTE_COMMAND"
fi
