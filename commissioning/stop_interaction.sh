#!/bin/bash
# Stop checkout-owned nodes even when the launch leader or pidfile is gone.
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
    LAUNCH=x2_grasp_head.launch.py
else
    STATE="$INTERACTION_ROOT/log/agent_interaction"
    NAME=pipeline
    LAUNCH=embodied_pipeline.launch.py
fi
STATE="${IBROBOT_INTERACTION_STATE:-$STATE}"
mkdir -p "$STATE"
exec 9>"$STATE/lifecycle.lock"
flock -x 9
FILE="$STATE/$NAME.pgid"
CONFIG="$INTERACTION_ROOT/src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"
RESULT=0
if [[ -f "$FILE" ]]; then
    GROUP=$(cat "$FILE")
    if [[ ! "$GROUP" =~ ^[0-9]+$ || "$GROUP" -le 1 ]]; then
        echo "Invalid process group: $GROUP" >&2
        RESULT=1
    elif kill -0 "$GROUP" 2>/dev/null; then
        OWNER=$(ps -o args= -p "$GROUP")
        if [[ "$OWNER" != *"$LAUNCH"* || "$OWNER" != *"config_path:=$CONFIG"* ]]; then
            echo "PID ownership mismatch; refusing to signal pgid $GROUP" >&2
            RESULT=1
        fi
    fi
fi
# Never signal an unverified whole group. The helper verifies every process by
# its exact executable/config argv and rechecks its acquisition identity.
if python3 "$INTERACTION_ROOT/commissioning/interaction_processes.py" stop --root "$INTERACTION_ROOT" --launch "$LAUNCH"; then
    if [[ "$RESULT" == 0 ]]; then
        rm -f "$FILE"
        echo "$NAME stopped; no checkout-owned nodes remain"
    fi
else
    echo "$NAME stop incomplete; retaining pidfile" >&2
    RESULT=1
fi
# Attempt the head even if local shutdown failed; never hide either failure.
if [[ "$ROLE" == pipeline && "${IBROBOT_MANAGE_GRASP_HEAD:-1}" == 1 ]]; then
    HEAD_WORKSPACE="${IBROBOT_GRASP_HEAD_WORKSPACE:-ibrobot_x2_skill_pause_8b68e9b9}"
    printf -v REMOTE_COMMAND '%q --head' "$HEAD_WORKSPACE/commissioning/stop_interaction.sh"
    ssh -nT -o BatchMode=yes -o ConnectTimeout=5 \
        "${IBROBOT_GRASP_HEAD_HOST:-agi@10.0.1.41}" "$REMOTE_COMMAND" || RESULT=1
fi
exit "$RESULT"
