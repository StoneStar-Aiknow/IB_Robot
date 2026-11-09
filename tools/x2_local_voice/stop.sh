#!/bin/bash
# Stop the local voice rig of this checkout (only its own process groups) and remove the
# configuration files start.sh generated inside the source tree.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"
RUN="${X2_LOCAL_VOICE_STATE:-$WS/log/x2_local_voice}/run"
for name in audio pipeline; do
    file="$RUN/$name.pgid"
    [[ -f "$file" ]] || continue
    group=$(cat "$file")
    if pgrep -g "$group" >/dev/null 2>&1; then
        kill -INT -"$group" 2>/dev/null
        for _ in $(seq 1 20); do pgrep -g "$group" >/dev/null || break; sleep 0.5; done
        pgrep -g "$group" >/dev/null && kill -TERM -"$group" 2>/dev/null
        for _ in $(seq 1 10); do pgrep -g "$group" >/dev/null || break; sleep 0.5; done
        pgrep -g "$group" >/dev/null && kill -KILL -"$group" 2>/dev/null
    fi
    rm -f "$file"
    echo "stopped $name (pgid $group)"
done
rm -f "$WS/src/robot_config/config/robots/aimdk_x2_summit_local_voice.yaml" \
    "$WS/src/agent/skill_catalog/config/profiles/aimdk_x2_summit_local_voice.yaml"
