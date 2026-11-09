#!/bin/bash
# Run one command in the local voice rig's clean environment:
#   .shrc_local of this checkout + the AimDK x86 overlay + this checkout's install,
#   ROS isolated to localhost (the LAN robots run their pipelines in domain 0).
# Usage: tools/x2_local_voice/rig_env.sh <command> [args...]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"
AIMDK_OVERLAY="${AIMDK_OVERLAY:-$HOME/aimdk/overlay_x86/install/local_setup.bash}"
DOMAIN="${X2_LOCAL_VOICE_DOMAIN:-91}"
KEY="${X2_PLANNER_API_KEY:-}"
if [[ -z "$KEY" && -n "${X2_PLANNER_API_KEY_FILE:-}" ]]; then
    KEY="$(cat "$X2_PLANNER_API_KEY_FILE")"
fi
exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-xterm}" LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}" \
    LOCAL_VOICE_PICK="${LOCAL_VOICE_PICK:-success}" LOCAL_VOICE_PLACE="${LOCAL_VOICE_PLACE:-success}" \
    RIG_WS="$WS" RIG_HERE="$HERE" RIG_OVERLAY="$AIMDK_OVERLAY" RIG_DOMAIN="$DOMAIN" X2_PLANNER_API_KEY="$KEY" \
    bash --noprofile --norc -c '
cd "$RIG_WS" || exit 1
source .shrc_local >/dev/null 2>&1 || { echo ".shrc_local failed in $RIG_WS" >&2; exit 1; }
source "$RIG_OVERLAY" || { echo "AimDK overlay not found: $RIG_OVERLAY (set AIMDK_OVERLAY)" >&2; exit 1; }
# Reassert this checkout over the overlay: only the vendor messages come from AimDK.
source "$RIG_WS/install/local_setup.bash" || exit 1
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="$RIG_DOMAIN"
export PYTHONPATH="$RIG_HERE:$PYTHONPATH"
unset RIG_WS RIG_HERE RIG_OVERLAY RIG_DOMAIN
exec "$@"
' _ "$@"
