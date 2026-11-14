#!/bin/bash
# Stop this pipeline and duplicate X2 runtimes from any checkout on the 310P.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "${1:-}" in
    --head|-h|--help) exec bash "$ROOT/commissioning/stop_interaction.sh" "$@" ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
export IBROBOT_GRASP_HEAD_HOST=agi@10.0.1.41
export IBROBOT_GRASP_HEAD_WORKSPACE=/home/agi/IB_Robot_oee_summit_2026_8ad9364a
RESULT=0
bash "$ROOT/commissioning/stop_interaction.sh" || RESULT=1
# Always attempt duplicate cleanup, even if local or remote shutdown failed.
STATE="${IBROBOT_INTERACTION_STATE:-$ROOT/log/agent_interaction}"
mkdir -p "$STATE"
exec 9>"$STATE/lifecycle.lock"
flock -x 9
if python3 "$ROOT/commissioning/interaction_processes.py" stop --root "$ROOT" \
    --launch embodied_pipeline.launch.py --all-x2-runtimes; then
    echo "No checkout-owned nodes or duplicate X2 runtimes remain"
else
    RESULT=1
fi
exit "$RESULT"
