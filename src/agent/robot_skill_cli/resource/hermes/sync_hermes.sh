#!/usr/bin/env bash
set -euo pipefail

if ! command -v hermes-robot-configure >/dev/null 2>&1; then
    echo "error: hermes-robot-configure is unavailable; build robot_skill_cli and source install/setup.bash" >&2
    exit 4
fi

exec hermes-robot-configure "$@"
