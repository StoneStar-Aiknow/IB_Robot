#!/bin/bash
# Reuse the checkout environment. Set IBROBOT_FTSERVO_SOURCE for an offline
# ftservo source cache; an existing CMake cache remains usable without it.
set -eo pipefail
INTERACTION_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$INTERACTION_ROOT"
SITE_ENV="$INTERACTION_ROOT/commissioning/env.sh"
[[ -f "$SITE_ENV" ]] || SITE_ENV="$INTERACTION_ROOT/.shrc_local"
source "${IBROBOT_INTERACTION_ENV:-$SITE_ENV}"
[[ "${WORKSPACE:-}" -ef "$INTERACTION_ROOT" ]] || { echo "Environment points at another checkout" >&2; exit 1; }
cd "$INTERACTION_ROOT"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export MAKEFLAGS="${MAKEFLAGS:--j4}"
EXTRA=()
if [[ -n "${IBROBOT_FTSERVO_SOURCE:-}" ]]; then
    [[ -d "$IBROBOT_FTSERVO_SOURCE" ]] || { echo "ftservo source directory is missing" >&2; exit 1; }
    EXTRA+=(--cmake-args "-DFETCHCONTENT_SOURCE_DIR_FTSERVO=$IBROBOT_FTSERVO_SOURCE")
fi
./scripts/build.sh -- --packages-select aimdk_grasp aimdk_robot embodied_bringup embodied_common \
    ibrobot_agent pose_composer robot_config robot_runtime skill_catalog skill_library \
    --parallel-workers "${IBROBOT_BUILD_WORKERS:-3}" "${EXTRA[@]}" "$@"
