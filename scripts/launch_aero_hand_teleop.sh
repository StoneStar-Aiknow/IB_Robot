#!/usr/bin/env bash
# Convenience wrapper for the unified right/left/dual Aero Hand profile.
# Profile selection and P-pose lifecycle are owned by robot_config; this script
# only supplies local SDK/device paths and performs friendly preflight checks.

# ROS 2 setup files probe optional variables and are not compatible with
# Bash nounset (`set -u`). Keep error, ERR-trap, and pipeline propagation.
set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SDK_CHECKSUMS="${WORKSPACE_ROOT}/third_party/vendor/mhandpro/3.0.20/SHA256SUMS"
LOCAL_CONFIG="${AERO_HAND_TELEOP_CONFIG:-${WORKSPACE_ROOT}/.aero_hand_teleop.env}"
if [[ -n "${AERO_HAND_TELEOP_CONFIG:-}" && ! -f "$LOCAL_CONFIG" ]]; then
  echo "Aero Hand local config not found: $LOCAL_CONFIG" >&2
  exit 1
fi
if [[ -f "$LOCAL_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$LOCAL_CONFIG"
fi

CONFIG_NAME="${AERO_HAND_CONFIG_NAME:-aero_hand_teleop}"
HAND_PROFILE="${AERO_HAND_PROFILE:-right}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
MHANDPRO_SDK_LIB="${MHANDPRO_SDK_LIB:-}"
AERO_HAND_RIGHT_PORT="${AERO_HAND_RIGHT_PORT:-}"
AERO_HAND_LEFT_PORT="${AERO_HAND_LEFT_PORT:-}"

usage() {
  cat <<'EOF'
Usage: scripts/launch_aero_hand_teleop.sh [options]

Options:
  --profile NAME      right, left, or dual (default: right)
  --sdk PATH          External mHandPro SDK .so path
  --right-port PATH   Right Aero Hand serial device
  --left-port PATH    Left Aero Hand serial device
  --domain ID         ROS_DOMAIN_ID (default: 42)
  Local defaults are loaded from .aero_hand_teleop.env or
  AERO_HAND_TELEOP_CONFIG. See .aero_hand_teleop.env.example.
  -h, --help          Show this help
EOF
}

while (($#)); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires a name" >&2; exit 2; }
      HAND_PROFILE="$2"
      shift 2
      ;;
    --sdk)
      [[ $# -ge 2 ]] || { echo "--sdk requires a path" >&2; exit 2; }
      MHANDPRO_SDK_LIB="$2"
      shift 2
      ;;
    --right-port)
      [[ $# -ge 2 ]] || { echo "--right-port requires a path" >&2; exit 2; }
      AERO_HAND_RIGHT_PORT="$2"
      shift 2
      ;;
    --left-port)
      [[ $# -ge 2 ]] || { echo "--left-port requires a path" >&2; exit 2; }
      AERO_HAND_LEFT_PORT="$2"
      shift 2
      ;;
    --domain)
      [[ $# -ge 2 ]] || { echo "--domain requires an ID" >&2; exit 2; }
      ROS_DOMAIN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$HAND_PROFILE" in
  right|left|dual) ;;
  *) echo "Invalid profile: $HAND_PROFILE (expected right, left, or dual)" >&2; exit 2 ;;
esac

cd "$WORKSPACE_ROOT"
source .shrc_local
export ROS_DOMAIN_ID MHANDPRO_SDK_LIB AERO_HAND_RIGHT_PORT AERO_HAND_LEFT_PORT

if [[ ! -f "$MHANDPRO_SDK_LIB" ]]; then
  echo "mHandPro SDK not found: ${MHANDPRO_SDK_LIB:-<unset>}" >&2
  echo "Configure external MHANDPRO_SDK_LIB or pass --sdk." >&2
  exit 1
fi
if [[ "$MHANDPRO_SDK_LIB" != /* ]]; then
  echo "mHandPro SDK path must be absolute: $MHANDPRO_SDK_LIB" >&2
  exit 1
fi
if [[ ! -f "$SDK_CHECKSUMS" ]]; then
  echo "mHandPro SDK checksum manifest not found: $SDK_CHECKSUMS" >&2
  exit 1
fi
read -r expected_sdk_sha256 _ < "$SDK_CHECKSUMS"
if [[ ! "$expected_sdk_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid mHandPro SDK checksum manifest: $SDK_CHECKSUMS" >&2
  exit 1
fi
actual_sdk_sha256="$(sha256sum "$MHANDPRO_SDK_LIB" | cut -d' ' -f1)"
if [[ "$actual_sdk_sha256" != "$expected_sdk_sha256" ]]; then
  echo "mHandPro SDK 3.0.20 checksum mismatch: $actual_sdk_sha256" >&2
  exit 1
fi

check_side() {
  local side="$1"
  local port calibration
  if [[ "$side" == "right" ]]; then
    port="$AERO_HAND_RIGHT_PORT"
  else
    port="$AERO_HAND_LEFT_PORT"
  fi
  calibration="$HOME/.calibrate/aero_hand_${side}_calibrate.json"
  if [[ -z "$port" || ! -e "$port" ]]; then
    echo "${side^} Aero Hand serial device not found: ${port:-<unset>}" >&2
    echo "Set AERO_HAND_${side^^}_PORT or pass --${side}-port." >&2
    exit 1
  fi
  if [[ ! -f "$calibration" ]]; then
    echo "Missing ${side} glove calibration: $calibration" >&2
    exit 1
  fi
}

if [[ "$HAND_PROFILE" == "right" || "$HAND_PROFILE" == "dual" ]]; then
  check_side right
fi
if [[ "$HAND_PROFILE" == "left" || "$HAND_PROFILE" == "dual" ]]; then
  check_side left
fi
if [[ "$HAND_PROFILE" == "dual" && "$AERO_HAND_LEFT_PORT" -ef "$AERO_HAND_RIGHT_PORT" ]]; then
  echo "Left and right Aero Hands must use different serial devices." >&2
  exit 1
fi

echo "Starting Aero Hand teleoperation: profile=$HAND_PROFILE, ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
exec ros2 launch robot_config robot.launch.py \
  robot_config:="$CONFIG_NAME" \
  hand_profile:="$HAND_PROFILE" \
  control_mode:=teleop \
  use_sim:=false \
  auto_start_controllers:=false \
  with_inference:=false \
  with_moveit:=false \
  with_navigation:=false
