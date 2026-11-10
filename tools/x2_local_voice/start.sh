#!/bin/bash
# Start the local X2 voice rig of this checkout: this PC's microphone and headphones, the real
# Agent/ASR/TTS/runtime and a simulated robot. Design and setup: docs/aimdk_x2_local_voice_rig_zh.md
#
#   tools/x2_local_voice/start.sh [--source <pulse source>] [--sink <pulse sink>] [--show-level]
#
# Environment (all optional):
#   X2_LOCAL_VOICE_MODELS   model root (default: <checkout>/models)
#   X2_LOCAL_VOICE_STATE    logs, ledgers and pid files (default: <checkout>/log/x2_local_voice)
#   X2_LOCAL_VOICE_DOMAIN   ROS domain, localhost only (default: 91)
#   X2_PLANNER_API_KEY or X2_PLANNER_API_KEY_FILE   planner credential
#   AIMDK_OVERLAY           AimDK x86 overlay setup (default: ~/aimdk/overlay_x86/install/local_setup.bash)
#   LOCAL_VOICE_PICK=empty  LOCAL_VOICE_PLACE=unseen   fake pick/place failures
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"
STATE="${X2_LOCAL_VOICE_STATE:-$WS/log/x2_local_voice}"
MODELS="${X2_LOCAL_VOICE_MODELS:-$WS/models}"
LOGS="$STATE/logs"
RUN="$STATE/run"
ROBOTS="$WS/src/robot_config/config/robots"
PROFILES="$WS/src/agent/skill_catalog/config/profiles"
mkdir -p "$LOGS" "$RUN"

if [[ -f "$RUN/pipeline.pgid" ]] && pgrep -g "$(cat "$RUN/pipeline.pgid")" >/dev/null 2>&1; then
    echo "already running (pgid $(cat "$RUN/pipeline.pgid")); run $HERE/stop.sh first" >&2
    exit 1
fi

ASR_BUNDLE="$MODELS/voice_asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05"
TTS_BUNDLE="$MODELS/zipvoice"
SPEAKER_BUNDLE="$MODELS/campplus-speaker"
INTENT_MODEL="$MODELS/addressee/addressee_zh_v2.json"
VAD_MODEL="$MODELS/silero-vad/assets/silero_vad.onnx"
missing=0
for path in "$ASR_BUNDLE/inference_manifest.json" "$TTS_BUNDLE/inference_manifest.json" \
    "$SPEAKER_BUNDLE/inference_manifest.json" "$INTENT_MODEL" "$VAD_MODEL"; do
    [[ -e "$path" ]] || { echo "missing model file: $path" >&2; missing=1; }
done
[[ $missing -eq 0 ]] || { echo "set X2_LOCAL_VOICE_MODELS to a directory with this layout" >&2; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
# Each run starts from empty Agent/Gateway ledgers; the previous ones are kept for reading.
AGENT="$STATE/agent"
[[ -d "$AGENT" ]] && mv "$AGENT" "$LOGS/agent_state_before_$STAMP"
mkdir -p "$AGENT"

OPTIONS=()
if "$HERE/rig_env.sh" python3 -c "import piper_phonemize" >/dev/null 2>&1; then
    OPTIONS+=(--keep-english)
fi
# Written next to the shipped robot configs (relative references resolve as for the summit file)
# and the catalog profiles; stop.sh removes both.
CONFIG="$ROBOTS/aimdk_x2_summit_local_voice.yaml"
"$HERE/rig_env.sh" python3 "$HERE/make_config.py" --models "$MODELS" --state "$AGENT" --out "$CONFIG" \
    --catalog-out "$PROFILES/aimdk_x2_summit_local_voice.yaml" "${OPTIONS[@]}" || exit 1

# A failed startup must not leave another microphone reader behind.
trap '"$HERE/stop.sh" >/dev/null 2>&1' EXIT
setsid "$HERE/rig_env.sh" ros2 launch embodied_bringup embodied_pipeline.launch.py \
    config_path:="$CONFIG" control_mode:=moveit_planning with_moveit:=false use_sim:=true authorize_motion:=true \
    > "$LOGS/pipeline_$STAMP.log" 2>&1 < /dev/null &
echo $! > "$RUN/pipeline.pgid"
ln -sfn "$LOGS/pipeline_$STAMP.log" "$LOGS/pipeline_latest.log"
setsid "$HERE/rig_env.sh" python3 "$HERE/local_audio_hal.py" --vad-model "$VAD_MODEL" "$@" \
    > "$LOGS/audio_hal_$STAMP.log" 2>&1 < /dev/null &
echo $! > "$RUN/audio.pgid"
ln -sfn "$LOGS/audio_hal_$STAMP.log" "$LOGS/audio_hal_latest.log"

echo "starting (${OPTIONS[*]:-defaults}); logs: $LOGS"
for _ in $(seq 1 180); do
    if grep -qE "process has died|Runtime interface binding failed" "$LOGS/pipeline_$STAMP.log" 2>/dev/null; then
        echo "pipeline startup failed; see $LOGS/pipeline_$STAMP.log" >&2
        exit 1
    fi
    if grep -q "ASR deployment loaded" "$LOGS/pipeline_$STAMP.log" 2>/dev/null && \
        grep -q "voice bridge ready:" "$LOGS/pipeline_$STAMP.log" 2>/dev/null && \
        grep -q "pose executor ready" "$LOGS/pipeline_$STAMP.log" 2>/dev/null && \
        grep -q "local audio HAL ready:" "$LOGS/audio_hal_$STAMP.log" 2>/dev/null; then
        break
    fi
    if ! pgrep -g "$(cat "$RUN/pipeline.pgid")" >/dev/null; then
        echo "pipeline exited; see $LOGS/pipeline_$STAMP.log" >&2
        exit 1
    fi
    if ! pgrep -g "$(cat "$RUN/audio.pgid")" >/dev/null; then
        echo "audio HAL exited; see $LOGS/audio_hal_$STAMP.log" >&2
        exit 1
    fi
    sleep 1
done
if grep -q "process has died" "$LOGS/pipeline_$STAMP.log"; then
    echo "a node died during startup; see $LOGS/pipeline_$STAMP.log" >&2
    exit 1
fi
if grep -q "ASR deployment loaded" "$LOGS/pipeline_$STAMP.log" && \
    grep -q "voice bridge ready:" "$LOGS/pipeline_$STAMP.log" && \
    grep -q "pose executor ready" "$LOGS/pipeline_$STAMP.log" && \
    grep -q "local audio HAL ready:" "$LOGS/audio_hal_$STAMP.log"; then
    echo "ready. Say \"欧拉欧拉\" and a request into the microphone."
else
    echo "ASR, voice bridge, pose executor or audio HAL not ready after 180 s; see $LOGS" >&2
    exit 1
fi
trap - EXIT
echo "watch: $HERE/watch.sh    type instead of speaking: $HERE/say.sh \"欧拉欧拉，自我介绍\"    stop: $HERE/stop.sh"
