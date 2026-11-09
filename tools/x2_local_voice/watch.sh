#!/bin/bash
# Follow the interaction: microphone segments, ASR results, gate verdicts, plans, skill
# terminals and what was spoken or played.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"
LOGS="${X2_LOCAL_VOICE_STATE:-$WS/log/x2_local_voice}/logs"
tail -n 0 -F "$LOGS/pipeline_latest.log" "$LOGS/audio_hal_latest.log" 2>/dev/null \
    | sed -u -E 's/\x1b\[[0-9;]*m//g' \
    | grep --line-buffered -E "speech (BEGIN|END)|Command:|request voice-[0-9a-f]+(:| accepted| not admitted| conversation| quarantined)|not addressed|introduction speech|\[say\]|exact plan|\"event\":\"(requested|terminal)\"|grasp_pipeline_terminal|place_pipeline_terminal|playback (start|idle)|drop reason|E06|ERROR|Traceback|process has died" \
    | sed -u -E 's/^\[[^]]+\] \[(INFO|WARN|ERROR)\] \[[0-9.]+\] //' | cut -c1-300
