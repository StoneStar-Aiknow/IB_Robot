#!/bin/bash
# Type instead of speaking: the text enters the voice bridge as an ASR transcript would.
#   tools/x2_local_voice/say.sh "欧拉欧拉，自我介绍" [timeout_s]
# Typed text has no voiceprint, which the speaker check lets through: say.sh exercises the wake
# phrase, the intent gate and everything after them, but never the voiceprint (use inject_wav.py).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/rig_env.sh" python3 "$HERE/tools/send_text.py" "$1" --timeout "${2:-120}"
