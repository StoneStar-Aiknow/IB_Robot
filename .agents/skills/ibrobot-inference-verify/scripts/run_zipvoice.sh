#!/bin/zsh
# ZipVoice host verification via the typed model service.
#
# Standard model_service_node form (unlike fullsubnet):
#   /voice_tts/synthesize (ibrobot_msgs/srv/SynthesizeSpeech)
#   bundle: models/zipvoice (deployments: ubuntu_onnx + ascend_310p)
#
# CPU mode: works out of the box (providers CPUExecutionProvider).
# CUDA mode requires ALL of (see references/troubleshooting.md):
#   1. onnxruntime-gpu installed (pip uninstall onnxruntime; pip install onnxruntime-gpu==1.23.2)
#   2. providers flipped to CUDAExecutionProvider in assets/zipvoice_onnx.json (done + restored here)
#   3. LD_LIBRARY_PATH pointing at torch-cu126's bundled nvidia libs (done here)
# CUDA only accelerates text_encoder + fm_decoder; the Vocos vocoder is
# hardcoded torch-CPU.
#
# Usage: run_zipvoice.sh <cpu|cuda> <ros_domain_id>
set -e
MODE="$1"; DOMAIN="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
source .shrc_local >/dev/null 2>&1
# nvidia pip wheels live at site-packages/nvidia (NOT under torch/)
NVLIB="$(python3 -c 'import nvidia, pathlib; print(pathlib.Path(nvidia.__file__).parent)' 2>/dev/null || true)"
if [ -z "$NVLIB" ] || [ ! -d "$NVLIB" ]; then
  NVLIB="$(python3 -c 'import sysconfig, pathlib; p = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"; print(p if p.is_dir() else "")')"
fi
NVLIBPATHS=""
for d in cublas cudnn curand cuda_runtime cuda_nvrtc cufft cusolver cusparse nvjitlink; do
  [ -d "$NVLIB/$d/lib" ] && NVLIBPATHS="$NVLIBPATHS:$NVLIB/$d/lib"
done
export LD_LIBRARY_PATH="${NVLIBPATHS#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROS_DOMAIN_ID="$DOMAIN"
export PYTHONUNBUFFERED=1
LOG=/tmp/opencode/zipvoice_${MODE}.log
BUNDLE="$(pwd)/models/zipvoice"
CONFIG_JSON="$BUNDLE/assets/zipvoice_onnx.json"

restore_providers() {
  python3 - "$CONFIG_JSON" <<'EOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["providers"] = ["CPUExecutionProvider"]
json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
EOF
}

if [[ "$MODE" == "cuda" ]]; then
  python3 - "$CONFIG_JSON" <<'EOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
print("providers -> CUDA-first (will restore on exit)")
EOF
fi

ros2 launch voice_tts_service voice_tts.launch.py \
  bundle_path:=$BUNDLE deployment:=ubuntu_onnx >"$LOG" 2>&1 &
LPID=$!
trap 'pkill -TERM -P $LPID 2>/dev/null; kill -TERM $LPID 2>/dev/null; [[ "$MODE" == "cuda" ]] && restore_providers' EXIT
START=$(date +%s)
STATE="LAUNCH_TIMEOUT"
while true; do
  sleep 3
  if (( $(date +%s) - START > 120 )); then break; fi
  if ros2 service list 2>/dev/null | grep -q "/voice_tts/synthesize"; then STATE="READY"; break; fi
done
echo "LAUNCH_STATE=$STATE ELAPSED=$(( $(date +%s) - START ))s"
if [[ "$STATE" == "READY" ]]; then
  sleep 3
  if [[ "$MODE" == "cuda" ]]; then
    echo "=== GPU binding (pid, mem) ==="
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -4
  fi
  python3 "$SCRIPT_DIR/call_tts.py"
fi
sleep 1
if [[ "$MODE" == "cuda" ]] && grep -aq "Failed to create CUDAExecutionProvider" "$LOG"; then
  echo "WARNING: CUDAExecutionProvider FAILED to initialize — the run above was a CPU FALLBACK"
fi
grep -aE "ZipVoice ONNX loaded|initialization failed|Failed to create CUDA|Traceback" "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | head -6
