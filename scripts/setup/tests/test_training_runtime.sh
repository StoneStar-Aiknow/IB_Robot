#!/usr/bin/env bash
# Hermetic training video verification checks.
# No package downloads, ROS initialization or system mutations.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap '[[ "${TMP_ROOT}" == /tmp/* ]] && rm -rf -- "${TMP_ROOT}"' EXIT
PASS=0
ok() { PASS=$((PASS + 1)); printf 'PASS %s\n' "$*"; }
cd "${REPO_ROOT}"
source scripts/setup.sh
set -euo pipefail

# Execute the real verify_training_env heredoc with lightweight Python module
# fixtures. The wrapper is the venv interpreter boundary, not a replacement
# for the production shell function or its failure handling.
WORKSPACE="${TMP_ROOT}/video"
mkdir -p "${WORKSPACE}/venv/bin" "${WORKSPACE}/modules/torchcodec"
touch "${WORKSPACE}/venv/bin/lerobot-train"
printf '' > "${WORKSPACE}/modules/torchcodec/__init__.py"
cat > "${WORKSPACE}/venv/bin/python3" <<'PY'
#!/usr/bin/python3
import importlib.util
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules'))
sys.executable = str(Path(__file__).resolve())
for name in ('numpy', 'torch', 'lerobot', 'lerobot.configs', 'lerobot.policies', 'lerobot.policies.act'):
    mod = types.ModuleType(name)
    mod.__version__ = 'fixture'
    mod.__file__ = '<fixture>'
    sys.modules[name] = mod
class PreTrainedConfig:
    pass
class ACTConfig(PreTrainedConfig):
    pass
class ACTPolicy:
    config_class = ACTConfig
    def reset(self):
        pass
sys.modules['lerobot.configs'].PreTrainedConfig = PreTrainedConfig
sys.modules['lerobot.policies.act'].ACTConfig = ACTConfig
sys.modules['lerobot.policies.act'].ACTPolicy = ACTPolicy
if os.environ['VIDEO_CASE'] == 'absent':
    original = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *args, **kwargs: None if name == 'torchcodec' else original(name, *args, **kwargs)
exec(compile(sys.stdin.read(), '<production-training-verification>', 'exec'))
PY
chmod +x "${WORKSPACE}/venv/bin/python3"
VENV_PYTHON="${WORKSPACE}/venv/bin/python3"
SETUP_PROFILE=training
SETUP_PLATFORM_ID=ubuntu-22.04
USE_SUDO=false
source scripts/setup/verify_env.sh
export VIDEO_CASE=absent
verify_training_env > "${TMP_ROOT}/absent.log" 2>&1
grep -q 'check not applicable' "${TMP_ROOT}/absent.log"
ok 'platform without torchcodec does not require it'

printf 'raise RuntimeError("fixture missing FFmpeg libraries")\n' > "${WORKSPACE}/modules/torchcodec/decoders.py"
export VIDEO_CASE=broken
if verify_training_env > "${TMP_ROOT}/video-broken.log" 2>&1; then
    echo 'broken decoder incorrectly accepted' >&2; exit 1
fi
grep -q 'fixture missing FFmpeg libraries' "${TMP_ROOT}/video-broken.log"
grep -q 'training profile environment is incomplete' "${TMP_ROOT}/video-broken.log"
ok 'installed but unloadable torchcodec fails the training gate'
printf 'class VideoDecoder: pass\n' > "${WORKSPACE}/modules/torchcodec/decoders.py"
export VIDEO_CASE=ready
verify_training_env > "${TMP_ROOT}/video-ready.log" 2>&1
grep -q 'torchcodec FFmpeg libraries loadable' "${TMP_ROOT}/video-ready.log"
ok 'loadable torchcodec passes the real training verification function'
printf '\n%s passed, 0 failed\n' "${PASS}"
