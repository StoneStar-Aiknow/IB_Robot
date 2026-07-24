#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/convert_hmm.sh <policy>

Required environment:
  MODEL_BUNDLE_ROOT  Workspace-relative input model bundle path

Supported policies:
  pi05
  smolvla
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    pi05)
        exec "${WORKSPACE}/src/model_utils/model_utils/pi05_export/convert_hmm.sh"
        ;;
    smolvla)
        exec "${WORKSPACE}/src/model_utils/model_utils/smolvla_export/convert_hmm.sh"
        ;;
    -h | --help)
        usage
        ;;
    *)
        printf 'Unsupported HMM policy: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac
