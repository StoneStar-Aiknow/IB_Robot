#!/bin/bash
# Quick-start an LTTng trace session for IB-Robot.
# Usage: bash scripts/tracing/start_trace.sh [session_name] [output_dir]
set -euo pipefail

DEFAULT_SESSION="ib_robot_trace"
REQUESTED_SESSION="${1:-${DEFAULT_SESSION}}"
OUTPUT_DIR="${2:-${HOME}/.ros/tracing}"
mkdir -p "${OUTPUT_DIR}"

session_exists() {
    lttng list "$1" >/dev/null 2>&1
}

SESSION="${REQUESTED_SESSION}"
TRACE_DIR="${OUTPUT_DIR}/${SESSION}"
if session_exists "${SESSION}" || [[ -e "${TRACE_DIR}" ]]; then
    if [[ $# -gt 0 ]]; then
        echo "Error: tracing session '${SESSION}' already exists or output directory already exists at ${TRACE_DIR}." >&2
        echo "Choose a different session name or clean up the existing trace first." >&2
        exit 1
    fi

    suffix="$(date +%Y%m%d_%H%M%S)"
    index=0
    while true; do
        candidate_suffix="${suffix}"
        if [[ "${index}" -gt 0 ]]; then
            candidate_suffix="${suffix}_${index}"
        fi
        SESSION="${DEFAULT_SESSION}_${candidate_suffix}"
        TRACE_DIR="${OUTPUT_DIR}/${SESSION}"
        if ! session_exists "${SESSION}" && [[ ! -e "${TRACE_DIR}" ]]; then
            break
        fi
        index=$((index + 1))
    done
    echo "Note: default tracing session already exists; using unique session '${SESSION}'."
fi

lttng create "${SESSION}" --output="${TRACE_DIR}"
lttng enable-event --session "${SESSION}" --userspace 'ros2:*'
lttng enable-event --session "${SESSION}" --python 'ib_trace.*'
lttng start "${SESSION}"

echo "Trace '${SESSION}' ACTIVE → ${TRACE_DIR}"
echo "Stop with: bash scripts/tracing/stop_trace.sh ${SESSION}"
