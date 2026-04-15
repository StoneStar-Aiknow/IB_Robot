#!/bin/bash
# Stop an LTTng trace session and show analysis hints.
# Usage: bash scripts/tracing/stop_trace.sh [session_name]
set -euo pipefail

SESSION="${1:-ib_robot_trace}"
TRACE_DIR="${HOME}/.ros/tracing/${SESSION}"

lttng stop "${SESSION}" 2>/dev/null || echo "Session may not be active"
lttng destroy "${SESSION}" 2>/dev/null || echo "Session may not exist"

if [ -d "${TRACE_DIR}" ]; then
    echo "Trace saved: ${TRACE_DIR} ($(du -sh "${TRACE_DIR}" | cut -f1))"
    echo ""
    echo "Analyze:"
    echo "  python3 scripts/tracing/analyze_trace.py --trace-dir ${TRACE_DIR}"
    echo "  babeltrace2 ${TRACE_DIR} | head -50"
    echo "  # Or open with Trace Compass"
fi
