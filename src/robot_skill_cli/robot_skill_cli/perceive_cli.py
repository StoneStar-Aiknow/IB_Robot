"""ibrobot-perceive: read-only perception topic reader with hard allowlist and audit log.

This wrapper is the only sanctioned path for an LLM to read ROS perception topics.
It enforces a hard-coded topic/field allowlist, logs every read for audit, runs
with a bounded timeout, and extracts the requested field from the ``ros2 topic echo``
YAML output. Direct ``ros2`` subcommands remain forbidden by POLICY.md.

Allowlisted topics map to real publishers in the IB-Robot stack:
  /voice/speech_direction  — voice_asr_service SpeechDirection (azimuth_rad, seq_id)
  /joint_states            — sensor_msgs JointState (position)

Point-in-time read, not a persistent snapshot:
  The wrapper calls ``ros2 topic echo --once`` which returns the *next* message
  published on the topic within the timeout. For volatile event topics such as
  ``/voice/speech_direction`` (published only when voice activity is detected)
  there is no latched/persistent state to read; the returned value is a single
  point-in-time sample that may already be stale by the time it is consumed.
  Callers must treat the value as open-loop and may not assume it reflects the
  current world state at execution time. This is a deliberate contract, not a
  cache; do not widen it by wrapping the wrapper in a cache without declaring
  the freshness/liveness semantics here and in POLICY.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PERCEPTION_ALLOWLIST: dict[str, set[str]] = {
    "/joint_states": {"position"},
    "/voice/speech_direction": {"azimuth_rad", "seq_id"},
}

LOG_PATH = Path("/tmp/hermes-perceive.log")
TIMEOUT_SEC = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibrobot-perceive",
        description="Read one perception topic field via the allowlisted wrapper.",
    )
    parser.add_argument("--topic", required=True, help="ROS topic name (must be in the allowlist)")
    parser.add_argument("--field", required=True, help="top-level YAML field to extract")
    return parser


def _log(topic: str, field: str, status: str) -> None:
    ts = datetime.now().astimezone().isoformat()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(f"{ts} topic={topic} field={field} status={status}\n")


def _extract_field(stdout: str, field: str) -> object | None:
    marker = stdout.find("---\n")
    payload = stdout[marker:] if marker >= 0 else stdout
    docs = list(yaml.safe_load_all(payload))
    for doc in docs:
        if isinstance(doc, dict) and field in doc:
            return doc[field]
    return None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.topic not in PERCEPTION_ALLOWLIST:
        _log(args.topic, args.field, f"rejected_topic (allowlist={sorted(PERCEPTION_ALLOWLIST)})")
        print(f"ERROR: topic {args.topic} not in perception allowlist", file=sys.stderr)
        print(f"Allowlist: {', '.join(sorted(PERCEPTION_ALLOWLIST))}", file=sys.stderr)
        return 1

    if args.field not in PERCEPTION_ALLOWLIST[args.topic]:
        allowed = ", ".join(sorted(PERCEPTION_ALLOWLIST[args.topic]))
        _log(args.topic, args.field, f"rejected_field (allowed={allowed})")
        print(f"ERROR: field {args.field} not allowed for {args.topic}; allowed: {allowed}", file=sys.stderr)
        return 1

    try:
        result = subprocess.run(
            ["ros2", "topic", "echo", "--once", args.topic],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _log(args.topic, args.field, "timeout")
        print(f"ERROR: timed out reading {args.topic} ({TIMEOUT_SEC}s)", file=sys.stderr)
        return 1

    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[:1]
        detail_str = detail[0] if detail else "unknown"
        _log(args.topic, args.field, f"ros2_error: {detail_str[:200]}")
        print(f"ERROR: ros2 topic echo failed: {detail_str[:200]}", file=sys.stderr)
        return 1

    try:
        value = _extract_field(result.stdout, args.field)
    except yaml.YAMLError as e:
        _log(args.topic, args.field, f"parse_error: {e}")
        print(f"ERROR: failed to parse topic output: {e}", file=sys.stderr)
        return 1

    if value is None:
        _log(args.topic, args.field, "missing_field")
        print(f"ERROR: field {args.field} not found in topic data", file=sys.stderr)
        return 1

    _log(args.topic, args.field, f"ok: {value}")
    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
