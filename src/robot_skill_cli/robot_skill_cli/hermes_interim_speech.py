"""Compatibility entry for structured interim events; never speaks raw prose."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from robot_skill_cli.hermes_response import ResponseFormatError, SpeechEvent, load_json, parse_response


def handle(payload: dict) -> None:
    if payload.get("hook_event_name") != "on_interim_message":
        return
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    session, turn = payload.get("session_id"), extra.get("turn_id")
    speaker = os.environ.get("IBROBOT_INTERIM_SPEAKER", "")
    if not session or not turn or not speaker:
        return
    try:
        response = parse_response(extra.get("text"))
    except ResponseFormatError:
        return
    if not response.speech:
        return
    event = SpeechEvent(session, turn, "interim", response)
    process = subprocess.Popen(
        [speaker],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
        close_fds=True,
    )
    if process.stdin is not None:
        process.stdin.write(json.dumps(event.to_dict(), ensure_ascii=False))
        process.stdin.close()


def main() -> int:
    try:
        value = load_json(sys.stdin.read())
        if isinstance(value, dict):
            handle(value)
    except (ResponseFormatError, OSError, subprocess.SubprocessError):
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
