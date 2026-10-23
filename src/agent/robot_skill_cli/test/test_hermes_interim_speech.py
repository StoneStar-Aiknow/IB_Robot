import io
import json
from types import SimpleNamespace

import pytest

from robot_skill_cli import hermes_interim_speech as speech


@pytest.mark.parametrize(
    "text", ["正在查询。", '{"schema_version":1,"speech":"正在', "`/semantic_mapping/get_objects`"]
)
def test_interim_ignores_unstructured_or_partial_text(monkeypatch, text):
    monkeypatch.setenv("IBROBOT_INTERIM_SPEAKER", "/test/speak")
    calls = []
    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    speech.handle({"hook_event_name": "on_interim_message", "session_id": "s", "extra": {"turn_id": "t", "text": text}})
    assert calls == []


def test_interim_emits_same_event_contract(monkeypatch):
    monkeypatch.setenv("IBROBOT_INTERIM_SPEAKER", "/test/speak")

    class Pipe(io.StringIO):
        def close(self):
            pass

    pipe = Pipe()
    monkeypatch.setattr(speech.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(stdin=pipe))
    response = {"schema_version": 1, "speech": "正在查询。", "details": ""}
    speech.handle(
        {
            "hook_event_name": "on_interim_message",
            "session_id": "s",
            "extra": {"turn_id": "t", "text": json.dumps(response)},
        }
    )
    event = json.loads(pipe.getvalue())
    assert (event["session_id"], event["turn_id"], event["phase"]) == ("s", "t", "interim")
    assert event["response"] == response
