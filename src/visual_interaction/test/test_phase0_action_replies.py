"""Replay wiring for natural replies, not a claim about model language quality."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from visual_interaction.phase0 import action_probe


def test_natural_reply_has_frame_slots_without_demonstration_utterances():
    prompt = action_probe._variants(2)["natural_reply"]

    assert prompt.count("<image>") == 2
    assert "<|im_start|>assistant\n" in prompt
    # These were copied into unrelated poses in the seven-action replay.
    assert "你正在整理的样子很专注" not in prompt
    assert "例如" not in prompt


@pytest.mark.parametrize(
    "raw,kind,category,reply",
    [
        ('{"kind":"none","category":"","reply":""}', "none", "", ""),
        ('{"kind":"uncertain","category":"","reply":""}', "uncertain", "", ""),
        ('{"kind":"action","category":"gesture","reply":""}', "action", "gesture", ""),
        ('{"kind":"action","category":"gesture","reply":"你好呀！"}', "action", "gesture", "你好呀！"),
    ],
)
def test_replay_preserves_silence_or_a_direct_response(raw, kind, category, reply):
    answer = action_probe._answer(raw)

    assert answer == {"parsed": True, "kind": kind, "category": category, "reply": reply}


@pytest.mark.parametrize("selected", [None, ["bare_list", "two_step_category"]])
def test_replay_runs_only_selected_prompts_and_keeps_replies(tmp_path, monkeypatch, selected):
    frames = tmp_path / "frames" / "0001"
    frames.mkdir(parents=True)
    for index in range(2):
        (frames / f"{index}.jpg").write_bytes(b"fake-jpeg")
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2]
    generator = Mock()
    generator.run.return_value = {"tokens": [3]}
    answer = '{"kind":"action","category":"gesture","reply":"你好呀！"}'
    detokenizer = Mock()
    detokenizer.decode.return_value = answer
    vision = Mock(return_value=[np.zeros((4, 256, 1), dtype=np.float32)])
    live_chat = SimpleNamespace(preprocess=lambda _: np.zeros((2, 3, 448, 448), dtype=np.float32))
    monkeypatch.setattr(action_probe, "_load", lambda _: (live_chat, tokenizer, detokenizer, vision, generator))
    output = tmp_path / "measurement.json"
    args = ["--demo-root", str(tmp_path), "--frames-root", str(frames.parent), "--output", str(output)]
    if selected:
        args += ["--variants", *selected]

    assert action_probe.main(args) == 0

    report = json.loads(output.read_text())
    expected = selected or ["natural_reply"]
    assert report["conditions"]["variants"] == expected
    records = report["results"]["records"]
    assert [record["variant"] for record in records] == expected
    assert all(record["reply"] == "你好呀！" for record in records)
    assert generator.run.call_count == len(expected)
    assert [call.args[0] for call in tokenizer.encode.call_args_list] == [
        action_probe._variants(2)[name].replace("<image>", "<img>" + "<IMG_CONTEXT>" * 256 + "</img>")
        for name in expected
    ]
