"""Opt-in camera frames for the VLM planner (planner.vision_topics)."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest
from sensor_msgs.msg import Image

from ibrobot_agent.node import _LatestCameraFrames, _planner_from_config, _planner_vision_topics

CONFIG = {
    "mode": "vlm",
    "provider": "openai_compatible",
    "base_url": "http://127.0.0.1:9/v1",
    "api_key_env": "",
    "model": "unit-vlm",
}


class _FakeNode:
    def __init__(self):
        self.subscriptions = {}

    def create_subscription(self, _type, topic, callback, _qos):
        self.subscriptions[topic] = callback


def _frame(width=4, height=2):
    msg = Image()
    msg.width, msg.height, msg.encoding, msg.step = width, height, "rgb8", width * 3
    msg.data = bytes([10, 20, 30] * width * height)
    return msg


def test_without_vision_topics_the_planner_stays_text_only():
    assert _planner_vision_topics(json.dumps(CONFIG)) == []
    planner = _planner_from_config(json.dumps(CONFIG))
    assert planner._model_client._multimodal is False


def test_vision_topics_must_be_absolute():
    with pytest.raises(ValueError):
        _planner_vision_topics(json.dumps({**CONFIG, "vision_topics": ["camera/image"]}))


def test_latest_frames_are_attached_to_the_user_message():
    node = _FakeNode()
    frames = _LatestCameraFrames(node, ["/cam/head", "/cam/wrist"])
    node.subscriptions["/cam/head"](_frame())
    planner = _planner_from_config(json.dumps({**CONFIG, "vision_topics": ["/cam/head", "/cam/wrist"]}), images=frames)
    assert planner._model_client._multimodal is True

    request = SimpleNamespace(text="pick it up", reply_to_request_id="")
    messages = frames.prompt_builder(request=request, context={}, catalog={"skills": []})
    content = messages[-1]["content"]
    assert content[0]["type"] == "text" and "pick it up" in content[0]["text"]
    assert content[1] == {"type": "text", "text": "[camera /cam/head]"}
    url = content[2]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,") and base64.b64decode(url.split(",", 1)[1])[:2] == b"\xff\xd8"
    assert content[3] == {"type": "text", "text": "[camera /cam/wrist: no frame]"}


def _stamped(sec):
    msg = _frame()
    msg.header.stamp.sec = int(sec)
    msg.header.stamp.nanosec = int((sec - int(sec)) * 1e9)
    return msg


def _attached(frames):
    request = SimpleNamespace(text="t", reply_to_request_id="")
    return [
        p["text"]
        for p in frames.prompt_builder(request=request, context={}, catalog={"skills": []})[-1]["content"]
        if p["type"] == "text"
    ][1:]


def test_frames_of_an_earlier_session_never_reach_the_prompt():
    node = _FakeNode()
    frames = _LatestCameraFrames(node, ["/cam/head", "/cam/wrist"], epoch_topic="/simulation/clock")
    frames.on_epoch({"session_id": "s0001", "generation": 1, "sim_time_total_s": 0.0, "sim_time_s": 0.0})
    node.subscriptions["/cam/head"](_stamped(4.0))
    node.subscriptions["/cam/wrist"](_stamped(4.0))
    assert _attached(frames) == ["[camera /cam/head]", "[camera /cam/wrist]"]
    # reset at simulated 12.0 s: the cache is emptied; a late frame of the old session is refused
    frames.on_epoch({"session_id": "s0003", "generation": 3, "sim_time_total_s": 12.0, "sim_time_s": 0.0})
    assert _attached(frames) == ["[camera /cam/head: no frame]", "[camera /cam/wrist: no frame]"]
    node.subscriptions["/cam/head"](_stamped(4.0))
    node.subscriptions["/cam/head"](_stamped(11.96))
    assert frames.dropped_frames == 2
    # the new session's first observation may carry the old session's last stamp (time stands still)
    node.subscriptions["/cam/head"](_stamped(12.0))
    assert _attached(frames) == ["[camera /cam/head]", "[camera /cam/wrist: no frame]"]  # missing camera stays missing
    # later clock messages of the same session (time moved on) keep the cache
    frames.on_epoch({"session_id": "s0003", "generation": 3, "sim_time_total_s": 13.0, "sim_time_s": 1.0})
    assert _attached(frames) == ["[camera /cam/head]", "[camera /cam/wrist: no frame]"]


def test_the_epoch_topic_must_be_absolute():
    from ibrobot_agent.node import _planner_vision_epoch_topic

    assert _planner_vision_epoch_topic(json.dumps({**CONFIG, "vision_epoch_topic": "/simulation/clock"})) == (
        "/simulation/clock"
    )
    with pytest.raises(ValueError):
        _planner_vision_epoch_topic(json.dumps({**CONFIG, "vision_epoch_topic": "clock"}))
