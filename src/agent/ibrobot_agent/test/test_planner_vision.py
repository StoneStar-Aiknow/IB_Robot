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
