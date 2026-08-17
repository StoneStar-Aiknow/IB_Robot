"""Tests for the OpenAI-compatible Responses API transport."""

import json
from unittest.mock import patch

from embodied_common.vlm_api_client import VLMAPIClient


class _FakeResponse:
    def __init__(self, value):
        self._body = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._body


def _client() -> VLMAPIClient:
    return VLMAPIClient(
        provider="openai_compatible",
        base_url="https://example.test/v1",
        api_key_env="",
        model="gpt-5.6-sol",
        timeout_sec=30.0,
        api_protocol="responses",
    )


def test_responses_api_uses_responses_endpoint_and_multimodal_input() -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "bin"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            }
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
                {"type": "text", "text": "identify"},
            ],
        }
    ]
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _client().complete(messages)

    assert result["status"] == "ok"
    assert result["content"] == "bin"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["payload"]["input"][0]["content"] == [
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="},
        {"type": "input_text", "text": "identify"},
    ]


def test_responses_api_normalizes_function_calls() -> None:
    response = {"output": [{"type": "function_call", "call_id": "call-1", "name": "pick", "arguments": '{"id":1}'}]}
    with patch("urllib.request.urlopen", return_value=_FakeResponse(response)):
        result = _client().complete(
            [{"role": "user", "content": "pick"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "pick", "description": "Pick", "parameters": {"type": "object"}},
                }
            ],
        )

    assert result["status"] == "ok"
    assert result["tool_calls"][0]["function"] == {"name": "pick", "arguments": '{"id":1}'}
