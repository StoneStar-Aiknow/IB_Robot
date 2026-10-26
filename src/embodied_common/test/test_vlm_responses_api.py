"""Tests for the OpenAI-compatible Responses API transport."""

import json
from unittest.mock import patch

from embodied_common.vlm_api_client import VLMAPIClient


class _FakeHTTPResponse:
    """``http.client.HTTPResponse``-shaped stand-in."""

    def __init__(self, value, status: int = 200, will_close: bool = False) -> None:
        self._body = json.dumps(value).encode()
        self.status = status
        self.will_close = will_close

    def read(self):
        return self._body


class _FakeConnection:
    """Records the requests the client makes over one kept-alive connection."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.requests = []
        self.closed = False
        self.timeout = None
        self.sock = None

    def request(self, method, path, body=None, headers=None):
        self.requests.append(
            {"method": method, "path": path, "payload": json.loads(body), "headers": dict(headers or {})}
        )

    def getresponse(self):
        return self._responses.pop(0)

    def close(self):
        self.closed = True


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
    connection = _FakeConnection(
        [
            _FakeHTTPResponse(
                {
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "bin"}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                }
            )
        ]
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
    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = _client().complete(messages)

    assert result["status"] == "ok"
    assert result["content"] == "bin"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert connection.requests[0]["path"] == "/v1/responses"
    assert connection.requests[0]["headers"]["Host"] == "example.test"
    assert connection.requests[0]["payload"]["input"][0]["content"] == [
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="},
        {"type": "input_text", "text": "identify"},
    ]


def test_responses_api_normalizes_function_calls() -> None:
    response = {"output": [{"type": "function_call", "call_id": "call-1", "name": "pick", "arguments": '{"id":1}'}]}
    with patch.object(VLMAPIClient, "_open_connection", return_value=_FakeConnection([_FakeHTTPResponse(response)])):
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
