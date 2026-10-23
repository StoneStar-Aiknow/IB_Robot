from unittest.mock import patch

import pytest

from perception_service.api_client import VLMAPIClient


@pytest.fixture(autouse=True)
def _no_host_proxy(monkeypatch):
    """A host exporting http_proxy would rewrite the request target as an absolute URI."""
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


class _FakeHTTPResponse:
    """``http.client.HTTPResponse``-shaped stand-in."""

    def __init__(self, body: str, status: int = 200):
        self._body = body.encode("utf-8")
        self.status = status
        self.will_close = False

    def read(self):
        return self._body


class _FakeConnection:
    """Records the requests the client makes over one kept-alive connection."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.timeout = None
        self.sock = None

    def request(self, method, path, body=None, headers=None):
        self.requests.append({"method": method, "path": path, "headers": dict(headers or {})})

    def getresponse(self):
        return self._responses.pop(0)

    def close(self):
        return None


def test_openai_compatible_allows_empty_api_key_env():
    connection = _FakeConnection([_FakeHTTPResponse('{"choices":[{"message":{"content":"scene summary"}}]}')])

    client = VLMAPIClient(
        provider="openai_compatible",
        base_url="http://localhost:8000/v1",
        api_key_env="",
        model="Qwen3.5-9B",
        timeout_sec=120.0,
    )

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        content, response = client.analyze([{"role": "user", "content": "describe"}])

    assert content == "scene summary"
    assert response["choices"][0]["message"]["content"] == "scene summary"
    assert connection.requests[0]["path"] == "/v1/chat/completions"
    assert connection.requests[0]["headers"]["Host"] == "localhost:8000"
    assert "Authorization" not in connection.requests[0]["headers"]


def test_invalid_json_response_raises_friendly_runtime_error():
    """Non-JSON HTTP bodies (e.g. HTML error pages) must surface as a
    :class:`RuntimeError` carrying a body preview, not a raw JSONDecodeError."""

    connection = _FakeConnection([_FakeHTTPResponse("<html>500 Internal Server Error</html>")])

    client = VLMAPIClient(
        provider="openai_compatible",
        base_url="http://localhost:8000/v1",
        api_key_env="",
        model="Qwen3.5-9B",
        timeout_sec=120.0,
    )

    with (
        patch.object(VLMAPIClient, "_open_connection", return_value=connection),
        pytest.raises(RuntimeError) as exc_info,
    ):
        client.analyze([{"role": "user", "content": "describe"}])

    message = str(exc_info.value)
    assert "invalid JSON" in message
    assert "<html>" in message
