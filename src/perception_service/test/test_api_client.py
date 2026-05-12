from unittest.mock import patch

from perception_service.api_client import VLMAPIClient


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def test_openai_compatible_allows_empty_api_key_env():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return _FakeResponse('{"choices":[{"message":{"content":"scene summary"}}]}')

    client = VLMAPIClient(
        provider="openai_compatible",
        base_url="http://localhost:8000/v1",
        api_key_env="",
        model="Qwen3.5-9B",
        timeout_sec=120.0,
    )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        content, response = client.analyze([{"role": "user", "content": "describe"}])

    assert content == "scene summary"
    assert response["choices"][0]["message"]["content"] == "scene summary"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in captured["headers"]
