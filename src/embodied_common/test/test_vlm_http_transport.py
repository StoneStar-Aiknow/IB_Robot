"""HTTP transport behaviour of the shared VLM client.

Every planning turn is one non-streaming request, so the connection setup is
paid in full on the user-visible delay. These tests pin the two things that
decide that cost -- the connection is reused, and a bounded answer can be
requested -- plus the failure mode reuse introduces: a socket the peer closed
while idle.
"""

import base64
import http.client
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from embodied_common.vlm_api_client import VLMAPIClient

_CHAT_REPLY = {"choices": [{"message": {"content": "ok"}}]}


class _FakeHTTPResponse:
    def __init__(self, value, status: int = 200, will_close: bool = False) -> None:
        self._body = json.dumps(value).encode()
        self.status = status
        self.will_close = will_close

    def read(self):
        return self._body


class _TruncatedHTTPResponse(_FakeHTTPResponse):
    """The status line arrived, then the peer dropped mid-body."""

    def read(self):
        raise http.client.IncompleteRead(b"{")


class _FakeConnection:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.requests = []
        self.closed = False
        self.timeout = None
        self.sock = None

    def request(self, method, path, body=None, headers=None):
        outcome = self._responses[len(self.requests)]
        self.requests.append(
            {"method": method, "path": path, "payload": json.loads(body), "headers": dict(headers or {})}
        )
        if isinstance(outcome, Exception):
            raise outcome

    def getresponse(self):
        return self._responses[len(self.requests) - 1]

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _no_host_proxy(monkeypatch):
    """Start every test without the host's proxy settings.

    urllib's getproxies() prefers the lowercase variables, so a developer
    machine exporting http_proxy would otherwise reroute the loopback servers
    here and override the uppercase ones the proxy tests set.
    """
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


def _client(**overrides) -> VLMAPIClient:
    options = {
        "provider": "openai_compatible",
        "base_url": "https://example.test/v1",
        "api_key_env": "",
        "model": "test-model",
        "timeout_sec": 30.0,
        "temperature": 0.0,
        "multimodal": False,
    }
    options.update(overrides)
    return VLMAPIClient(**options)


def test_one_connection_is_reused_across_calls():
    """A second turn must not pay DNS, TCP and TLS again."""
    connection = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY), _FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection) as opened:
        assert client.complete([{"role": "user", "content": "a"}])["status"] == "ok"
        assert client.complete([{"role": "user", "content": "b"}])["status"] == "ok"

    assert opened.call_count == 1
    assert len(connection.requests) == 2
    assert connection.requests[0]["headers"]["Connection"] == "keep-alive"


@pytest.mark.parametrize(
    "stale_error",
    [http.client.RemoteDisconnected("closed"), ConnectionResetError("reset"), ssl.SSLEOFError("EOF")],
)
def test_a_kept_alive_connection_the_peer_closed_is_retried_once(stale_error):
    """An idle-closed socket is not a model error; it must not surface as one.

    A load balancer that drops an idle TLS connection without close_notify
    surfaces as SSLEOFError rather than a reset, so it counts too.
    """
    stale = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY), stale_error])
    fresh = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", side_effect=[stale, fresh]):
        assert client.complete([{"role": "user", "content": "a"}])["status"] == "ok"
        result = client.complete([{"role": "user", "content": "b"}])

    assert result["status"] == "ok"
    assert stale.closed is True
    assert len(fresh.requests) == 1


def test_a_fresh_connection_failure_is_reported_not_retried():
    """Only a reused socket can be stale; a new one failing is a network error."""
    attempts = [
        _FakeConnection([http.client.RemoteDisconnected("closed")]),
        _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)]),
    ]
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", side_effect=attempts) as opened:
        result = client.complete([{"role": "user", "content": "a"}])

    assert result["status"] == "error"
    assert "network error" in result["error"]
    assert opened.call_count == 1


def test_a_second_connection_failure_is_reported():
    client = _client()
    attempts = [
        _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY), http.client.RemoteDisconnected("closed")]),
        _FakeConnection([http.client.RemoteDisconnected("closed again")]),
    ]

    with patch.object(VLMAPIClient, "_open_connection", side_effect=attempts):
        client.complete([{"role": "user", "content": "a"}])
        result = client.complete([{"role": "user", "content": "b"}])

    assert result["status"] == "error"
    assert "network error" in result["error"]


def test_a_failure_after_the_status_line_is_not_sent_again():
    """The model already has the request; re-sending would pay for it twice."""
    connection = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY), _TruncatedHTTPResponse(_CHAT_REPLY)])
    spare = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", side_effect=[connection, spare]):
        client.complete([{"role": "user", "content": "a"}])
        result = client.complete([{"role": "user", "content": "b"}])

    assert result["status"] == "error"
    assert len(connection.requests) == 2
    assert spare.requests == []


def test_a_response_marked_will_close_drops_the_connection():
    first = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY, will_close=True)])
    second = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", side_effect=[first, second]) as opened:
        client.complete([{"role": "user", "content": "a"}])
        client.complete([{"role": "user", "content": "b"}])

    assert first.closed is True
    assert opened.call_count == 2


def test_http_error_status_is_reported_with_a_body_preview():
    connection = _FakeConnection([_FakeHTTPResponse({"error": "bad model"}, status=404)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}])

    assert result["status"] == "error"
    assert "HTTP 404" in result["error"]
    assert "bad model" in result["error"]


@pytest.mark.parametrize(
    ("protocol", "field"),
    [("chat_completions", "max_tokens"), ("responses", "max_output_tokens")],
)
def test_max_tokens_bounds_the_answer_per_protocol(protocol, field):
    reply = _CHAT_REPLY if protocol == "chat_completions" else {"output_text": "ok"}
    connection = _FakeConnection([_FakeHTTPResponse(reply)])
    client = _client(api_protocol=protocol)

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        client.complete([{"role": "user", "content": "a"}], max_tokens=256)

    assert connection.requests[0]["payload"][field] == 256


def test_max_tokens_is_absent_when_not_configured():
    connection = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        client.complete([{"role": "user", "content": "a"}])

    assert "max_tokens" not in connection.requests[0]["payload"]


def test_extra_body_carries_a_provider_specific_thinking_switch():
    """Providers spell "do not think" differently; the switch comes from config."""
    connection = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        client.complete(
            [{"role": "user", "content": "a"}],
            enable_thinking=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

    assert connection.requests[0]["payload"]["thinking"] == {"type": "disabled"}
    # The dashscope-only field must not leak to other providers, which reject
    # unknown request fields.
    assert "enable_thinking" not in connection.requests[0]["payload"]


def test_dashscope_keeps_its_native_thinking_switch():
    connection = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        client.complete([{"role": "user", "content": "a"}], enable_thinking=False)

    assert connection.requests[0]["payload"]["enable_thinking"] is False


def test_close_releases_the_connection():
    connection = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        client.complete([{"role": "user", "content": "a"}])
    client.close()

    assert connection.closed is True


class _StreamResponse:
    """http.client.HTTPResponse-shaped SSE body."""

    def __init__(self, chunks, status=200):
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self._body = body.encode()
        self._offset = 0
        self.status = status
        self.will_close = False

    def read(self, size=None):
        if size is None:
            block, self._offset = self._body[self._offset :], len(self._body)
            return block
        block = self._body[self._offset : self._offset + size]
        self._offset += len(block)
        return block

    read1 = read

    def isclosed(self):
        return self._offset >= len(self._body)


def _delta(content=None, reasoning=None, usage=None):
    chunk = {"choices": [{"delta": {}}]}
    if content is not None:
        chunk["choices"][0]["delta"]["content"] = content
    if reasoning is not None:
        chunk["choices"][0]["delta"]["reasoning_content"] = reasoning
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def test_streaming_returns_the_same_shape_as_a_blocking_call():
    """Streaming is a transport choice; callers that ignore the callback
    must not be able to tell the difference."""
    connection = _FakeConnection(
        [
            _StreamResponse(
                [
                    _delta('{"user_message": '),
                    _delta('"你好"}'),
                    _delta(usage={"prompt_tokens": 700, "completion_tokens": 9, "total_tokens": 709}),
                ]
            )
        ]
    )
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}], stream=True)

    assert result["status"] == "ok"
    assert result["content"] == '{"user_message": "你好"}'
    assert result["usage"]["prompt_tokens"] == 700
    assert connection.requests[0]["payload"]["stream"] is True


def test_streaming_feeds_the_observer_the_text_accumulated_so_far():
    """The observer needs a growing prefix, not isolated fragments, so it can
    tell when a field has closed."""
    connection = _FakeConnection(
        [
            _StreamResponse(
                [
                    _delta('{"user_message"'),
                    _delta(': "好的"'),
                    _delta(', "steps": []}'),
                ]
            )
        ]
    )
    client = _client()
    seen = []

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        client.complete([{"role": "user", "content": "a"}], stream=True, on_content_delta=seen.append)

    assert seen == ['{"user_message"', '{"user_message": "好的"', '{"user_message": "好的", "steps": []}']


def test_streaming_separates_thinking_from_content():
    connection = _FakeConnection(
        [
            _StreamResponse(
                [
                    _delta(reasoning="想一想"),
                    _delta(reasoning="再想想"),
                    _delta('{"a": 1}'),
                ]
            )
        ]
    )
    client = _client()
    seen = []

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}], stream=True, on_content_delta=seen.append)

    assert result["content"] == '{"a": 1}'
    assert result["reasoning"] == "想一想再想想"
    assert seen == ['{"a": 1}'], "thinking must not reach the speech observer"


def test_a_broken_observer_cannot_fail_the_call():
    connection = _FakeConnection([_StreamResponse([_delta('{"a": 1}')])])
    client = _client()

    def explode(_partial):
        raise RuntimeError("tts down")

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}], stream=True, on_content_delta=explode)

    assert result["status"] == "ok"


def test_a_streaming_http_error_is_reported_not_raised():
    connection = _FakeConnection([_StreamResponse([], status=429)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}], stream=True)

    assert result["status"] == "error"
    assert "HTTP 429" in result["error"]


def test_an_empty_stream_is_an_error_not_an_empty_success():
    connection = _FakeConnection([_StreamResponse([])])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}], stream=True)

    assert result["status"] == "error"


def test_a_stream_read_to_its_end_keeps_the_connection():
    """A streaming planner is the one that most needs keep-alive."""
    connection = _FakeConnection([_StreamResponse([_delta('{"a": 1}')]), _StreamResponse([_delta('{"b": 2}')])])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection) as opened:
        client.complete([{"role": "user", "content": "a"}], stream=True)
        result = client.complete([{"role": "user", "content": "b"}], stream=True)

    assert result["content"] == '{"b": 2}'
    assert opened.call_count == 1
    assert connection.closed is False


class _BrokenStreamResponse(_StreamResponse):
    def read(self, size=None):
        raise http.client.IncompleteRead(b"")

    read1 = read


def test_an_abandoned_stream_drops_the_connection():
    """Unread bytes would be read by the next request as its own response."""
    first = _FakeConnection([_BrokenStreamResponse([_delta('{"a": 1}')])])
    second = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", side_effect=[first, second]) as opened:
        assert client.complete([{"role": "user", "content": "a"}], stream=True)["status"] == "error"
        client.complete([{"role": "user", "content": "b"}])

    assert first.closed is True
    assert opened.call_count == 2


def test_streaming_is_refused_for_the_responses_protocol():
    """Its events are not chat deltas; it would read as an empty answer."""
    client = _client(api_protocol="responses")

    with patch.object(VLMAPIClient, "_open_connection") as opened:
        result = client.complete([{"role": "user", "content": "a"}], stream=True)

    assert result["status"] == "error"
    assert "chat_completions" in result["error"]
    opened.assert_not_called()


def _proxied(monkeypatch, proxy, base_url):
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    monkeypatch.setenv("HTTP_PROXY", proxy)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    return _client(base_url=base_url)


def test_an_https_target_tunnels_through_the_proxy_with_its_credentials(monkeypatch):
    client = _proxied(monkeypatch, "http://user:p%40ss@proxy.test", "https://ark.test/api/v3")

    connection = client._open_connection(5.0)

    assert isinstance(connection, http.client.HTTPSConnection)
    # As urllib (this client's transport before keep-alive): a proxy without
    # a port gets the target connection's default, 443 for an HTTPS target.
    assert (connection.host, connection.port) == ("proxy.test", 443)
    assert connection._tunnel_host == "ark.test"
    expected = "Basic " + base64.b64encode(b"user:p@ss").decode()
    assert connection._tunnel_headers == {"Proxy-Authorization": expected}
    assert client._forwarding() == ("", {})


def test_an_http_target_is_forwarded_by_absolute_uri(monkeypatch):
    """Plain HTTP is not tunnelled: many proxies refuse CONNECT to port 80."""
    client = _proxied(monkeypatch, "http://user:pw@proxy.test:3128", "http://llm.test:8000/v1")
    connection = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])

    with patch.object(http.client, "HTTPConnection", return_value=connection) as built:
        assert client.complete([{"role": "user", "content": "a"}])["status"] == "ok"

    built.assert_called_once_with("proxy.test", 3128, timeout=30.0)
    request = connection.requests[0]
    assert request["path"] == "http://llm.test:8000/v1/chat/completions"
    assert request["headers"]["Proxy-Authorization"] == "Basic " + base64.b64encode(b"user:pw").decode()


def test_an_https_proxy_url_still_tunnels_an_https_target(monkeypatch):
    """urllib ignores the proxy URL scheme for an HTTPS target and CONNECTs in
    plain TCP, so HTTPS_PROXY=https://... worked before keep-alive; it must
    keep working."""
    client = _proxied(monkeypatch, "https://proxy.test:3128", "https://ark.test/api/v3")

    connection = client._open_connection(5.0)

    assert (connection.host, connection.port) == ("proxy.test", 3128)
    assert connection._tunnel_host == "ark.test"


@pytest.mark.parametrize("proxy", ["http://user:@proxy.test:3128", "http://user@proxy.test:3128"])
def test_proxy_credentials_need_both_user_and_password(monkeypatch, proxy):
    """urllib sends Proxy-Authorization only when both parts are non-empty."""
    client = _proxied(monkeypatch, proxy, "https://ark.test/api/v3")

    connection = client._open_connection(5.0)

    assert connection._tunnel_headers == {}


def test_a_tls_proxy_for_a_plain_http_target_is_refused(monkeypatch):
    client = _proxied(monkeypatch, "https://proxy.test:8443", "http://llm.test:8000/v1")

    result = client.complete([{"role": "user", "content": "a"}])

    assert result["status"] == "error"
    assert "must be http://" in result["error"]


class _BlockingConnection(_FakeConnection):
    """Holds its response until released, like a model still generating."""

    def __init__(self, responses) -> None:
        super().__init__(responses)
        self.started = threading.Event()
        self.release = threading.Event()

    def getresponse(self):
        self.started.set()
        assert self.release.wait(5)
        return super().getresponse()


def test_a_concurrent_call_does_not_wait_for_another_callers_request():
    """perception_service shares one client across request threads; a call
    must not queue, unbounded by its own timeout, behind another's model
    call."""
    busy = _BlockingConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    other = _FakeConnection([_FakeHTTPResponse(_CHAT_REPLY)])
    client = _client()
    results = {}

    with patch.object(VLMAPIClient, "_open_connection", side_effect=[busy, other]):
        worker = threading.Thread(target=lambda: results.update(a=client.complete([{"role": "user", "content": "a"}])))
        worker.start()
        assert busy.started.wait(5)
        results["b"] = client.complete([{"role": "user", "content": "b"}])
        busy.release.set()
        worker.join(5)

    assert results["a"]["status"] == "ok"
    assert results["b"]["status"] == "ok"
    assert len(other.requests) == 1
    # One idle connection is kept; the surplus one is closed, not leaked.
    assert [busy.closed, other.closed].count(True) == 1


def test_streamed_tool_call_fragments_are_merged_like_a_blocking_call():
    fragments = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "type": "function", "function": {"name": "wave", "arguments": ""}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"times"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 2}"}}]}}]},
    ]
    connection = _FakeConnection([_StreamResponse(fragments)])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}], stream=True)

    assert result["status"] == "ok"
    assert result["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "wave", "arguments": '{"times": 2}'}}
    ]


@pytest.mark.parametrize("event", ['"text"', "[1, 2]", '{"choices": ["x"]}', '{"choices": [{"delta": "x"}]}'])
def test_a_malformed_stream_event_is_skipped_not_raised(event):
    """complete() reports failures as a status, never as an exception."""

    class _RawStream(_StreamResponse):
        def __init__(self) -> None:
            super().__init__([])
            self._body = f"data: {event}\n\n".encode() + self._body

    connection = _FakeConnection([_RawStream()])
    client = _client()

    with patch.object(VLMAPIClient, "_open_connection", return_value=connection):
        result = client.complete([{"role": "user", "content": "a"}], stream=True)

    assert result["status"] == "error"
    assert "empty content" in result["error"]


def test_a_streamed_delta_reaches_the_observer_before_the_stream_ends():
    """Over a real chunked response, not a fake one: read(n) on a chunked
    body keeps reading across chunks until it has n bytes, so a delta would
    wait for the events behind it and the early reply would arrive late."""
    observed = threading.Event()
    held = {}

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(text):
                data = text.encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            send("data: " + json.dumps(_delta('{"user_message": "hi"}')) + "\n\n")
            # The rest of the answer is withheld until the observer has seen
            # the first delta (or a generous deadline passes).
            held["seen_in_time"] = observed.wait(5)
            send("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = _client(base_url=f"http://127.0.0.1:{server.server_port}/v1")
        with patch.dict("os.environ", {"NO_PROXY": "*", "no_proxy": "*"}):
            started = time.monotonic()
            result = client.complete(
                [{"role": "user", "content": "a"}], stream=True, on_content_delta=lambda _text: observed.set()
            )
        client.close()
    finally:
        server.shutdown()
        server.server_close()

    assert result["status"] == "ok"
    assert held["seen_in_time"] is True
    assert time.monotonic() - started < 4


def test_a_stream_sent_with_content_length_leaves_a_usable_connection():
    """A buffering gateway, or a provider ignoring ``stream``, answers with
    Content-Length. Before Python 3.13 read1() returns b"" at its end without
    closing the response, and a connection parked like that failed the next
    request as ResponseNotReady after sending it -- every other call failed
    and was billed."""
    received = []

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            received.append(self.client_address)
            body = ("data: " + json.dumps(_delta('{"a": 1}')) + "\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = _client(base_url=f"http://127.0.0.1:{server.server_port}/v1")
        results = [client.complete([{"role": "user", "content": str(n)}], stream=True) for n in range(3)]
        client.close()
    finally:
        server.shutdown()
        server.server_close()

    assert [result["status"] for result in results] == ["ok", "ok", "ok"]
    assert len(received) == 3
    # Still one kept-alive connection, not a fresh one per call.
    assert len(set(received)) == 1
