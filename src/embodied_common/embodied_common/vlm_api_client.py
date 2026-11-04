"""OpenAI-compatible VLM API client shared by embodied runtime packages."""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import os
import pathlib
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from embodied_common.model_timeout import resolve_model_output_idle_timeout

_YAML_BASENAME = "vlm_models.yaml"

#: Failures of a reused keep-alive socket before any response byte arrived: the
#: peer closed it while idle (with or without a TLS close_notify). Only these,
#: and only on a reused socket, are retried -- once a status line has arrived
#: the request reached the model, and re-sending it would pay for it twice.
_STALE_CONNECTION_ERRORS = (
    http.client.RemoteDisconnected,
    ConnectionResetError,
    BrokenPipeError,
    ssl.SSLEOFError,
    ssl.SSLZeroReturnError,
)


def _close_quietly(connection: http.client.HTTPConnection | None) -> None:
    if connection is not None:
        with contextlib.suppress(OSError):
            connection.close()


def _merge_tool_call_deltas(calls: dict[int, dict[str, Any]], deltas: list[Any]) -> None:
    """Fold streamed tool-call fragments into whole calls, keyed by ``index``.

    A streamed call arrives as fragments: the first carries ``id``, ``type``
    and the function name, later ones only pieces of ``function.arguments``.
    Merging them gives the same shape a blocking call returns.
    """
    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        index = delta.get("index")
        if index is None:
            # Without an index, an id starts a new call and anything else
            # continues the latest one.
            index = len(calls) if delta.get("id") or not calls else max(calls)
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if isinstance(delta.get("id"), str) and delta["id"]:
            call["id"] = delta["id"]
        if isinstance(delta.get("type"), str) and delta["type"]:
            call["type"] = delta["type"]
        function = delta.get("function")
        if isinstance(function, dict):
            if isinstance(function.get("name"), str) and function["name"]:
                call["function"]["name"] = function["name"]
            if isinstance(function.get("arguments"), str):
                call["function"]["arguments"] += function["arguments"]


def _default_yaml_path() -> pathlib.Path:
    """Resolve the installed vlm_models.yaml under the package share space.

    Falls back to the in-tree copy when the package is not installed (e.g. running
    straight from a source checkout during development).
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        share = pathlib.Path(get_package_share_directory("embodied_common"))
        installed = share / "config" / _YAML_BASENAME
        if installed.exists():
            return installed
    except Exception:
        pass
    return pathlib.Path(__file__).parent / _YAML_BASENAME


def _parse_response_json(body: str) -> dict[str, Any]:
    """Parse the HTTP response body as JSON with a friendly error on failure."""
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body.strip()[:200]
        raise RuntimeError(f"VLM API returned invalid JSON: {preview!r}") from exc


class VLMAPIClient:
    """Call an OpenAI-compatible Chat Completions or Responses API."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key_env: str,
        model: str,
        timeout_sec: float,
        temperature: float = 0.1,
        multimodal: bool = True,
        api_protocol: str = "chat_completions",
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._model = model
        self._timeout_sec = timeout_sec
        self._temperature = temperature
        self._multimodal = multimodal
        if api_protocol not in {"chat_completions", "responses"}:
            raise ValueError("api_protocol must be 'chat_completions' or 'responses'")
        self._api_protocol = api_protocol
        # One idle keep-alive connection per client instance. A call borrows
        # it for the whole request and hands it back afterwards; the lock only
        # guards that slot, never the request. One client is shared by
        # concurrent callers (perception_service runs a thread per request),
        # so a call that finds the slot taken opens its own connection rather
        # than waiting, unbounded, for another caller's model call to finish.
        self._connection: http.client.HTTPConnection | None = None
        self._connection_lock = threading.Lock()

    def close(self) -> None:
        """Drop the idle keep-alive connection, if any."""
        with self._connection_lock:
            connection, self._connection = self._connection, None
        _close_quietly(connection)

    def _borrow_connection(self, timeout_sec: float) -> tuple[http.client.HTTPConnection, bool]:
        """Take the idle connection, or open one; return it and whether it was reused."""
        with self._connection_lock:
            connection, self._connection = self._connection, None
        if connection is None:
            return self._open_connection(timeout_sec), False
        # The timeout can differ from the call that created the socket.
        connection.timeout = timeout_sec
        if connection.sock is not None:
            connection.sock.settimeout(timeout_sec)
        return connection, True

    def _return_connection(self, connection: http.client.HTTPConnection, *, reusable: bool) -> None:
        """Park a connection whose response was read to its end, or close it."""
        if reusable:
            with self._connection_lock:
                if self._connection is None:
                    self._connection = connection
                    return
        _close_quietly(connection)

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key_env = self._api_key_env.strip()
        api_key = ""
        if api_key_env:
            api_key = os.getenv(api_key_env, "").strip()
        if self._provider == "kimicode":
            if not api_key_env:
                raise RuntimeError("kimicode provider requires api_key_env to be configured")
            if not api_key:
                raise RuntimeError(f"missing API key environment variable: {api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
            headers["User-Agent"] = "KimiCLI/1.3"
        elif api_key_env:
            if not api_key:
                raise RuntimeError(f"missing API key environment variable: {api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request_target(self) -> tuple[urllib.parse.SplitResult, str]:
        endpoint = "/responses" if self._api_protocol == "responses" else "/chat/completions"
        split = urllib.parse.urlsplit(f"{self._base_url}{endpoint}")
        path = split.path or "/"
        if split.query:
            path = f"{path}?{split.query}"
        return split, path

    def _proxy_route(self) -> tuple[str, int, dict[str, str]] | None:
        """Return ``(host, port, auth headers)`` of the proxy for the target, if any.

        Before keep-alive this client went through urllib, which honours
        ``*_PROXY``; http.client does not, so the same choices are made here
        explicitly, matching urllib rather than any other client (the X2
        deployment reaches the model through a CONNECT proxy). urllib ignores
        the proxy URL's scheme for an HTTPS target and CONNECTs in plain TCP,
        defaults a missing proxy port to the target connection's (443 for
        HTTPS), and sends credentials only when both user and password are
        non-empty.
        """
        split, _ = self._request_target()
        host = split.hostname or ""
        proxy = urllib.request.getproxies().get(split.scheme) or ""
        if not proxy or urllib.request.proxy_bypass(host):
            return None
        proxy_split = urllib.parse.urlsplit(proxy if "//" in proxy else f"http://{proxy}")
        proxy_host = proxy_split.hostname or ""
        if not proxy_host:
            raise RuntimeError(f"VLM API proxy has no host: {proxy!r}")
        if split.scheme == "http" and proxy_split.scheme not in {"", "http"}:
            # urllib would speak TLS to such a proxy for a plain-HTTP target;
            # nothing deployed needs that, so refuse it instead of guessing.
            raise RuntimeError(f"VLM API proxy for an http:// target must be http://: {proxy!r}")
        proxy_port = proxy_split.port or (443 if split.scheme == "https" else 80)
        auth: dict[str, str] = {}
        user = urllib.parse.unquote(proxy_split.username or "")
        password = urllib.parse.unquote(proxy_split.password or "")
        if user and password:
            credentials = f"{user}:{password}".encode()
            auth["Proxy-Authorization"] = "Basic " + base64.b64encode(credentials).decode("ascii")
        return proxy_host, proxy_port, auth

    def _forwarding(self) -> tuple[str, dict[str, str]]:
        """Return the request-URI prefix and headers a forwarding proxy needs.

        A plain-HTTP target behind a proxy is forwarded, as urllib does:
        absolute-form request URI, credentials on the request itself. A
        tunnelled HTTPS target needs neither.
        """
        split, _ = self._request_target()
        route = self._proxy_route() if split.scheme == "http" else None
        if route is None:
            return "", {}
        return f"{split.scheme}://{split.netloc}", route[2]

    def _open_connection(self, timeout_sec: float) -> http.client.HTTPConnection:
        split, _ = self._request_target()
        if split.scheme not in {"http", "https"}:
            raise RuntimeError(f"VLM API base_url must be http(s): {self._base_url!r}")
        host = split.hostname or ""
        if not host:
            raise RuntimeError(f"VLM API base_url has no host: {self._base_url!r}")
        route = self._proxy_route()
        if route is not None:
            proxy_host, proxy_port, auth = route
            if split.scheme == "https":
                # TLS end to end with the target: CONNECT through the proxy,
                # then verify the target's certificate and SNI.
                connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                    proxy_host, proxy_port, timeout=timeout_sec
                )
                connection.set_tunnel(host, split.port, headers=auth)
                return connection
            return http.client.HTTPConnection(proxy_host, proxy_port, timeout=timeout_sec)
        if split.scheme == "https":
            return http.client.HTTPSConnection(host, split.port, timeout=timeout_sec)
        return http.client.HTTPConnection(host, split.port, timeout=timeout_sec)

    def _send(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        """Send one POST; return the borrowed connection and its response.

        The status line has arrived when this returns. A keep-alive socket
        the peer closed while idle fails on send or on the status line; that
        is not a model error, so it is retried once on a fresh socket. Nothing
        else is retried: a fresh socket that fails is a real network error,
        and a failure after the status line means the model already received
        the request. The caller owns the connection and must return or close
        it.
        """
        prefix, proxy_headers = self._forwarding()
        for attempt in (1, 2):
            connection, reused = self._borrow_connection(timeout_sec)
            try:
                connection.request("POST", f"{prefix}{path}", body=body, headers={**headers, **proxy_headers})
                return connection, connection.getresponse()
            except _STALE_CONNECTION_ERRORS:
                _close_quietly(connection)
                if not reused or attempt == 2:
                    raise
            except BaseException:
                _close_quietly(connection)
                raise
        raise AssertionError("unreachable")

    def _post_once(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, str]:
        connection, response = self._send(path, body, headers, timeout_sec)
        complete = False
        try:
            payload = response.read().decode("utf-8", errors="replace")
            complete = True
        finally:
            self._return_connection(connection, reusable=complete and not response.will_close)
        return response.status, payload

    def _post_stream(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        timeout_sec: float,
        on_delta,
    ) -> tuple[int, dict[str, Any]]:
        """POST with ``stream: true`` and assemble the SSE deltas.

        ``on_delta`` is called with the full content accumulated so far after
        every content chunk. That is what streaming buys a planner that has to
        produce a whole JSON document before anything downstream can start
        here: a short field that closes early can be acted on before the rest
        of the document exists.

        Bytes are taken with ``read1``, which returns what has arrived.
        ``read(n)`` on a chunked response keeps reading across chunks until it
        has ``n`` bytes, so each event would wait for the next ~13 behind it
        and a short answer would surface only when the stream ended.

        The connection is kept only when the stream was read to its end. One
        abandoned partway leaves unread bytes in the socket, and the next
        request would read them as its own response.
        """
        connection, response = self._send(path, body, headers, timeout_sec)
        if response.status >= 400:
            try:
                detail = response.read().decode("utf-8", errors="replace")
            finally:
                _close_quietly(connection)
            return response.status, {"_error_body": detail}

        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        pending = b""
        complete = False
        try:
            while True:
                block = response.read1(4096)
                if not block:
                    break
                pending += block
                while b"\n" in pending:
                    raw, pending = pending.split(b"\n", 1)
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        pending = b""
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # A malformed event is skipped like an unparsable one; it
                    # must not escape complete() as an exception.
                    if not isinstance(chunk, dict):
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta")
                    if not isinstance(delta, dict):
                        continue
                    thinking = delta.get("reasoning_content") or delta.get("thinking") or ""
                    if isinstance(thinking, str) and thinking:
                        reasoning.append(thinking)
                    calls = delta.get("tool_calls")
                    if isinstance(calls, list):
                        _merge_tool_call_deltas(tool_calls, calls)
                    piece = delta.get("content") or ""
                    if not isinstance(piece, str) or not piece:
                        continue
                    content.append(piece)
                    if on_delta is not None:
                        # An observer must never be able to fail the call.
                        with contextlib.suppress(Exception):
                            on_delta("".join(content))
            # read1 returning b"" ends a chunked body but not a Content-Length
            # one (a buffering gateway, or a provider ignoring stream): before
            # Python 3.13 that response stays open, and a connection parked
            # with it makes the next request fail as ResponseNotReady after the
            # model has already received it. read() finishes either kind.
            response.read()
            complete = response.isclosed()
        finally:
            self._return_connection(connection, reusable=complete and not response.will_close)
        return response.status, {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
            "usage": usage,
        }

    def _http_post(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_sec: float,
    ) -> dict[str, Any]:
        """POST one request over a connection kept alive across calls.

        Each planning turn used to pay a fresh DNS lookup, TCP handshake and
        TLS handshake before the model saw a single token. Reusing the
        connection removes that from every turn after the first.
        """
        _, path = self._request_target()
        body = json.dumps(payload).encode("utf-8")
        request_headers = {
            **headers,
            "Host": urllib.parse.urlsplit(self._base_url).netloc,
            "Connection": "keep-alive",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
        }
        try:
            status, response_body = self._post_once(path, body, request_headers, timeout_sec)
        except TimeoutError as exc:
            raise RuntimeError(f"VLM API timeout after {timeout_sec}s") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise RuntimeError(f"VLM API network error: {exc}") from exc
        if status >= 400:
            raise RuntimeError(f"VLM API HTTP {status}: {response_body.strip()[:200]}")
        return _parse_response_json(response_body)

    def analyze(
        self,
        messages: list[dict[str, Any]],
        timeout_sec: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self._provider not in {"kimicode", "openai_compatible"}:
            raise RuntimeError(f"unsupported VLM API provider: {self._provider}")

        if self._api_protocol == "responses":
            payload = {"model": self._model, "input": self._responses_input(messages)}
        else:
            payload = {"model": self._model, "messages": messages, "temperature": 0.1}
        headers = self._build_headers()
        output_idle_timeout_sec = resolve_model_output_idle_timeout(
            configured_timeout_sec=self._timeout_sec,
            override_timeout_sec=timeout_sec,
        )
        response_json = self._http_post(payload, headers, output_idle_timeout_sec)

        if self._api_protocol == "responses":
            content = response_json.get("output_text") or self._responses_output_text(response_json)
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("VLM API returned empty content")
            return content, response_json

        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError("VLM API returned no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("VLM API returned empty content")

        return content, response_json

    def complete(
        self,
        messages: list[dict[str, Any]],
        timeout_sec: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        force_json: bool = False,
        enable_thinking: bool = True,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        on_content_delta=None,
    ) -> dict[str, Any]:
        """Call the configured API protocol and normalize its result.

        ``stream`` changes only the transport: the returned mapping has the
        same shape either way, so a caller that ignores ``on_content_delta``
        cannot tell the difference. Streaming exists for that callback -- it
        receives the content accumulated so far on every chunk, which lets a
        field that closes early be acted on before the rest of the answer
        exists.
        """
        t_start = time.monotonic()

        def _err(msg: str) -> dict[str, Any]:
            return {
                "status": "error",
                "model": self._model,
                "content": "",
                "tool_calls": [],
                "reasoning": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "timing_ms": int((time.monotonic() - t_start) * 1000),
                "error": msg,
            }

        if self._provider not in {"kimicode", "openai_compatible"}:
            return _err(f"unsupported VLM API provider: {self._provider}")
        if stream and self._api_protocol == "responses":
            # The SSE reader assembles chat-completions deltas; Responses
            # events have another shape and would read as an empty answer.
            return _err("VLM API streaming requires api_protocol=chat_completions")

        send_messages = messages
        if not self._multimodal:
            stripped = []
            for msg in send_messages:
                if isinstance(msg.get("content"), list):
                    text = "\n".join(str(p.get("text", "")) for p in msg["content"] if p.get("type") == "text")
                    stripped.append({**msg, "content": text})
                else:
                    stripped.append(msg)
            send_messages = stripped

        if self._api_protocol == "responses":
            payload = {"model": self._model, "input": self._responses_input(send_messages)}
        else:
            payload = {"model": self._model, "messages": send_messages, "temperature": self._temperature}

        if tools:
            payload["tools"] = self._responses_tools(tools) if self._api_protocol == "responses" else tools
        elif force_json:
            if self._api_protocol == "responses":
                payload["text"] = {"format": {"type": "json_object"}}
            else:
                payload["response_format"] = {"type": "json_object"}

        if max_tokens is not None:
            # A planner answer is one small JSON object. An unbounded ceiling
            # lets a chatty model keep generating, and in a non-streaming call
            # every extra token is added to the wait before the first word is
            # spoken.
            payload["max_output_tokens" if self._api_protocol == "responses" else "max_tokens"] = int(max_tokens)

        if not enable_thinking and "dashscope.aliyuncs.com" in self._base_url:
            payload["enable_thinking"] = False

        # Providers spell "do not think" differently and most reject an unknown
        # field outright, so there is no safe neutral default: the per-provider
        # switch is supplied as extra_body from configuration and applied last.
        if extra_body:
            payload.update(extra_body)

        try:
            headers = self._build_headers()
        except RuntimeError as exc:
            return _err(str(exc))

        output_idle_timeout_sec = resolve_model_output_idle_timeout(
            configured_timeout_sec=self._timeout_sec,
            override_timeout_sec=timeout_sec,
        )

        if stream:
            payload["stream"] = True
            _, path = self._request_target()
            body = json.dumps(payload).encode("utf-8")
            request_headers = {
                **headers,
                "Host": urllib.parse.urlsplit(self._base_url).netloc,
                "Content-Length": str(len(body)),
                "Accept": "text/event-stream",
            }
            try:
                status, assembled = self._post_stream(
                    path, body, request_headers, output_idle_timeout_sec, on_content_delta
                )
            except (http.client.HTTPException, ConnectionError, OSError, TimeoutError) as exc:
                return _err(f"VLM API stream error: {exc}")
            if status >= 400:
                return _err(f"VLM API HTTP {status}: {assembled.get('_error_body', '').strip()[:200]}")
            content = assembled["content"]
            tool_calls = assembled["tool_calls"]
            if not content.strip() and not tool_calls:
                return _err("VLM API stream returned empty content and no tool_calls")
            usage_raw = assembled["usage"]
            return {
                "status": "ok",
                "model": self._model,
                "content": content,
                "tool_calls": tool_calls,
                "reasoning": assembled["reasoning"],
                "usage": {
                    "prompt_tokens": usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)),
                    "completion_tokens": usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)),
                    "total_tokens": usage_raw.get("total_tokens", 0),
                },
                "timing_ms": int((time.monotonic() - t_start) * 1000),
                "error": None,
            }

        try:
            response_json = self._http_post(payload, headers, output_idle_timeout_sec)
        except RuntimeError as exc:
            return _err(str(exc))

        if self._api_protocol == "responses":
            content = response_json.get("output_text") or self._responses_output_text(response_json)
            tool_calls = self._responses_tool_calls(response_json)
            reasoning = ""
        else:
            choices = response_json.get("choices", [])
            if not choices:
                return _err("VLM API returned no choices")
            message = choices[0].get("message", {})
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(item.get("text", "")) for item in content if item.get("type") == "text")
            tool_calls = message.get("tool_calls") or []
            reasoning = message.get("reasoning_content") or message.get("thinking") or ""

        # A valid response must carry either text content or tool calls; empty both is an error.
        if not content.strip() and not tool_calls:
            return _err("VLM API returned empty content and no tool_calls")

        usage_raw = response_json.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)),
            "completion_tokens": usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }

        return {
            "status": "ok",
            "model": self._model,
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "usage": usage,
            "timing_ms": int((time.monotonic() - t_start) * 1000),
            "error": None,
        }

    @staticmethod
    def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"type": "input_text", "text": str(item.get("text", ""))})
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        parts.append({"type": "input_image", "image_url": image_url})
            else:
                parts = [{"type": "input_text", "text": str(content)}]
            converted.append({"role": message.get("role", "user"), "content": parts})
        return converted

    @staticmethod
    def _responses_output_text(response: dict[str, Any]) -> str:
        parts = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    parts.append(str(content.get("text", "")))
        return "\n".join(parts)

    @staticmethod
    def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            function = tool.get("function") if tool.get("type") == "function" else None
            converted.append({"type": "function", **function} if isinstance(function, dict) else tool)
        return converted

    @staticmethod
    def _responses_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": item.get("call_id", item.get("id", "")),
                "type": "function",
                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "")},
            }
            for item in response.get("output", [])
            if item.get("type") == "function_call"
        ]


def _load_yaml_config(yaml_path: pathlib.Path) -> dict[str, Any]:
    import yaml

    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise RuntimeError(f"VLM models config not found: {yaml_path}") from None


def _detect_image_mime(image: bytes) -> str:
    """Detect image MIME type from magic bytes; raise on unsupported formats."""
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image format (expected PNG / JPEG / WEBP)")


class VLMClient:
    """High-level VLM client: one-line access with auto routing, message building and context."""

    def __init__(self, yaml_path: pathlib.Path | None = None, system: str | None = None) -> None:
        resolved_path = yaml_path or _default_yaml_path()
        self._config = _load_yaml_config(resolved_path)
        self._defaults = self._config.get("defaults", {})
        self._models_cfg = self._config.get("models", {})
        if not self._models_cfg:
            raise RuntimeError(f"no 'models' configured in {resolved_path}")
        self._default_model: str = self._defaults.get("model", next(iter(self._models_cfg)))
        if self._default_model not in self._models_cfg:
            raise RuntimeError(f"defaults.model '{self._default_model}' not found under 'models' in {resolved_path}")
        self._client_cache: dict[str, VLMAPIClient] = {}
        self._history: list[dict[str, Any]] = []
        # A persistent system prompt is prepended to every request but kept OUT of
        # _history, so a long conversation never pushes it out of context and
        # clear_history() never drops it. None means no system message is sent.
        self._system = system

    def _resolve_model(self, model: str | None) -> str:
        if model is None:
            return self._default_model
        if model not in self._models_cfg:
            # An explicitly requested but unknown model is a configuration error, not a
            # runtime condition to paper over. Silently falling back would send the
            # conversation (and its history) to an unintended model/endpoint.
            raise ValueError(f"unknown model '{model}'; available models: {', '.join(self._models_cfg.keys())}")
        return model

    def _get_client(self, model_name: str) -> VLMAPIClient:
        if model_name not in self._client_cache:
            m = self._models_cfg[model_name]
            self._client_cache[model_name] = VLMAPIClient(
                provider=m["provider"],
                base_url=m["base_url"],
                api_key_env=m["api_key_env"],
                model=model_name,
                timeout_sec=float(m.get("timeout_sec", self._defaults.get("timeout_sec", 90))),
                temperature=float(m.get("temperature", self._defaults.get("temperature", 0.1))),
                multimodal=bool(m.get("multimodal", True)),
                api_protocol=str(m.get("api_protocol", "chat_completions")),
            )
        return self._client_cache[model_name]

    def _build_user_message(self, text: str, image: bytes | None) -> dict[str, Any]:
        if image is None:
            return {"role": "user", "content": text}
        mime = _detect_image_mime(image)
        b64 = base64.b64encode(image).decode()
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": text},
            ],
        }

    def chat(
        self,
        text: str,
        image: bytes | None = None,
        model: str | None = None,
        clear_history: bool = False,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a message and return the model response. Manages history automatically."""
        if clear_history:
            self._history = []

        model_name = self._resolve_model(model)
        client = self._get_client(model_name)

        # Current turn sends the real image; history stores a text placeholder so
        # past images are not re-sent every turn (token blowup / non-multimodal break).
        current_message = self._build_user_message(text, image)
        history_message = {"role": "user", "content": f"{text}\n[图片已省略]"} if image is not None else current_message

        # System prompt (if any) leads every request but is never stored in _history.
        system_prefix = [{"role": "system", "content": self._system}] if self._system else []
        send_messages = [*system_prefix, *self._history, current_message]

        result = client.complete(
            send_messages,
            tools=tools,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

        # Only commit to history on success; failed requests (network / quota / empty
        # response) must not leave a half-written turn behind. Callers that want to
        # retry or fall back to another model start from a clean, consistent history.
        if result["status"] == "ok":
            self._history.append(history_message)
            assistant_message: dict[str, Any] = {"role": "assistant", "content": result["content"]}
            if result["tool_calls"]:
                assistant_message["tool_calls"] = result["tool_calls"]
            self._history.append(assistant_message)

        return result

    def clear_history(self) -> None:
        self._history = []
