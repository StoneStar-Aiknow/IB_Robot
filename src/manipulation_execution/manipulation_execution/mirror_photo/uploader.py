"""The outbox to the upper PC over HTTP, until each photo is confirmed; no ROS.

``POST <target>/x2/photo``: the body is the JPEG exactly as the camera sent it,
the metadata is one ASCII JSON header. A photo counts as delivered only on a
200 whose JSON repeats the body's sha256; only then is the local copy removed.
Anything else is retried without end: the newest photo first, so the person who
just finished is served before a backlog, waiting 1, 2, 5, 10, then every 30 s.
A new photo or a changed target retries at once.

The target is read from ``target_file`` before every send, so moving to another
upper PC takes one line in one file and no restart. A missing or empty file
falls back to ``upload.default_target``; a file with a bad address does not (a
typo must not send photos somewhere else), the photos wait until it is fixed.
Proxy variables are ignored: the upper PC is on the local network.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .config import normalize_target
from .outbox import Log, Outbox

ENDPOINT = "/x2/photo"
META_HEADER = "X-X2-Photo-Meta"
BACKOFF_SEC = (1.0, 2.0, 5.0, 10.0, 30.0)
# While waiting: how often the target file is looked at, and the queue with nothing to send.
POLL_SEC = 1.0

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Refused(Exception):
    """The receiver answered, but did not confirm this photo."""


def read_target(target_file: str, default: str = "") -> tuple[str, str]:
    """``(base url, "")``, or ``("", why there is none)``. The file's first line not empty or ``#``."""
    try:
        lines = Path(target_file).read_text(encoding="utf-8").splitlines() if target_file else []
    except FileNotFoundError:
        lines = []
    except (OSError, UnicodeDecodeError) as exc:
        return "", f"cannot read {target_file}: {exc}"
    text = next((line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")), "")
    origin = target_file
    if not text:
        text, origin = default, "upload.default_target"
    if not text:
        return "", f"no upper PC address: {target_file or 'target_file'} is empty and upload.default_target is not set"
    try:
        return normalize_target(text), ""
    except ValueError as exc:
        return "", f"{origin}: {exc}"


def post_photo(target: str, jpeg: bytes, meta: dict, timeout_sec: float) -> None:
    """One attempt; returns only if confirmed. ``Refused``, or ``OSError`` / ``HTTPException`` for no answer."""
    request = urllib.request.Request(
        target + ENDPOINT,
        data=jpeg,
        headers={"Content-Type": "image/jpeg", META_HEADER: json.dumps(meta, ensure_ascii=True, sort_keys=True)},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout_sec) as response:
            body = response.read(4096)
    except urllib.error.HTTPError as exc:
        raise Refused(f"HTTP {exc.code}") from exc
    try:
        answer = json.loads(body)
    except ValueError as exc:
        raise Refused("the answer is not JSON") from exc
    expected = hashlib.sha256(jpeg).hexdigest()
    if not isinstance(answer, dict) or answer.get("sha256") != expected:
        raise Refused("the answer does not confirm the photo's sha256")


class Uploader:
    def __init__(
        self,
        outbox: Outbox,
        *,
        target_file: str,
        default_target: str,
        timeout_sec: float,
        log: Log,
        post: Callable[[str, bytes, dict, float], None] = post_photo,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._outbox = outbox
        self._target_file = target_file
        self._default = default_target
        self._timeout = timeout_sec
        self._log = log
        self._post = post
        self._clock = clock
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._target = ""
        self._failures = 0
        self._error = ""
        self._delivered = 0
        self._last_delivered = ""
        self._last_send_ms: float | None = None

    def wake(self) -> None:
        """A new photo is in: stop waiting."""
        self._wake.set()

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    def run(self) -> None:
        """Until ``stop``; never raises. For one thread (a send may block for ``timeout_sec``, nothing waits on it)."""
        while not self._stopping.is_set():
            try:
                delay = self.step()
            except Exception as exc:  # noqa: BLE001 - this thread has to outlive any one bad pass
                delay = self._failed(f"{type(exc).__name__}: {exc}")
            self._wait(delay)

    def step(self) -> float:
        """Send until the queue is empty or a send fails; returns how long to wait before the next pass."""
        refused: set[str] = set()
        while not self._stopping.is_set():
            name = next((name for name in self._outbox.names() if name not in refused), None)
            target, problem = read_target(self._target_file, self._default)
            self._target = target
            if name is None:
                break
            if not target:
                return self._failed(problem, POLL_SEC)
            loaded = self._outbox.load(name)
            if loaded is None:
                refused.add(name)
                continue
            jpeg, meta = loaded
            started = self._clock()
            try:
                self._post(target, jpeg, meta, self._timeout)
            except Refused as exc:
                refused.add(name)
                self._failed(f"{target} refused {name}: {exc}")
                continue
            except (OSError, http.client.HTTPException) as exc:
                # Nobody there: the next photo would only wait out the same timeout.
                return self._failed(f"{target} did not answer: {exc}")
            self._last_send_ms = (self._clock() - started) * 1000.0
            self._outbox.remove(name)
            self._delivered += 1
            self._last_delivered = name
            self._log("info", f"photo {name} delivered to {target} ({len(jpeg)} bytes, {self._last_send_ms:.0f} ms)")
            refused.discard(name)
            if self._failures or self._error:
                self._log("info", f"photo upload back after {self._failures} failed attempt(s)")
            self._failures = 0
            self._error = ""
        return self._backoff() if refused else POLL_SEC

    def status(self) -> dict:
        return {
            "target": self._target,
            "delivered": self._delivered,
            "last_delivered": self._last_delivered,
            "last_send_ms": round(self._last_send_ms, 1) if self._last_send_ms is not None else None,
            "failures": self._failures,
            "last_error": self._error,
        }

    def _failed(self, why: str, delay: float | None = None) -> float:
        # Once per kind of failure, however long it lasts: a retry forever must not fill the log.
        if why != self._error:
            self._log("warning", f"photo upload: {why}; {len(self._outbox.names())} photo(s) waiting, still trying")
        self._error = why
        self._failures += 1
        return self._backoff() if delay is None else delay

    def _backoff(self) -> float:
        return BACKOFF_SEC[min(max(self._failures, 1), len(BACKOFF_SEC)) - 1]

    def _wait(self, delay: float) -> None:
        deadline = self._clock() + delay
        tried = self._target
        while not self._stopping.is_set():
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                return
            if self._wake.wait(min(remaining, POLL_SEC)):
                self._wake.clear()
                return
            if read_target(self._target_file, self._default)[0] != tried:
                return
