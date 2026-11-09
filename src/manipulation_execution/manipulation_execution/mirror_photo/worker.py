"""The node's second thread: a finished round becomes its photo in the outbox; no ROS.

``PhotoCore`` runs in the ROS callbacks and must never wait, so everything
that touches the disk or decodes a JPEG comes here through one FIFO queue:
the provisional best of the running round (kept on disk as ``current``) and
the finished round (chosen, put in the outbox, ``current`` cleared, the
uploader woken). FIFO keeps a round's last provisional copy ahead of its own
end. Nothing here raises out of ``run``: a failure while choosing still sends
the round's heaviest candidate (``emergency_photo``), any other failure is
logged and the next item is taken.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .config import PhotoSection
from .outbox import Log, Outbox
from .scoring import Candidate
from .selector import FinishedRound, choose_photo, emergency_photo, provisional_photo

_STOP = object()


class RoundWorker:
    def __init__(
        self,
        section: PhotoSection,
        outbox: Outbox,
        *,
        wake_uploader: Callable[[], None],
        log: Log,
        choose=choose_photo,
    ):
        self._section = section
        self._outbox = outbox
        self._wake = wake_uploader
        self._log = log
        self._choose = choose
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._sent = 0
        self._last_photo: dict = {}
        self._provisional_error = ""

    # -- from the ROS thread (never blocks) --------------------------------------------

    def provisional(self, round_id: str, started_at: float, candidate: Candidate) -> None:
        self._queue.put(("provisional", (round_id, started_at, candidate)))

    def finished(self, finished: FinishedRound) -> None:
        self._queue.put(("finished", finished))

    def stop(self) -> None:
        """Finish what is queued, then return from ``run``."""
        self._queue.put(_STOP)

    def status(self) -> dict:
        with self._lock:
            return {"backlog": self._queue.qsize(), "rounds_sent": self._sent, "last_photo": dict(self._last_photo)}

    # -- the thread ----------------------------------------------------------------------

    def run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            kind, payload = item
            try:
                if kind == "provisional":
                    self._keep_provisional(*payload)
                else:
                    self._send(payload)
            except Exception as exc:  # noqa: BLE001 - one bad round must not stop the next
                self._log("error", f"photo worker: {kind} failed: {type(exc).__name__}: {exc}")

    def _keep_provisional(self, round_id: str, started_at: float, candidate: Candidate) -> None:
        try:
            self._outbox.write_current(provisional_photo(round_id, started_at, candidate))
        except OSError as exc:
            why = str(exc)
            # Once per kind: a disk that refuses refuses every few seconds.
            if why != self._provisional_error:
                self._log("warning", f"photo round {round_id}: no crash copy on disk ({why}); the round goes on")
            self._provisional_error = why
            return
        self._provisional_error = ""

    def _send(self, finished: FinishedRound) -> None:
        try:
            photo, _ = self._choose(finished, self._section)
        except Exception as exc:  # noqa: BLE001 - the round's photo goes out whatever broke the choice
            why = f"{type(exc).__name__}: {exc}"
            self._log("error", f"photo round {finished.round_id}: choosing failed ({why}); sending it unscored")
            photo = emergency_photo(finished, why)
        if photo is None:
            self._outbox.clear_current()
            self._log(
                "warning",
                f"photo round {finished.round_id} ({finished.reason}): no camera frame came in the round, "
                "nothing to send",
            )
            return
        where = self._outbox.put(photo)
        self._outbox.clear_current()
        self._wake()
        meta = photo.meta
        summary = {
            "round_id": photo.name,
            "end_reason": finished.reason,
            "quality": meta.get("quality"),
            "source": meta.get("source"),
            "label": meta.get("label"),
            "scoring": meta.get("scoring"),
            "scoring_ms": meta.get("scoring_ms"),
            "bytes": len(photo.jpeg),
            "stored": where,
        }
        with self._lock:
            self._sent += 1
            self._last_photo = summary
        self._log(
            "info",
            f"photo round {photo.name} ({finished.reason}, {finished.frames} frames, "
            f"{len(finished.candidates)} candidates): {summary['quality']} "
            f"{summary['source'] or '-'}/{summary['label'] or '-'}, scoring {summary['scoring']}"
            + (f" {summary['scoring_ms']} ms" if summary["scoring_ms"] is not None else "")
            + f", {summary['bytes']} bytes, queued on {where}",
        )
