"""The composer's state, free of ROS: compose a description, keep it, hand it back to execute.

A composition is made while the Agent prepares a plan and executed when the
Gateway runs the plan's step for the same text. ``Composer`` keeps the
prepared compositions (one per text, newest wins, expiring after
``PREPARED_TTL_SEC``) and the short history the language model sees for
relative instructions ("再高一点").
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from embodied_common.pose_composer_contract import MAX_TEXT_CHARS, PREPARED_TTL_SEC
from pose_composer import spec

HISTORY_TURNS = 3

# (messages) -> reply text; raises on transport failure.
LanguageModel = Callable[[list[dict]], str]


class ComposeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Prepared:
    composition_id: str
    text: str
    composition: spec.Composition
    description: dict
    created: float = field(default_factory=time.monotonic)


class Composer:
    def __init__(self, model: LanguageModel, *, ttl_sec: float = PREPARED_TTL_SEC) -> None:
        self._model = model
        self._ttl_sec = ttl_sec
        self._lock = threading.Lock()
        self._prepared: dict[str, Prepared] = {}
        self._history: deque[tuple[str, dict]] = deque(maxlen=HISTORY_TURNS)

    def compose(self, text: str, start_q: dict[str, float]) -> Prepared:
        """Ask the model, solve, check; keep the result for ``take``."""
        text = text.strip()
        if not text:
            raise ComposeError("POSE_EMPTY_TEXT", "no pose description")
        if len(text) > MAX_TEXT_CHARS:
            raise ComposeError("POSE_NOT_EXECUTABLE", f"pose description longer than {MAX_TEXT_CHARS} characters")
        with self._lock:
            history = list(self._history)
        try:
            reply = self._model(spec.build_messages(text, start_q, history))
        except Exception as exc:  # noqa: BLE001 - any transport failure is the model's
            raise ComposeError("POSE_MODEL_FAILED", f"language model: {exc}") from exc
        try:
            data = spec.parse(reply)
            composition = spec.compose(data, start_q)
        except (spec.SpecError, ValueError, KeyError, TypeError) as exc:
            raise ComposeError("POSE_NOT_EXECUTABLE", f"description not executable: {exc}") from exc
        prepared = Prepared(uuid.uuid4().hex[:12], text, composition, data)
        if composition.kind in ("pose", "sequence"):
            with self._lock:
                self._expire_locked()
                self._prepared[text] = prepared
        return prepared

    def take(self, text: str, composition_id: str = "") -> Prepared | None:
        """The fresh prepared composition for ``text`` (or that id), removed so it runs once."""
        with self._lock:
            self._expire_locked()
            if composition_id:
                for key, item in list(self._prepared.items()):
                    if item.composition_id == composition_id:
                        return self._prepared.pop(key)
                return None
            return self._prepared.pop(text.strip(), None)

    def remember(self, text: str, final_q: dict[str, float]) -> None:
        """A pose that was actually reached becomes context for the next instruction."""
        with self._lock:
            self._history.append((text, dict(final_q)))

    def _expire_locked(self) -> None:
        now = time.monotonic()
        for key in [k for k, v in self._prepared.items() if now - v.created > self._ttl_sec]:
            del self._prepared[key]


def description_json(prepared: Prepared) -> str:
    return json.dumps(prepared.description, ensure_ascii=False, sort_keys=True)
