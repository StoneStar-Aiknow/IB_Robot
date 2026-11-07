"""The composer's state, free of ROS: compose, keep for execution, remember what worked.

A composition is made while the Agent prepares a plan and executed when the
Gateway runs the plan's step for the same text. ``Composer`` keeps

* the prepared compositions (one per text, newest wins, expiring after
  ``PREPARED_TTL_SEC``), so what was presented is what runs;
* per conversation, the last few poses reached, which the model reads
  relative instructions ("再高一点") against, and the last remembered pose,
  which the user's next words confirm or reject;
* optionally a ``PoseMemory``: a pose recalled from it needs no model call, and
  a pose that succeeds is remembered for its words.
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
from pose_composer.errors import ComposeError
from pose_composer.memory import PoseMemory, is_relative

HISTORY_TURNS = 3

# (messages) -> reply text; raises on transport failure.
LanguageModel = Callable[[list[dict]], str]


@dataclass
class Prepared:
    composition_id: str
    text: str
    composition: spec.Composition
    description: dict
    session_id: str = ""
    utterance: str = ""
    from_memory: bool = False
    # The remembered pose recalled, or "" (a memorable new pose gets its id when it succeeds).
    pose_id: str = ""
    memorable: bool = False
    arms: tuple[str, ...] = ()
    created: float = field(default_factory=time.monotonic)


@dataclass
class _Pending:
    """The last remembered pose a conversation executed, awaiting the user's verdict."""

    pose_id: str
    phrases: tuple[str, ...]
    created: float = field(default_factory=time.monotonic)
    offered: bool = False
    approximate: bool = False
    targets: dict[str, float] = field(default_factory=dict)


class Composer:
    def __init__(
        self, model: LanguageModel, *, ttl_sec: float = PREPARED_TTL_SEC, memory: PoseMemory | None = None
    ) -> None:
        self._model = model
        self._ttl_sec = ttl_sec
        self._memory = memory
        self._lock = threading.Lock()
        self._prepared: dict[str, Prepared] = {}
        self._history: dict[str, deque[tuple[str, dict]]] = {}
        self._pending: dict[str, _Pending] = {}

    @property
    def memory(self) -> PoseMemory | None:
        return self._memory

    def compose(self, text: str, start_q: dict[str, float], *, session_id: str = "", utterance: str = "") -> Prepared:
        """Recall or ask the model, solve, check; keep the result for ``take``."""
        text = text.strip()
        if not text:
            raise ComposeError("POSE_EMPTY_TEXT", "no pose description")
        if len(text) > MAX_TEXT_CHARS:
            raise ComposeError("POSE_NOT_EXECUTABLE", f"pose description longer than {MAX_TEXT_CHARS} characters")
        # A normalized target must not hide relative requirements in the original utterance.
        relative = bool(utterance.strip() and is_relative(utterance))
        prepared = (None if relative else self._recall(text, utterance, start_q)) or self._ask(
            utterance if relative else text, start_q, session_id
        )
        prepared.text = text
        prepared.session_id, prepared.utterance = session_id, utterance.strip()
        if prepared.composition.kind in ("pose", "sequence"):
            with self._lock:
                self._expire_locked()
                self._prepared[text] = prepared
        return prepared

    def _recall(self, text: str, utterance: str, start_q) -> Prepared | None:
        if self._memory is None:
            return None
        record = self._memory.lookup(text, confirmed_only=True) or (
            self._memory.lookup(utterance, confirmed_only=True) if utterance.strip() else None
        )
        if record is None:
            return None
        # One pose answers many phrasings ("秀肌肉", "投降"): say back the user's own words.
        spoken = f"好的，{(utterance or text).strip()}"
        try:
            composition = spec.recall(record.targets, spoken, start_q)
        except spec.SpecError:
            return None
        self._memory.note_hit(record.pose_id)
        return Prepared(uuid.uuid4().hex[:12], text, composition, record.description, from_memory=True,
                        pose_id=record.pose_id, memorable=True, arms=record.arms)  # fmt: skip

    def _ask(self, text: str, start_q, session_id: str) -> Prepared:
        with self._lock:
            history = list(self._history.get(session_id, ()))
        try:
            reply = self._model(spec.build_messages(text, start_q, history))
        except Exception as exc:  # noqa: BLE001 - any transport failure is the model's
            raise ComposeError("POSE_MODEL_FAILED", f"language model: {exc}") from exc
        try:
            data = spec.parse(reply)
            composition = spec.compose(data, start_q)
        except (spec.SpecError, ValueError, KeyError, TypeError) as exc:
            raise ComposeError("POSE_NOT_EXECUTABLE", f"description not executable: {exc}") from exc
        # Approximate absolute poses may be staged, but require explicit acceptance
        # of the achieved pose. They are never recalled while still candidates.
        memorable = self._memory is not None and spec.memorable(data, composition, text, allow_approximate=True)
        return Prepared(uuid.uuid4().hex[:12], text, composition, data, memorable=memorable,
                        arms=tuple(spec.moved_arms(data)))  # fmt: skip

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

    def succeeded(self, prepared: Prepared) -> str:
        """The pose was reached: context for the next instruction, and remembered if memorable.

        Returns the remembered pose id, or "".
        """
        final = prepared.composition.frames[-1]
        with self._lock:
            self._history.setdefault(prepared.session_id, deque(maxlen=HISTORY_TURNS)).append(
                (prepared.text, dict(final))
            )
        with self._lock:
            self._pending.pop(prepared.session_id, None)
        if self._memory is None or not prepared.memorable:
            return ""
        phrases = [(prepared.text, "step")]
        if prepared.utterance and prepared.utterance != prepared.text:
            phrases.append((prepared.utterance, "utterance"))
        pose_id = self._memory.record_success(
            phrases,
            description=prepared.description,
            arms=prepared.arms,
            targets=spec.targets_of(final, prepared.arms),
            user_message=prepared.composition.user_message,
            pose_id=prepared.pose_id,
            signature=spec.signature_of(final, prepared.arms),
        )
        with self._lock:
            self._pending[prepared.session_id] = _Pending(
                pose_id,
                tuple(p for p, _ in phrases),
                approximate=prepared.composition.approx,
                targets=spec.targets_of(final, prepared.arms),
            )
        return pose_id

    def feedback(self, session_id: str, verdict: str) -> tuple[str, str]:
        """The user's verdict on the conversation's last remembered pose: (pose_id, outcome)."""
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is not None and time.monotonic() - pending.created > 60.0:
                self._pending.pop(session_id, None)
                pending = None
        if pending is None or self._memory is None:
            return "", "no pose awaiting a verdict"
        record = self._memory.get(pending.pose_id)
        if record is not None and record.state == "candidate" and record.targets != pending.targets:
            # Another session may have completed a near-equivalent candidate.
            # Never let this session consent to a different achieved target.
            with self._lock:
                self._pending.pop(session_id, None)
            return "", "pose changed since execution; confirmation cancelled"
        if verdict in ("offer", "status"):
            if record is None or record.state != "candidate" or record.successes < 2:
                return "", "not ready for saving"
            if verdict == "offer":
                pending.offered = True
                pending.created = time.monotonic()
            elif not pending.offered:
                return "", "no save confirmation pending"
            return pending.pose_id, (
                "刚才实际完成的是近似姿态，到位吗？可以保存吗？"
                if pending.approximate
                else "刚才这个动作到位吗？可以保存吗？"
            )
        with self._lock:
            self._pending.pop(session_id, None)
        if verdict == "good" and (not pending.offered or record is None or record.successes < 2):
            return "", "no save confirmation pending"
        return pending.pose_id, self._memory.feedback(pending.pose_id, verdict, pending.phrases)

    def _expire_locked(self) -> None:
        now = time.monotonic()
        for key in [k for k, v in self._prepared.items() if now - v.created > self._ttl_sec]:
            del self._prepared[key]


def description_json(prepared: Prepared) -> str:
    return json.dumps(prepared.description, ensure_ascii=False, sort_keys=True)
