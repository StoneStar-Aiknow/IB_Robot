"""An optional source fence; health heartbeats never create gesture votes."""

from __future__ import annotations

from embodied_common.hand_gesture_contract import SourceState, identifier, integer


class GestureSourceGate:
    def __init__(self, started_ns: int, health_s: float = 2.5):
        self.started_ns = integer(started_ns, "started_ns")
        self.required_after_ns = started_ns
        self.health_ns = int(health_s * 1e9)
        self.state: SourceState | None = None
        self.available = False
        self.result_seq = 0
        self.reason = "source_unregistered"

    def invalidate(self, now_ns: int):
        self.required_after_ns = max(self.required_after_ns, now_ns)
        self.available = False

    def observe(self, text, *, now_ns: int, token: str):
        """(accepted, reset_votes). Rejects never mutate the registered fence."""
        try:
            state = SourceState.from_json(text)
            if not self.required_after_ns <= state.published_mono_ns <= now_ns:
                raise ValueError("source_before_barrier")
            if now_ns - state.published_mono_ns > self.health_ns:
                raise ValueError("source_stale")
            old = self.state
            if old is not None:
                if state.producer_start_mono_ns < old.producer_start_mono_ns:
                    raise ValueError("old_source_boot")
                if state.producer_start_mono_ns == old.producer_start_mono_ns:
                    if state.requester_boot_id != old.requester_boot_id:
                        raise ValueError("ambiguous_source_boot")
                    if state.stream_generation < old.stream_generation or state.state_seq <= old.state_seq:
                        raise ValueError("source_reordered")
            if state.available and (not token or state.session_token != token):
                raise ValueError("source_context")
        except (ValueError, TypeError, UnicodeError):
            self.reason = "source_rejected"
            return False, False
        changed = old is None or state.source_key != old.source_key
        reset = changed or (self.available and not state.available)
        if changed or (not self.available and state.available):
            self.required_after_ns = max(self.required_after_ns, state.published_mono_ns)
        if changed:
            self.result_seq = 0
        self.state, self.available = state, state.available
        self.reason = "" if state.available else "source_unavailable"
        return True, reset

    def tick(self, now_ns: int):
        if self.available and self.state is not None and now_ns - self.state.published_mono_ns > self.health_ns:
            self.available = False
            self.reason = "source_stale"
            return True
        return False

    def allows(self, message, *, now_ns: int, token: str):
        state = self.state
        if not self.available or state is None or now_ns - state.published_mono_ns > self.health_ns:
            self.reason = "source_unavailable"
            return False
        try:
            key = (
                integer(message.get("producer_start_mono_ns"), "producer_start_mono_ns"),
                identifier(message.get("requester_boot_id"), "requester_boot_id"),
                integer(message.get("stream_generation"), "stream_generation", 1),
            )
            seq = integer(message.get("result_seq"), "result_seq", 1)
            published = integer(message.get("published_mono_ns"), "published_mono_ns")
            if key != state.source_key or seq <= self.result_seq:
                raise ValueError("old_result")
            if not self.required_after_ns <= published <= now_ns:
                raise ValueError("result_before_barrier")
            if state.session_token != token:
                raise ValueError("result_context")
            if message.get("session_token") != token:
                raise ValueError("result_context")
        except (ValueError, TypeError, AttributeError):
            self.reason = "source_result_rejected"
            return False
        self.reason = ""
        return True

    def commit(self, message):
        self.result_seq = message["result_seq"]
