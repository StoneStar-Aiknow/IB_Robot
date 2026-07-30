"""Atomic priority-zero deadline reservations by hardware resource."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class DeadlineReservation:
    token: int
    pipeline_id: str
    hardware_resource_id: str
    estimated_start_ns: int
    estimated_finish_ns: int

    @property
    def estimate_ns(self) -> int:
        return self.estimated_finish_ns - self.estimated_start_ns


class DeadlineReservationTable:
    """Serialize equal-priority estimates without imposing a pipeline count limit."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._next_token = 1
        self._reservations: dict[int, DeadlineReservation] = {}
        self._resource_tokens: dict[str, list[int]] = {}
        self._unknown_tokens: set[int] = set()

    def try_reserve(
        self,
        *,
        pipeline_id: str,
        hardware_resource_id: str,
        now_ns: int,
        deadline_ns: int,
        estimate_ns: int,
    ) -> DeadlineReservation | None:
        if not hardware_resource_id:
            raise ValueError("hardware_resource_id must be non-empty")
        if estimate_ns <= 0:
            raise ValueError("estimate_ns must be positive")
        with self._condition:
            tokens = self._resource_tokens.get(hardware_resource_id, [])
            if any(token in self._unknown_tokens for token in tokens):
                return None
            tail_ns = max(
                (self._reservations[token].estimated_finish_ns for token in tokens),
                default=now_ns,
            )
            start_ns = max(now_ns, tail_ns)
            finish_ns = start_ns + estimate_ns
            if finish_ns > deadline_ns:
                return None
            token = self._next_token
            self._next_token += 1
            reservation = DeadlineReservation(
                token=token,
                pipeline_id=pipeline_id,
                hardware_resource_id=hardware_resource_id,
                estimated_start_ns=start_ns,
                estimated_finish_ns=finish_ns,
            )
            self._reservations[token] = reservation
            self._resource_tokens.setdefault(hardware_resource_id, []).append(token)
            return reservation

    def wait_for_turn(
        self,
        reservation: DeadlineReservation,
        *,
        deadline_ns: int,
        cancel_requested: Callable[[], bool] | None = None,
        now_ns: Callable[[], int] = time.monotonic_ns,
    ) -> str:
        """Wait until this reservation owns the resource dispatch turn.

        Returns ``ready``, ``deadline_exceeded``, ``request_canceled``, or
        ``reservation_released``. Deadline feasibility is checked again against
        the actual monotonic time when the turn becomes available.
        """

        with self._condition:
            while True:
                current = self._reservations.get(reservation.token)
                if current is None:
                    return "reservation_released"
                if cancel_requested is not None and cancel_requested():
                    self._remove_locked(current)
                    self._condition.notify_all()
                    return "request_canceled"
                current_time_ns = now_ns()
                if current_time_ns + current.estimate_ns > deadline_ns:
                    self._remove_locked(current)
                    self._condition.notify_all()
                    return "deadline_exceeded"
                tokens = self._resource_tokens[current.hardware_resource_id]
                if tokens and tokens[0] == current.token:
                    return "ready"
                remaining_ns = deadline_ns - current_time_ns
                if remaining_ns <= 0:
                    self._remove_locked(current)
                    self._condition.notify_all()
                    return "deadline_exceeded"
                self._condition.wait(min(0.05, remaining_ns / 1_000_000_000))

    def release(self, reservation: DeadlineReservation) -> None:
        """Release work known not to have started or known to have completed."""

        with self._condition:
            current = self._reservations.pop(reservation.token, None)
            self._unknown_tokens.discard(reservation.token)
            if current is None:
                return
            self._remove_resource_token_locked(current)
            self._condition.notify_all()

    def mark_unknown(self, reservation: DeadlineReservation) -> None:
        """Retain uncertain work so the resource fails closed until reconciliation."""

        with self._condition:
            if reservation.token in self._reservations:
                self._unknown_tokens.add(reservation.token)
                self._condition.notify_all()

    def reconcile_pipeline(self, pipeline_id: str) -> None:
        """Clear uncertain work fenced by a pipeline reboot."""

        with self._condition:
            tokens = [token for token in self._unknown_tokens if self._reservations[token].pipeline_id == pipeline_id]
            for token in tokens:
                reservation = self._reservations.pop(token)
                self._unknown_tokens.remove(token)
                self._remove_resource_token_locked(reservation)
            self._condition.notify_all()

    def _remove_locked(self, reservation: DeadlineReservation) -> None:
        self._reservations.pop(reservation.token, None)
        self._unknown_tokens.discard(reservation.token)
        self._remove_resource_token_locked(reservation)

    def _remove_resource_token_locked(self, reservation: DeadlineReservation) -> None:
        resource_tokens = self._resource_tokens.get(reservation.hardware_resource_id)
        if resource_tokens is None:
            return
        if reservation.token in resource_tokens:
            resource_tokens.remove(reservation.token)
        if not resource_tokens:
            self._resource_tokens.pop(reservation.hardware_resource_id, None)


__all__ = ["DeadlineReservation", "DeadlineReservationTable"]
