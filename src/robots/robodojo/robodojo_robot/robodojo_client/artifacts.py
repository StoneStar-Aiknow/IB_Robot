"""Per-episode artifacts: identity, the tick ledger, decision blocks and RoboDojo's own verdict.

This module never decides success. ``finalize`` takes the values RoboDojo computed
(``TASK_ENV.success[env_idx]`` and the flags around it) and writes them out unchanged.
``failure_reason`` only classifies *which* RoboDojo signal ended a failed episode.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


class LedgerError(RuntimeError):
    """The tick ledger is not contiguous or the recorder is used out of order."""


@dataclass
class NativeOutcome:
    """Values read from RoboDojo after ``is_episode_end()``; nothing here is computed by the bridge."""

    success: bool
    take_action_cnt: int
    step_lim: int
    layout_valid: bool = True
    unstable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def classify_failure(outcome: NativeOutcome, bridge_error: str | None = None) -> str | None:
    """Name the native signal that ended a failed episode; ``None`` when RoboDojo reported success."""
    if outcome.success:
        return None
    if bridge_error:
        return "bridge_error"
    if not outcome.layout_valid:
        return "layout_invalid"
    if outcome.unstable:
        return "env_unstable"
    if outcome.take_action_cnt >= outcome.step_lim:
        return "timeout"
    return "early_terminated"  # RoboDojo cleared success before step_lim (task-side failure check)


class EpisodeRecorder:
    def __init__(
        self,
        *,
        run_id: str,
        task: str,
        seed: int | None,
        layout: Any,
        episode_id: str,
        session_id: str | None = None,
        clock=time.time,
    ):
        self._clock = clock
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "task": task,
            "seed": seed,
            "layout": layout,
            "episode_id": episode_id,
            "session_id": session_id,
            "started_at": clock(),
            "finished_at": None,
            "ticks": [],
            "decisions": [],
            "native": None,
            "success": None,
            "failure_reason": None,
            "bridge_error": None,
            "decision_count": 0,
        }
        self._finalized = False

    @property
    def ticks(self) -> list[dict[str, Any]]:
        return self.data["ticks"]

    def record_tick(
        self,
        tick: int,
        action: Mapping[str, Sequence[float]],
        *,
        decision_id: int | None = None,
        index_in_block: int | None = None,
        latency_s: float | None = None,
    ) -> None:
        if self._finalized:
            raise LedgerError("episode already finalized")
        expected = self.ticks[-1]["tick"] + 1 if self.ticks else tick
        if tick != expected:
            raise LedgerError(f"tick {tick} recorded, expected {expected}")
        self.ticks.append(
            {
                "tick": int(tick),
                "t": self._clock(),
                "decision_id": decision_id,
                "index_in_block": index_in_block,
                "latency_s": latency_s,
                "action": {k: [float(x) for x in v] for k, v in action.items()},
            }
        )

    def record_decision(self, decision_id: int, block_start_tick: int, **info: Any) -> None:
        if self._finalized:
            raise LedgerError("episode already finalized")
        self.data["decisions"].append(
            {"decision_id": int(decision_id), "block_start_tick": int(block_start_tick), "t": self._clock(), **info}
        )

    def finalize(self, outcome: NativeOutcome, bridge_error: str | None = None) -> dict[str, Any]:
        if self._finalized:
            raise LedgerError("episode already finalized")
        self._close_decision_blocks()
        self.data["native"] = asdict(outcome)
        self.data["success"] = bool(outcome.success)
        self.data["bridge_error"] = bridge_error
        self.data["failure_reason"] = classify_failure(outcome, bridge_error)
        self.data["decision_count"] = len(self.data["decisions"])
        self.data["finished_at"] = self._clock()
        problems = ledger_problems(self.data)
        if problems:
            raise LedgerError("; ".join(problems))
        self._finalized = True
        return self.data

    def finalize_interrupted(self, reason: str, *, take_action_cnt: int) -> dict[str, Any]:
        """The episode was stopped before RoboDojo judged it: no native verdict exists."""
        if self._finalized:
            raise LedgerError("episode already finalized")
        self._close_decision_blocks()
        self.data["native"] = None
        self.data["success"] = None
        self.data["interrupted"] = True
        self.data["bridge_error"] = reason
        self.data["failure_reason"] = "interrupted"
        self.data["take_action_cnt"] = int(take_action_cnt)
        self.data["decision_count"] = len(self.data["decisions"])
        self.data["finished_at"] = self._clock()
        self._finalized = True
        return self.data

    def _close_decision_blocks(self) -> None:
        last_by_decision: dict[int, int] = {}
        for t in self.ticks:
            if t["decision_id"] is not None:
                last_by_decision[t["decision_id"]] = t["tick"]
        for d in self.data["decisions"]:
            d["block_end_tick"] = last_by_decision.get(d["decision_id"])

    def write(self, out_dir: str) -> str:
        if not self._finalized:
            raise LedgerError("finalize() before write()")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.data['episode_id']}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, path)
        return path


def ledger_problems(data: Mapping[str, Any]) -> list[str]:
    """Integrity checks on a finished artifact; an empty list means it is complete."""
    problems = []
    for key in ("task", "episode_id", "run_id"):
        if not data.get(key):
            problems.append(f"missing {key}")
    ticks = [t["tick"] for t in data.get("ticks", [])]
    for prev, cur in zip(ticks, ticks[1:], strict=False):
        if cur != prev + 1:
            problems.append(f"tick gap {prev} -> {cur}")
    known = {d["decision_id"] for d in data.get("decisions", [])}
    for t in data.get("ticks", []):
        if t.get("decision_id") is not None and t["decision_id"] not in known:
            problems.append(f"tick {t['tick']} references unknown decision {t['decision_id']}")
    native = data.get("native")
    # tick k is the action that took RoboDojo's counter from k to k+1
    if native is not None and ticks and native.get("take_action_cnt") not in (None, ticks[-1] + 1):
        problems.append(f"native take_action_cnt {native['take_action_cnt']} != ledger ticks {ticks[-1] + 1}")
    return problems
