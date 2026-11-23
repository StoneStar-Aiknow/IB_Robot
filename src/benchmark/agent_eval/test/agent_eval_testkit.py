"""Shared test doubles for agent_eval: fixture body, a scripted platform + Agent behind RunnerIO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from agent_eval.body import body_from_documents
from agent_eval.episode_context import EpisodeContext
from agent_eval.runner import EpisodeRunner, EventBook, RunnerSettings

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
CONFIG = HERE.parent / "config"


def fixture_body(name: str = "dual_arm_fixture"):
    robot_path = FIXTURES / "robots" / f"{name}.yaml"
    profile_path = FIXTURES / "profiles" / f"{name}.yaml"
    robot = yaml.safe_load(robot_path.read_text(encoding="utf-8"))["robot"]
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    return body_from_documents(
        robot, profile, robot_config=name, robot_config_path=str(robot_path), profile_path=str(profile_path)
    )


class FakeWall:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class ScriptedPlatform:
    """RunnerIO double: each request gets the next scripted Agent reply and robot state.

    A decision entry is ``{"reply": {...}, "after": <state or None>, "end_episode": bool}``;
    ``reply.kind`` is ``events`` / ``refused`` / ``timeout``. ``native`` is published with the episode end (evaluation side only).
    """

    def __init__(self, book: EventBook, body, decisions, native: dict[str, Any] | None = None):
        self.book = book
        self.body = body
        self.decisions = iter(decisions)
        self.native = native or {}
        self.key: tuple[str, Any] | None = None
        self.texts: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []
        self.finish_calls = 0
        self.admitted: list[tuple[str, Any]] = []
        self.aborts = 0

    DT_S = 0.04

    def start(self, wall: FakeWall, session_id: str, generation: int | None, initial, **fields) -> dict[str, Any]:
        event = {"event": "session_start", "session_id": session_id, "generation": generation, "wall_time": wall.t}
        event.update(fields)
        self.book.on_session(event)
        self.key = (session_id, generation)
        self.time_base = float(generation or 0) * 100.0  # simulated time never goes back
        wall.t += 1.0
        self.apply(initial)
        return event

    def clock(self, tick: int, state: str = "RUNNING") -> dict[str, Any]:
        session_id, generation = self.key
        within = round(tick * self.DT_S, 6)
        return {
            "session_id": session_id,
            "generation": generation,
            "state": state,
            "tick": tick,
            "sim_time_s": within,
            "sim_time_total_s": self.time_base + within,
        }

    def apply(self, state) -> None:
        """The runtime's publishing order for one observation: its clock, then its state."""
        if state is None:
            return
        clock = self.clock(state["tick"])
        self.book.on_clock(clock)
        stamp = clock["sim_time_total_s"]
        for side, arm in state["arms"].items():
            if arm.get("xyz") is not None:
                self.book.on_ee(side, arm["xyz"], arm["quat"], stamp)
            body_arm = self.body.arm(side)
            self.book.on_joints(
                [body_arm.gripper_joint, body_arm.wrist_joint], [float(arm["gripper"]), float(arm["wrist"])], stamp
            )

    def end(self) -> None:
        session_id, generation = self.key
        self.book.on_session({"event": "session_end", "session_id": session_id, "generation": generation})
        self.book.on_native_result({"session_id": session_id, "generation": generation, **self.native})

    def publish_request(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)
        self.texts.append(payload["text"])
        entry = next(self.decisions)
        rid = payload["request_id"]
        reply = entry["reply"]
        if reply["kind"] == "events":
            for event in reply["events"]:
                self.book.on_agent_event({**event, "request_key": {"request_id": rid}})
        elif reply["kind"] == "refused":
            self.book.on_agent_response({**reply["response"], "request_id": rid})
        self.apply(entry.get("after"))
        if entry.get("end_episode"):
            self.end()

    def publish_control(self, payload: dict[str, Any]) -> None:
        self.controls.append(payload)

    def call_finish(self) -> str:
        self.finish_calls += 1
        self.end()
        return "finishing"

    def abort_session(self) -> str:
        self.aborts += 1
        session_id, generation = self.key
        self.book.on_session(
            {
                "event": "session_end",
                "session_id": session_id,
                "generation": generation,
                "interrupted": True,
                "reason": "session aborted by the caller",
            }
        )
        self.book.on_native_result(
            {
                "session_id": session_id,
                "generation": generation,
                "native_success": None,
                "failure_reason": "interrupted",
                "interrupted": True,
            }
        )
        return "aborted"

    def admit_episode(self, session_id: str, generation) -> tuple[bool, str]:
        self.admitted.append((session_id, generation))
        return True, "admitted"

    def agent_ready(self, timeout_s: float) -> bool:
        return True


def make_runner(tmp_path, body, recipe_facts=None, ids=None, **settings_kw):
    from agent_eval.task_spec import load_recipe

    wall = FakeWall()
    book = EventBook(body, wall_clock=wall)
    recipe_facts = recipe_facts or load_recipe(str(CONFIG / "recipes" / "general_pickup_x5.yaml")).facts()
    context = EpisodeContext(recipe_facts, body.facts())
    defaults = {
        "out_dir": str(tmp_path),
        "decision_timeout_s": 0.2,
        "fresh_wait_s": 0.05,
        "end_wait_s": 0.5,
    }
    defaults.update(settings_kw)
    hexes = iter(ids or [f"{k:02x}" * 16 for k in range(100)])
    runner = EpisodeRunner(
        book,
        None,
        context,
        RunnerSettings(**defaults),
        sleep=lambda _s: None,
        new_hex=lambda: next(hexes),
        log=lambda _t: None,
    )
    return runner, book, wall
