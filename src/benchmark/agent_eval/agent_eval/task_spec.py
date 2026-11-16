"""TaskSpec and Recipe files: strict, allowlisted schemas.

A TaskSpec says what an evaluation task lets the Agent see and do (observation policy,
decision budget, recipe). It must never carry the answer: object poses, target coordinates,
layouts, seeds or the evaluator's success thresholds are refused, as is every key outside the
schema. A Recipe is the method knowledge each request carries (the rules text and the
fingertip height reference of :mod:`agent_eval.episode_context`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .episode_context import ARM_OBSERVATION_FIELDS, OBSERVATION_FIELDS, RecipeFacts

PACKAGE = "agent_eval"
_SOURCE_CONFIG = Path(__file__).resolve().parents[1] / "config"

#: Observation fields a TaskSpec may grant (the episode context's allowlist, flattened).
GRANTABLE_FIELDS = frozenset(f for f in OBSERVATION_FIELDS if f != "arms") | frozenset(ARM_OBSERVATION_FIELDS)
INSTRUCTION_SOURCES = frozenset({"platform"})

# Key fragments that name an answer or the evaluator's judgement; never allowed in a TaskSpec.
_ANSWER_FRAGMENTS = (
    "answer",
    "truth",
    "object",
    "target",
    "goal",
    "success",
    "threshold",
    "score",
    "reward",
    "layout",
    "seed",
    "position",
    "pose",
)


class SpecError(ValueError):
    """A task spec, recipe or suite file does not match its schema."""


def _check_keys(where: str, data: Any, required: set[str], optional: set[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise SpecError(f"{where}: expected a mapping, got {type(data).__name__}")
    keys = set(data)
    missing = required - keys
    if missing:
        raise SpecError(f"{where}: missing {sorted(missing)}")
    unknown = keys - required - set(optional)
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(required | set(optional))})")
    return data


def _reject_answer_fields(where: str, data: Any) -> None:
    if isinstance(data, Mapping):
        for key, value in data.items():
            folded = str(key).lower()
            hit = next((frag for frag in _ANSWER_FRAGMENTS if frag in folded), None)
            if hit is not None:
                raise SpecError(f"{where}.{key}: answer-like field ({hit!r}) is not allowed in a task spec")
            _reject_answer_fields(f"{where}.{key}", value)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            _reject_answer_fields(f"{where}[{index}]", item)


def _string_list(where: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise SpecError(f"{where}: expected a list of non-empty strings")
    return tuple(value)


def _positive(where: str, value: Any, kind=float) -> Any:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise SpecError(f"{where}: expected a positive number")
    if kind is int and int(value) != value:
        raise SpecError(f"{where}: expected an integer")
    return kind(value)


def canonical_sha256(data: Any) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no whitespace)."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    task: str
    platform_task: str
    instruction_source: str
    cameras: tuple[str, ...]
    observation_fields: tuple[str, ...]
    max_decisions: int
    decision_timeout_s: float
    recipe: str
    source: dict[str, Any]
    path: str | None = None

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.source)


def parse_task_spec(data: Any, path: str | None = None) -> TaskSpec:
    """Validate a task spec mapping; answer-like and unknown fields are refused."""
    _reject_answer_fields("task_spec", data)
    top = _check_keys(
        "task_spec",
        data,
        {
            "schema_version",
            "kind",
            "task",
            "platform_task",
            "instruction_source",
            "observation",
            "decision_budget",
            "recipe",
        },
    )
    if top["schema_version"] != 1 or top["kind"] != "task_spec":
        raise SpecError("task_spec: expected schema_version 1 and kind task_spec")
    if top["instruction_source"] not in INSTRUCTION_SOURCES:
        raise SpecError(f"task_spec.instruction_source: expected one of {sorted(INSTRUCTION_SOURCES)}")
    observation = _check_keys("task_spec.observation", top["observation"], {"cameras", "fields"})
    fields = _string_list("task_spec.observation.fields", observation["fields"])
    extra = set(fields) - GRANTABLE_FIELDS
    if extra:
        raise SpecError(
            f"task_spec.observation.fields: not grantable {sorted(extra)} (allowed {sorted(GRANTABLE_FIELDS)})"
        )
    budget = _check_keys("task_spec.decision_budget", top["decision_budget"], {"max_decisions", "decision_timeout_s"})
    for key in ("task", "platform_task", "recipe"):
        if not isinstance(top[key], str) or not top[key]:
            raise SpecError(f"task_spec.{key}: expected a non-empty string")
    return TaskSpec(
        task=top["task"],
        platform_task=top["platform_task"],
        instruction_source=top["instruction_source"],
        cameras=_string_list("task_spec.observation.cameras", observation["cameras"]),
        observation_fields=fields,
        max_decisions=_positive("task_spec.decision_budget.max_decisions", budget["max_decisions"], int),
        decision_timeout_s=_positive("task_spec.decision_budget.decision_timeout_s", budget["decision_timeout_s"]),
        recipe=top["recipe"],
        source=dict(top),
        path=path,
    )


_THRESHOLDS = {"fingers_on_table_z"}


@dataclass(frozen=True)
class Recipe:
    name: str
    guide: str
    done_word: str
    max_idle_replies: int
    thresholds: dict[str, float]
    source: dict[str, Any]
    path: str | None = None

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.source)

    def facts(self) -> RecipeFacts:
        """What the episode context consumes."""
        return RecipeFacts(
            guide=self.guide,
            done_word=self.done_word,
            max_idle_replies=self.max_idle_replies,
            **{key: self.thresholds[key] for key in sorted(_THRESHOLDS)},
        )


def parse_recipe(data: Any, path: str | None = None) -> Recipe:
    top = _check_keys(
        "recipe", data, {"schema_version", "kind", "name", "guide", "done_word", "max_idle_replies", "thresholds"}
    )
    if top["schema_version"] != 1 or top["kind"] != "recipe":
        raise SpecError("recipe: expected schema_version 1 and kind recipe")
    if not isinstance(top["guide"], str) or not top["guide"].strip():
        raise SpecError("recipe.guide: expected non-empty text")
    if not isinstance(top["done_word"], str) or top["done_word"] not in top["guide"]:
        raise SpecError("recipe.done_word: must be a string the guide tells the Agent to use")
    thresholds = _check_keys("recipe.thresholds", top["thresholds"], set(_THRESHOLDS))
    numbers = {}
    for key, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SpecError(f"recipe.thresholds.{key}: expected a number")
        numbers[key] = float(value)
    recipe = Recipe(
        name=str(top["name"]),
        guide=top["guide"],
        done_word=top["done_word"],
        max_idle_replies=_positive("recipe.max_idle_replies", top["max_idle_replies"], int),
        thresholds=numbers,
        source=dict(top),
        path=path,
    )
    try:
        recipe.facts()
    except ValueError as exc:
        raise SpecError(f"recipe: {exc}") from exc
    return recipe


def config_path(kind: str, name_or_path: str) -> Path:
    """A config file given as a path, or by name under share/agent_eval/config/<kind>/."""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    file_name = name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml"
    roots = []
    with suppress(ImportError):
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        with suppress(PackageNotFoundError):
            roots.append(Path(get_package_share_directory(PACKAGE)) / "config")
    roots.append(_SOURCE_CONFIG)
    for root in roots:
        path = root / kind / file_name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{kind} config not found: {name_or_path!r} (searched {[str(r / kind) for r in roots]})")


def _load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_task_spec(name_or_path: str) -> TaskSpec:
    path = config_path("tasks", name_or_path)
    return parse_task_spec(_load_yaml(path), str(path))


def load_recipe(name_or_path: str) -> Recipe:
    path = config_path("recipes", name_or_path)
    return parse_recipe(_load_yaml(path), str(path))
