"""Exact seed aliases only; unknown descriptions never fall back to generation."""

import time
import uuid
from collections.abc import Mapping
from types import SimpleNamespace

from embodied_common.pose_composer_contract import PREPARED_TTL_SEC
from pose_composer.errors import ComposeError
from pose_composer.memory import normalize
from robot_runtime.pose_backend import Composition


def display_pose_policy(value, library=None):
    """Opt in named display targets without changing functional task-pose arrival."""
    if value is None or (isinstance(value, Mapping) and not value):
        return {}
    keys = {"pose_ids", "mode"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"pose_composer.display_completion requires exactly {sorted(keys)}")
    names = value["pose_ids"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("display_completion.pose_ids must be a non-empty list of pose IDs")
    if len(names) != len(set(names)):
        raise ValueError("display_completion.pose_ids must be distinct")
    allowed = {"arms_forward", "arms_up", "arms_out", "arms_v", "flex", "surrender", "right_forward", "left_forward"}
    if any(name not in allowed for name in names) or value["mode"] != "playback_only":
        raise ValueError("only supported display poses may use playback_only completion")
    if library is not None:
        for name in names:
            if name not in library["poses"]:
                raise ValueError(f"display completion pose {name!r} is not in the fixed library")
            description = library["poses"][name]["description"]
            if all(description[side].get("task_pose") for side in ("left", "right")):
                raise ValueError("display completion cannot override a functional task-pose anchor")
    return dict(value)


class FixedComposer:
    memory = None

    def __init__(self, library):
        self.library = library
        self.aliases, self.prepared = {}, {}
        for name, record in library["poses"].items():
            for phrase in [name, *record["phrases"]]:
                key = normalize(phrase)
                if key in self.aliases and self.aliases[key] != name:
                    raise ValueError(f"ambiguous fixed-pose alias: {phrase}")
                self.aliases[key] = name

    def compose(self, text, start, **_kwargs):
        name = self.aliases.get(normalize(text))
        if name is None:
            raise ComposeError("POSE_NOT_IN_LIBRARY", "only compiled seed actions are enabled; generation is disabled")
        record = self.library["poses"][name]
        prepared = SimpleNamespace(
            text=text,
            composition_id=uuid.uuid4().hex,
            created=time.monotonic(),
            composition=Composition("pose", [dict(record["target"])], [0.0], record["user_message"]),
            from_memory=True,
            pose_id=name,
            description=record["description"],
        )
        self.prepared[normalize(text)] = prepared
        return prepared

    def take(self, text, composition_id=""):
        result = self.prepared.pop(normalize(text), None)
        if result is None or (composition_id and result.composition_id != composition_id):
            return None
        return result if time.monotonic() - result.created <= PREPARED_TTL_SEC else None

    def lookup(self, request, response):
        import json

        name = self.aliases.get(normalize(request.text))
        response.hit = name is not None
        response.pose_id = name or ""
        response.state = "pinned" if name else ""
        response.user_message = self.library["poses"][name]["user_message"] if name else ""
        response.catalog_json = json.dumps(
            [
                {
                    "pose_id": key,
                    "target_name": record["phrases"][0],
                    "phrases": record["phrases"],
                    "arms": ["left", "right"],
                    "description": record["description"],
                }
                for key, record in self.library["poses"].items()
            ],
            ensure_ascii=False,
        )
        return response

    def feedback(self, *_args):
        return "", "fixed library is read-only"

    def succeeded(self, _prepared):
        return ""
