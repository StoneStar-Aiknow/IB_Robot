"""OmniHand joint mapping between native positions and planning-model aliases, per side."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aimdk_grasp.joints import DIP_FOLLOWERS, HAND_ALIASES, HAND_PARTS, check_side, hand_alias_signs

#: How far (rad) a measured joint may be from the open preset and still count
#: as open: before planning a grasp, after opening and on release.
OPEN_HAND_TOLERANCE_RAD = 0.025
#: Except the thumb abduction, which can stop short near the end of its travel
#: toward the palm: 0.088 rad short after a release (2026-11-08) failed the
#: following regrasp. Every other joint keeps the tight bound.
THUMB_ABAD_OPEN_TOLERANCE_RAD = 0.1
#: Per native joint (HAND_PARTS order, either hand).
OPEN_TOLERANCES_RAD = np.full(len(HAND_PARTS), OPEN_HAND_TOLERANCE_RAD)
OPEN_TOLERANCES_RAD[HAND_PARTS.index("thumb_abad")] = THUMB_ABAD_OPEN_TOLERANCE_RAD
OPEN_TOLERANCES_RAD.setflags(write=False)
#: How far (rad) the other, preserved hand may drift while one hand is driven.
#: A drift check, not an "is open" check.
PRESERVED_HAND_DRIFT_RAD = 0.025


def is_open(native, open_pose) -> bool:
    """Every joint within its open tolerance of ``open_pose`` (both native)."""
    return bool(np.all(np.abs(np.asarray(native, float) - np.asarray(open_pose, float)) <= OPEN_TOLERANCES_RAD))


def hand_links(model: ET.Element) -> tuple[str, ...]:
    """The hand's rigid bodies in the planning model: the palm and every link a
    finger joint moves. Fixed frames on them (fingertip markers, the middle
    finger's knuckle) belong to their parent."""
    moved = {
        joint.find("child").get("link")
        for joint in model.findall("joint")
        if joint.get("type") != "fixed" and joint.find("child").get("link").startswith("R_")
    }
    return ("R_palm", *(link.get("name") for link in model.findall("link") if link.get("name") in moved))


def load_gripper_description(gripper: Path, model: ET.Element) -> dict:
    """The GraspGenX gripper description (``config.json``) the deployed grasp model was trained for.

    Its hand presets, palm mount (``base_rotation``) and closing sweep belong to
    the grasp model, so they are read from the deployed description rather than
    kept here. Its link list must name this model's hand links.
    """
    annotation = json.loads((Path(gripper) / "config.json").read_text(encoding="utf-8"))
    if set(annotation["links"]) != set(hand_links(model)):
        raise ValueError(f"{gripper}: gripper links do not match the planning model's hand")
    return annotation


def load_gripper_presets(gripper: Path, model: ET.Element) -> dict[str, dict[str, float]]:
    """The ``open`` and ``close`` hand presets of a GraspGenX gripper description.

    The presets belong to the grasp model: candidates are generated for the hand
    shape they describe.
    """
    annotation = load_gripper_description(gripper, model)
    return {name: dict(annotation[name]) for name in ("open", "close")}


class HandMapping:
    """Native positions of ``side``'s hand (``hand_joints(side)`` order) to and from mounted ``R_*`` aliases.

    ``model`` is that side's planning model: the left hand is mounted mirrored
    (alias = native x ``HAND_SIGNS``), the right hand natively. Limits come
    from the planning model; presets, when given, from the gripper description
    (:func:`load_gripper_presets`), in alias space, so one description serves
    both hands.
    """

    def __init__(self, model: ET.Element, presets: Mapping[str, Mapping[str, float]] | None = None, side: str = "left"):
        self.side = check_side(side)
        self.signs = hand_alias_signs(side)

        def limit(name: str) -> tuple[float, float]:
            element = model.find(f"joint[@name='{name}']/limit")
            if element is None:
                raise ValueError(f"planning model has no limits for {name}")
            return float(element.get("lower")), float(element.get("upper"))

        bounds = np.array(
            [sorted(sign * value for value in limit(name)) for name, sign in zip(HAND_ALIASES, self.signs, strict=True)]
        )
        self.lower, self.upper = bounds[:, 0].copy(), bounds[:, 1].copy()
        self.distal_limits = {name: limit(name) for name in DIP_FOLLOWERS}
        self.presets = {name: dict(values) for name, values in (presets or {}).items()}
        for name, values in self.presets.items():
            if not set(HAND_ALIASES) <= set(values):
                raise ValueError(f"hand preset {name!r} lacks driven joints")

    def check(self, native) -> np.ndarray:
        q = np.asarray(native, dtype=np.float64)
        if q.shape != (10,) or not np.isfinite(q).all():
            raise ValueError(f"Exactly ten finite active {self.side} joints required")
        if np.any(q < self.lower - 1e-6) or np.any(q > self.upper + 1e-6):
            raise ValueError(f"{self.side.capitalize()} hand exceeds native URDF limits")
        return q

    def to_native(self, alias: Mapping[str, float]) -> np.ndarray:
        if set(alias) != set(HAND_ALIASES):
            raise ValueError("Active alias mapping must contain exactly the ten commanded joints")
        return self.check([sign * alias[name] for name, sign in zip(HAND_ALIASES, self.signs, strict=True)])

    def to_alias(self, native) -> dict[str, float]:
        return dict(zip(HAND_ALIASES, self.check(native) * self.signs, strict=True))

    def kinematic_alias(self, native) -> dict[str, float]:
        """Active aliases plus the five model-derived, unmeasured DIP angles."""
        values = self.to_alias(native)
        for name, (master, coefficients) in DIP_FOLLOWERS.items():
            angle = float(np.polynomial.polynomial.polyval(values[master], coefficients))
            lower, upper = self.distal_limits[name]
            if not lower - 1e-6 <= angle <= upper + 1e-6:
                raise ValueError(f"Coupled hand joint exceeds URDF limits: {name}")
            values[name] = angle
        return values

    def preset(self, name: str) -> np.ndarray:
        if name not in self.presets:
            raise KeyError(f"no hand preset {name!r}; load the gripper description's presets")
        return self.to_native({joint: self.presets[name][joint] for joint in HAND_ALIASES})

    def require_open(self, native) -> float:
        """Require the measured pregrasp pose; never replace it with a preset."""
        self.kinematic_alias(native)
        measured = self.check(native)
        error = float(np.max(np.abs(measured - self.preset("open"))))
        if not is_open(measured, self.preset("open")):
            raise RuntimeError(
                f"Initial {self.side} hand is not open; open the hand and confirm fresh feedback "
                f"before capturing/planning (max error {error:.4f} rad, tolerance {OPEN_HAND_TOLERANCE_RAD:.3f} rad, "
                f"thumb abduction {THUMB_ABAD_OPEN_TOLERANCE_RAD:.3f} rad)"
            )
        return error
