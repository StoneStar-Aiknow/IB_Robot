"""Left OmniHand joint mapping between native positions and planning-model aliases."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aimdk_grasp.joints import DIP_FOLLOWERS, HAND_ALIASES, HAND_SIGNS

OPEN_HAND_TOLERANCE_RAD = 0.025


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


def load_gripper_presets(gripper: Path, model: ET.Element) -> dict[str, dict[str, float]]:
    """The ``open`` and ``close`` hand presets of a GraspGenX gripper description.

    The presets belong to the grasp model: candidates are generated for the hand
    shape they describe, so they are read from the deployed gripper description
    rather than kept here. Its link list must name this model's hand links.
    """
    annotation = json.loads((Path(gripper) / "config.json").read_text(encoding="utf-8"))
    if set(annotation["links"]) != set(hand_links(model)):
        raise ValueError(f"{gripper}: gripper links do not match the planning model's hand")
    return {name: dict(annotation[name]) for name in ("open", "close")}


class HandMapping:
    """Native left-hand positions (``L_*`` order) to and from mounted ``R_*`` aliases.

    Limits come from the planning model; presets, when given, from the gripper
    description (:func:`load_gripper_presets`).
    """

    def __init__(self, model: ET.Element, presets: Mapping[str, Mapping[str, float]] | None = None):
        def limit(name: str) -> tuple[float, float]:
            element = model.find(f"joint[@name='{name}']/limit")
            if element is None:
                raise ValueError(f"planning model has no limits for {name}")
            return float(element.get("lower")), float(element.get("upper"))

        bounds = np.array(
            [sorted(sign * value for value in limit(name)) for name, sign in zip(HAND_ALIASES, HAND_SIGNS, strict=True)]
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
            raise ValueError("Exactly ten finite active left joints required")
        if np.any(q < self.lower - 1e-6) or np.any(q > self.upper + 1e-6):
            raise ValueError("Left hand exceeds native URDF limits")
        return q

    def to_native(self, alias: Mapping[str, float]) -> np.ndarray:
        if set(alias) != set(HAND_ALIASES):
            raise ValueError("Active alias mapping must contain exactly the ten commanded joints")
        return self.check([sign * alias[name] for name, sign in zip(HAND_ALIASES, HAND_SIGNS, strict=True)])

    def to_alias(self, native) -> dict[str, float]:
        return dict(zip(HAND_ALIASES, self.check(native) * HAND_SIGNS, strict=True))

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
        error = float(np.max(np.abs(self.check(native) - self.preset("open"))))
        if error > OPEN_HAND_TOLERANCE_RAD:
            raise RuntimeError(
                "Initial left hand is not open; open the hand and confirm fresh feedback "
                f"before capturing/planning (max error {error:.4f} rad, "
                f"tolerance {OPEN_HAND_TOLERANCE_RAD:.3f} rad)"
            )
        return error
