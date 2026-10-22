"""Is the camera covered in this frame: the business half of the cover gestures. No ROS.

The gesture timing (double cover, long cover) is the framework's
(:mod:`.gesture`); deciding that one frame is "covered" is the business's.
The cover judge is chosen by name (``cover_judge``), like the business hooks:

``""``               ``CoverJudge``: never covered, so no gesture fires
``"demo"``           ``DemoCoverJudge``: the demo rule below
``"pkg.mod:Class"``  ``Class(cover_judge section)``; must subclass ``CoverJudge``

The ``cover_judge`` section of the business config goes to the class as is.
A judge may leave the value it compared in ``level`` for ``~/status``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy

from manipulation_execution.mirror_session.business import DEMO, BusinessConfig, load_class

DEMO_COVER_DEFAULTS = {"percentile": 95.0, "max_value": 45.0}
_DEMO_COVER_RANGES = {"percentile": 100.0, "max_value": 255.0}


@dataclass(frozen=True)
class GrayFrame:
    image: numpy.ndarray  # the reduced uint8 gray image
    mean: float  # 0..255
    max: float  # 0..255
    stamp_s: float  # camera time

    @classmethod
    def of(cls, image: numpy.ndarray, stamp_s: float) -> GrayFrame:
        if image.size == 0:
            raise ValueError("empty image")
        return cls(image, float(image.mean()), float(image.max()), stamp_s)


class CoverJudge:
    """The default and the base class: never covered."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.level: float | None = None

    def covered(self, frame: GrayFrame) -> bool:
        return False


class DemoCoverJudge(CoverJudge):
    """Covered when the ``percentile``-th percentile of the gray image is at most ``max_value``.

    With the defaults (95, 45): covered when no more than 5 % of the pixels
    are brighter than 45. A few bright pixels (a lamp at the edge of the hand)
    still count as covered; a dark room with a lit window does not. A room
    dark everywhere would; that is for the real machine to show.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(config)
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("the demo cover_judge config must be a mapping")
        unknown = sorted(str(key) for key in set(config) - set(DEMO_COVER_DEFAULTS))
        if unknown:
            known = list(DEMO_COVER_DEFAULTS)
            raise ValueError(f"the demo cover_judge config has unknown keys {unknown}; known: {known}")
        settings = {}
        for name, default in DEMO_COVER_DEFAULTS.items():
            value = config.get(name, default)
            high = _DEMO_COVER_RANGES[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= high
            ):
                raise ValueError(f"cover_judge.{name} must be a number from 0 to {high:g}")
            settings[name] = float(value)
        self._percentile = settings["percentile"]
        self._max_value = settings["max_value"]

    def covered(self, frame: GrayFrame) -> bool:
        self.level = float(numpy.percentile(frame.image, self._percentile))
        return self.level <= self._max_value


def build_cover_judge(name: str, business: BusinessConfig) -> CoverJudge:
    """The cover judge named ``name`` (see the module docstring)."""
    if not isinstance(name, str):
        raise ValueError("cover_judge must be a string")
    if not name:
        return CoverJudge()
    if name == DEMO:
        return DemoCoverJudge(business.cover_judge)
    cls = load_class(name)
    if not issubclass(cls, CoverJudge):
        raise ValueError(f"cover_judge {name!r} is not a CoverJudge subclass")
    return cls(business.cover_judge)
