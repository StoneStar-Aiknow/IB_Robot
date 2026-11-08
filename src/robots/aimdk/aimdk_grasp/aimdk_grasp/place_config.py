"""Where in a measured opening the X2 releases a held object (``grasp_execution.backend_options.place``)."""

import math
from dataclasses import dataclass

#: Release positions per orientation besides the centre, by default.
PLACE_EXTRA_POSITIONS = 4
MAX_EXTRA_POSITIONS = 12
MAX_CLEARANCE_M = 0.3


def _metres(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class PlaceTargetConfig:
    #: Gap between the held object's lowest observed point and the rim at release;
    #: a further fixed 1 cm covers the margins of the one-sided object cloud.
    bottom_clearance_m: float = 0.045
    #: The transfer ("above") pose sits this far over the release pose along the opening's normal.
    transfer_clearance_m: float = 0.08
    #: Release positions besides the centre; 0 releases at the centre only.
    extra_positions: int = PLACE_EXTRA_POSITIONS
    #: Where the extra positions are taken: this fraction of the way from the centre to the rim.
    #: Positions that leave the held object too close to the rim are dropped either way.
    edge_fraction: float = 0.25

    def __post_init__(self):
        if not _metres(self.bottom_clearance_m) or not 0.0 <= self.bottom_clearance_m <= MAX_CLEARANCE_M:
            raise ValueError(f"backend_options.place.bottom_clearance_m must be metres in [0, {MAX_CLEARANCE_M}]")
        if not _metres(self.transfer_clearance_m) or not 0.0 < self.transfer_clearance_m <= MAX_CLEARANCE_M:
            raise ValueError(f"backend_options.place.transfer_clearance_m must be metres in (0, {MAX_CLEARANCE_M}]")
        extra = self.extra_positions
        if isinstance(extra, bool) or not isinstance(extra, int) or not 0 <= extra <= MAX_EXTRA_POSITIONS:
            raise ValueError(f"backend_options.place.extra_positions must be an integer in [0, {MAX_EXTRA_POSITIONS}]")
        if not _metres(self.edge_fraction) or not 0.0 < self.edge_fraction < 1.0:
            raise ValueError("backend_options.place.edge_fraction must be a number in (0, 1)")
