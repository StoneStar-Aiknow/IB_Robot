"""Mode, base-stability, and scan-epoch gates for persistent map mutation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WriteAdmission:
    allowed: bool
    reason: str = ""


class SemanticWritePolicy:
    WRITABLE_MODES = {"mapping", "navigation"}
    PAUSED_MODES = {"manipulation", "imitation"}

    def admit(
        self,
        *,
        mode: str,
        base_stable: bool,
        frame_scan_epoch: int,
        active_scan_epoch: int,
        override: bool = False,
    ) -> WriteAdmission:
        if override:
            return WriteAdmission(True)
        if mode in self.PAUSED_MODES:
            return WriteAdmission(False, f"semantic writes paused in {mode} mode")
        if mode not in self.WRITABLE_MODES:
            return WriteAdmission(False, f"semantic writes are not enabled in {mode} mode")
        if not base_stable:
            return WriteAdmission(False, "semantic writes require a stable base")
        if frame_scan_epoch != active_scan_epoch:
            return WriteAdmission(False, "frame belongs to a stale scan epoch")
        return WriteAdmission(True)
