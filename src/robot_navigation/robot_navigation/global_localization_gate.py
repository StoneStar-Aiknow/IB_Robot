from dataclasses import dataclass


@dataclass
class GlobalLocalizationGate:
    startup_delay: float
    required_scans: int
    scan_count: int = 0
    triggered: bool = False

    def record_scan(self) -> None:
        self.scan_count += 1

    def should_trigger(self, elapsed: float) -> bool:
        return not self.triggered and elapsed >= self.startup_delay and self.scan_count >= self.required_scans

    def mark_triggered(self) -> None:
        self.triggered = True
