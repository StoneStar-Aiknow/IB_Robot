"""Base class and result types for sim scene tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    success: bool
    reason: str  # "contact" | "flew_out" | "timeout" | "no_data"
    banana_final_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    duration_s: float = 0.0


class SceneTask(ABC):
    """Abstract base for per-scene task implementations.

    Subclasses must set ``scene_name`` as a class attribute and implement
    ``randomize``, ``reset``, and ``evaluate``.
    """

    scene_name: str  # class-level; used as registry key

    @abstractmethod
    def randomize(self) -> tuple[bool, str]:
        """Randomise the scene (e.g. place banana at a new position).

        Returns:
            (success, message)
        """

    @abstractmethod
    def reset(self) -> tuple[bool, str]:
        """Reset the scene to its default state.

        Returns:
            (success, message)
        """

    @abstractmethod
    def evaluate(self, duration_s: float = 30.0) -> EvalResult:
        """Run evaluation for up to *duration_s* seconds and return result."""
