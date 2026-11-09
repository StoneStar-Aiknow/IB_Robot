"""Neutral, immutable identity for an executor-owned stage handoff."""

from dataclasses import asdict, dataclass
from typing import ClassVar

from .contracts import ModelRequest


@dataclass(frozen=True)
class StageTicket:
    generation: int
    snapshot_id: str = ""
    snapshot_version: int = 0

    metadata_key: ClassVar[str] = "stage_ticket"

    def attach(self, request: ModelRequest) -> ModelRequest:
        return ModelRequest(
            request.inputs,
            {**request.metadata, self.metadata_key: asdict(self)},
        )

    @classmethod
    def from_request(cls, request: ModelRequest) -> "StageTicket":
        return cls(**request.metadata[cls.metadata_key])

    def __post_init__(self):
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("stage generation must be a positive integer")
        if type(self.snapshot_version) is not int or self.snapshot_version < 0:
            raise ValueError("snapshot version must be a nonnegative integer")
        if not isinstance(self.snapshot_id, str) or bool(self.snapshot_id) != (self.snapshot_version > 0):
            raise ValueError("snapshot identity and version must be provided together")
