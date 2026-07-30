"""Model-semantic adapter boundary for shared inference sessions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from inference_service.generic_runtime import NamedTensorResult


@dataclass(frozen=True)
class AdapterIdentity:
    family: str
    preprocessing: str
    postprocessing: str
    supported_deployments: frozenset[str]


class PerceptionAdapter(ABC):
    """Own canonical model semantics while the runtime owns device lifecycle."""

    identity: AdapterIdentity

    def validate_deployment(self, deployment: str) -> None:
        if deployment not in self.identity.supported_deployments:
            raise RuntimeError(
                f"{self.identity.family} deployment {deployment!r} is not supported; "
                f"supported deployments: {sorted(self.identity.supported_deployments)}"
            )

    @abstractmethod
    def preprocess(self, value: object) -> Mapping[str, np.ndarray]: ...

    @abstractmethod
    def postprocess(self, result: NamedTensorResult, **options): ...
