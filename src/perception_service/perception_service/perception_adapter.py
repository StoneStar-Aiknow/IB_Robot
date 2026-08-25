"""Model-semantic adapter boundary for shared inference sessions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from inference_manifest import SemanticIdentity
from inference_service.unified_runtime import ModelResult


@dataclass(frozen=True)
class AdapterIdentity:
    model_type: str
    preprocessing: str
    postprocessing: str
    supported_deployments: frozenset[str]
    # ``operation`` distinguishes one service contract of a model type from another
    # (SAM2 automatic vs prompt). Empty for single-contract adapters.
    operation: str = ""


class PerceptionAdapter(ABC):
    """Own canonical model semantics while the runtime owns device lifecycle."""

    identity: AdapterIdentity

    def validate_deployment(self, deployment: str) -> None:
        if deployment not in self.identity.supported_deployments:
            raise RuntimeError(
                f"{self.identity.model_type} deployment {deployment!r} is not supported; "
                f"supported deployments: {sorted(self.identity.supported_deployments)}"
            )

    def validate_identity(self, identity: SemanticIdentity | None) -> None:
        if identity is None:
            raise ValueError(f"{self.identity.model_type} manifest must declare semantic_identity")
        if identity.preprocessing_contract != self.identity.preprocessing:
            raise ValueError(
                f"{self.identity.model_type} preprocessing identity mismatch: "
                f"expected {self.identity.preprocessing!r}, got {identity.preprocessing_contract!r}"
            )
        if identity.output_semantics != self.identity.postprocessing:
            raise ValueError(
                f"{self.identity.model_type} output identity mismatch: "
                f"expected {self.identity.postprocessing!r}, got {identity.output_semantics!r}"
            )

    @abstractmethod
    def preprocess(self, value: object) -> Mapping[str, np.ndarray]: ...

    @abstractmethod
    def postprocess(self, result: ModelResult, **options): ...
