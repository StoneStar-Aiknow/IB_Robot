"""Backend-independent policy codec contracts and canonical value objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np

from inference_manifest import ArtifactBindings

TensorValue: TypeAlias = Any
RuntimeOutputs: TypeAlias = Mapping[str | int, TensorValue] | Sequence[TensorValue] | TensorValue


@dataclass(frozen=True)
class CodecRequest:
    """Canonical semantic tensors supplied by policy preprocessing."""

    semantic_tensors: Mapping[str, TensorValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_tensors", MappingProxyType(dict(self.semantic_tensors)))


@dataclass(frozen=True)
class BoundTensor:
    """One converted tensor associated with its declared runtime slot."""

    semantic: str
    runtime_name: str | None
    index: int | None
    value: np.ndarray


@dataclass(frozen=True)
class BoundInputs:
    """Converted runtime inputs addressable by manifest name or index."""

    tensors: tuple[BoundTensor, ...]

    @property
    def by_runtime_name(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {tensor.runtime_name: tensor.value for tensor in self.tensors if tensor.runtime_name is not None}
        )

    @property
    def ordered_values(self) -> tuple[np.ndarray, ...]:
        if any(tensor.index is None for tensor in self.tensors):
            raise ValueError("ordered runtime inputs require an index on every input binding")
        return tuple(tensor.value for tensor in sorted(self.tensors, key=lambda tensor: int(tensor.index)))


@dataclass(frozen=True)
class CodecResult:
    """Canonical semantic outputs with the selected policy action."""

    action: np.ndarray
    semantic_tensors: Mapping[str, np.ndarray]
    actual_chunk_size: int = field(init=False)

    def __post_init__(self) -> None:
        if self.action.ndim == 0:
            raise ValueError("codec action output must have at least one dimension")
        chunk_size = int(self.action.shape[-2]) if self.action.ndim >= 2 else 1
        if chunk_size < 1:
            raise ValueError("codec action output must contain at least one action")
        object.__setattr__(self, "semantic_tensors", MappingProxyType(dict(self.semantic_tensors)))
        object.__setattr__(self, "actual_chunk_size", chunk_size)


@runtime_checkable
class PolicyCodec(Protocol):
    """Maps canonical policy semantics to and from manifest-bound tensors."""

    def encode_inputs(self, request: CodecRequest, bindings: ArtifactBindings) -> BoundInputs: ...

    def decode_outputs(self, outputs: RuntimeOutputs, bindings: ArtifactBindings) -> CodecResult: ...
