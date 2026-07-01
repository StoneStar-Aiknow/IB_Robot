from __future__ import annotations

from inference_service.core.compiled_policy import (
    CompiledPolicyWrapper,
    normalize_backend_name,
)
from inference_service.core.pure_inference_engine import PolicyWrapper


class HMMPolicyWrapper(CompiledPolicyWrapper):
    """PolicyWrapper facade for the Houmo HMM (LQ50 / M50 xh2) backend.

    Delegates input preparation, execution and output decoding to the shared
    ``CompiledPolicyWrapper`` machinery (ACT adapter + ``HMMRuntimeSession``),
    keeping the backend surface uniform with RKNN / Ascend OM.
    """

    def __init__(self) -> None:
        super().__init__("hmm")


def create_hmm_policy_wrapper(device: str) -> PolicyWrapper:
    normalized = normalize_backend_name(device)
    if normalized == "hmm":
        return HMMPolicyWrapper()
    raise ValueError(f"Unsupported HMM inference device: {device}")
