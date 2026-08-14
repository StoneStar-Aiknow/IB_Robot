"""Manifest-backed session adapters for speech direction models."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np

from inference_service.backends import RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions import ModelSession


class SpeechDirectionRoleRunner:
    """Protocol adapter retained for the Host FullSubNet and VAD layers."""

    def __init__(self, session: ModelSession, context: RuntimeContext) -> None:
        self.session = session
        self.context = context
        self.backend = "stateful_raw_acl"
        self._request_counter = 0
        self._execution = None

    @property
    def last_timing_ms(self) -> dict[str, float]:
        return {}

    def _invoke(self, role: str, values: Mapping[str, object]) -> Mapping[str, object]:
        if self._execution is not None:
            return self._execution.invoke(role, values)
        self._request_counter += 1
        request = NamedTensorRequest(
            f"speech-direction-{self._request_counter}",
            {"observation.audio_4ch": np.zeros((1, 4), dtype=np.float32)},
        )
        return self.session.execute_role(role, values, request)

    @contextmanager
    def execution_scope(self):
        """Keep FB, Host feature assembly, and SB in one admitted request."""
        if self._execution is not None:
            raise RuntimeError("speech direction execution scope is already active")
        self._request_counter += 1
        request = NamedTensorRequest(
            f"speech-direction-{self._request_counter}",
            {"observation.audio_4ch": np.zeros((1, 4), dtype=np.float32)},
        )
        with self.session.execution(request) as execution:
            self._execution = execution
            try:
                yield
            finally:
                self._execution = None

    def infer_named(self, values: Mapping[str, object]) -> Mapping[str, object]:
        return self._invoke("silero_vad", values)

    def run_fb(self, frame: np.ndarray) -> np.ndarray:
        output = self._invoke("fullsubnet_fb", {"host.fullsubnet.fb_spectrum": np.ascontiguousarray(frame)})
        return np.asarray(output["host.fullsubnet.fb_features"], dtype=np.float32)

    def run_sb(self, frame: np.ndarray) -> np.ndarray:
        output = self._invoke("fullsubnet_sb", {"host.fullsubnet.sb_features": np.ascontiguousarray(frame)})
        return np.asarray(output["host.fullsubnet.sb_mask"], dtype=np.float32)

    def infer(self, audio: np.ndarray) -> float:
        output = self._invoke("silero_vad", {"host.silero.audio": np.ascontiguousarray(audio)})
        return float(np.asarray(output["host.silero.prob"]).reshape(-1)[0])

    inference = infer

    def reset_state(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.session.reset()

    def close(self) -> None:
        self.session.close()


__all__ = ["SpeechDirectionRoleRunner"]
