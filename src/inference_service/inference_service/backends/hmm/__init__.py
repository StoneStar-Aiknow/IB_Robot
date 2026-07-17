"""Houmo HMM backend loaded lazily by the static backend registry."""

from inference_service.backends.hmm.backend import HMMBackend, create_backend

__all__ = ["HMMBackend", "create_backend"]
