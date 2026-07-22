"""Ascend ACL backend loaded lazily by the static backend registry."""

from inference_service.backends.ascend.backend import AscendBackend, create_backend

__all__ = ["AscendBackend", "create_backend"]
