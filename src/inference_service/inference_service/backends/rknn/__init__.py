"""RKNNLite backend loaded lazily by the static backend registry."""

from inference_service.backends.rknn.backend import RKNNBackend, create_backend

__all__ = ["RKNNBackend", "create_backend"]
