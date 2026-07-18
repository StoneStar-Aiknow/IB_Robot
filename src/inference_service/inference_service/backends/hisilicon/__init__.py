"""Hisilicon worker backend with lazy SD3403 protocol construction."""

from inference_service.backends.hisilicon.backend import HisiliconBackend, create_backend

__all__ = ["HisiliconBackend", "create_backend"]
