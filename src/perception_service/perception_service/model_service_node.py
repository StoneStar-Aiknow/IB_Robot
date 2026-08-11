"""Compatibility imports for the model-service host moved to inference_service."""

from inference_service.model_service_node import (
    ModelServiceNode,
    _instantiate_plugin,
    _load_class,
    _load_service_type,
    _runtime_info,
    _validate_service_contract,
    main,
)

__all__ = [
    "ModelServiceNode",
    "_instantiate_plugin",
    "_load_class",
    "_load_service_type",
    "_runtime_info",
    "_validate_service_contract",
    "main",
]


if __name__ == "__main__":
    main()
