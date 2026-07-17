"""Read-only LeRobot processor adapters for compiled deployments."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Mapping
from typing import Any

from inference_manifest import ValidatedManifest
from inference_service.backends import RuntimeContext
from inference_service.lerobot_assets import TOKENIZER_REFERENCE_KEYS, resolve_local_semantic_reference
from inference_service.pipeline.errors import PipelineConfigurationError


class _ProcessorPair:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._preprocessor: Any | None = None
        self._postprocessor: Any | None = None
        self._closed = False

    def load(self, context: RuntimeContext | ValidatedManifest) -> None:
        with self._lock:
            if self._closed:
                raise PipelineConfigurationError("LeRobot processor pair is closed", code="processor_pair_closed")
            if self._preprocessor is not None and self._postprocessor is not None:
                return

            try:
                policy_config_module = importlib.import_module("lerobot.configs.policies")
                factory_module = importlib.import_module("lerobot.policies.factory")
                pretrained_config_type = policy_config_module.PreTrainedConfig
                make_processors = factory_module.make_pre_post_processors
            except (AttributeError, ImportError, OSError) as exc:
                raise PipelineConfigurationError(
                    f"LeRobot processor dependencies are unavailable: {exc}",
                    code="processor_dependency_unavailable",
                ) from exc

            validated_manifest = context.validated_manifest if isinstance(context, RuntimeContext) else context
            bundle_path = str(validated_manifest.bundle_root)
            policy_config_resolver = getattr(factory_module, "_get_builtin_policy_config_class", None)
            if callable(policy_config_resolver):
                policy_config_resolver(validated_manifest.policy.policy_type)
            policy_config = pretrained_config_type.from_pretrained(bundle_path, local_files_only=True)
            try:
                policy_config.device = "cpu"
            except (AttributeError, TypeError) as exc:
                raise PipelineConfigurationError(
                    "LeRobot policy config does not permit in-memory CPU processor placement",
                    code="incompatible_policy_config",
                ) from exc
            preprocessor_overrides: dict[str, dict[str, object]] = {"device_processor": {"device": "cpu"}}
            try:
                tokenizer_path = resolve_local_semantic_reference(
                    validated_manifest.bundle_root,
                    "policy_preprocessor.json",
                    TOKENIZER_REFERENCE_KEYS,
                )
            except ValueError as exc:
                raise PipelineConfigurationError(
                    str(exc),
                    code="invalid_processor_metadata",
                ) from exc
            if tokenizer_path is not None:
                preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}
            postprocessor_overrides = {"device_processor": {"device": "cpu"}}
            self._preprocessor, self._postprocessor = make_processors(
                policy_cfg=policy_config,
                pretrained_path=bundle_path,
                preprocessor_overrides=preprocessor_overrides,
                postprocessor_overrides=postprocessor_overrides,
            )

    def preprocess(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        if self._preprocessor is None:
            raise PipelineConfigurationError("LeRobot preprocessor is not loaded", code="processor_not_loaded")
        return self._preprocessor(dict(inputs))

    def postprocess(self, action: object) -> object:
        if self._postprocessor is None:
            raise PipelineConfigurationError("LeRobot postprocessor is not loaded", code="processor_not_loaded")
        torch_module = importlib.import_module("torch")
        if not torch_module.is_tensor(action):
            action = torch_module.as_tensor(action)
        return self._postprocessor(action)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._preprocessor = None
            self._postprocessor = None


class _PreprocessorView:
    def __init__(self, pair: _ProcessorPair) -> None:
        self._pair = pair

    def load(self, context: RuntimeContext | ValidatedManifest) -> None:
        self._pair.load(context)

    def __call__(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        return self._pair.preprocess(inputs)

    def close(self) -> None:
        self._pair.close()


class _PostprocessorView:
    def __init__(self, pair: _ProcessorPair) -> None:
        self._pair = pair

    def load(self, context: RuntimeContext | ValidatedManifest) -> None:
        self._pair.load(context)

    def __call__(self, action: object) -> object:
        return self._pair.postprocess(action)

    def close(self) -> None:
        self._pair.close()


def create_lerobot_processor_views() -> tuple[_PreprocessorView, _PostprocessorView]:
    pair = _ProcessorPair()
    return _PreprocessorView(pair), _PostprocessorView(pair)
