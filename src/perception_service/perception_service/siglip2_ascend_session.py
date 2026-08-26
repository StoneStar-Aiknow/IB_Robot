"""Host-orchestrated SigLIP2 dual-encoder execution on Ascend."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.model_sessions import AscendOmModelSession
from inference_service.unified_runtime import ExecutionContext, ModelRequest


class SigLIP2AscendSession(AscendOmModelSession):
    """Adapt dynamic semantic batches to fixed-batch vision and text OM roles."""

    def _load(self, context, rollback) -> None:
        deployment = context.deployment
        if tuple(deployment.execution) != ("vision", "text"):
            raise BackendLoadError(
                f"SigLIP2 Ascend requires execution ['vision', 'text'], got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        super()._load(context, rollback)

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        images = np.asarray(request.inputs["masked_images"], dtype=np.float32)
        tokens = np.asarray(request.inputs["text_tokens"], dtype=np.int64)
        attention = np.asarray(request.inputs["text_attention_mask"], dtype=np.int64)
        if tokens.shape != attention.shape:
            raise BackendInferenceError(
                f"SigLIP2 token and attention shapes differ: {tokens.shape} != {attention.shape}",
                code="input_shape_mismatch",
            )

        image_rows = [
            self._run_role(0, "vision", {"host.siglip2.image": image[None]})["host.siglip2.image_embedding"]
            for image in images
        ]
        image_embeddings = (
            np.concatenate(image_rows, axis=0).astype(np.float32, copy=False)
            if image_rows
            else np.empty((0, self._embedding_dimension("vision")), dtype=np.float32)
        )

        text_batch = self._text_batch_size()
        text_rows = []
        for start in range(0, len(tokens), text_batch):
            chunk = tokens[start : start + text_batch]
            padded = np.zeros((text_batch, tokens.shape[1]), dtype=np.int64)
            padded[: len(chunk)] = chunk
            output = self._run_role(1, "text", {"host.siglip2.input_ids": padded})["host.siglip2.text_embeddings"]
            text_rows.append(np.asarray(output, dtype=np.float32)[: len(chunk)])
        text_embeddings = (
            np.concatenate(text_rows, axis=0)
            if text_rows
            else np.empty((0, self._embedding_dimension("text")), dtype=np.float32)
        )
        return {
            "image_embeddings": np.ascontiguousarray(image_embeddings, dtype=np.float32),
            "text_embeddings": np.ascontiguousarray(text_embeddings, dtype=np.float32),
        }

    def _text_batch_size(self) -> int:
        binding = self._loaded_deployment().bindings["text"].inputs[0]
        if len(binding.shape) != 2 or binding.shape[0] < 1:
            raise BackendInferenceError(
                f"SigLIP2 text OM requires a fixed rank-2 input, got {binding.shape}",
                code="invalid_input_bindings",
            )
        return int(binding.shape[0])

    def _embedding_dimension(self, role: str) -> int:
        binding = self._loaded_deployment().bindings[role].outputs[-1]
        if len(binding.shape) != 2 or binding.shape[-1] < 1:
            raise BackendInferenceError(
                f"SigLIP2 {role} OM has invalid embedding output shape {binding.shape}",
                code="invalid_output_bindings",
            )
        return int(binding.shape[-1])


__all__ = ["SigLIP2AscendSession"]
