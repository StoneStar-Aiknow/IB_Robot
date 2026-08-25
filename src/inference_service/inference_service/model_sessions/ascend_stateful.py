"""Stateful manifest-bound Ascend OM model sessions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_manifest import CompiledDeployment
from inference_service.backends.ascend.model import AclDeviceBuffer, AclModel
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.types import RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions.ascend import AscendOmModelSession


class StatefulAscendOmModelSession(AscendOmModelSession):
    """Keep recurrent role state in double-buffered Ascend device datasets.

    State links come from the deployment manifest. All remaining inputs and
    outputs use the normal manifest host ABI.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._state_indices: dict[str, tuple[tuple[int, int], ...]] = {}
        self._state_buffers: dict[str, tuple[tuple[AclDeviceBuffer, ...], ...]] = {}
        self._state_banks: dict[str, int] = {}

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment):
            raise BackendLoadError("stateful Ascend sessions require a compiled deployment")
        typed_links = deployment.execution_contract.state_links
        if not typed_links:
            raise BackendLoadError(
                "stateful Ascend sessions require manifest state_links", code="invalid_state_contract"
            )
        # The v3 state-link contract intentionally carries logical ownership,
        # not runtime tensor names or indices.  The old double-buffer adapter
        # still needs an explicit ABI binding map; do not infer one from names.
        raise BackendLoadError(
            "stateful Ascend tensor-bank execution requires a v3 state-link ABI adapter",
            code="state_link_abi_adapter_unavailable",
        )

    def _prepare_models(self, deployment: CompiledDeployment, models: Mapping[str, AclModel]) -> None:
        if deployment.device_links:
            raise BackendLoadError(
                "stateful Ascend sessions do not combine recurrent banks with device links",
                code="unsupported_device_links",
            )
        state_buffers: dict[str, tuple[tuple[AclDeviceBuffer, ...], ...]] = {}
        for role in deployment.execution:
            model = models[role]
            state_indices = self._state_indices.get(role, ())
            input_count = len(model.input_descriptors)
            output_count = len(model.output_descriptors)
            if any(
                input_index < 0 or input_index >= input_count or output_index < 0 or output_index >= output_count
                for input_index, output_index in state_indices
            ):
                raise BackendLoadError(f"state indices for role {role!r} are outside its runtime ABI")
            for input_index, output_index in state_indices:
                if model.input_descriptors[input_index].size != model.output_descriptors[output_index].size:
                    raise BackendLoadError(
                        f"state link ({input_index}, {output_index}) for role {role!r} changes size across inference",
                        code="state_size_mismatch",
                    )
            banks = tuple(
                tuple(
                    model.allocate_device_buffer(model.input_descriptors[input_index].size)
                    for input_index, _output_index in state_indices
                )
                for _ in range(2)
            )
            input_banks = tuple(
                {input_index: banks[bank][offset] for offset, (input_index, _output_index) in enumerate(state_indices)}
                for bank in range(2)
            )
            output_banks = tuple(
                {
                    output_index: banks[1 - bank][offset]
                    for offset, (_input_index, output_index) in enumerate(state_indices)
                }
                for bank in range(2)
            )
            host_outputs = set(range(output_count)) - {output_index for _input_index, output_index in state_indices}
            model.prepare_dataset_banks(input_banks, output_banks, host_output_indices=host_outputs)
            state_buffers[role] = banks
        self._state_buffers = state_buffers
        self._state_banks = dict.fromkeys(deployment.execution, 0)
        for role, banks in state_buffers.items():
            for bank in banks:
                for buffer in bank:
                    models[role].zero_device_buffer(buffer)

    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]:
        deployment = self._loaded_deployment()
        if len(deployment.execution) != 1:
            raise BackendInferenceError(
                "multi-role stateful sessions require explicit role execution",
                code="host_orchestration_required",
            )
        role = deployment.execution[0]
        return self._execute_role(role, request.inputs, request)

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: NamedTensorRequest,
    ) -> Mapping[str, object]:
        deployment = self._loaded_deployment()
        if role not in self._state_indices:
            raise BackendInferenceError(f"unknown stateful role {role!r}", code="unknown_execution_role")
        stream = self._request_stream(request)
        if stream is not None:
            raise BackendInferenceError(
                "stateful dataset-bank execution does not support priority streams",
                code="hardware_priority_unavailable",
            )
        bindings = deployment.bindings[role]
        state_input_indices = {input_index for input_index, _output_index in self._state_indices.get(role, ())}
        state_output_indices = {output_index for _input_index, output_index in self._state_indices.get(role, ())}
        role_inputs: dict[int, np.ndarray] = {}
        for binding in bindings.inputs:
            if binding.index is None or int(binding.index) in state_input_indices:
                continue
            try:
                role_inputs[int(binding.index)] = np.asarray(inputs[binding.semantic])
            except KeyError as exc:
                raise BackendInferenceError(
                    f"stateful role {role!r} is missing semantic input {binding.semantic!r}",
                    code="missing_semantic_input",
                ) from exc
        self._capture_role_inputs(role, role_inputs)
        bank = self._state_banks[role]
        try:
            runtime_outputs = self._models[role].execute_bank(bank, role_inputs)
        except Exception as exc:
            raise BackendInferenceError(
                f"stateful role {role!r} failed and requires recovery: {exc}",
                code="state_outcome_unknown",
                recoverable=True,
                operation_started=True,
                outcome_known=False,
            ) from exc
        self._state_banks[role] = 1 - bank
        outputs = {
            binding.semantic: self._bound_output(role, binding, runtime_outputs)
            for binding in bindings.outputs
            if binding.index is not None and int(binding.index) not in state_output_indices
        }
        self._capture_role_outputs(role, outputs)
        return outputs

    def _validate_role_values(
        self,
        role: str,
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
    ) -> None:
        deployment = self._require_context().deployment
        bindings = deployment.bindings[role]
        state_input_indices = {input_index for input_index, _output_index in self._state_indices.get(role, ())}
        state_output_indices = {output_index for _input_index, output_index in self._state_indices.get(role, ())}
        host_inputs = tuple(binding for binding in bindings.inputs if binding.index not in state_input_indices)
        host_outputs = tuple(binding for binding in bindings.outputs if binding.index not in state_output_indices)
        self._validate_values(inputs, host_inputs, f"role_{role}_input")
        self._validate_values(outputs, host_outputs, f"role_{role}_output")

    def execute_role(
        self, role: str, inputs: Mapping[str, object], request: NamedTensorRequest
    ) -> Mapping[str, object]:
        """Execute one role inside an already admitted request scope.

        Host-orchestrated callers should normally use ``session.execution``;
        this small public adapter is useful for legacy streaming facades while
        keeping admission and lifecycle ownership in the session.
        """

        with self.execution(request) as execution:
            return execution.invoke(role, inputs)

    def _reset(self) -> None:
        self._zero_state()

    def _recover(self) -> None:
        self._zero_state()

    def _zero_state(self) -> None:
        for role, banks in self._state_buffers.items():
            model = self._models[role]
            for bank in banks:
                for buffer in bank:
                    model.zero_device_buffer(buffer)
            self._state_banks[role] = 0

    def _close(self) -> None:
        self._state_buffers = {}
        self._state_banks = {}
        super()._close()


__all__ = ["StatefulAscendOmModelSession"]
