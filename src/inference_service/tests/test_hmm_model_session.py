from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import (
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
    BackendState,
    RuntimeContext,
)
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions import HMMModelSession
from tests.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    TEST_DEPLOYMENT_UUID,
    create_non_policy_bundle,
    make_non_policy_manifest,
    v3_runtime_deployment,
    write_manifest,
)
from tests.test_hmm_backend import FakeModuleSpec, FakeTCIMEnvironment, _tensor


def _binding(
    semantic: str,
    name: str,
    index: int,
    dtype: str,
    shape: list[int],
    *,
    layout: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "semantic": semantic,
        "runtime_name": name,
        "index": index,
        "dtype": dtype,
        "shape": shape,
    }
    if layout is not None:
        result["layout"] = layout
    return result


def _hmm_artifact(root: Path, role: str, fmt: str = "hmm") -> dict[str, str]:
    suffix = ".pt" if fmt in {"pt", "pytorch"} else ".hmm"
    path = root / "artifacts" / f"{role}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"host-role-weights" if fmt in {"pt", "pytorch"} else role.encode())
    return {"path": str(path.relative_to(root)), "format": fmt}


def _write_hmm_manifest(
    root: Path,
    deployment: dict[str, object],
    *,
    model: dict[str, object],
    runtime_options: dict[str, object] | None = None,
) -> RuntimeContext:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "model.marker"
    marker.write_bytes(b"hmm-session")
    entries = [BundleFile(path="model.marker")]
    full_deployment = {"uuid": TEST_DEPLOYMENT_UUID, "revision": 1, **deployment}
    full_deployment = v3_runtime_deployment(full_deployment, default_backend="hmm")
    write_manifest(
        root,
        {
            "schema_version": 3,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": "hmm-session-test",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "hmm-session-test", entries),
                },
            },
            "model": model,
            "deployments": {"houmo": full_deployment},
        },
    )
    return RuntimeContext(load_inference_manifest(root, "houmo"), runtime_options=runtime_options or {})


_LINKED_MODEL = {
    "interface": "tensor_model",
    "model_type": "linked_hmm",
    "operation": "infer",
    "inputs": [
        {"semantic": "observation.image", "dtype": "float16", "shape": [1, 3, 4, 4], "layout": "NCHW"},
        {"semantic": "bias", "dtype": "float16", "shape": [1, 4]},
    ],
    "outputs": [{"semantic": "scores", "dtype": "float16", "shape": [1, 4]}],
}


def _linked_deployment(root: Path) -> dict[str, object]:
    return {
        "backend": "hmm",
        "target": {"soc": "lq50", "runtime": "tcim"},
        "artifacts": {
            "encoder": _hmm_artifact(root, "encoder"),
            "head": _hmm_artifact(root, "head"),
        },
        "execution": ["encoder", "head"],
        "bindings": {
            "encoder": {
                "inputs": [_binding("observation.image", "pixel_values", 0, "float16", [1, 3, 4, 4], layout="NCHW")],
                "outputs": [_binding("internal.hidden", "hidden", 0, "float16", [1, 4])],
            },
            "head": {
                "inputs": [
                    _binding("internal.hidden", "hidden", 0, "float16", [1, 4]),
                    _binding("bias", "bias", 1, "float16", [1, 4]),
                ],
                "outputs": [_binding("scores", "scores", 0, "float16", [1, 4])],
            },
        },
        "device_links": [
            {
                "semantic": "internal.hidden",
                "producer": "encoder",
                "consumer": "head",
                "transport": "device_pointer",
                "owner": "producer",
            }
        ],
    }


def _linked_environment(context: RuntimeContext) -> FakeTCIMEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def encoder(inputs):
        (image,) = inputs
        return [np.full((1, 4), float(np.asarray(image).mean()), dtype=np.float16)]

    def head(inputs):
        hidden, bias = inputs
        return [np.asarray(hidden, dtype=np.float16) + np.asarray(bias, dtype=np.float16)]

    return FakeTCIMEnvironment(
        {
            paths["encoder"]: FakeModuleSpec(
                inputs=(_tensor("pixel_values", "float16", (1, 3, 4, 4)),),
                outputs=(_tensor("hidden", "float16", (1, 4)),),
                callback=encoder,
            ),
            paths["head"]: FakeModuleSpec(
                inputs=(_tensor("hidden", "float16", (1, 4)), _tensor("bias", "float16", (1, 4))),
                outputs=(_tensor("scores", "float16", (1, 4)),),
                callback=head,
            ),
        }
    )


def test_hmm_session_invokes_linked_roles_without_materializing_intermediate(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)

    image = np.ones((1, 3, 4, 4), dtype=np.float16)
    bias = np.ones((1, 4), dtype=np.float16)
    request = NamedTensorRequest("linked", {"observation.image": image, "bias": bias})

    with session.execution(request) as execution:
        encoder_outputs = execution.invoke("encoder", {"observation.image": image})
        head_outputs = execution.invoke("head", {"bias": bias})

    assert encoder_outputs == {}
    np.testing.assert_array_equal(head_outputs["scores"], np.full((1, 4), 2.0, dtype=np.float16))
    assert session.health().state is BackendState.READY
    assert session.health().failure_count == 0
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.modules[paths["encoder"]].runs == 1
    assert environment.modules[paths["head"]].runs == 1
    assert len(environment.device_links) == 1
    _, target_name, handle = environment.device_links[0]
    assert target_name == "hidden"
    assert handle.direction == "output"
    session.close()


def test_hmm_session_sequential_execute_threads_device_linked_roles(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)

    result = session.infer(
        NamedTensorRequest(
            "sequential",
            {
                "observation.image": np.ones((1, 3, 4, 4), dtype=np.float16),
                "bias": np.ones((1, 4), dtype=np.float16),
            },
        )
    )

    np.testing.assert_array_equal(result.outputs["scores"], np.full((1, 4), 2.0, dtype=np.float16))
    assert result.deployment.backend == "hmm"
    assert result.deployment.deployment_fingerprint == context.deployment_fingerprint
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.modules[paths["encoder"]].runs == 1
    assert environment.modules[paths["head"]].runs == 1
    session.close()


_PRODUCER_INPUT_MODEL = {
    "interface": "tensor_model",
    "model_type": "producer_input_hmm",
    "operation": "infer",
    "inputs": [{"semantic": "features", "dtype": "float16", "shape": [1, 4]}],
    "outputs": [{"semantic": "scores", "dtype": "float16", "shape": [1, 4]}],
}


def _producer_input_deployment(root: Path) -> dict[str, object]:
    return {
        "backend": "hmm",
        "target": {"soc": "lq50", "runtime": "tcim"},
        "artifacts": {
            "feeder": _hmm_artifact(root, "feeder"),
            "reader": _hmm_artifact(root, "reader"),
        },
        "execution": ["feeder", "reader"],
        "bindings": {
            "feeder": {
                "inputs": [
                    _binding("internal.seed", "seed", 0, "float16", [1, 4]),
                    _binding("features", "features", 1, "float16", [1, 4]),
                ],
                "outputs": [_binding("internal.hidden", "hidden", 0, "float16", [1, 4])],
            },
            "reader": {
                "inputs": [
                    _binding("internal.seed", "seed", 0, "float16", [1, 4]),
                    _binding("internal.hidden", "hidden", 1, "float16", [1, 4]),
                ],
                "outputs": [_binding("scores", "scores", 0, "float16", [1, 4])],
            },
        },
        "device_links": [
            {
                "semantic": "internal.seed",
                "producer": "feeder",
                "consumer": "reader",
                "producer_binding": "input",
                "transport": "device_pointer",
                "owner": "producer",
            }
        ],
    }


def _producer_input_environment(context: RuntimeContext) -> FakeTCIMEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def feeder(inputs):
        _seed, features = inputs
        return [np.asarray(features, dtype=np.float16) * 2.0]

    def reader(inputs):
        _seed, hidden = inputs
        return [np.asarray(hidden, dtype=np.float16)]

    return FakeTCIMEnvironment(
        {
            paths["feeder"]: FakeModuleSpec(
                inputs=(_tensor("seed", "float16", (1, 4)), _tensor("features", "float16", (1, 4))),
                outputs=(_tensor("hidden", "float16", (1, 4)),),
                callback=feeder,
            ),
            paths["reader"]: FakeModuleSpec(
                inputs=(_tensor("seed", "float16", (1, 4)), _tensor("hidden", "float16", (1, 4))),
                outputs=(_tensor("scores", "float16", (1, 4)),),
                callback=reader,
            ),
        }
    )


def test_hmm_session_supports_producer_input_device_links(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _producer_input_deployment(tmp_path), model=_PRODUCER_INPUT_MODEL)
    environment = _producer_input_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)

    features = np.full((1, 4), 3.0, dtype=np.float16)
    request = NamedTensorRequest("producer-input", {"features": features})

    with session.execution(request) as execution:
        feeder_outputs = execution.invoke("feeder", {"features": features})
        reader_outputs = execution.invoke("reader", feeder_outputs)

    np.testing.assert_array_equal(feeder_outputs["internal.hidden"], np.full((1, 4), 6.0, dtype=np.float16))
    np.testing.assert_array_equal(reader_outputs["scores"], np.full((1, 4), 6.0, dtype=np.float16))
    assert len(environment.device_links) == 1
    _, target_name, handle = environment.device_links[0]
    assert target_name == "seed"
    assert handle.direction == "input"
    session.close()


_HOST_ROLE_MODEL = {
    "interface": "tensor_model",
    "model_type": "host_role_hmm",
    "operation": "infer",
    "inputs": [{"semantic": "observation.image", "dtype": "float16", "shape": [1, 3, 4, 4], "layout": "NCHW"}],
    "outputs": [{"semantic": "scores", "dtype": "float16", "shape": [1, 4]}],
}


def _host_role_deployment(root: Path) -> dict[str, object]:
    return {
        "backend": "hmm",
        "target": {"soc": "lq50", "runtime": "tcim"},
        "artifacts": {
            "encoder": _hmm_artifact(root, "encoder"),
            "embedding": _hmm_artifact(root, "embedding", "pt"),
            "decoder": _hmm_artifact(root, "decoder"),
        },
        "execution": ["encoder", "embedding", "decoder"],
        "bindings": {
            "encoder": {
                "inputs": [_binding("observation.image", "pixel_values", 0, "float16", [1, 3, 4, 4], layout="NCHW")],
                "outputs": [_binding("internal.features", "features", 0, "float16", [1, 4])],
            },
            "embedding": {
                "inputs": [_binding("internal.features", "features", 0, "float16", [1, 4])],
                "outputs": [_binding("internal.prefix", "prefix", 0, "float16", [1, 4])],
            },
            "decoder": {
                "inputs": [_binding("internal.prefix", "prefix", 0, "float16", [1, 4])],
                "outputs": [_binding("scores", "scores", 0, "float16", [1, 4])],
            },
        },
    }


def _host_role_environment(context: RuntimeContext) -> FakeTCIMEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def encoder(inputs):
        (image,) = inputs
        return [np.full((1, 4), float(np.asarray(image).mean()), dtype=np.float16)]

    def decoder(inputs):
        (prefix,) = inputs
        return [np.asarray(prefix, dtype=np.float16) + 4.0]

    return FakeTCIMEnvironment(
        {
            paths["encoder"]: FakeModuleSpec(
                inputs=(_tensor("pixel_values", "float16", (1, 3, 4, 4)),),
                outputs=(_tensor("features", "float16", (1, 4)),),
                callback=encoder,
            ),
            paths["decoder"]: FakeModuleSpec(
                inputs=(_tensor("prefix", "float16", (1, 4)),),
                outputs=(_tensor("scores", "float16", (1, 4)),),
                callback=decoder,
            ),
        }
    )


def test_hmm_session_skips_host_role_artifacts_and_rejects_host_invocation(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _host_role_deployment(tmp_path), model=_HOST_ROLE_MODEL)
    environment = _host_role_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)

    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert paths["embedding"] not in environment.load_order
    assert paths["encoder"] in environment.load_order
    assert paths["decoder"] in environment.load_order

    image = np.ones((1, 3, 4, 4), dtype=np.float16)
    request = NamedTensorRequest("host-role", {"observation.image": image})
    with session.execution(request) as execution:
        features = execution.invoke("encoder", {"observation.image": image})
        np.testing.assert_array_equal(features["internal.features"], np.full((1, 4), 1.0, dtype=np.float16))
        outputs = execution.invoke("decoder", {"internal.prefix": np.zeros((1, 4), dtype=np.float16)})
        np.testing.assert_array_equal(outputs["scores"], np.full((1, 4), 4.0, dtype=np.float16))

    with pytest.raises(BackendInferenceError, match="host role") as error:
        session._execute_role("embedding", {"internal.features": np.zeros((1, 4), dtype=np.float16)}, request)
    assert error.value.code == "host_role_not_executable"
    assert session.health().state is BackendState.READY
    session.close()


def test_hmm_session_reports_runtime_version(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    environment.runtime.__version__ = "tcim-fake-3.2"
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)

    assert session.runtime_version == ""
    session.load(context)
    assert session.runtime_version == "tcim-fake-3.2"
    session.close()
    assert session.runtime_version == ""


def test_hmm_session_partial_load_failure_releases_loaded_resources(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    environment.fail_paths.add(paths["head"])
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)

    with pytest.raises(BackendLoadError, match="fake TCIM load failure") as error:
        session.load(context)

    assert error.value.code == "runtime_load_failed"
    assert session.health().state is BackendState.FAILED
    assert environment.release_order == [paths["encoder"]]
    assert environment.weight_manager_releases == 1
    session.close()


def test_hmm_session_close_releases_resources_in_reverse_order_and_is_idempotent(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)

    session.close()
    session.close()
    assert session.health().state is BackendState.CLOSED
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.release_order == [paths["head"], paths["encoder"]]
    assert environment.weight_manager_releases == 1


def test_hmm_session_close_reports_structured_error_after_releasing_resources(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def fail() -> None:
        raise RuntimeError("encoder close failed")

    environment.modules[paths["encoder"]].release = fail

    with pytest.raises(BackendLifecycleError, match="encoder close failed") as error:
        session.close()

    assert error.value.code == "close_failed"
    assert session.health().state is BackendState.CLOSED
    assert session.health().reason_code == "close_failed"
    assert environment.release_order == [paths["head"]]
    assert environment.weight_manager_releases == 1


def test_hmm_session_rejects_non_hmm_deployment(tmp_path) -> None:
    bundle_files = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, bundle_files)
    write_manifest(tmp_path, manifest)
    context = RuntimeContext(load_inference_manifest(tmp_path, "ascend"))
    session = HMMModelSession(runtime_loader=lambda: FakeTCIMEnvironment({}).runtime)

    with pytest.raises(BackendLoadError, match="compiled hmm deployment") as error:
        session.load(context)

    assert error.value.code == "invalid_deployment"
    session.close()


def test_hmm_session_rejects_incompatible_target_runtime(tmp_path) -> None:
    deployment = _linked_deployment(tmp_path)
    deployment["target"]["runtime"] = "rknn-lite"
    context = _write_hmm_manifest(tmp_path, deployment, model=_LINKED_MODEL)
    session = HMMModelSession(runtime_loader=lambda: FakeTCIMEnvironment({}).runtime)

    with pytest.raises(BackendLoadError, match="TCIM runtime family") as error:
        session.load(context)

    assert error.value.code == "incompatible_backend_target"
    session.close()


def test_hmm_session_rejects_device_id_mismatch(tmp_path) -> None:
    context = _write_hmm_manifest(
        tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL, runtime_options={"device_id": 1}
    )
    session = HMMModelSession(device_id=0, runtime_loader=lambda: FakeTCIMEnvironment({}).runtime)

    with pytest.raises(BackendLoadError, match="device_id does not match") as error:
        session.load(context)

    assert error.value.code == "deployment_context_mismatch"
    session.close()


def test_hmm_session_rejects_unknown_runtime_options(tmp_path) -> None:
    context = _write_hmm_manifest(
        tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL, runtime_options={"random_seed": 7}
    )
    session = HMMModelSession(runtime_loader=lambda: FakeTCIMEnvironment({}).runtime)

    with pytest.raises(BackendLoadError, match="unknown HMM model-session options") as error:
        session.load(context)

    assert error.value.code == "invalid_runtime_options"
    session.close()


def test_hmm_session_rejects_non_hmm_device_artifact_format(tmp_path) -> None:
    deployment = _linked_deployment(tmp_path)
    deployment["artifacts"]["encoder"]["format"] = "om"
    context = _write_hmm_manifest(tmp_path, deployment, model=_LINKED_MODEL)
    environment = FakeTCIMEnvironment({})
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)

    with pytest.raises(BackendLoadError, match="format must be") as error:
        session.load(context)

    assert error.value.code == "invalid_artifact_format"
    assert environment.weight_manager_releases == 1
    session.close()


def test_hmm_session_rejects_missing_artifact_file(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    paths = {role: path for role, path in context.resolved_artifacts.items()}
    paths["head"].unlink()
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)

    with pytest.raises(BackendLoadError, match="artifact 'head' is unavailable") as error:
        session.load(context)

    assert error.value.code == "invalid_artifact"
    assert environment.weight_manager_releases == 1
    session.close()


def test_hmm_session_rejects_unknown_role_invocation(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)

    request = NamedTensorRequest(
        "unknown-role",
        {
            "observation.image": np.ones((1, 3, 4, 4), dtype=np.float16),
            "bias": np.ones((1, 4), dtype=np.float16),
        },
    )
    with session.execution(request) as execution:
        with pytest.raises(BackendInferenceError, match="unknown or unloaded") as error:
            execution.invoke("ghost", {"observation.image": request.inputs["observation.image"]})
        assert error.value.code == "unknown_execution_role"

    assert session.health().state is BackendState.FAILED
    session.close()


def test_hmm_session_rejects_unsupported_controls_without_claiming_capabilities(tmp_path) -> None:
    context = _write_hmm_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = HMMModelSession(runtime_loader=lambda: environment.runtime)
    session.load(context)

    assert not session.capabilities.resettable
    assert not session.capabilities.supports_cancellation
    for operation, capability in (
        (session.reset, "reset"),
        (lambda: session.cancel("request"), "cancellation"),
        (session.recover, "recovery"),
    ):
        with pytest.raises(Exception) as error:
            operation()
        assert getattr(error.value, "capability", None) == capability
    assert session.health().state is BackendState.READY
    assert session.health().failure_count == 0
    session.close()
