from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_manifest.models import DeviceLink
from inference_service.backends import (
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
    BackendState,
    RuntimeContext,
)
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.model_sessions import RKNNModelSession
from tests.manifest_fixtures import (
    TEST_BUNDLE_UUID,
    TEST_DEPLOYMENT_UUID,
    create_non_policy_bundle,
    make_non_policy_manifest,
    v3_runtime_deployment,
    write_manifest,
)
from tests.test_rknn_backend import FakeRKNNEnvironment, FakeRKNNModel


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


def _rknn_artifact(root: Path, role: str, fmt: str = "rknn") -> dict[str, str]:
    suffix = ".pt" if fmt in {"pt", "pytorch"} else ".rknn"
    path = root / "artifacts" / f"{role}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"host-role-weights" if fmt in {"pt", "pytorch"} else role.encode())
    return {"path": str(path.relative_to(root)), "format": fmt}


def _write_rknn_manifest(
    root: Path,
    deployment: dict[str, object],
    *,
    model: dict[str, object],
    runtime_options: dict[str, object] | None = None,
) -> RuntimeContext:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "model.marker"
    marker.write_bytes(b"rknn-session")
    entries = [BundleFile(path="model.marker")]
    full_deployment = {"uuid": TEST_DEPLOYMENT_UUID, "revision": 1, **deployment}
    full_deployment = v3_runtime_deployment(full_deployment, default_backend="rknn")
    write_manifest(
        root,
        {
            "schema_version": 3,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": "rknn-session-test",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "rknn-session-test", entries),
                },
            },
            "model": model,
            "deployments": {"rk3588": full_deployment},
        },
    )
    return RuntimeContext(load_inference_manifest(root, "rk3588"), runtime_options=runtime_options or {})


_LINKED_MODEL = {
    "interface": "tensor_model",
    "model_type": "linked_rknn",
    "operation": "infer",
    "inputs": [
        {"semantic": "observation.image", "dtype": "float32", "shape": [1, 3, 4, 4], "layout": "NCHW"},
        {"semantic": "bias", "dtype": "float32", "shape": [1, 4]},
    ],
    "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1, 4]}],
}


def _linked_deployment(root: Path) -> dict[str, object]:
    return {
        "backend": "rknn",
        "target": {"soc": "rk3588", "runtime": "rknn-lite"},
        "artifacts": {
            "encoder": _rknn_artifact(root, "encoder"),
            "head": _rknn_artifact(root, "head"),
        },
        "execution": ["encoder", "head"],
        "bindings": {
            "encoder": {
                "inputs": [_binding("observation.image", "pixel_values", 0, "float32", [1, 3, 4, 4], layout="NCHW")],
                "outputs": [_binding("internal.hidden", "hidden", 0, "float32", [1, 4])],
            },
            "head": {
                "inputs": [
                    _binding("internal.hidden", "hidden", 0, "float32", [1, 4]),
                    _binding("bias", "bias", 1, "float32", [1, 4]),
                ],
                "outputs": [_binding("scores", "scores", 0, "float32", [1, 4])],
            },
        },
    }


def _linked_environment(context: RuntimeContext) -> FakeRKNNEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def encoder(inputs):
        (image,) = inputs
        return [np.full((1, 4), float(np.asarray(image).mean()), dtype=np.float32)]

    def head(inputs):
        hidden, bias = inputs
        return [np.asarray(hidden, dtype=np.float32) + np.asarray(bias, dtype=np.float32)]

    return FakeRKNNEnvironment(
        {
            paths["encoder"]: FakeRKNNModel(encoder),
            paths["head"]: FakeRKNNModel(head),
        }
    )


def test_rknn_session_invokes_roles_semantically_with_per_role_data_format(tmp_path) -> None:
    context = _write_rknn_manifest(
        tmp_path,
        _linked_deployment(tmp_path),
        model=_LINKED_MODEL,
        runtime_options={"target": "rk3588", "core_mask": "0"},
    )
    environment = _linked_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
    session.load(context)

    image = np.ones((1, 3, 4, 4), dtype=np.float32)
    bias = np.full((1, 4), 2.0, dtype=np.float32)
    request = NamedTensorRequest("linked", {"observation.image": image, "bias": bias})

    with session.execution(request) as execution:
        encoder_outputs = execution.invoke("encoder", {"observation.image": image})
        head_outputs = execution.invoke(
            "head",
            {"internal.hidden": encoder_outputs["internal.hidden"], "bias": bias},
        )

    np.testing.assert_array_equal(encoder_outputs["internal.hidden"], np.full((1, 4), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(head_outputs["scores"], np.full((1, 4), 3.0, dtype=np.float32))
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.inference_formats[paths["encoder"]] == ["nchw"]
    assert environment.inference_formats[paths["head"]] == [None]
    assert environment.init_calls == [(paths["encoder"], "rk3588", 1), (paths["head"], "rk3588", 1)]
    assert session.health().state is BackendState.READY
    session.close()


def test_rknn_session_sequential_execute_threads_roles_and_returns_semantic_outputs(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
    session.load(context)

    result = session.infer(
        NamedTensorRequest(
            "sequential",
            {
                "observation.image": np.ones((1, 3, 4, 4), dtype=np.float32),
                "bias": np.ones((1, 4), dtype=np.float32),
            },
        )
    )

    np.testing.assert_array_equal(result.outputs["scores"], np.full((1, 4), 2.0, dtype=np.float32))
    assert result.deployment.backend == "rknn"
    assert result.deployment.deployment_fingerprint == context.deployment_fingerprint
    session.close()


_SHARED_MODEL = {
    "interface": "tensor_model",
    "model_type": "shared_rknn",
    "operation": "infer",
    "inputs": [
        {"semantic": "observation.images.a", "dtype": "float32", "shape": [1, 2, 2, 3], "layout": "NHWC"},
        {"semantic": "observation.images.b", "dtype": "float32", "shape": [1, 2, 2, 3], "layout": "NHWC"},
    ],
    "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1, 2]}],
}


def _shared_deployment(root: Path) -> dict[str, object]:
    shared_path = root / "artifacts" / "vision.rknn"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_bytes(b"shared-vision-rknn")
    fuse_path = root / "artifacts" / "fuse.rknn"
    fuse_path.write_bytes(b"fuse-rknn")
    rel_shared = str(shared_path.relative_to(root))
    rel_fuse = str(fuse_path.relative_to(root))

    def vision(semantic: str, output: str) -> dict[str, object]:
        return {
            "inputs": [_binding(semantic, "pixel_values", 0, "float32", [1, 2, 2, 3], layout="NHWC")],
            "outputs": [_binding(output, "image_embeddings", 0, "float32", [1, 2])],
        }

    return {
        "backend": "rknn",
        "target": {"soc": "rk3588", "runtime": "rknn-lite"},
        "artifacts": {
            "cam_a": {"path": rel_shared, "format": "rknn", "share_group": "vision"},
            "cam_b": {"path": rel_shared, "format": "rknn", "share_group": "vision"},
            "fuse": {"path": rel_fuse, "format": "rknn"},
        },
        "execution": ["cam_a", "cam_b", "fuse"],
        "bindings": {
            "cam_a": vision("observation.images.a", "internal.embed_a"),
            "cam_b": vision("observation.images.b", "internal.embed_b"),
            "fuse": {
                "inputs": [
                    _binding("internal.embed_a", "embed_a", 0, "float32", [1, 2]),
                    _binding("internal.embed_b", "embed_b", 1, "float32", [1, 2]),
                ],
                "outputs": [_binding("scores", "scores", 0, "float32", [1, 2])],
            },
        },
    }


def _shared_environment(context: RuntimeContext) -> FakeRKNNEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def vision(inputs):
        (image,) = inputs
        return [np.full((1, 2), float(np.asarray(image).mean()), dtype=np.float32)]

    def fuse(inputs):
        embed_a, embed_b = inputs
        return [np.asarray(embed_a, dtype=np.float32) + np.asarray(embed_b, dtype=np.float32)]

    return FakeRKNNEnvironment(
        {
            paths["cam_a"]: FakeRKNNModel(vision),
            paths["fuse"]: FakeRKNNModel(fuse),
        }
    )


def test_rknn_session_reuses_share_group_artifact_for_shared_roles(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _shared_deployment(tmp_path), model=_SHARED_MODEL)
    environment = _shared_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
    session.load(context)

    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert paths["cam_a"] == paths["cam_b"]
    assert environment.load_order == [paths["cam_a"], paths["fuse"]]

    image_a = np.full((1, 2, 2, 3), 1.0, dtype=np.float32)
    image_b = np.full((1, 2, 2, 3), 3.0, dtype=np.float32)
    request = NamedTensorRequest("shared", {"observation.images.a": image_a, "observation.images.b": image_b})
    with session.execution(request) as execution:
        embed_a = execution.invoke("cam_a", {"observation.images.a": image_a})
        embed_b = execution.invoke("cam_b", {"observation.images.b": image_b})
        scores = execution.invoke(
            "fuse",
            {
                "internal.embed_a": embed_a["internal.embed_a"],
                "internal.embed_b": embed_b["internal.embed_b"],
            },
        )

    np.testing.assert_array_equal(embed_a["internal.embed_a"], np.full((1, 2), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(embed_b["internal.embed_b"], np.full((1, 2), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(scores["scores"], np.full((1, 2), 4.0, dtype=np.float32))
    assert len(environment.inference_inputs[paths["cam_a"]]) == 2
    assert len(environment.inference_inputs[paths["fuse"]]) == 1
    session.close()


_HOST_ROLE_MODEL = {
    "interface": "tensor_model",
    "model_type": "host_role_rknn",
    "operation": "infer",
    "inputs": [{"semantic": "observation.image", "dtype": "float32", "shape": [1, 3, 4, 4], "layout": "NCHW"}],
    "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1, 4]}],
}


def _host_role_deployment(root: Path) -> dict[str, object]:
    return {
        "backend": "rknn",
        "target": {"soc": "rk3588", "runtime": "rknn-lite"},
        "artifacts": {
            "encoder": _rknn_artifact(root, "encoder"),
            "embedding": _rknn_artifact(root, "embedding", "pt"),
            "decoder": _rknn_artifact(root, "decoder"),
        },
        "execution": ["encoder", "embedding", "decoder"],
        "bindings": {
            "encoder": {
                "inputs": [_binding("observation.image", "pixel_values", 0, "float32", [1, 3, 4, 4], layout="NCHW")],
                "outputs": [_binding("internal.features", "features", 0, "float32", [1, 4])],
            },
            "embedding": {
                "inputs": [_binding("internal.features", "features", 0, "float32", [1, 4])],
                "outputs": [_binding("internal.prefix", "prefix", 0, "float32", [1, 4])],
            },
            "decoder": {
                "inputs": [_binding("internal.prefix", "prefix", 0, "float32", [1, 4])],
                "outputs": [_binding("scores", "scores", 0, "float32", [1, 4])],
            },
        },
    }


def _host_role_environment(context: RuntimeContext) -> FakeRKNNEnvironment:
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}

    def encoder(inputs):
        (image,) = inputs
        return [np.full((1, 4), float(np.asarray(image).mean()), dtype=np.float32)]

    def decoder(inputs):
        (prefix,) = inputs
        return [np.asarray(prefix, dtype=np.float32) + 4.0]

    return FakeRKNNEnvironment(
        {
            paths["encoder"]: FakeRKNNModel(encoder),
            paths["decoder"]: FakeRKNNModel(decoder),
        }
    )


def test_rknn_session_skips_host_role_artifacts_and_rejects_host_invocation(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _host_role_deployment(tmp_path), model=_HOST_ROLE_MODEL)
    environment = _host_role_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
    session.load(context)

    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert paths["embedding"] not in environment.load_order
    assert paths["encoder"] in environment.load_order
    assert paths["decoder"] in environment.load_order

    image = np.ones((1, 3, 4, 4), dtype=np.float32)
    request = NamedTensorRequest("host-role", {"observation.image": image})
    with session.execution(request) as execution:
        features = execution.invoke("encoder", {"observation.image": image})
        np.testing.assert_array_equal(features["internal.features"], np.full((1, 4), 1.0, dtype=np.float32))
        outputs = execution.invoke("decoder", {"internal.prefix": np.zeros((1, 4), dtype=np.float32)})
        np.testing.assert_array_equal(outputs["scores"], np.full((1, 4), 4.0, dtype=np.float32))

    with pytest.raises(BackendInferenceError, match="host role") as error:
        session._execute_role("embedding", {"internal.features": np.zeros((1, 4), dtype=np.float32)}, request)
    assert error.value.code == "host_role_not_executable"
    assert session.health().state is BackendState.READY
    session.close()


def test_rknn_session_partial_load_failure_releases_loaded_sessions(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    environment.fail_init_paths.add(paths["head"])
    session = RKNNModelSession(rknn_loader=environment.runtime_type)

    with pytest.raises(BackendLoadError, match="init_runtime") as error:
        session.load(context)

    assert error.value.code == "runtime_load_failed"
    assert session.health().state is BackendState.FAILED
    assert environment.load_order == [paths["encoder"], paths["head"]]
    assert environment.release_calls == [paths["head"], paths["encoder"]]
    session.close()


def test_rknn_session_close_releases_sessions_in_reverse_order_and_is_idempotent(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
    session.load(context)

    session.close()
    session.close()
    assert session.health().state is BackendState.CLOSED
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.release_calls == [paths["head"], paths["encoder"]]


def test_rknn_session_close_reports_structured_error_after_releasing_sessions(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
    session.load(context)

    def fail() -> None:
        raise RuntimeError("encoder release failed")

    environment.instances[0].release = fail

    with pytest.raises(BackendLifecycleError, match="encoder release failed") as error:
        session.close()

    assert error.value.code == "close_failed"
    assert session.health().state is BackendState.CLOSED
    assert session.health().reason_code == "close_failed"
    paths = {role: str(path) for role, path in context.resolved_artifacts.items()}
    assert environment.release_calls == [paths["head"]]


@pytest.mark.parametrize(
    "runtime_options",
    [
        {"target": ""},
        {"core_mask": -1},
        {"core_mask": "invalid"},
        {"random_seed": 1.5},
        {"unknown": True},
    ],
)
def test_rknn_session_rejects_invalid_runtime_options(tmp_path, runtime_options) -> None:
    context = _write_rknn_manifest(
        tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL, runtime_options=runtime_options
    )

    def load_sdk():
        raise AssertionError("RKNN SDK must not load for invalid runtime options")

    session = RKNNModelSession(rknn_loader=load_sdk)

    with pytest.raises(BackendLoadError) as error:
        session.load(context)

    assert error.value.code == "invalid_runtime_options"
    session.close()


def test_rknn_session_rejects_device_links_before_loading_sdk(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    deployment = context.deployment.model_copy(
        update={
            "device_links": (
                DeviceLink(
                    semantic="internal.hidden",
                    producer="encoder",
                    consumer="head",
                    transport="device_pointer",
                    owner="producer",
                    lifetime="inference",
                ),
            )
        }
    )
    validated = replace(context.validated_manifest, deployment=deployment)
    invalid_context = RuntimeContext(validated)
    sdk_loaded = False

    def load_sdk():
        nonlocal sdk_loaded
        sdk_loaded = True
        raise AssertionError("RKNN SDK must not load when device links are declared")

    session = RKNNModelSession(rknn_loader=load_sdk)

    with pytest.raises(BackendLoadError) as error:
        session.load(invalid_context)

    assert error.value.code == "unsupported_device_links"
    assert sdk_loaded is False
    session.close()


def test_rknn_session_rejects_non_rknn_deployment(tmp_path) -> None:
    bundle_files = create_non_policy_bundle(tmp_path)
    manifest = make_non_policy_manifest(tmp_path, bundle_files)
    write_manifest(tmp_path, manifest)
    context = RuntimeContext(load_inference_manifest(tmp_path, "ascend"))
    session = RKNNModelSession(rknn_loader=lambda: FakeRKNNEnvironment({}).runtime_type())

    with pytest.raises(BackendLoadError, match="compiled rknn deployment") as error:
        session.load(context)

    assert error.value.code == "invalid_deployment"
    session.close()


def test_rknn_session_rejects_unknown_role_invocation(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
    session.load(context)

    request = NamedTensorRequest(
        "unknown-role",
        {
            "observation.image": np.ones((1, 3, 4, 4), dtype=np.float32),
            "bias": np.ones((1, 4), dtype=np.float32),
        },
    )
    with session.execution(request) as execution:
        with pytest.raises(BackendInferenceError, match="unknown or unloaded") as error:
            execution.invoke("ghost", {"observation.image": request.inputs["observation.image"]})
        assert error.value.code == "unknown_execution_role"

    assert session.health().state is BackendState.FAILED
    session.close()


def test_rknn_session_rejects_unsupported_controls_without_claiming_capabilities(tmp_path) -> None:
    context = _write_rknn_manifest(tmp_path, _linked_deployment(tmp_path), model=_LINKED_MODEL)
    environment = _linked_environment(context)
    session = RKNNModelSession(rknn_loader=environment.runtime_type)
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
