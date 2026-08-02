from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from inference_manifest import BundleFile, ValidatedManifest, canonical_bundle_digest
from robot_config import InferenceConfigError, parse_inference_config

_BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
_DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _create_bundle(root: Path, deployments: dict[str, dict[str, str]] | None = None) -> Path:
    root.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "type": "act",
            "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "policy_preprocessor", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "policy_postprocessor", "steps": []})
    (root / "model.safetensors").write_bytes(b"test-policy-weights")

    bundle_paths = (
        "config.json",
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_preprocessor.json",
    )
    entries = [BundleFile(path=path) for path in bundle_paths]
    deployment_values = deployments or {"cpu": {"backend": "torch", "device": "cpu"}}
    deployment_values = {
        name: {"uuid": _DEPLOYMENT_UUID, "revision": 1, **value} for name, value in deployment_values.items()
    }
    _write_json(
        root / "inference_manifest.json",
        {
            "schema_version": 2,
            "bundle": {
                "uuid": _BUNDLE_UUID,
                "revision": 1,
                "name": root.name,
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(_BUNDLE_UUID, 1, root.name, entries),
                },
            },
            "deployments": deployment_values,
        },
    )
    return root


def _pipeline(
    model_path: str | Path,
    *,
    deployment: str = "cpu",
    execution_mode: str = "monolithic",
    **overrides: Any,
) -> dict[str, Any]:
    value = {
        "model_path": str(model_path),
        "deployment": deployment,
        "execution_mode": execution_mode,
    }
    value.update(overrides)
    return value


def _robot_config(pipelines: dict[Any, Any], *, enabled: bool = True) -> dict[str, Any]:
    return {
        "control_modes": {
            "model_inference": {
                "inference": {
                    "enabled": enabled,
                    "pipelines": pipelines,
                }
            }
        }
    }


def test_parse_one_pipeline_returns_immutable_typed_validated_config(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    config = parse_inference_config(_robot_config({"policy": _pipeline(bundle)}), "model_inference")

    assert config.enabled is True
    assert tuple(config.pipelines) == ("policy",)
    pipeline = config.pipelines["policy"]
    assert pipeline.pipeline_id == "policy"
    assert pipeline.model_path == bundle.resolve()
    assert pipeline.deployment == "cpu"
    assert pipeline.execution_mode == "monolithic"
    assert pipeline.request_timeout == 5.0
    assert pipeline.default_task == ""
    assert pipeline.runtime_options == {}
    assert isinstance(pipeline.validated_manifest, ValidatedManifest)
    assert pipeline.validated_manifest.deployment_name == "cpu"
    with pytest.raises(FrozenInstanceError):
        pipeline.default_task = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.pipelines["other"] = pipeline  # type: ignore[index]
    with pytest.raises(TypeError):
        pipeline.runtime_options["device_id"] = 0  # type: ignore[index]


def test_parse_multiple_pipelines_with_mixed_execution_modes(tmp_path: Path) -> None:
    bundle = _create_bundle(
        tmp_path / "bundle",
        {
            "cpu": {"backend": "torch", "device": "cpu"},
            "gpu": {"backend": "torch", "device": "cuda"},
        },
    )

    config = parse_inference_config(
        _robot_config(
            {
                "policy": _pipeline(
                    bundle,
                    request_timeout=1.25,
                    default_task="pick the banana",
                    runtime_options={"perf_enabled": True, "perf_log_every": 2},
                ),
                "planner": _pipeline(bundle, deployment="gpu", execution_mode="distributed", request_timeout=9),
            }
        ),
        "model_inference",
    )

    assert config.pipelines["policy"].execution_mode == "monolithic"
    assert config.pipelines["policy"].request_timeout == 1.25
    assert config.pipelines["policy"].default_task == "pick the banana"
    assert config.pipelines["policy"].runtime_options == {"perf_enabled": True, "perf_log_every": 2}
    assert config.pipelines["planner"].execution_mode == "distributed"
    assert config.pipelines["planner"].request_timeout == 9.0


@pytest.mark.parametrize(
    "pipeline_id",
    ["", "Policy", "1policy", "_policy", "policy-name", "policy.name", "a" * 64, 1],
)
def test_reject_invalid_pipeline_ids(tmp_path: Path, pipeline_id: Any) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="invalid pipeline ID"):
        parse_inference_config(_robot_config({pipeline_id: _pipeline(bundle)}), "model_inference")


def test_accept_maximum_length_pipeline_id_verbatim(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline_id = "a" * 63

    config = parse_inference_config(_robot_config({pipeline_id: _pipeline(bundle)}), "model_inference")

    assert config.pipelines[pipeline_id].pipeline_id == pipeline_id


def test_relative_model_path_resolves_only_against_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _create_bundle(tmp_path / "workspace" / "models" / "bundle")
    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.setenv("WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.chdir(unrelated_cwd)

    config = parse_inference_config(
        _robot_config({"policy": _pipeline("models/bundle")}),
        "model_inference",
    )

    assert config.pipelines["policy"].model_path == bundle.resolve()


def test_reject_relative_model_path_when_workspace_is_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _create_bundle(tmp_path / "models" / "bundle")
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InferenceConfigError, match="WORKSPACE is unset"):
        parse_inference_config(
            _robot_config({"policy": _pipeline("models/bundle")}),
            "model_inference",
        )


def test_reject_relative_workspace_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _create_bundle(tmp_path / "models" / "bundle")
    monkeypatch.setenv("WORKSPACE", ".")

    with pytest.raises(InferenceConfigError, match="WORKSPACE must be an absolute path"):
        parse_inference_config(
            _robot_config({"policy": _pipeline("models/bundle")}),
            "model_inference",
        )


def test_absolute_model_path_does_not_require_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    monkeypatch.delenv("WORKSPACE", raising=False)

    config = parse_inference_config(_robot_config({"policy": _pipeline(bundle)}), "model_inference")

    assert config.pipelines["policy"].model_path == bundle.resolve()


def test_absolute_model_path_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WORKSPACE", raising=False)

    config = parse_inference_config(_robot_config({"policy": _pipeline("~/bundle")}), "model_inference")

    assert config.pipelines["policy"].model_path == bundle.resolve()


def test_reject_missing_model_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(InferenceConfigError, match="model_path does not exist"):
        parse_inference_config(_robot_config({"policy": _pipeline(missing)}), "model_inference")


def test_reject_model_path_that_is_not_directory(tmp_path: Path) -> None:
    model_file = tmp_path / "model"
    model_file.write_text("not a bundle", encoding="utf-8")

    with pytest.raises(InferenceConfigError, match="model_path is not a directory"):
        parse_inference_config(_robot_config({"policy": _pipeline(model_file)}), "model_inference")


def test_reject_missing_config_json(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    (bundle / "config.json").unlink()

    with pytest.raises(InferenceConfigError, match="missing config.json"):
        parse_inference_config(_robot_config({"policy": _pipeline(bundle)}), "model_inference")


def test_reject_missing_manifest(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    (bundle / "inference_manifest.json").unlink()

    with pytest.raises(InferenceConfigError, match="missing inference_manifest.json"):
        parse_inference_config(_robot_config({"policy": _pipeline(bundle)}), "model_inference")


def test_reject_missing_deployment(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="Deployment 'missing' is not present"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, deployment="missing")}),
            "model_inference",
        )


def test_reject_invalid_manifest(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    (bundle / "inference_manifest.json").write_text("{invalid", encoding="utf-8")

    with pytest.raises(InferenceConfigError, match="Invalid JSON"):
        parse_inference_config(_robot_config({"policy": _pipeline(bundle)}), "model_inference")


def test_reject_legacy_inference_model_field(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    robot_config = _robot_config({"policy": _pipeline(bundle)})
    robot_config["control_modes"]["model_inference"]["inference"]["model"] = "legacy_model"

    with pytest.raises(InferenceConfigError, match="legacy field"):
        parse_inference_config(robot_config, "model_inference")


@pytest.mark.parametrize("legacy_field", ["device", "concurrency"])
def test_reject_pipeline_device_and_concurrency(tmp_path: Path, legacy_field: str) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match=legacy_field):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, **{legacy_field: "cpu"})}),
            "model_inference",
        )


def test_reject_enabled_inference_with_empty_pipelines() -> None:
    with pytest.raises(InferenceConfigError, match="must be non-empty"):
        parse_inference_config(_robot_config({}), "model_inference")


@pytest.mark.parametrize("missing_field", ["model_path", "deployment", "execution_mode"])
def test_reject_missing_required_pipeline_fields(tmp_path: Path, missing_field: str) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline = _pipeline(bundle)
    del pipeline[missing_field]

    with pytest.raises(InferenceConfigError, match=rf"{missing_field} is required"):
        parse_inference_config(_robot_config({"policy": pipeline}), "model_inference")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), "5", True])
def test_reject_non_positive_or_non_finite_request_timeout(tmp_path: Path, timeout: Any) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="positive finite number"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, request_timeout=timeout)}),
            "model_inference",
        )


def test_reject_non_string_default_task(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="default_task must be a string"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, default_task=1)}),
            "model_inference",
        )


@pytest.mark.parametrize(
    "runtime_options",
    [[], {1: "invalid"}, {"timeout": float("nan")}, {"value": object()}],
)
def test_reject_invalid_runtime_options(tmp_path: Path, runtime_options: Any) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="runtime_options"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, runtime_options=runtime_options)}),
            "model_inference",
        )


def test_monolithic_transport_defaults_are_exact(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    transport = (
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle)}),
            "model_inference",
        )
        .pipelines["policy"]
        .transport
    )

    assert transport.node_name == "inference_policy"
    assert transport.cloud_node_name is None
    assert transport.action_server == "/inference/policy/dispatch"
    assert transport.reset_service == "/inference/policy/reset"
    assert transport.health_topic == "/inference/policy/health"
    assert transport.action_topic == "/actions/policy"
    assert transport.request_topic is None
    assert transport.result_topic is None
    assert transport.heartbeat_topic is None


def test_distributed_transport_defaults_are_exact(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    transport = (
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, execution_mode="distributed")}),
            "model_inference",
        )
        .pipelines["policy"]
        .transport
    )

    assert transport.node_name == "inference_policy"
    assert transport.cloud_node_name == "inference_policy_cloud"
    assert transport.action_server == "/inference/policy/dispatch"
    assert transport.reset_service == "/inference/policy/reset"
    assert transport.health_topic == "/inference/policy/health"
    assert transport.action_topic == "/actions/policy"
    assert transport.request_topic == "/inference/policy/request"
    assert transport.result_topic == "/inference/policy/result"
    assert transport.heartbeat_topic == "/inference/policy/heartbeat"
    assert transport.video_descriptor_topic == "/inference/policy/video/descriptors"
    assert transport.video_status_topic == "/inference/policy/video/status"


def test_distributed_transport_overrides_are_typed_and_exact(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    overrides = {
        "node_name": "edge_policy",
        "cloud_node_name": "cloud_policy",
        "action_server": "/custom/dispatch",
        "reset_service": "/custom/reset",
        "health_topic": "/custom/health",
        "action_topic": "/custom/action",
        "request_topic": "/custom/request",
        "result_topic": "/custom/result",
        "heartbeat_topic": "/custom/heartbeat",
    }

    transport = (
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, execution_mode="distributed", transport=overrides)}),
            "model_inference",
        )
        .pipelines["policy"]
        .transport
    )

    for field, value in overrides.items():
        assert getattr(transport, field) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("cloud_node_name", "cloud_policy"),
        ("request_topic", "/custom/request"),
        ("result_topic", "/custom/result"),
        ("heartbeat_topic", "/custom/heartbeat"),
    ],
)
def test_monolithic_rejects_distributed_transport_overrides(tmp_path: Path, field: str, value: str) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="monolithic pipeline"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, transport={field: value})}),
            "model_inference",
        )


@pytest.mark.parametrize(
    "transport",
    [
        {"node_name": "/not_a_node"},
        {"node_name": "1invalid"},
        {"action_server": "relative/name"},
        {"health_topic": "/invalid-name"},
        {"result_topic": "/trailing/"},
    ],
)
def test_reject_invalid_ros_transport_names(tmp_path: Path, transport: dict[str, str]) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="valid ROS|absolute ROS"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, execution_mode="distributed", transport=transport)}),
            "model_inference",
        )


@pytest.mark.parametrize(
    "field,conflicting_value",
    [
        ("node_name", "inference_alpha"),
        ("cloud_node_name", "inference_alpha_cloud"),
        ("action_server", "/inference/alpha/dispatch"),
        ("reset_service", "/inference/alpha/reset"),
        ("health_topic", "/inference/alpha/health"),
        ("action_topic", "/actions/alpha"),
        ("request_topic", "/inference/alpha/request"),
        ("result_topic", "/inference/alpha/result"),
        ("heartbeat_topic", "/inference/alpha/heartbeat"),
        ("video_descriptor_topic", "/inference/alpha/video/descriptors"),
        ("video_status_topic", "/inference/alpha/video/status"),
    ],
)
def test_reject_endpoint_conflicts_between_pipelines(tmp_path: Path, field: str, conflicting_value: str) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="endpoint conflict"):
        parse_inference_config(
            _robot_config(
                {
                    "alpha": _pipeline(bundle, execution_mode="distributed"),
                    "beta": _pipeline(
                        bundle,
                        execution_mode="distributed",
                        transport={field: conflicting_value},
                    ),
                }
            ),
            "model_inference",
        )


def test_detect_endpoint_conflicts_across_interface_fields(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="endpoint conflict"):
        parse_inference_config(
            _robot_config(
                {
                    "alpha": _pipeline(bundle),
                    "beta": _pipeline(bundle, transport={"action_topic": "/inference/alpha/health"}),
                }
            ),
            "model_inference",
        )


def test_same_pipeline_transport_interfaces_must_be_distinct(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")

    with pytest.raises(InferenceConfigError, match="endpoint conflict"):
        parse_inference_config(
            _robot_config(
                {
                    "policy": _pipeline(
                        bundle,
                        execution_mode="distributed",
                        transport={
                            "request_topic": "/shared/policy_transport",
                            "result_topic": "/shared/policy_transport",
                        },
                    )
                }
            ),
            "model_inference",
        )


def test_endpoint_conflicts_are_scoped_to_selected_control_mode(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    robot_config = {
        "control_modes": {
            "first": {"inference": {"enabled": True, "pipelines": {"policy": _pipeline(bundle)}}},
            "second": {"inference": {"enabled": True, "pipelines": {"policy": _pipeline(bundle)}}},
        }
    }

    first = parse_inference_config(robot_config, "first")
    second = parse_inference_config(robot_config, "second")

    assert first.pipelines["policy"].transport.action_server == second.pipelines["policy"].transport.action_server
