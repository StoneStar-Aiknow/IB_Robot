"""Resource and target compilation; business launch migration is separate."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from inference_manifest import BundleFile, canonical_bundle_digest
from robot_config.runtime_target_config import build_runtime_resources


def _coverage(config, targets=None):
    from robot_config.runtime_target_config import build_runtime_target_directory

    if targets is None:
        targets = build_runtime_target_directory(config)
    return [resource["managed_coverage"] for resource in build_runtime_resources(config, targets).values()]


@pytest.mark.parametrize("endpoint", ["relative", "/bad-name", "/a//b", "/" + "a" * 247, "/" + "a" * 241])
def test_global_endpoint_and_readiness_name_are_validated_before_compilation(endpoint):
    from robot_config.runtime_target_config import validate_runtime_fields

    with pytest.raises(ValueError, match="endpoint"):
        validate_runtime_fields({"runtime_endpoint": endpoint})
    validate_runtime_fields({"runtime_endpoint": "/" + "a" * 240})


@pytest.mark.parametrize("resource_id", ["a" * 127, "a" * 128, "gpu:0/x-y.z_1", "-gpu", "gpu name", "设备"])
def test_resource_identifiers_share_target_rules_and_keep_config_path(resource_id):
    from inference_manifest.runtime_target import identifier

    config = {"runtime_scheduler": {"enable": True, "resources": {resource_id: {"physical_device": "local/cpu"}}}}
    try:
        identifier(resource_id, "hardware resource ID")
    except ValueError:
        with pytest.raises(ValueError) as error:
            build_runtime_resources(config)
        assert f"runtime_scheduler.resources.{resource_id}" in str(error.value)
        assert "invalid hardware resource ID" in str(error.value)
    else:
        assert resource_id in build_runtime_resources(config)


def _bundle(root: Path, family: str, *, deployment: str = "torch_cpu") -> Path:
    root.mkdir(parents=True)
    marker = root / "assets" / "identity.txt"
    marker.parent.mkdir()
    marker.write_text(family, encoding="utf-8")
    entry = BundleFile(path="assets/identity.txt")
    manifest = {
        "schema_version": 3,
        "bundle": {
            "uuid": "123e4567-e89b-42d3-a456-426614174000",
            "revision": 1,
            "name": f"test-{family}",
            "files": [entry.model_dump(mode="json")],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest("123e4567-e89b-42d3-a456-426614174000", 1, f"test-{family}", [entry]),
            },
        },
        "model": {
            "interface": "tensor_model",
            "model_type": family,
            "operation": "infer",
            "inputs": [{"semantic": "features", "dtype": "float32", "shape": [1]}],
            "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1]}],
            "semantic_identity": {
                "logical_model_revision": f"{family}@v1",
                "preprocessing_contract": "test-pre-v1",
                "output_semantics": "test-output-v1",
            },
        },
        "deployments": {
            deployment: {
                "uuid": "123e4567-e89b-42d3-a456-426614174001",
                "revision": 1,
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {
                    "backend": "torch",
                    "target": {"runtime": "torch"},
                    "profile": {"device": "cpu"},
                },
            }
        },
    }
    (root / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _service(bundle: Path, endpoint: str, *, instance_id: str = "depth_front", required: bool = True) -> dict:
    return {
        "id": instance_id,
        "enabled": True,
        "required": required,
        "bundle_path": str(bundle),
        "deployment": "torch_cpu",
        "adapter_class": "example_depth.plugin:DepthServicePlugin",
        "service_type": "example_depth_msgs/srv/EstimateDepth",
        "endpoint": endpoint,
        "runtime_options": {"device_id": 0},
    }


def _config(bundle: Path) -> dict:
    return {
        "runtime_scheduler": {"enable": True},
        "perception_services": {"services": [_service(bundle, "/depth/front")]},
    }


@pytest.mark.parametrize(
    "resources,expected",
    [
        ({"cpu": {"physical_device": "local/cpu:0"}}, "canonical local device"),
        ({"cpu": {"physical_device": "local/cpu", "mode": "parallel"}}, "unsupported resource mode"),
        (
            {"cpu": {"physical_device": "local/cpu", "mode": "ascend_concurrent", "verified_concurrency": True}},
            "verified Ascend",
        ),
        ({"npu": {"physical_device": "local/npu:0", "mode": "ascend_concurrent"}}, "verified Ascend"),
        ({"cpu": {"physical_device": "local/cpu", "verified_concurrency": 1}}, "must be a boolean"),
        ({"cpu": {"physical_device": "local/cpu", "managed_coverage": 1}}, "must be a boolean"),
        (
            {"cpu": {"physical_device": "local/cpu"}, "alias": {"physical_device": "local/cpu"}},
            "multiple resource aliases",
        ),
        ({"bad id": {"physical_device": "local/cpu"}}, "invalid hardware resource ID"),
        ({"cpu": {"physical_device": "local/cpu", "predictive_admission": True}}, "unsupported fields"),
    ],
)
def test_public_resource_validation_does_not_require_participants(resources, expected):
    from robot_config.runtime_target_config import build_runtime_resources

    with pytest.raises(ValueError, match=expected):
        build_runtime_resources({"runtime_scheduler": {"enable": True, "resources": resources}})


@pytest.mark.parametrize("field", ["hardware_profile_fingerprint"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_predictive_resource_requires_complete_runtime_identity(field, value):
    from robot_config.runtime_target_config import build_runtime_resources

    resource = {
        "physical_device": "local/cpu",
        "managed_coverage": True,
        "hardware_profile_fingerprint": "runtime-hardware",
    }
    if value is None:
        resource.pop(field)
    else:
        resource[field] = value
    config = {
        "runtime_scheduler": {
            "enable": True,
            "resources": {"cpu": resource},
            "priority_zero_deadline_admission": {"enable": True},
        }
    }
    with pytest.raises(ValueError, match=rf"{field} must be nonempty"):
        build_runtime_resources(config)


@pytest.mark.parametrize("coverage", [None, False])
def test_predictive_resource_requires_central_coverage_declaration(coverage):
    from robot_config.runtime_target_config import build_runtime_resources

    resource = {
        "physical_device": "local/cpu",
        "hardware_profile_fingerprint": "runtime-hardware",
    }
    if coverage is not None:
        resource["managed_coverage"] = coverage
    config = {
        "runtime_scheduler": {
            "enable": True,
            "resources": {"cpu": resource},
            "priority_zero_deadline_admission": {"enable": True},
        }
    }
    with pytest.raises(ValueError, match="explicit managed_coverage=true"):
        build_runtime_resources(config)


@pytest.mark.parametrize(
    "field,value", [("max_pending_requests", 0), ("default_priority", True), ("retry_window_ms", 300001)]
)
def test_runtime_member_bounds_fail_before_launch(tmp_path, field, value):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _config(_bundle(tmp_path / "depth", "test_depth"))
    config["runtime_scheduler"] = {
        "enable": True,
        "resources": {"cpu": {"physical_device": "local/cpu"}},
        "managed_perception": {"depth_front": {"hardware_resource_id": "cpu", field: value}},
    }
    with pytest.raises(ValueError):
        build_runtime_target_directory(config)


def test_runtime_resource_mapping_and_prediction_coverage(tmp_path):
    from robot_config.runtime_target_config import build_runtime_target_directory

    bundle = _bundle(tmp_path / "depth", "test_depth")
    config = _config(bundle)
    config["perception_services"]["services"].append(_service(bundle, "/depth/other", instance_id="other"))
    config["runtime_scheduler"] = {
        "enable": True,
        "resources": {"cpu": {"physical_device": "local/cpu", "mode": "serial"}},
        "managed_perception": {"depth_front": {"hardware_resource_id": "cpu"}},
    }
    assert all(value for value in _coverage(config))
    config["runtime_scheduler"]["resources"]["cpu"]["physical_device"] = "local/cuda:0"
    with pytest.raises(ValueError, match="physical device disagrees"):
        build_runtime_target_directory(config)
    config["runtime_scheduler"]["resources"]["cpu"]["physical_device"] = "local/cpu"
    config["runtime_scheduler"]["priority_zero_deadline_admission"] = {"enable": True}
    for resource in config["runtime_scheduler"]["resources"].values():
        resource["managed_coverage"] = True
        resource.setdefault("hardware_profile_fingerprint", "runtime-hardware")
    config["runtime_scheduler"]["managed_perception"]["depth_front"].update(
        profile_path=str(tmp_path / "runtime-profile.json"), runtime_version="runtime-v1"
    )
    (tmp_path / "runtime-profile.json").write_text("{}")
    with pytest.raises(ValueError, match="runtime_execution profiles for every target"):
        build_runtime_target_directory(config)


@pytest.mark.parametrize("stateful", [True, False])
def test_ascend_concurrent_rejects_state_bank_execution_contract(tmp_path, stateful):
    from robot_config.runtime_target_config import build_runtime_target_directory

    bundle = _bundle(tmp_path / "stream", "test_depth")
    manifest_path = bundle / "inference_manifest.json"
    data = json.loads(manifest_path.read_text())
    data["deployments"]["torch_cpu"]["runtime_profile"] = {
        "backend": "ascend",
        "target": {"runtime": "acl"},
        "profile": {"device_id": 0},
    }
    data["deployments"]["torch_cpu"]["execution_contract"] = {
        "state_scope": "stream",
        "execution_structure": "direct",
        "cancellation_granularity": "checkpoint",
        "stateful": stateful,
        "state_bank_mode": "runtime_exclusive",
        "max_open_streams": 1,
        "state_links": [
            {
                "role": "__runtime__",
                "state_name": "aggregate.state",
                "owner": "streaming_runtime",
                "source": "aggregate.input",
                "target": "aggregate.output",
                "scope": "stream",
                "state_bank": "aggregate.bank",
            }
        ],
    }
    _write_compiled_manifest(manifest_path, data)
    config = _config(bundle)
    config["runtime_scheduler"] = {
        "enable": True,
        "resources": {
            "npu": {
                "physical_device": "local/npu:0",
                "mode": "ascend_concurrent",
                "verified_concurrency": True,
            }
        },
        "managed_perception": {"depth_front": {"hardware_resource_id": "npu"}},
    }
    with pytest.raises(ValueError, match="ascend_concurrent.*stateful.*state_links"):
        build_runtime_target_directory(config)
    config["runtime_scheduler"]["resources"]["npu"]["mode"] = "serial"
    assert build_runtime_target_directory(config)[0]["resource_mode"] == "serial"
    data["deployments"]["torch_cpu"]["execution_contract"] = {
        "state_scope": "request",
        "execution_structure": "direct",
        "cancellation_granularity": "request_boundary",
    }
    _write_compiled_manifest(manifest_path, data)
    config["runtime_scheduler"]["resources"]["npu"]["mode"] = "ascend_concurrent"
    assert build_runtime_target_directory(config)[0]["resource_mode"] == "ascend_concurrent"
    config["runtime_scheduler"]["managed_perception"]["depth_front"]["default_priority"] = 8
    with pytest.raises(ValueError, match="Ascend range 0..7"):
        build_runtime_target_directory(config)


def test_robot_scheduler_member_typo():
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = {"runtime_scheduler": {"enable": True, "managed_perception": {"sam": {"hardware_resorce_id": "cpu"}}}}
    with pytest.raises(ValueError, match=r"managed_perception\.sam.*hardware_resorce_id"):
        build_runtime_target_directory(config)


def _managed_config(tmp_path):
    config = _config(_bundle(tmp_path / "depth", "test_depth"))
    config["runtime_scheduler"] = {
        "enable": True,
        "resources": {"cpu": {"physical_device": "local/cpu"}},
        "managed_perception": {
            "depth_front": {
                "hardware_resource_id": "cpu",
                "profile_path": str(tmp_path / "profile.json"),
                "runtime_version": "runtime-v1",
            }
        },
    }
    (tmp_path / "profile.json").write_text("{}")
    return config


@pytest.mark.parametrize("endpoint", ["/depth/front", "/inference/runtime/depth_front"])
def test_global_runtime_endpoint_cannot_alias_other_services(tmp_path, endpoint):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _config(_bundle(tmp_path / "depth", "test_depth"))
    original = build_runtime_target_directory(config)[0]
    # Exercise the actual resolved host endpoint as well as the business one.
    if endpoint == "/inference/runtime/depth_front":
        endpoint = original["endpoint"]
    config["runtime_scheduler"] = {"enable": True, "runtime_endpoint": endpoint}
    with pytest.raises(ValueError, match="endpoint"):
        build_runtime_target_directory(config)


@pytest.mark.parametrize("second_version", ["onnx-1.20", "", None])
def test_shared_device_predictive_targets_have_independent_backend_versions(tmp_path, second_version):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _managed_config(tmp_path)
    bundle = _bundle(tmp_path / "onnx-peer", "other")
    manifest_path = bundle / "inference_manifest.json"
    document = json.loads(manifest_path.read_text())
    document["deployments"]["torch_cpu"]["runtime_profile"] = {
        "backend": "onnx",
        "target": {"runtime": "onnxruntime"},
        "profile": {"provider": "cpu"},
    }
    _write_compiled_manifest(manifest_path, document)
    config["perception_services"]["services"].append(_service(bundle, "/other", instance_id="other"))
    block = config["runtime_scheduler"]
    block["priority_zero_deadline_admission"] = {"enable": True}
    block["resources"]["cpu"].update(managed_coverage=True, hardware_profile_fingerprint="cpu-hardware")
    (tmp_path / "torch.json").write_text("{}")
    (tmp_path / "onnx.json").write_text("{}")
    block["managed_perception"]["depth_front"].update(
        profile_path=str(tmp_path / "torch.json"), runtime_version="torch-2.7"
    )
    block["managed_perception"]["other"] = {
        "profile_path": str(tmp_path / "onnx.json"),
        "runtime_version": second_version,
    }
    if not second_version:
        with pytest.raises(ValueError, match="runtime_version must be"):
            build_runtime_target_directory(config)
        return
    targets = build_runtime_target_directory(config)
    assert {target["runtime_version"] for target in targets} == {"torch-2.7", "onnx-1.20"}
    assert {target["hardware_resource_id"] for target in targets} == {"cpu"}
    assert all(target["predictive_admission"] for target in targets)


def _write_compiled_manifest(path, document):
    deployment = document["deployments"]["torch_cpu"]
    backend = deployment["runtime_profile"]["backend"]
    extension = {"ascend": "om", "onnx": "onnx"}[backend]
    artifact = path.parent / f"artifacts/model.{extension}"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_bytes(b"configuration-only compiled fixture")
    deployment["artifacts"] = {
        "model": {
            "path": f"artifacts/model.{extension}",
            "format": extension,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    }
    path.write_text(json.dumps(document))


def test_policy_compiler_keeps_legacy_and_runtime_profiles_separate(tmp_path):
    from robot_config.inference_config import parse_inference_config
    from robot_config.runtime_target_config import build_runtime_target_directory

    from .test_inference_scheduler_config import (
        _create_bundle,
        _pipeline,
        _profile_file,
        _robot_config,
        _scheduler_block,
    )

    pipeline = _pipeline(_create_bundle(tmp_path / "policy"), profile=_profile_file(tmp_path))
    (tmp_path / "runtime-v2.json").write_text("{}")
    pipeline.update(runtime_profile_path=str(tmp_path / "runtime-v2.json"), runtime_version="test-v1")
    block = _scheduler_block()
    block["resources"] = {"ascend:0": {"physical_device": "local/cpu"}}
    config = _robot_config({"policy": pipeline}, scheduler=block)
    parsed = parse_inference_config(config, "model_inference")
    targets = build_runtime_target_directory(config, control_mode="model_inference")
    assert len(targets) == 1 and targets[0]["target_id"] == "policy/policy"
    assert targets[0]["profile_path"] == str(tmp_path / "runtime-v2.json")
    assert parsed.pipelines["policy"].profile_path == Path(pipeline["profile_path"]).resolve()
    config["runtime_scheduler"]["enable"] = False
    assert build_runtime_target_directory(config, control_mode="model_inference") == ()


@pytest.mark.parametrize("devices", [("local/cpu", "local/cuda:0"), ("local/cuda:0", "local/cpu")])
def test_unmanaged_participant_disables_coverage_on_every_resource(devices):
    config = {
        "runtime_scheduler": {
            "enable": True,
            "resources": {
                str(index): {"physical_device": device, "managed_coverage": True}
                for index, device in enumerate(devices)
            },
        },
        "voice_asr": {"enabled": True},
    }
    resources = build_runtime_resources(config, ())
    assert {value["physical_device"]: value["managed_coverage"] for value in resources.values()} == {
        "local/cpu": False,
        "local/cuda:0": False,
    }


def test_runtime_readiness_endpoint_is_reserved(tmp_path):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _config(_bundle(tmp_path / "depth", "test_depth"))
    config["perception_services"]["services"][0]["endpoint"] = "/inference/runtime/v1/ready"
    with pytest.raises(ValueError, match="endpoint conflict"):
        build_runtime_target_directory(config)


@pytest.mark.parametrize("selection", ["depth_front", ["depth_fron"], [None], [""]])
def test_explicit_perception_selection_rejects_invalid_ids(tmp_path, selection):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _config(_bundle(tmp_path / "depth", "test_depth"))
    with pytest.raises(ValueError, match="instance_ids"):
        build_runtime_target_directory(config, instance_ids=selection)


@pytest.mark.parametrize("profile", ["profiles/measured.json", "~/profiles/measured.json"])
def test_perception_profile_uses_workspace_and_user_path_resolution(tmp_path, monkeypatch, profile):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _managed_config(tmp_path)
    measured = tmp_path / "profiles" / "measured.json"
    measured.parent.mkdir()
    measured.write_text("{}")
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    if profile.startswith("~/"):
        profile = str(Path("~") / os.path.relpath(measured, Path.home()))
    monkeypatch.chdir(tmp_path.parent)
    config["runtime_scheduler"]["managed_perception"]["depth_front"]["profile_path"] = profile
    assert build_runtime_target_directory(config)[0]["profile_path"] == str(measured.resolve())


@pytest.mark.parametrize("profile", [123, "missing.json", ""])
def test_perception_profile_rejects_invalid_paths_before_launch(tmp_path, profile):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _managed_config(tmp_path)
    config["runtime_scheduler"]["managed_perception"]["depth_front"]["profile_path"] = profile
    with pytest.raises(ValueError, match="profile_path"):
        build_runtime_target_directory(config)


@pytest.mark.parametrize("accelerator", ["cuda", "npu"])
def test_managed_torch_explicit_index_reaches_compiled_target(tmp_path, monkeypatch, accelerator):
    from robot_config.runtime_target_config import build_runtime_target_directory

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    bundle = _bundle(tmp_path / accelerator, "test_depth")
    path = bundle / "inference_manifest.json"
    document = json.loads(path.read_text())
    document["deployments"]["torch_cpu"]["runtime_profile"]["profile"]["device"] = accelerator
    path.write_text(json.dumps(document))
    config = _config(bundle)
    config["perception_services"]["services"][0]["runtime_options"] = {"torch_device_index": 2}
    targets = build_runtime_target_directory(config)
    assert targets[0]["physical_device"] == f"local/{accelerator}:2"


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_version", "v" * 257),
        ("hardware_profile_fingerprint", "h" * 257),
        ("profile_observation_inputs", [str(i) + "图" * 80 for i in range(64)]),
    ],
)
def test_compiler_rejects_values_outside_control_descriptor_budget(tmp_path, field, value):
    from robot_config.runtime_target_config import build_runtime_target_directory

    config = _config(_bundle(tmp_path / "bounded", "test_depth"))
    config["runtime_scheduler"] = {
        "enable": True,
        "resources": {"cpu": {"physical_device": "local/cpu"}},
        "managed_perception": {"depth_front": {"hardware_resource_id": "cpu"}},
    }
    if field == "hardware_profile_fingerprint":
        config["runtime_scheduler"]["resources"]["cpu"][field] = value
    else:
        config["runtime_scheduler"]["managed_perception"]["depth_front"][field] = value
    with pytest.raises(ValueError, match="runtime_version|hardware_profile_fingerprint|descriptor budget"):
        build_runtime_target_directory(config)


def test_positive_submillisecond_policy_timeout_rounds_up(tmp_path):
    from robot_config.runtime_target_config import build_runtime_target_directory

    from .test_inference_scheduler_config import (
        _create_bundle,
        _pipeline,
        _profile_file,
        _robot_config,
        _scheduler_block,
    )

    pipeline = _pipeline(_create_bundle(tmp_path / "policy"), profile=_profile_file(tmp_path))
    pipeline["request_timeout"] = 0.0005
    block = _scheduler_block()
    block["resources"] = {"ascend:0": {"physical_device": "local/cpu"}}
    config = _robot_config({"policy": pipeline}, scheduler=block)
    assert build_runtime_target_directory(config, control_mode="model_inference")[0]["request_timeout_ms"] == 1


@pytest.mark.parametrize("use_sim", [True, False])
@pytest.mark.parametrize("runtime_target", [None, "hardware", "simulation"])
def test_raw_policy_and_public_runtime_switch_agree_on_target_resolution(use_sim, runtime_target):
    from robot_config.inference_config import scheduler_enabled_from_raw_config
    from robot_config.runtime_scheduling import scheduling_available

    config = {
        "runtime_scheduler": {"enable": True},
        "use_sim": use_sim,
        "control_modes": {"model_inference": {"inference": {"enabled": True}}},
    }
    if runtime_target is not None:
        config["runtime"] = {"target": runtime_target}
    assert scheduler_enabled_from_raw_config(config, "model_inference") is scheduling_available(config)
    assert scheduling_available(config) is (runtime_target != "simulation")


@pytest.mark.parametrize("value", [True, False, 0, -1, "1", float("inf"), float("nan")])
def test_perception_connection_rejects_invalid_shared_startup_timeout(value):
    from robot_config.launch_builders.model_runtime import model_runtime_connection_parameters

    config = {"runtime_scheduler": {"enable": True, "startup_readiness_timeout": value}}
    with pytest.raises(ValueError, match="startup_readiness_timeout"):
        model_runtime_connection_parameters(config)


@pytest.mark.parametrize("limits", [{"max_records": 7}, {"retry_window_ms": 300001}, {"max_pending_requests": True}])
def test_config_and_target_reject_the_same_capacity_limits(tmp_path, limits):
    from inference_manifest.runtime_target import RuntimeExecutionTarget
    from robot_config.runtime_target_config import build_runtime_target_directory, validate_runtime_limits

    config = _config(_bundle(tmp_path / "depth", "depth"))
    target = build_runtime_target_directory(config)[0]
    with pytest.raises(ValueError):
        RuntimeExecutionTarget(**{**target, **limits})
    with pytest.raises(ValueError, match="example_path"):
        validate_runtime_limits(limits, "example_path")
