"""Registry-contribution tests for inference_service's perception_models entry.

The entry is an extension contribution of this package (entry point group
``ibrobot.launch_builders``), so its participation and the ownership of its
``with_model_services`` argument are pinned here, through the real builtin
registry with this package's entry point injected (hermetic: no ambient
installed extensions).
"""

from __future__ import annotations

import json
import shutil
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from inference_service.launch_builders.perception_models import PERCEPTION_MODELS_BUILDER
from robot_config.launch_registry.builtin_registry import build_launch_registry
from robot_config.launch_registry.launch_options import compose_argument_specs, read_launch_arguments
from robot_config.launch_registry.preparation import prepare_launch
from robot_config.launch_registry.registry import LaunchBuildContext
from robot_config.runtime_target import RuntimeTarget

_PERCEPTION_MODELS_ENTRY_POINT = EntryPoint(
    name="perception_models",
    value="inference_service.launch_builders.perception_models:PERCEPTION_MODELS_BUILDER",
    group="ibrobot.launch_builders",
)


def _registry():
    return build_launch_registry(entry_points=[_PERCEPTION_MODELS_ENTRY_POINT])


def _plan_ids(launch_configurations: dict[str, str]) -> list[str]:
    registry = _registry()
    specs = compose_argument_specs(registry.contributed_arguments())
    context = LaunchBuildContext(
        robot_config={"name": "test_robot"},
        runtime_target=RuntimeTarget.HARDWARE,
        control_mode="moveit_planning",
        arguments=read_launch_arguments(launch_configurations, specs),
    )
    return [spec.id for spec in registry.plan(context).specs]


class TestParticipation:
    def test_model_services_start_unless_the_launch_says_otherwise(self):
        assert "perception_models" in _plan_ids({})
        assert "perception_models" in _plan_ids({"with_model_services": "true"})

    @pytest.mark.parametrize("value", ["false", "False", "0"])
    def test_a_host_whose_models_are_served_elsewhere_starts_none(self, value):
        assert "perception_models" not in _plan_ids({"with_model_services": value})


class TestArgumentOwnership:
    def test_with_model_services_is_owned_by_the_perception_models_entry(self):
        registry = _registry()
        contributed = {argument.name: spec.id for spec in registry.snapshot() for argument in spec.arguments}
        assert contributed["with_model_services"] == "perception_models"
        (argument,) = PERCEPTION_MODELS_BUILDER.arguments
        assert (argument.declared_default, argument.absent_fallback) == ("true", "true")


def _metadata_only_bundle(root: Path) -> Path:
    """A bundle a consumer host holds: the manifest, not the weights.

    ``_bundle`` here writes the whole manifest the way robot_config's
    perception tests do, then drops the one declared bundle file so only the
    metadata remains.
    """
    from inference_manifest import BundleFile, canonical_bundle_digest

    root.mkdir(parents=True)
    marker = root / "assets" / "identity.txt"
    marker.parent.mkdir()
    marker.write_text("metric_depth", encoding="utf-8")
    entry = BundleFile(path="assets/identity.txt")
    family = "metric_depth"
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
            "torch_cpu": {
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
    shutil.rmtree(root / "assets")
    return root


def _split_robot_config(tmp_path: Path) -> Path:
    """A consumer host's YAML: the service is configured, its artifacts are not here."""
    config_path = tmp_path / "split_consumer.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "split_consumer",
                    "type": "test",
                    "perception_services": {
                        "services": [
                            {
                                "id": "depth_front",
                                "enabled": True,
                                "required": True,
                                "bundle_path": str(_metadata_only_bundle(tmp_path / "depth")),
                                "deployment": "torch_cpu",
                                "adapter_class": "example_depth.plugin:DepthServicePlugin",
                                "service_type": "example_depth_msgs/srv/EstimateDepth",
                                "endpoint": "/depth/front",
                                "runtime_options": {},
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _launch_context(config_path: Path, **arguments) -> SimpleNamespace:
    return SimpleNamespace(
        launch_configurations={"config_path": str(config_path), **arguments},
    )


class TestSplitDeploymentLoading:
    """``with_model_services:=false`` must survive real config loading, not just planning.

    The switch would be useless if loading still demanded every enabled
    service's local weights first: a consumer host holds the bundle metadata
    only, and the launch has to get past ``_load_config`` before the plan that
    skips ``perception_models`` ever freezes.
    """

    def test_a_non_hosting_launch_loads_metadata_only_bundles(self, tmp_path: Path):
        prepared = prepare_launch(
            _launch_context(_split_robot_config(tmp_path), with_model_services="false"), _registry()
        )
        assert "perception_models" not in [spec.id for spec in prepared.plan.specs]

    def test_the_hosting_default_still_demands_local_artifacts(self, tmp_path: Path):
        with pytest.raises(ValueError, match="failed bundle validation"):
            prepare_launch(_launch_context(_split_robot_config(tmp_path)), _registry())
