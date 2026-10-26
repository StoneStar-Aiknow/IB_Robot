"""Launch arbitrary typed model-service plugins independently.

This module is ``inference_service``'s contribution to the launch-builder
registry (openspec change ``migrate-business-launch-builders``): the
``perception_models`` entry is declared as an entry point in the
``ibrobot.launch_builders`` group (see ``setup.py``), so the model-service
nodes' owning package registers its own launch wiring without any
robot_config change. The content is a verbatim migration of the former
``robot_config/launch_builders/perception_models.py``.
"""

import json
import os
from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from robot_config.launch_registry.launch_options import ArgumentSpec
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
from robot_config.perception_runtime_config import parse_perception_runtime_config
from robot_config.utils import parse_bool

# The operator location is owned by torch_models.graspgenx_ascend_310p
# .custom_ops, which cannot be imported here: its package imports torch_npu.
# Rather than keep a second copy of the path that has to be edited in step with
# it, this forwards the variable that module already reads.
_GRASPGENX_OPS_VARIABLE = "IBROBOT_GRASPGENX_ASCEND_OPS"
_GRASPGENX_OPS_VENDORS = ("graspgenx_subm", "graspgenx_grid")


def _graspgenx_npu_environment(service) -> dict[str, str]:
    """Operator paths for a GraspGen-X deployment on an Ascend device.

    Its PTV3 encoder calls custom Ascend C operators that setup builds on the
    host. CANN resolves their packages through ASCEND_CUSTOM_OPP_PATH, which it
    reads once at initialisation: a process cannot set it for itself, so the
    launch has to carry it. Deployments without that need get nothing extra.
    """
    validated = getattr(service, "validated_manifest", None)
    if validated is None or validated.manifest.model.model_type != "graspgenx":
        return {}
    if getattr(validated.deployment, "device", None) != "npu":
        return {}
    configured = os.environ.get(_GRASPGENX_OPS_VARIABLE)
    if not configured:
        # Nothing to carry: the operators were never pointed at. The loader in
        # the served process reports that with the command that builds them.
        return {}
    root = Path(configured).expanduser()
    vendors = [root / "vendors" / name for name in _GRASPGENX_OPS_VENDORS]
    present = [str(path) for path in vendors if path.is_dir()]
    if not present:
        # Leave the environment alone rather than point at nothing: the loader
        # reports the missing operators with the command that builds them.
        return {}
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    return {
        _GRASPGENX_OPS_VARIABLE: str(root),
        "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(present + ([existing] if existing else [])),
    }


def generate_perception_model_nodes(
    robot_config: dict[str, Any],
    *,
    instance_ids: set[str] | None = None,
    configuration_generation: int = 0,
    require_semantic_identity: bool = False,
) -> list[Node]:
    runtime = parse_perception_runtime_config(robot_config)
    mapping = robot_config.get("semantic_mapping", {})
    excluded_ids = (
        set(mapping.get("perception", {}).get("semantic_roles", {}).values())
        if mapping.get("query_only", False)
        else set()
    )
    return [
        Node(
            package="inference_service",
            executable="model_service_node",
            name=service.node_name,
            output="screen",
            parameters=[
                {
                    "instance_id": service.instance_id,
                    "bundle_path": str(service.bundle_path),
                    "deployment": service.deployment,
                    "adapter_class": service.adapter_class,
                    "service_type": service.service_type,
                    "service_endpoint": service.endpoint,
                    "required": service.required,
                    "configuration_generation": configuration_generation,
                    "require_semantic_identity": require_semantic_identity,
                    "runtime_options_json": json.dumps(dict(service.runtime_options), sort_keys=True),
                }
            ],
            additional_env=_graspgenx_npu_environment(service) or None,
        )
        for service in runtime.enabled_services
        if service.instance_id not in excluded_ids and (instance_ids is None or service.instance_id in instance_ids)
    ]


__all__ = ["generate_perception_model_nodes"]


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration. Openspec change
# ``migrate-business-launch-builders`` moved this module from
# ``robot_config/launch_builders/perception_models.py`` to its owning package.
# ---------------------------------------------------------------------------


# A robot whose models are served from another host (the pipeline next to the
# camera, the models on a GPU machine) keeps its perception_services as they
# are: the endpoints its clients call stay configured, only the local hosts of
# those services are not started. Config loading then validates each enabled
# service's identity, endpoints and semantic contract from the bundle's manifest
# metadata, so this host needs the bundle directory with its
# inference_manifest.json but not the weights or deployment artifacts.
_PERCEPTION_MODELS_ARGUMENTS = (
    ArgumentSpec(
        name="with_model_services",
        declared_default="true",
        absent_fallback="true",
        description="Start the configured perception model services on this host. "
        "false when another host serves them; the configured endpoints are still called, "
        "and only each bundle's manifest metadata has to be present here.",
    ),
)


def model_services_requested(context: LaunchBuildContext) -> bool:
    return parse_bool(context.arguments.get("with_model_services", "true"), default=True)


def build_perception_models(context: LaunchBuildContext) -> BuildResult:
    from inference_service.launch_builders.perception_models import generate_perception_model_nodes

    return BuildResult(actions=list(generate_perception_model_nodes(context.robot_config)))


PERCEPTION_MODELS_BUILDER = BuilderSpec(
    id="perception_models",
    factory=f"{__name__}:build_perception_models",
    timing="runtime_ready",
    arguments=_PERCEPTION_MODELS_ARGUMENTS,
    enabled=model_services_requested,
)
