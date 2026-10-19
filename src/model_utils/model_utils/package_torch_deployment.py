"""Generate native Torch deployments for a LeRobot policy bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from inference_manifest import (
    DeploymentTarget,
    ExecutionContract,
    RoleRuntimeProfile,
    TorchDeployment,
    TorchRuntimeProfile,
    ValidatedManifest,
)
from model_utils.inference_manifest_export import upsert_deployment

_TORCH_DEVICES = ("cpu", "cuda", "mps", "npu")


def package_torch_deployments(
    bundle_root: str | Path,
    *,
    devices: Sequence[str] = ("cpu", "cuda"),
    deployment_prefix: str = "torch",
    deployment_name: str | None = None,
    extra_parameters: Mapping[str, object] | None = None,
) -> tuple[ValidatedManifest, ...]:
    """Add one named native Torch deployment for each requested device."""

    selected_devices = tuple(devices)
    if not selected_devices:
        raise ValueError("at least one Torch device is required")
    if len(selected_devices) != len(set(selected_devices)):
        raise ValueError("Torch devices must not contain duplicates")
    unsupported = sorted(set(selected_devices) - set(_TORCH_DEVICES))
    if unsupported:
        raise ValueError(f"unsupported Torch devices: {unsupported}")
    if not deployment_prefix:
        raise ValueError("deployment prefix must not be empty")
    if deployment_name is not None and (not deployment_name or len(selected_devices) != 1):
        raise ValueError("deployment_name requires one device and a non-empty name")

    return tuple(
        upsert_deployment(
            bundle_root,
            deployment_name if deployment_name is not None else f"{deployment_prefix}-{device}",
            TorchDeployment(
                extra_parameters={} if extra_parameters is None else extra_parameters,
                execution_contract=ExecutionContract(
                    state_scope="request",
                    execution_structure="direct",
                    cancellation_granularity="request_boundary",
                ),
                runtime_profile=RoleRuntimeProfile(
                    backend="torch",
                    target=DeploymentTarget(runtime="torch"),
                    profile=TorchRuntimeProfile(device=device),
                ),
            ),
        )
        for device in selected_devices
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, help="LeRobot save_pretrained policy directory")
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=_TORCH_DEVICES,
        default=("cpu", "cuda"),
        help="Torch devices to register (default: cpu cuda)",
    )
    parser.add_argument(
        "--deployment-prefix",
        default="torch",
        help="Deployment name prefix (default: torch)",
    )
    parser.add_argument("--deployment-name", help="Exact deployment name; requires a single --devices value")
    parser.add_argument(
        "--extra-parameters-json",
        default="{}",
        help="Inline JSON object owned by the model provider; no parameter file or checksum required",
    )
    args = parser.parse_args()

    try:
        parameters = json.loads(args.extra_parameters_json)
    except ValueError as exc:
        parser.error(f"invalid --extra-parameters-json: {exc}")
    if not isinstance(parameters, dict):
        parser.error("--extra-parameters-json must be a JSON object")

    validated = package_torch_deployments(
        args.bundle_root,
        devices=args.devices,
        deployment_prefix=args.deployment_prefix,
        deployment_name=args.deployment_name,
        extra_parameters=parameters,
    )
    print(validated[-1].manifest_path)
    for item in validated:
        print(f"{item.deployment_name}: {item.deployment.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
