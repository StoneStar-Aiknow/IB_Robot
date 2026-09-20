"""Publish PI05 Ascend310P fidelity/fast deployments over shared native weights."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from inference_manifest import ValidatedManifest, load_policy_metadata
from model_utils.package_torch_deployment import package_torch_deployments
from torch_models.pi05_ascend_310p.inference_config import deployment_parameters


def package_pi05_ascend_deployments(
    bundle_root: str | Path, *, modes: Sequence[str] = ("fidelity", "fast")
) -> tuple[ValidatedManifest, ...]:
    selected = tuple(modes)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("PI05 modes must be non-empty and unique")
    # Resolve all presets before writing so an invalid mode cannot partially publish.
    parameters = {mode: deployment_parameters(mode) for mode in selected}
    policy = load_policy_metadata(Path(bundle_root).expanduser().resolve(), require_native_weights=True)
    if policy.policy_type != "pi05":
        raise ValueError(f"PI05 Ascend310P presets require a pi05 bundle, got {policy.policy_type!r}")
    return tuple(
        package_torch_deployments(
            bundle_root,
            devices=("npu",),
            deployment_name=f"torch-npu-{mode}",
            extra_parameters=parameters[mode],
        )[0]
        for mode in selected
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--modes", nargs="+", choices=("fidelity", "fast"), default=("fidelity", "fast"))
    args = parser.parse_args()
    deployments = package_pi05_ascend_deployments(args.bundle_root, modes=args.modes)
    print(deployments[-1].manifest_path)
    for deployment in deployments:
        print(f"{deployment.deployment_name}: {deployment.deployment.extra_parameters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
