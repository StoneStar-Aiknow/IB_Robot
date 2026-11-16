"""Robot execution purpose and resource alias selection."""

from .runtime_target import RuntimeTarget, resolve_runtime_target


def scheduling_available(config):
    from collections.abc import Mapping

    block = config.get("runtime_scheduler", {})
    if not isinstance(block, Mapping):
        raise ValueError("runtime_scheduler must be a mapping")
    enabled = block.get("enable", False)
    if type(enabled) is not bool:
        raise ValueError("runtime_scheduler.enable must be a boolean")
    # Only an explicit launch use_sim argument constrains target resolution.
    # Raw YAML readers use runtime.target, just like the policy config reader.
    return enabled and resolve_runtime_target(config, "", None) is RuntimeTarget.HARDWARE


def with_runtime_target(config, runtime_target=None, use_sim=None):
    """Carry the resolved launch target through every downstream config reader."""
    target = resolve_runtime_target(config, runtime_target.value if runtime_target else "", use_sim)
    return {
        **config,
        "runtime": {**config.get("runtime", {}), "target": target.value},
        "use_sim": target is not RuntimeTarget.HARDWARE,
    }


def resource_id_for_device(resources, device, explicit=None):
    if explicit is not None:
        if explicit not in resources:
            raise ValueError(f"runtime target references undefined resource {explicit!r}")
        if resources[explicit]["physical_device"] != device:
            raise ValueError(f"configured physical device disagrees with manifest {device}")
        return explicit
    return next((key for key, value in resources.items() if value["physical_device"] == device), device)
