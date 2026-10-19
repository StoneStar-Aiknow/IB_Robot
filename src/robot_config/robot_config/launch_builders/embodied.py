"""Embodied launch-builder entry and CLI config-override hook.

Owns the legacy embodied warning (robot_config no longer launches embodied
runtime nodes directly) and the ``with_embodied`` / ``with_perception`` CLI
config overrides. Both moved here from the retired ``adapters.py`` (openspec
change ``decentralize-launch-builder-registration``, design decision 1); the
bodies are verbatim migrations. A lightweight module: no business run
dependencies, so hooks run even when the entry never participates.
"""

from robot_config.launch_builders.launch_options import ArgumentSpec
from robot_config.launch_builders.registry import (
    BuilderSpec,
    BuildResult,
    HookSpec,
    LaunchBuildContext,
)
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.launch_builders.embodied")

# Business launch arguments owned by this entry (migrated verbatim from the
# former builtin_registry table; ownership decides who declares an argument).
_EMBODIED_ARGUMENTS = (
    ArgumentSpec(
        name="with_embodied",
        declared_default="",
        absent_fallback="",
        description="Enable the embodied minimal closure. If empty, uses robot.embodied.enabled from YAML.",
    ),
    ArgumentSpec(
        name="with_perception",
        declared_default="",
        absent_fallback="",
        description="Enable perception_service. If empty, uses robot.embodied.perception.enabled from YAML.",
    ),
)


def embodied_config_overrides_hook(robot_config, arguments) -> None:
    """config_overrides hook: with_embodied / with_perception CLI overrides.

    Migrated verbatim from the preparation core's
    ``_apply_embodied_overrides`` step (openspec change
    ``migrate-builtin-builder-metadata``, design decision 4). A lightweight
    hook living next to its entry (no business run dependencies).
    Ownership: ``embodied``.
    """
    with_embodied_str = arguments["with_embodied"]
    if with_embodied_str != "":
        embodied_cfg = robot_config.setdefault("embodied", {})
        embodied_cfg["enabled"] = parse_bool(with_embodied_str, default=False)
        logger.info(f"CLI override: embodied.enabled={embodied_cfg['enabled']}")
    with_perception_str = arguments["with_perception"]
    if with_perception_str != "":
        embodied_cfg = robot_config.setdefault("embodied", {})
        perception_cfg = embodied_cfg.setdefault("perception", {})
        perception_cfg["enabled"] = parse_bool(with_perception_str, default=False)
        logger.info(f"CLI override: embodied.perception.enabled={perception_cfg['enabled']}")


def build_embodied(context: LaunchBuildContext) -> BuildResult:
    """Legacy embodied warning only; no embodied runtime nodes here."""
    if context.robot_config.get("embodied", {}).get("enabled", False):
        logger.warning(
            "robot_config no longer launches embodied runtime nodes directly. "
            "Use the dedicated embodied bringup launch entry to start "
            "the base robot launch plus embodied runtime nodes."
        )
    return BuildResult()


EMBODIED_BUILDER = BuilderSpec(
    id="embodied",
    factory=f"{__name__}:build_embodied",
    arguments=_EMBODIED_ARGUMENTS,
    hooks=(
        # config_overrides: with_embodied / with_perception CLI
        # overrides (the legacy embodied phase, one owner).
        HookSpec(
            phase="config_overrides",
            factory=f"{__name__}:embodied_config_overrides_hook",
            config_keys=("embodied",),
        ),
    ),
)
