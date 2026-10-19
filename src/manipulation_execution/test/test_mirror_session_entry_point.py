"""End-to-end entry point test: mirror_session joins the launch through registration.

Openspec change ``decentralize-launch-builder-registration`` (design decision
4): with ``manipulation_execution`` installed, the ``aimdk_x2_magicmirror``
configuration must reach the mirror session WITHOUT any robot_config change —
the entry point group ``ibrobot.launch_builders`` is discovered for real by
``importlib.metadata`` (not injected), the entry is appended after every
builtin entry, and the frozen plan for the magic-mirror robot ends with
``mirror_session``, matching the pre-migration f0aded6e baseline.

The full action-graph equivalence against the baseline is verified separately
by the Phase 2 equivalence script (task 2.9); this test pins the structure
that matters for the migration: discovery, plan order, and the deferred
provider wrapper that carries the plan.

Requires the two perception model bundles referenced by the configuration
(``yolox_x_640`` / ``pear_parameter_network``); without them the test skips
(the root conftest's ``model_bundle`` marker), mirroring the other suites
that drive real robot configs.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from types import SimpleNamespace

import pytest
from launch_ros.actions import Node

from robot_config.launch_builders.builtin_registry import (
    LAUNCH_BUILDER_ENTRY_POINT_GROUP,
    build_builtin_registry,
    build_launch_registry,
)

# The aimdk_x2_magicmirror plan after openspec change
# ``migrate-business-launch-builders`` (Phase B): the retained builtin
# entries keep their relative order, and every entry that moved to an
# owning package (voice_asr, voice_tts, speech_direction,
# interaction_demo, mirror_session, motion_imitation_stream,
# perception_models, recording, semantic_mapping) participates from the
# extension block, appended sorted by entry point name — the plan-order
# change that design decision 7 of that change accepts. The plan SET is
# unchanged from the pre-migration baseline (pinned by the change's
# baseline equivalence run); only the block placement moved.
# ``migrate-hand-sources-to-robot-teleop`` later moved hand_sources from
# the builtin segment to the extension block (its accepted plan-order
# change); this list was not updated then and is synced here.
_BASELINE_PLAN = [
    "control",
    "auxiliary_actuators",
    "simulation",
    "controller_startup",
    "perception",
    "audio_io",
    "motion_planning",
    "embodied",
    "hand_sources",
    "interaction_demo",
    "mirror_session",
    "motion_imitation_stream",
    "perception_models",
    "recording",
    "semantic_mapping",
    "speech_direction",
    "voice_asr",
    "voice_tts",
]

# The f0aded6e baseline's top-level action summary for this configuration:
# the live-binding provider wrapper (aimdk_robot) that defers the frozen plan
# behind the runtime's readiness. The mirror session's own node is built
# inside the deferred consumer once the runtime binds the live interface
# description, so it is not part of this list by design.
_BASELINE_ACTION_SUMMARY = [
    "RegisterEventHandler",
    "RegisterEventHandler",
    "IncludeLaunchDescription",
    "Node: robot_runtime/wait_for_runtime name=wait_for_runtime",
]


def _action_summary(actions):
    lines = []
    for action in actions:
        kind = type(action).__name__
        package = getattr(action, "_Node__package", None)
        if package is not None:
            executable = getattr(action, "_Node__node_executable", None)
            name = getattr(action, "_Node__node_name", None)
            lines.append(f"{kind}: {package}/{executable} name={name}")
        else:
            lines.append(kind)
    return lines


def test_entry_point_group_declares_mirror_session():
    """The installed package metadata really carries the registration: the
    group contains exactly the mirror_session entry pointing at this
    package's builder constant (this is what build_launch_registry finds)."""
    discovered = {entry.name: entry.value for entry in entry_points(group=LAUNCH_BUILDER_ENTRY_POINT_GROUP)}
    assert (
        discovered.get("mirror_session")
        == "manipulation_execution.launch_builders.mirror_session:MIRROR_SESSION_BUILDER"
    )


def test_mirror_session_is_not_a_builtin_entry_anymore():
    """robot_config's builtin table no longer contains the mirror session —
    its only registration path is the entry point above."""
    assert "mirror_session" not in build_builtin_registry()


@pytest.mark.model_bundle("yolox_x_640")
@pytest.mark.model_bundle("pear_parameter_network")
def test_magicmirror_plan_reaches_mirror_session_through_the_entry_point():
    """The full production preparation path for aimdk_x2_magicmirror: real
    discovery (no injection) appends mirror_session after every builtin
    entry, and the frozen plan matches the pre-migration baseline."""
    registry = build_launch_registry()

    builtin_ids = [spec.id for spec in build_builtin_registry().snapshot()]
    registry_ids = [spec.id for spec in registry.snapshot()]
    assert registry_ids[: len(builtin_ids)] == builtin_ids
    # The extension block is sorted by entry point name and contains
    # mirror_session; its exact membership follows whatever packages are
    # installed, so only the ordering rule is pinned here.
    extension_ids = registry_ids[len(builtin_ids) :]
    assert extension_ids == sorted(extension_ids)
    assert "mirror_session" in extension_ids

    from robot_config.launch_builders.preparation import prepare_launch

    context = SimpleNamespace(
        launch_configurations={"robot_config": "aimdk_x2_magicmirror", "use_sim": "true"},
    )
    prepared = prepare_launch(context, registry)
    plan_ids = [spec.id for spec in prepared.plan.specs]
    assert plan_ids == _BASELINE_PLAN
    # The migration moved mirror_session from the plan tail into the
    # alphabetically-sorted extension block (design decision 7 accepts the
    # order change); it still participates for this config.
    assert "mirror_session" in plan_ids


@pytest.mark.model_bundle("yolox_x_640")
@pytest.mark.model_bundle("pear_parameter_network")
def test_magicmirror_actions_match_the_baseline_wrapper_structure():
    """build_prepared_launch returns the same provider wrapper structure as
    the f0aded6e baseline: the plan rides behind the aimdk runtime
    provider's readiness handler, with no extra or missing top-level
    actions caused by the registration move."""
    from robot_config.launch_builders.orchestrator import build_prepared_launch
    from robot_config.launch_builders.preparation import prepare_launch

    context = SimpleNamespace(
        launch_configurations={"robot_config": "aimdk_x2_magicmirror", "use_sim": "true"},
    )
    prepared = prepare_launch(context, build_launch_registry())
    actions = build_prepared_launch(prepared)

    assert _action_summary(actions) == _BASELINE_ACTION_SUMMARY
    # The wrapper's visible node is the runtime readiness gate (private
    # accessors: the public properties resolve only during launch execution).
    nodes = [action for action in actions if isinstance(action, Node)]
    assert [(n._Node__package, n._Node__node_executable, n._Node__node_name) for n in nodes] == [
        ("robot_runtime", "wait_for_runtime", "wait_for_runtime")
    ]
