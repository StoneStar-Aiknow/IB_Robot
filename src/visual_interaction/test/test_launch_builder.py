"""Launch wiring: what the builder refuses, and what it passes to the node."""

from __future__ import annotations

import json
from importlib.metadata import entry_points

import pytest
import yaml
from launch_ros.actions import Node

from robot_config.interface_binding import InterfaceBindingError
from robot_config.launch_registry.registry import LaunchBuilderRegistry, resolve_builder_factory
from visual_interaction.launch_builders.visual_interaction import (
    VISUAL_INTERACTION_BUILDER,
    generate_visual_interaction_nodes,
)

CAMERA_ID = "camera.stereo_head_front_left.color_compressed"
CAMERA_ENDPOINT = "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed"

_DESCRIPTION = {
    "robot": {"id": "x2", "type": "x2_ultra_omnihand", "runtime_name": "aimdk", "runtime_version": "1"},
    "interfaces": {
        CAMERA_ID: {
            "kind": "topic",
            "direction": "publish",
            "message_type": "sensor_msgs/msg/CompressedImage",
            "endpoint": CAMERA_ENDPOINT,
            "qos": {"reliability": "reliable", "history": "keep_last", "depth": 5},
        }
    },
}


def node_parameters(node: Node) -> dict:
    """Decode a launch Node's parameters.

    Private accessors: the public properties only resolve after launch executes
    the action, the same pattern the mirror-session builder tests use.
    """

    def text(value):
        return "".join(item.text if hasattr(item, "text") else str(item) for item in value)

    def decode(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode(tuple(item)) for item in value]
        try:
            return yaml.safe_load(text(value))
        except yaml.YAMLError:
            return text(value)

    return {text(key): decode(value) for key, value in node._Node__parameters[0].items()}


def config(**overrides) -> dict:
    """A config that launches the feature, so a test can break one thing.

    The description sits where ``robot_config.interface_binding`` writes it:
    ``runtime.interface_description``. An earlier revision of this helper
    invented a top-level key, so the builder passed here and failed on every
    real config — a fixture has to mirror the framework, not the reader.
    """
    section = {
        "enabled": True,
        "frame_source": {"interface": CAMERA_ID},
        "vl": {"implementation": "fake"},
        "categories": ["toy", "sign", "hat", "badge"],
        "session": {"confirmation_count": 2, "min_gap_sec": 10.0},
    }
    section.update(overrides)
    return {
        "name": "visual_probe",
        "skill_required_control_mode": "observation",
        "control_modes": {"observation": {"inference": {"enabled": False}, "executor": {"enabled": False}}},
        "runtime": {"profile": "x2_ultra_omnihand", "interface_description": _DESCRIPTION},
        "visual_interaction": section,
    }


class TestSpecDeclaration:
    def test_the_spec_pins_its_id_factory_and_timing(self):
        assert VISUAL_INTERACTION_BUILDER.id == "visual_interaction"
        assert VISUAL_INTERACTION_BUILDER.factory == (
            "visual_interaction.launch_builders.visual_interaction:build_visual_interaction"
        )
        assert VISUAL_INTERACTION_BUILDER.timing == "runtime_ready"

    def test_it_declares_no_typed_output_and_no_registry_gate(self):
        """An extension may not declare provides; participation is the YAML's job."""
        assert VISUAL_INTERACTION_BUILDER.provides == ()
        assert VISUAL_INTERACTION_BUILDER.arguments == ()
        assert VISUAL_INTERACTION_BUILDER.hooks == ()
        assert VISUAL_INTERACTION_BUILDER.enabled is None
        assert VISUAL_INTERACTION_BUILDER.mock_skip_key is None

    def test_the_factory_string_stays_resolvable(self):
        assert callable(resolve_builder_factory(VISUAL_INTERACTION_BUILDER.factory))

    def test_the_spec_is_addable_to_a_registry(self):
        registry = LaunchBuilderRegistry()
        registry.add(VISUAL_INTERACTION_BUILDER)

        assert "visual_interaction" in registry


class TestEntryPoint:
    def test_the_group_declares_this_builder(self):
        """Installing the package registers it; the YAML still decides."""
        found = {entry.name: entry.value for entry in entry_points(group="ibrobot.launch_builders")}

        assert found.get("visual_interaction") == (
            "visual_interaction.launch_builders.visual_interaction:VISUAL_INTERACTION_BUILDER"
        )

    def test_it_is_not_a_builtin_entry(self):
        from robot_config.launch_registry.builtin_registry import build_builtin_registry

        assert "visual_interaction" not in build_builtin_registry()


class TestParticipation:
    def test_an_absent_section_launches_nothing(self):
        assert generate_visual_interaction_nodes({"name": "bare"}) == []

    def test_a_disabled_section_launches_nothing(self):
        assert generate_visual_interaction_nodes(config(enabled=False)) == []

    def test_a_missing_enabled_key_defaults_to_off(self):
        """Installing the package must not switch a deployment on by itself."""
        without_enabled = config()
        del without_enabled["visual_interaction"]["enabled"]

        assert generate_visual_interaction_nodes(without_enabled) == []

    def test_enabled_must_be_a_boolean(self):
        with pytest.raises(ValueError, match="must be a boolean"):
            generate_visual_interaction_nodes(config(enabled="yes"))

    def test_enabling_it_launches_exactly_one_node(self):
        actions = generate_visual_interaction_nodes(config())

        assert len(actions) == 1
        assert isinstance(actions[0], Node)

    def test_there_is_no_release_node_and_no_exit_handler(self):
        """Nothing takes over a runtime mode, so there is nothing to hand back."""
        actions = generate_visual_interaction_nodes(config())

        assert [type(action).__name__ for action in actions] == ["Node"]


class TestNodeIdentity:
    def test_it_launches_this_package_executable(self):
        node = generate_visual_interaction_nodes(config())[0]

        assert node._Node__package == "visual_interaction"
        assert node._Node__node_executable == "visual_interaction_node"
        assert node._Node__node_name == "visual_interaction"


class TestParameters:
    def parameters(self, **overrides) -> dict:
        return node_parameters(generate_visual_interaction_nodes(config(**overrides))[0])

    def test_the_bound_description_is_passed_whole(self):
        """So the node resolves the camera by ID rather than a hard-coded name."""
        passed = json.loads(self.parameters()["runtime_description_json"])

        assert passed["interfaces"][CAMERA_ID]["endpoint"] == CAMERA_ENDPOINT

    def test_the_camera_interface_id_is_passed_not_its_endpoint(self):
        assert self.parameters()["camera_interface"] == CAMERA_ID

    def test_the_default_camera_is_used_when_none_is_named(self):
        assert self.parameters(frame_source={})["camera_interface"] == CAMERA_ID

    def test_session_timings_reach_the_node(self):
        values = self.parameters(
            session={
                "confirmation_count": 3,
                "max_evidence_age_sec": 4.0,
                "min_gap_sec": 2.0,
                "absent_reset_sec": 15.0,
                "max_session_sec": 90.0,
            }
        )

        assert values["confirmation_count"] == 3
        assert values["max_evidence_age_sec"] == 4.0
        assert values["min_gap_sec"] == 2.0
        assert values["absent_reset_sec"] == 15.0
        assert values["max_session_sec"] == 90.0

    def test_internvl_reaches_the_node_with_its_deployment_path(self):
        """The deployment is on the board, so its path has to travel with it."""
        values = self.parameters(vl={"implementation": "internvl", "demo_root": "/root/vlx_internvl_310p"})

        assert values["vl_implementation"] == "internvl"
        assert values["vl_demo_root"] == "/root/vlx_internvl_310p"

    def test_the_category_whitelist_reaches_the_node(self):
        """As JSON: an empty ROS string array has no inferable element type,
        so rclpy declared BYTE_ARRAY and the node refused the builder's
        strings at startup."""
        assert json.loads(self.parameters(categories=["toy"])["categories_json"]) == ["toy"]

    def test_an_actor_topic_is_optional(self):
        assert self.parameters()["actor_state_topic"] == ""
        assert self.parameters(actor_state_topic="/stream/actor")["actor_state_topic"] == "/stream/actor"

    def test_the_mirror_gate_keys_reach_the_node(self):
        """The magicmirror entry wires both: the state topic, and how old a
        state message may be before it counts as unknown."""
        values = self.parameters(mirror_state_topic="/mirror_session/state", mirror_state_stale_sec=3.0)

        assert values["mirror_state_topic"] == "/mirror_session/state"
        assert values["mirror_state_stale_sec"] == 3.0

    def test_the_mirror_gate_defaults_to_ungated(self):
        """A deployment without a mirror gates nothing, on the contract's staleness."""
        from embodied_common.mirror_session_contract import STATE_STALE_SEC

        values = self.parameters()

        assert values["mirror_state_topic"] == ""
        assert values["mirror_state_stale_sec"] == STATE_STALE_SEC

    def test_every_passed_name_is_declared_by_the_node(self):
        """rclpy drops an override it has no declaration for without a warning."""
        from visual_interaction.visual_interaction_node import PARAMETERS

        assert set(self.parameters()) <= set(PARAMETERS)


class TestValidation:
    def test_an_unknown_key_is_refused_rather_than_ignored(self):
        """A rename that landed in one place only must not run on a default."""
        with pytest.raises(ValueError, match="unknown keys"):
            generate_visual_interaction_nodes(config(speak=True))

    def test_an_unknown_nested_key_is_refused(self):
        with pytest.raises(ValueError, match="visual_interaction.session has unknown keys"):
            generate_visual_interaction_nodes(config(session={"cooldown_sec": 1.0}))

    def test_an_unbound_description_is_refused_by_code(self):
        broken = config()
        del broken["runtime"]["interface_description"]

        with pytest.raises(InterfaceBindingError) as error:
            generate_visual_interaction_nodes(broken)

        assert error.value.code == "description_required"

    def test_a_description_at_the_wrong_key_is_not_accepted(self):
        """Guards the bug this test file used to hide.

        ``interface_binding`` writes ``runtime.interface_description``; a
        top-level key is written by nothing, so honouring one would pass here
        and fail at launch.
        """
        misplaced = config()
        misplaced["runtime_interface_description"] = misplaced["runtime"].pop("interface_description")

        with pytest.raises(InterfaceBindingError) as error:
            generate_visual_interaction_nodes(misplaced)

        assert error.value.path == "runtime.interface_description"

    def test_the_builder_reads_the_same_key_as_the_mirror_builder(self):
        """Both resolve endpoints by ID, so both read the one bound location."""
        import inspect

        from manipulation_execution.launch_builders import mirror_session
        from visual_interaction.launch_builders import visual_interaction as ours

        assert 'runtime.get("interface_description")' in inspect.getsource(mirror_session._runtime_description)
        assert 'runtime.get("interface_description")' in inspect.getsource(ours._runtime_description)

    def test_an_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="expected 'fake' or 'internvl'"):
            generate_visual_interaction_nodes(config(vl={"implementation": "llava"}))

    def test_internvl_without_a_deployment_path_is_refused(self):
        """It lives on the board, not in this repository, so it must be named.

        Refusing here rather than at the first round keeps a launch from coming
        up as a node that can never answer.
        """
        with pytest.raises(ValueError, match="demo_root must be an absolute path"):
            generate_visual_interaction_nodes(config(vl={"implementation": "internvl"}))

    def test_a_deployment_path_without_internvl_is_refused(self):
        """A stray key usually means the implementation was not switched over."""
        with pytest.raises(ValueError, match="only applies to implementation"):
            generate_visual_interaction_nodes(config(vl={"implementation": "fake", "demo_root": "/x"}))

    def test_a_relative_actor_topic_is_refused(self):
        with pytest.raises(ValueError, match="absolute ROS name"):
            generate_visual_interaction_nodes(config(actor_state_topic="stream/actor"))

    def test_a_relative_mirror_state_topic_is_refused(self):
        with pytest.raises(ValueError, match="absolute ROS name"):
            generate_visual_interaction_nodes(config(mirror_state_topic="mirror_session/state"))

    @pytest.mark.parametrize("value", [0, -1.0])
    def test_a_non_positive_stale_sec_is_refused(self, value):
        with pytest.raises(ValueError, match="positive number of seconds"):
            generate_visual_interaction_nodes(config(mirror_state_stale_sec=value))

    def test_a_boolean_stale_sec_is_refused(self):
        """A missing key reads as absent all over this config; a stray true must not pass as 1."""
        with pytest.raises(ValueError, match="positive number of seconds"):
            generate_visual_interaction_nodes(config(mirror_state_stale_sec=True))

    def test_an_empty_category_list_is_refused(self):
        with pytest.raises(ValueError, match="can never speak"):
            generate_visual_interaction_nodes(config(categories=[]))

    def test_a_category_outside_the_contract_is_refused(self):
        """Widening the whitelist is a configuration error, not a choice."""
        with pytest.raises(ValueError, match="inside the contract whitelist"):
            generate_visual_interaction_nodes(config(categories=["toy", "pendant"]))

    def test_a_non_string_category_is_refused(self):
        with pytest.raises(ValueError, match="list of strings"):
            generate_visual_interaction_nodes(config(categories=["toy", 7]))


class TestObservationOnlyGate:
    """The other half of the no-motion guarantee.

    The node refuses to resolve a motion interface, which stops it commanding
    anything itself. This stops the config putting a motion producer in the
    same entry point, which the node cannot see from inside one process.
    """

    @pytest.mark.parametrize("producer", ["inference", "executor"])
    def test_a_motion_producer_in_any_mode_is_refused(self, producer):
        broken = config()
        broken["control_modes"]["model_inference"] = {producer: {"enabled": True}}

        with pytest.raises(ValueError, match="observation-only"):
            generate_visual_interaction_nodes(broken)

    @pytest.mark.parametrize("producer", ["inference", "executor"])
    def test_a_producer_in_the_feature_s_own_mode_is_refused(self, producer):
        broken = config()
        broken["control_modes"]["observation"][producer] = {"enabled": True}

        with pytest.raises(ValueError, match="observation-only") as error:
            generate_visual_interaction_nodes(broken)

        assert producer in str(error.value)

    def test_an_explicitly_disabled_producer_is_fine(self):
        assert len(generate_visual_interaction_nodes(config())) == 1

    @pytest.mark.parametrize("producer", ["inference", "executor"])
    def test_a_producer_without_an_enabled_key_is_refused(self, producer):
        """A missing key counts as enabled, as robot_config reads it.

        ``execution._executor_enabled`` is ``executor.get("enabled", True)``, so
        treating absence as off would pass every inherited teleop and
        navigation mode — exactly the ones that drive the robot, and exactly
        the ones aimdk_x2_skills.yaml brings in.
        """
        broken = config()
        broken["control_modes"]["teleop"] = {producer: {"type": "topic", "mode": "teleop", "control_frequency": 50.0}}

        with pytest.raises(ValueError, match="observation-only"):
            generate_visual_interaction_nodes(broken)

    def test_an_empty_producer_block_is_refused(self):
        broken = config()
        broken["control_modes"]["teleop"] = {"executor": {}}

        with pytest.raises(ValueError, match="observation-only"):
            generate_visual_interaction_nodes(broken)

    def test_a_config_without_control_modes_is_fine(self):
        bare = config()
        del bare["control_modes"]

        assert len(generate_visual_interaction_nodes(bare)) == 1

    def test_the_gate_runs_only_once_the_feature_is_enabled(self):
        """A disabled feature must not veto a deployment it takes no part in."""
        disabled = config(enabled=False)
        disabled["control_modes"]["model_inference"] = {"inference": {"enabled": True}}

        assert generate_visual_interaction_nodes(disabled) == []
