"""Hand-source launch-builder behavior tests (openspec change ``migrate-hand-sources-to-robot-teleop``).

Migrated from robot_config's ``test_launch_readiness.py`` (the hand_sources
halves) and ``test_aero_hand_config.py`` / ``test_launch_preparation.py``
(the hand-profile behavior): assertions are preserved verbatim; only imports
point at this package's migrated module. The preparation-path tests inject
the ``hand_sources`` entry point through the registry seam, so the full
``prepare_launch`` pipeline is exercised exactly as production composes it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_config.launch_builders import control
from robot_config.launch_registry.builtin_registry import build_launch_registry
from robot_config.launch_registry.preparation import prepare_launch
from robot_teleop.launch_builders.hand_sources import (
    apply_hand_profile,
    confirm_interactive_startup_p_pose,
    generate_hand_source_nodes,
)

TELEOP_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "robot_config" / "config" / "robots" / "aero_hand_teleop.yaml"
)


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _node_parameters(node):
    def decode_parameter(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode_parameter(tuple(item)) for item in value]
        if all(isinstance(item, bool | int | float) for item in value):
            return list(value)
        text = _text(value)
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text.strip()

    raw = node._Node__parameters[0]
    parsed = {}
    for key, value in raw.items():
        name = _text(key)
        parsed[name] = decode_parameter(value)
    return parsed


def _hand_sources_registry():
    """The production builtin table plus this package's hand_sources entry."""
    from importlib.metadata import EntryPoint

    return build_launch_registry(
        entry_points=[
            EntryPoint(
                name="hand_sources",
                value="robot_teleop.launch_builders.hand_sources:HAND_SOURCES_BUILDER",
                group="ibrobot.launch_builders",
            )
        ]
    )


def _context(**arguments):
    return SimpleNamespace(launch_configurations=arguments)


# --- generate_hand_source_nodes (verbatim from robot_config's suite) --------


def test_generate_mock_mhandpro_source_for_both_sides():
    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": True,
                    "sides": ["left", "right"],
                    "topic_prefix": "/hands/mhandpro",
                }
            }
        }
    )

    assert len(nodes) == 1
    assert nodes[0].node_package == "robot_teleop"
    assert nodes[0].node_executable == "mhandpro_source_node"
    params = _node_parameters(nodes[0])
    assert params["sides"] == ["left", "right"]
    assert params["topic_prefix"] == "/hands/mhandpro"
    assert params["mock"] is True
    assert params["publish_raw_frame"] is False
    assert params["calibrate_p_pose_on_startup"] is False
    assert params["failure_policy"] == "require_all"
    assert params["auto_reconnect"] is True
    assert params["reconnect_initial_delay"] == 1.0
    assert params["reconnect_max_delay"] == 10.0
    assert params["reconnect_max_attempts"] == 0
    assert nodes[0]._ExecuteLocal__on_exit is None


def test_real_mhandpro_source_exits_shutdown_launch(tmp_path):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": False,
                    "lib_path": str(library),
                }
            }
        }
    )

    assert nodes[0]._ExecuteLocal__on_exit is not None


def test_mhandpro_raw_frame_publication_requires_explicit_opt_in():
    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": True,
                    "publish_raw_frame": True,
                }
            }
        }
    )

    assert _node_parameters(nodes[0])["publish_raw_frame"] is True


def test_unknown_startup_p_pose_mode_is_rejected():
    with pytest.raises(ValueError, match="startup_p_pose"):
        generate_hand_source_nodes(
            {
                "hand_sources": {
                    "mhandpro": {
                        "type": "mhandpro",
                        "mock": True,
                        "startup_p_pose": "automatic",
                    }
                }
            }
        )


def test_mhandpro_source_rejects_unknown_dual_hand_failure_policy():
    with pytest.raises(ValueError, match="failure_policy"):
        generate_hand_source_nodes(
            {
                "hand_sources": {
                    "mhandpro": {
                        "type": "mhandpro",
                        "mock": True,
                        "sides": ["left", "right"],
                        "failure_policy": "ignore_missing",
                    }
                }
            }
        )


def test_real_mhandpro_source_resolves_external_library(tmp_path, monkeypatch):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    monkeypatch.setenv("MHANDPRO_SDK_LIB", str(library))

    nodes = generate_hand_source_nodes(
        {
            "hand_sources": {
                "mhandpro": {
                    "type": "mhandpro",
                    "mock": False,
                    "lib_path": "$(env MHANDPRO_SDK_LIB)",
                }
            }
        }
    )

    assert _node_parameters(nodes[0])["lib_path"] == str(library)


def test_real_mhandpro_source_requires_external_library_when_environment_is_unset(monkeypatch):
    monkeypatch.delenv("MHANDPRO_SDK_LIB", raising=False)

    with pytest.raises(RuntimeError, match="external vendor library"):
        generate_hand_source_nodes(
            {
                "hand_sources": {
                    "mhandpro": {
                        "type": "mhandpro",
                        "mock": False,
                        "lib_path": "$(env MHANDPRO_SDK_LIB)",
                    }
                }
            }
        )


def test_simulation_skips_real_hand_sources(monkeypatch):
    """The hand_sources half of robot_config's mixed sim-skip test."""
    monkeypatch.delenv("MHANDPRO_SDK_LIB", raising=False)
    config = {
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": False,
                "lib_path": "$(env MHANDPRO_SDK_LIB)",
            }
        }
    }

    control.validate_runtime_resources(config, use_sim=True, control_mode="teleop")
    assert generate_hand_source_nodes(config, use_sim=True, control_mode="teleop") == []


def test_hand_sources_respect_active_control_modes():
    """The hand_sources half of robot_config's mixed control-mode test."""
    config = {
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": True,
                "active_control_modes": ["teleop"],
            }
        }
    }

    assert generate_hand_source_nodes(config, control_mode="model_inference") == []
    assert len(generate_hand_source_nodes(config, use_sim=True, control_mode="teleop")) == 1


# --- confirm_interactive_startup_p_pose (verbatim) ----------------------------


def test_interactive_startup_p_pose_prompts_and_enables_in_process_calibration(monkeypatch, tmp_path):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    config = {
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": False,
                "lib_path": str(library),
                "sides": ["right"],
                "require_p_pose": True,
                "startup_p_pose": "interactive",
            }
        }
    }
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt))

    confirm_interactive_startup_p_pose(config, use_sim=False, control_mode="teleop")
    nodes = generate_hand_source_nodes(config, use_sim=False, control_mode="teleop")

    assert len(prompts) == 1
    assert "P-pose" in prompts[0]
    assert _node_parameters(nodes[0])["calibrate_p_pose_on_startup"] is True


def test_interactive_startup_p_pose_requires_foreground_terminal(monkeypatch, tmp_path):
    library = tmp_path / "libVDMocapSDK_mHandPro.so"
    library.touch()
    config = {
        "hand_sources": {
            "mhandpro": {
                "type": "mhandpro",
                "mock": False,
                "lib_path": str(library),
                "startup_p_pose": "interactive",
            }
        }
    }
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

    with pytest.raises(RuntimeError, match="requires a terminal"):
        confirm_interactive_startup_p_pose(config, use_sim=False, control_mode="teleop")


# --- apply_hand_profile (verbatim) --------------------------------------------


@pytest.mark.parametrize(
    ("profile", "sides", "actuator_count", "device_count"),
    [
        ("right", ["right"], 1, 1),
        ("left", ["left"], 1, 1),
        ("dual", ["left", "right"], 2, 2),
    ],
)
def test_hand_profile_selects_sides_actuators_and_devices(profile, sides, actuator_count, device_count):
    config = {
        "joints": {"hand": []},
        "hand_profiles": {
            "default_profile": "right",
            "profiles": {
                "right": {
                    "sides": ["right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_right"],
                    "active_devices": ["aero_glove_right"],
                },
                "left": {
                    "sides": ["left"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_left"],
                    "active_devices": ["aero_glove_left"],
                },
                "dual": {
                    "sides": ["left", "right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_left", "aero_hand_right"],
                    "active_devices": ["aero_glove_left", "aero_glove_right"],
                },
            },
        },
        "hand_sources": {"mhandpro": {"type": "mhandpro", "mock": True}},
        "auxiliary_actuators": {
            "aero_hand_left": {
                "profile_managed": True,
                "enabled": False,
                "joint_names": [f"left_{index}" for index in range(7)],
            },
            "aero_hand_right": {
                "profile_managed": True,
                "enabled": False,
                "joint_names": [f"right_{index}" for index in range(7)],
            },
        },
        "teleoperation": {
            "devices": [{"name": "aero_glove_left"}, {"name": "aero_glove_right"}],
        },
    }

    assert apply_hand_profile(config, profile) == profile
    assert config["hand_sources"]["mhandpro"]["sides"] == sides
    assert sum(item["enabled"] for item in config["auxiliary_actuators"].values()) == actuator_count
    assert len(config["teleoperation"]["active_devices"]) == device_count
    assert len(config["joints"]["hand"]) == actuator_count * 7


def test_hand_profile_defaults_to_robot_yaml_selection():
    config = {
        "joints": {"hand": []},
        "hand_profiles": {
            "default_profile": "right",
            "profiles": {
                "right": {
                    "sides": ["right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aero_hand_right"],
                    "active_devices": ["aero_glove_right"],
                }
            },
        },
        "hand_sources": {"mhandpro": {}},
        "auxiliary_actuators": {
            "aero_hand_right": {
                "profile_managed": True,
                "enabled": False,
                "joint_names": [f"right_{index}" for index in range(7)],
            }
        },
        "teleoperation": {"devices": [{"name": "aero_glove_right"}]},
    }

    assert apply_hand_profile(config) == "right"
    assert config["teleoperation"]["active_devices"] == ["aero_glove_right"]


def test_unknown_hand_profile_is_rejected():
    config = {"hand_profiles": {"default_profile": "right", "profiles": {"right": {}}}}

    with pytest.raises(ValueError, match="unknown hand_profile"):
        apply_hand_profile(config, "triple")


# --- aero-hand SSOT profile application (from test_aero_hand_config.py) ------


def test_standalone_profiles_pass_safety_limits_to_each_aero_driver(monkeypatch):
    from robot_config.launch_builders.control import generate_auxiliary_actuator_nodes

    monkeypatch.setenv("AERO_HAND_LEFT_PORT", "/dev/aero-left")
    monkeypatch.setenv("AERO_HAND_RIGHT_PORT", "/dev/aero-right")
    source = yaml.safe_load(TELEOP_CONFIG_PATH.read_text(encoding="utf-8"))["robot"]
    for profile in ("left", "right", "dual"):
        config = yaml.safe_load(yaml.safe_dump(source))
        apply_hand_profile(config, profile)

        nodes = generate_auxiliary_actuator_nodes(config, control_mode="teleop")

        assert len(nodes) == (2 if profile == "dual" else 1)
        for node in nodes:
            parameters = _node_parameters(node)
            joint_names = parameters["joint_names"]
            limits = config["teleoperation"]["safety"]["joint_limits"]
            assert parameters["command_lower_limits"] == [limits[name]["min"] for name in joint_names]
            assert parameters["command_upper_limits"] == [limits[name]["max"] for name in joint_names]


# --- preparation path through the injected entry point -----------------------


def _hand_profile_loaded_config() -> dict:
    """A hand-profile selection config mirroring aero_hand_teleop.yaml."""
    return {
        "name": "hand_profile_robot",
        "hand_profiles": {
            "default_profile": "right",
            "profiles": {
                "right": {
                    "sides": ["right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aux_right"],
                    "active_devices": ["glove_right"],
                },
                "dual": {
                    "sides": ["left", "right"],
                    "hand_source": "mhandpro",
                    "active_actuators": ["aux_left", "aux_right"],
                    "active_devices": ["glove_left", "glove_right"],
                },
            },
        },
        "hand_sources": {"mhandpro": {"type": "mhandpro", "enabled": True, "mock": True, "sides": ["right"]}},
        "auxiliary_actuators": {
            "aux_left": {"profile_managed": True, "enabled": False, "joint_names": ["left_j1"]},
            "aux_right": {"profile_managed": True, "enabled": False, "joint_names": ["right_j1"]},
        },
        "teleoperation": {
            "enabled": True,
            "active_devices": ["glove_right"],
            "devices": [{"name": "glove_left"}, {"name": "glove_right"}],
        },
        "joints": {"arm": [], "hand": []},
    }


def test_hand_profile_hook_result_matches_legacy_preparation(monkeypatch):
    """Task 3.5 equivalence anchor (from robot_config's suite): through the
    production builtin table plus this package's injected entry point, the
    hand_sources config_overrides hook produces exactly the config the legacy
    core step produced."""
    import copy

    resource_configs = []
    monkeypatch.setattr(control, "validate_runtime_resources", lambda config, **kwargs: resource_configs.append(config))
    loaded = _hand_profile_loaded_config()
    before = copy.deepcopy(loaded)
    prepared = prepare_launch(_context(hand_profile=" DUAL "), _hand_sources_registry(), loaded_config=loaded)
    # The caller's loaded config is never mutated.
    assert loaded == before
    # Verbatim hand-profile semantics: the hook normalized the raw CLI value
    # (" DUAL " -> "dual") before applying the profile; sides,
    # profile-managed actuator enables, active teleop devices, hand joints.
    assert prepared.robot_config["active_hand_profile"] == "dual"
    assert prepared.robot_config["hand_sources"]["mhandpro"]["sides"] == ["left", "right"]
    assert prepared.robot_config["auxiliary_actuators"]["aux_left"]["enabled"] is True
    assert prepared.robot_config["auxiliary_actuators"]["aux_right"]["enabled"] is True
    assert prepared.robot_config["teleoperation"]["active_devices"] == ["glove_left", "glove_right"]
    assert prepared.robot_config["joints"]["hand"] == ["left_j1", "right_j1"]
    # resource_check runs AFTER the hook and observes the applied profile.
    assert len(resource_configs) == 1
    assert resource_configs[0]["active_hand_profile"] == "dual"
    # The hand_profiles INPUT section is consumed, never rewritten.
    assert prepared.robot_config["hand_profiles"] == before["hand_profiles"]


def test_default_hand_profile_applies_without_cli_argument(monkeypatch):
    monkeypatch.setattr(control, "validate_runtime_resources", lambda config, **kwargs: None)
    prepared = prepare_launch(_context(), _hand_sources_registry(), loaded_config=_hand_profile_loaded_config())
    assert prepared.robot_config["active_hand_profile"] == "right"
    assert prepared.robot_config["auxiliary_actuators"]["aux_left"]["enabled"] is False
    assert prepared.robot_config["auxiliary_actuators"]["aux_right"]["enabled"] is True


def test_preflight_p_pose_confirmation_runs_exactly_once(monkeypatch):
    """Migrated from robot_config's preparation suite: the P-pose
    confirmation runs exactly once per launch through this entry's preflight
    hook, before any provider startup, and does not rewrite the config."""
    from robot_teleop.launch_builders import hand_sources

    monkeypatch.setattr(control, "validate_runtime_resources", lambda config, **kwargs: None)
    confirmations = []
    monkeypatch.setattr(
        hand_sources,
        "confirm_interactive_startup_p_pose",
        lambda config, **kwargs: confirmations.append((config["active_hand_profile"], kwargs)),
    )
    prepared = prepare_launch(_context(), _hand_sources_registry(), loaded_config=_hand_profile_loaded_config())
    assert len(confirmations) == 1
    assert confirmations[0][0] == "right"
    assert confirmations[0][1] == {"use_sim": False, "control_mode": "model_inference"}
    assert prepared.robot_config["active_hand_profile"] == "right"


def test_preflight_p_pose_failure_fails_the_launch(monkeypatch):
    """Migrated from robot_config's preparation suite: cancel/EOF during the
    interactive confirmation fails the launch synchronously, before any
    provider factory runs."""
    from robot_teleop.launch_builders import hand_sources

    monkeypatch.setattr(control, "validate_runtime_resources", lambda config, **kwargs: None)

    def refusing(config, **kwargs):
        raise RuntimeError("Interactive startup P-pose requires a terminal.")

    monkeypatch.setattr(hand_sources, "confirm_interactive_startup_p_pose", refusing)
    with pytest.raises(RuntimeError, match=r"preparation hook 'hand_sources' \(preflight\) failed"):
        prepare_launch(_context(), _hand_sources_registry(), loaded_config=_hand_profile_loaded_config())


def test_hand_profile_argument_is_contributed_with_empty_fallback():
    """The hand_profile launch argument moved with the entry: it is declared
    by the injected extension with the same spec as the former builtin row
    (robot_config's composed-table assertions moved here)."""
    from robot_config.launch_registry.launch_options import compose_argument_specs

    specs = {spec.name: spec for spec in compose_argument_specs(_hand_sources_registry().contributed_arguments())}
    assert "hand_profile" in specs
    assert specs["hand_profile"].declared_default == ""
    assert specs["hand_profile"].absent_fallback == ""
    assert specs["hand_profile"].description.startswith("Select a hand profile")


def test_hand_sources_joins_the_plan_only_through_its_entry_point():
    """Spec scenarios: with the entry point installed the entry participates
    in the plan; without it the builtin-only registry carries no hand_sources
    entry (and no hand_profile argument is declared)."""
    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.launch_options import compose_argument_specs
    from robot_config.launch_registry.registry import LaunchBuildContext
    from robot_config.runtime_target import RuntimeTarget

    context = LaunchBuildContext(
        robot_config=_hand_profile_loaded_config(),
        runtime_target=RuntimeTarget.HARDWARE,
        use_sim=False,
        control_mode="teleop",
        with_inference=False,
        arguments={},
    )
    assert [spec.id for spec in _hand_sources_registry().plan(context).specs].count("hand_sources") == 1
    builtin = build_builtin_registry()
    assert "hand_sources" not in [spec.id for spec in builtin.plan(context).specs]
    assert "hand_profile" not in {spec.name for spec in compose_argument_specs(builtin.contributed_arguments())}
