"""Pure registration checks; all geometric fixtures are synthetic, not certificates."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from aimdk_robot.named_hold_registration import HAND_ORDER, NATIVE_ORDER, register_named_hold, verify_loaded_named_hold
from aimdk_robot.preset_geometry import NamedHoldGeometryGuard, canonical_digest
from aimdk_robot.projection import DEXTEROUS_JOINT_ORDER, VENDOR_ARM_ORDER
from aimdk_robot.trajectory import load_trajectory


@pytest.fixture
def assets(tmp_path):
    target = [0.0] * 34
    identity = hashlib.sha256(json.dumps(target, separators=(",", ":")).encode()).hexdigest()
    template = {
        "schema_version": 1,
        "name": "overhead_heart",
        "units": "rad",
        "visual_version": "synthetic-v1",
        "validated": True,
        "motion_enabled": True,
        "stop_bridge_certified": True,
        "required_head_clearance_m": 0.20,
        "visual_approval": {"approved": True, "source_candidate_sha256": "a" * 64},
        "arm_joint_names": list(VENDOR_ARM_ORDER),
        "hand_joint_names": list(HAND_ORDER),
        "arm_target_rad": target[:14],
        "hand_target_rad": target[14:],
        "recording_file": "fixed.json",
    }
    models = []
    for side, source, sign in (("left", 0, 1), ("right", 7, -1)):
        bounds = [[0.0] * 34 for _ in range(2)]
        for row in bounds:
            row[source] = 2.0
        models.append(
            {
                "side": side,
                "joints": [
                    {
                        "name": side + "_synthetic",
                        "parent": 0,
                        "origin": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        "axis": [0, 0, 1],
                        "source": source,
                        "coefficients": [0, 1],
                        "limits": [-2, 2],
                    }
                ],
                "spheres": [[sign, 0, 0, 0.05], [sign, -0.5, 0, 0.05]],
                "sphere_frames": [1, 1],
                "head_check_indices": [0, 1],
                "hand_indices": [0],
                "self_pairs": [[0, 1]],
                "motion_bounds_m_per_rad": bounds,
            }
        )
    geometry = {
        "schema_version": 1,
        "name": "overhead_heart",
        "validated": True,
        "visual_version": "synthetic-v1",
        "native_joint_names": list(NATIVE_ORDER),
        "native_target_sha256": identity,
        "required_head_clearance_m": 0.20,
        "native_limits_rad": [[-2.0, 2.0]] * 34,
        "native_speed_radps": [1.0] * 34,
        "native_uncertainty_rad": [0.001] * 34,
        "head_envelope_spheres": [[4, 0, 0, 0.1]],
        "head_envelope_covers_allowed_motion": True,
        "complete_articulated_hands": True,
        "continuous_motion_bounds_verified": True,
        "models": models,
        "sources": ["SYNTHETIC unit-test data; never a robot certificate"],
    }
    recording = {
        "format": "ibrobot.fixed_motion.delivery",
        "schema_version": 1,
        "name": "overhead_heart",
        "visual_version": "synthetic-v1",
        "validated": True,
        "source_candidate_sha256": "a" * 64,
        "native_target_sha256": identity,
        "time_from_start_s": [0.0, 1.0],
        "arms": {"units": "rad", "joints": list(VENDOR_ARM_ORDER), "positions_rad": [target[:14]] * 2},
        "hands": {
            "units": "rad",
            "joints": [f"{side}_{joint}" for side in ("left", "right") for joint in DEXTEROUS_JOINT_ORDER],
            "positions": [target[14:]] * 2,
        },
    }
    profile = {
        "runtime": {"name": "synthetic_runtime", "version": "0.0", "instance_id": "synthetic"},
        "joints": list(VENDOR_ARM_ORDER),
        "arm_joints": list(VENDOR_ARM_ORDER),
        "modes": {"idle": {}, "policy_stream": {}},
        "capabilities": {"motion.named": {"names": ["wave"]}},
        "interfaces": {
            "motion.named": {
                "capability": "motion.named",
                "kind": "action",
                "direction": "serve",
                "endpoint": "/motion/execute_named",
                "message_type": "ibrobot_msgs/action/ExecuteNamedMotion",
            }
        },
        "vendor": {
            "arm": {"limits": {name: [-2.0, 2.0] for name in VENDOR_ARM_ORDER}},
            "preset_motions": {"wave": {"motion": 1}},
            "postures": {"sit": "SIT"},
            "named_motion": {
                "trajectories": {"wave": "trajectories/wave.json"},
                "trajectory_modes": ["stream", "policy_stream"],
                "allowed_modes": ["idle"],
                "cancel_return_motions": ["wave"],
                "display_completion": {"motions": ["wave"]},
            },
        },
    }
    paths = {
        "template": tmp_path / "pose.yaml",
        "geometry": tmp_path / "geometry.json",
        "recording": tmp_path / "fixed.json",
    }

    def save():
        geometry["sha256"] = canonical_digest(geometry)
        paths["template"].write_text(yaml.safe_dump(template))
        paths["geometry"].write_text(json.dumps(geometry))
        paths["recording"].write_text(json.dumps(recording))

    save()
    payload = json.dumps(
        {
            "motion_name": "overhead_heart",
            "template_file": str(paths["template"]),
            "geometry_file": str(paths["geometry"]),
        }
    )
    return profile, payload, template, geometry, recording, paths, save


def test_empty_opt_in_never_reads_assets_and_returns_identical_profile(monkeypatch):
    profile = {"unchanged": object()}
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("disabled registration read an asset"))
    assert register_named_hold(profile, "") is profile


def test_registration_preserves_ordinary_actions_and_declares_exact_lease_interfaces(assets):
    profile, payload, _template, _geometry, _recording, paths, _save = assets
    before = copy.deepcopy(profile)
    result = register_named_hold(profile, payload)
    assert profile == before
    output = result["vendor"]["named_motion"]
    assert output["trajectories"] == {"wave": "trajectories/wave.json", "overhead_heart": str(paths["recording"])}
    for key in ("trajectory_modes", "allowed_modes", "cancel_return_motions", "display_completion"):
        assert output[key] == before["vendor"]["named_motion"][key]
    assert result["vendor"]["preset_motions"] == before["vendor"]["preset_motions"]
    assert output["hold_motions"] == ["overhead_heart"]
    assert output["hold_geometry_file"] == str(paths["geometry"])
    assert output["hold_hand_limits"] == [[-2.0, 2.0]] * 20
    assert output["hold_hand_speed_radps"] == [1.0] * 20
    for name, direction, endpoint in (
        ("motion.named_hold_lease", "subscribe", "/runtime/named_motion_hold_lease"),
        ("motion.named_hold_lease_ack", "publish", "/runtime/named_motion_hold_lease_ack"),
    ):
        item = result["interfaces"][name]
        assert (item["kind"], item["direction"], item["message_type"], item["endpoint"]) == (
            "topic",
            direction,
            "std_msgs/msg/String",
            endpoint,
        )
    assert result["capabilities"]["motion.named"]["names"] == ["wave", "overhead_heart"]


@pytest.mark.parametrize("field", ["validated", "motion_enabled", "stop_bridge_certified"])
def test_template_requires_explicit_certification(assets, field):
    profile, payload, template, _geometry, _recording, _paths, save = assets
    template[field] = False
    save()
    with pytest.raises(ValueError, match="validated template"):
        register_named_hold(profile, payload)


@pytest.mark.parametrize(
    "field",
    [
        "validated",
        "complete_articulated_hands",
        "continuous_motion_bounds_verified",
        "head_envelope_covers_allowed_motion",
    ],
)
def test_geometry_requires_independent_complete_evidence(assets, field):
    profile, payload, _template, geometry, _recording, _paths, save = assets
    geometry[field] = False
    save()
    with pytest.raises(ValueError, match="not certified"):
        register_named_hold(profile, payload)


@pytest.mark.parametrize("mutation", ["order", "target_digest", "version", "clearance", "sha"])
def test_geometry_identity_clearance_and_digest_cannot_drift(assets, mutation):
    profile, payload, _template, geometry, _recording, paths, save = assets
    if mutation == "order":
        geometry["native_joint_names"][0], geometry["native_joint_names"][1] = (
            geometry["native_joint_names"][1],
            geometry["native_joint_names"][0],
        )
    elif mutation == "target_digest":
        geometry["native_target_sha256"] = "0" * 64
    elif mutation == "version":
        geometry["visual_version"] = "different"
    elif mutation == "clearance":
        geometry["required_head_clearance_m"] = 0.15
    save()
    if mutation == "sha":
        geometry["native_speed_radps"][0] = 2.0
        paths["geometry"].write_text(json.dumps(geometry))
    with pytest.raises(ValueError):
        register_named_hold(profile, payload)


@pytest.mark.parametrize(
    "mutation", ["arm_order", "hand_order", "arm_value", "hand_value", "boolean", "name", "candidate"]
)
def test_fixed_recording_cannot_change_the_approved_34_joint_pose(assets, mutation):
    profile, payload, _template, _geometry, recording, _paths, save = assets
    if mutation == "arm_order":
        recording["arms"]["joints"].reverse()
    elif mutation == "hand_order":
        recording["hands"]["joints"].reverse()
    elif mutation in ("arm_value", "boolean"):
        recording["arms"]["positions_rad"][0][0] = True if mutation == "boolean" else 0.1
    elif mutation == "hand_value":
        recording["hands"]["positions"][0][0] = 0.1
    elif mutation == "name":
        recording["name"] = "wave"
    else:
        recording["source_candidate_sha256"] = "0" * 64
    save()
    with pytest.raises(ValueError):
        register_named_hold(profile, payload)


@pytest.mark.parametrize("where", ["presets", "recordings", "capability", "hold", "interface"])
def test_registration_refuses_existing_name_or_owner_conflicts(assets, where):
    profile, payload, *_rest = assets
    if where == "presets":
        profile["vendor"]["preset_motions"]["overhead_heart"] = {"motion": 2}
    elif where == "recordings":
        profile["vendor"]["named_motion"]["trajectories"]["overhead_heart"] = "old.json"
    elif where == "capability":
        profile["capabilities"]["motion.named"]["names"].append("overhead_heart")
    elif where == "hold":
        profile["vendor"]["named_motion"]["hold_motions"] = ["other"]
    else:
        profile["interfaces"]["other"] = {"endpoint": "/runtime/named_motion_hold_lease"}
    with pytest.raises(ValueError, match="conflict"):
        register_named_hold(profile, payload)


def test_registration_intersects_existing_hand_limits_and_speeds(assets):
    profile, payload, *_rest = assets
    config = profile["vendor"]["named_motion"]
    config["hold_hand_limits"] = [[-0.5, 0.5]] * 20
    config["hold_hand_speed_radps"] = 0.25
    output = register_named_hold(profile, payload)["vendor"]["named_motion"]
    assert output["hold_hand_limits"] == [[-0.5, 0.5]] * 20
    assert output["hold_hand_speed_radps"] == [0.25] * 20


def test_registration_does_not_silently_enable_another_replay_mode(assets):
    profile, payload, *_rest = assets
    profile["vendor"]["named_motion"]["trajectory_modes"] = ["stream"]
    with pytest.raises(ValueError, match="existing policy_stream"):
        register_named_hold(profile, payload)


def test_unvalidated_recording_remains_an_independent_gate(assets):
    profile, payload, _template, _geometry, recording, _paths, save = assets
    recording["validated"] = False
    save()
    with pytest.raises(ValueError, match="validated full arm-and-hand"):
        register_named_hold(profile, payload)


@pytest.mark.parametrize("asset", ["recording", "geometry", "recording_flag"])
def test_final_loaded_objects_reject_real_asset_replacement(assets, asset):
    profile, payload, _template, geometry, recording, paths, _save = assets
    registered = register_named_hold(profile, payload)
    if asset == "recording":
        recording["arms"]["positions_rad"] = [[0.1] * 14] * 2
        paths["recording"].write_text(json.dumps(recording))
    elif asset == "recording_flag":
        recording["validated"] = False
        paths["recording"].write_text(json.dumps(recording))
    else:
        geometry["head_envelope_spheres"] = [[5, 0, 0, 0.1]]
        geometry["sha256"] = canonical_digest(geometry)
        paths["geometry"].write_text(json.dumps(geometry))
    loaded = load_trajectory(paths["recording"], max_speed_radps=1.0)
    guard = NamedHoldGeometryGuard(json.loads(paths["geometry"].read_text()))
    with pytest.raises(ValueError, match="changed"):
        verify_loaded_named_hold(registered, {"overhead_heart": loaded}, guard)


def test_shipped_omnihand_profile_preserves_all_existing_motion_configuration(assets):
    _profile, payload, template, geometry, recording, paths, save = assets
    path = Path(__file__).resolve().parents[1] / "profiles/x2_ultra_omnihand.yaml"
    profile = yaml.safe_load(path.read_text())
    before = copy.deepcopy(profile)
    template["arm_target_rad"][3] = template["arm_target_rad"][10] = -1.0
    target = template["arm_target_rad"] + template["hand_target_rad"]
    identity = hashlib.sha256(json.dumps(target, separators=(",", ":")).encode()).hexdigest()
    geometry["native_target_sha256"] = recording["native_target_sha256"] = identity
    geometry["native_limits_rad"] = [profile["vendor"]["arm"]["limits"][name] for name in VENDOR_ARM_ORDER] + [
        [-2.0, 2.0]
    ] * 20
    recording["arms"]["positions_rad"] = [list(template["arm_target_rad"])] * 2
    save()
    result = register_named_hold(profile, payload)
    assert profile == before
    for name, value in before["vendor"]["named_motion"].items():
        if name == "trajectories":
            assert {key: result["vendor"]["named_motion"][name][key] for key in value} == value
        else:
            assert result["vendor"]["named_motion"][name] == value
    loaded = load_trajectory(paths["recording"], max_speed_radps=6.0)
    verify_loaded_named_hold(result, {"overhead_heart": loaded}, NamedHoldGeometryGuard(geometry))
    assert result["vendor"]["named_motion"]["hold_motions"] == ["overhead_heart"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("simulated", [False, True])
def test_provider_launch_and_runtime_reconstruct_the_same_opt_in_profile(assets, monkeypatch, enabled, simulated):
    from launch import LaunchContext

    from robot_runtime.interface_description import build_description

    profile, payload, *_rest = assets
    payload = payload if enabled else ""
    path = Path(__file__).resolve().parents[1] / "launch/runtime.launch.py"
    spec = importlib.util.spec_from_file_location("synthetic_named_hold_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes, planned = [], []

    def node(**kwargs):
        nodes.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(module, "Node", node)
    monkeypatch.setattr(module, "TimerAction", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "resolve_profile_argument", lambda *_args: Path("/unused/profile.yaml"))
    monkeypatch.setattr(module, "load_profile", lambda *_args: copy.deepcopy(profile))
    monkeypatch.setattr(
        module, "resolve_sim_selection", lambda *_args, **_kwargs: (simulated, "sdk" if simulated else "")
    )
    monkeypatch.setattr(module, "synthetic_perception_nodes", lambda *_args: [])
    monkeypatch.setattr(module.dataplane, "plan", lambda value: planned.append(copy.deepcopy(value)) or {})
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "profile": "unused",
            "use_sim": str(simulated).lower(),
            "sim_platform": "",
            "simulated": "",
            "peripherals": "",
            "instance_id": "",
            "initial_mode": "",
            "named_hold_json": payload,
        }
    )
    module._launch_setup(context)
    runtime = next(item for item in nodes if item["executable"] == "aimdk_runtime")
    params = runtime["parameters"][0]
    assert ("named_hold_json" in params) is enabled
    runtime_profile = register_named_hold(copy.deepcopy(profile), params.get("named_hold_json", ""))
    assert planned == [runtime_profile]
    interfaces = build_description(runtime_profile, use_sim=simulated, sim_platform="sdk" if simulated else None)[
        "interfaces"
    ]
    assert ("motion.named_hold_lease" in interfaces) is enabled
    assert ("motion.named_hold_lease_ack" in interfaces) is enabled
    if simulated:
        assert json.loads(params["interface_description_json"])["interfaces"] == interfaces
