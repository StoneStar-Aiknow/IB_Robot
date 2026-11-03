"""The open simulation platform registry (openspec sim-platform-registry D1-D5).

Commit-1 extensibility coverage (D10): the registry rules, the selection
checkpoints and the dispatch entries, all exercised with *injected* fake
platforms through the ``entry_points=`` seam — no real simulator is needed or
started. A fake platform is a ``SimPlatform`` whose embodiment is a test
module attribute, exactly how a third-party package would register one.
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import EntryPoint

import pytest

from robot_runtime.interface_description import (
    build_description,
    canonicalise_peripheral_entry,
    description_digest,
    merge_peripherals,
    validate_description,
)
from robot_runtime.launch_support import (
    RuntimeProfileError,
    render_sim_description,
    resolve_sim_selection,
    runtime_stack_actions,
    sim_embodiment_actions,
)
from robot_runtime.peripherals import perception_capabilities
from robot_runtime.sim_platforms import (
    SDK_PLATFORM,
    SIM_PLATFORM_ENTRY_POINT_GROUP,
    SimPlatform,
    SimPlatformError,
    SimSensorTraits,
    clear_sim_platform_cache,
    discover_sim_platforms,
    is_physics_selection,
    resolve_sim_embodiment,
    validate_embodiment_reference,
)

PLATFORM = "testsim"
EMBODIMENT_REF = "testsim_platform:EMBODIMENT"
SPEC_REF = "testsim_platform:SPEC"


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=SIM_PLATFORM_ENTRY_POINT_GROUP)


class FakeEmbodiment:
    """A minimal SimEmbodiment whose actions are unforgeable markers."""

    def __init__(self, sensor_traits: SimSensorTraits | None = None):
        self.sensor_traits = sensor_traits
        self.render_calls: list[dict] = []
        self.action_calls: list[dict] = []

    def render_description(self, profile, profile_path, *, peripherals=None, spawn_pose=(0.0, 0.0, 0.0, 0.0), world=""):
        self.render_calls.append({"world": world, "spawn_pose": spawn_pose})
        return "<robot name='testsim'/>"

    def actions(
        self,
        profile,
        profile_path,
        peripherals_file="",
        instance_id="",
        *,
        robot_description=None,
        interface_description=None,
        world="",
        spawn_pose=(0.0, 0.0, 0.0, 0.0),
    ):
        self.action_calls.append({"world": world, "spawn_pose": spawn_pose, "instance_id": instance_id})
        return ["testsim-embodiment-action"]


@pytest.fixture
def spec_module(monkeypatch) -> types.ModuleType:
    """An import-light fake spec module, the way a plugin package ships it."""
    module = types.ModuleType("testsim_platform")
    module.SPEC = SimPlatform(name=PLATFORM, package="testsim_pkg", embodiment=EMBODIMENT_REF)
    module.EMBODIMENT = FakeEmbodiment()
    monkeypatch.setitem(sys.modules, "testsim_platform", module)
    return module


@pytest.fixture
def fake_platform(spec_module, monkeypatch) -> SimPlatform:
    """Register a fake physics platform through the ``entry_points=`` seam.

    The platform is also seeded into the discovery path the selection
    checkpoints read (``resolve_sim_selection``, the facade and the
    description build look the platform up by name through the cached
    registry), and the lookup is pointed at the spec module so a test that
    rewrites ``spec_module.SPEC`` (e.g. with different traits) is seen too.
    """
    clear_sim_platform_cache()
    platforms = discover_sim_platforms(entry_points=[_entry_point(PLATFORM, SPEC_REF)])
    assert list(platforms) == [PLATFORM]

    def _discover(*, entry_points=None):
        if entry_points is None:
            return {PLATFORM: spec_module.SPEC}
        return {str(getattr(entry, "name", "")): spec_module.SPEC for entry in entry_points}

    import robot_runtime.sim_platforms as sim_platforms

    monkeypatch.setattr(sim_platforms, "discover_sim_platforms", _discover)
    return platforms[PLATFORM]


def _profile(**extra) -> dict:
    profile = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "use_sim": False,
        "sim_platforms": [SDK_PLATFORM, PLATFORM],
        "sim_platform": SDK_PLATFORM,
        "modes": {
            "initial": "idle",
            "idle": {"controllers": [], "transitions": ["trajectory"]},
            "trajectory": {
                "controllers": ["arm_trajectory_controller"],
                "allows_trajectory": True,
                "transitions": ["idle"],
            },
        },
        "capabilities": {"joint.state": {}, "joint.trajectory": {}},
        "joints": ["1"],
        "controller_manager": "controller_manager",
        "joint_state_topic": "/joint_states",
        "command_channels": [],
        "trajectory_actions": [],
        "description": {"xacro": "unused.xacro"},
        "ros2_control": {"controllers_config": "controllers.yaml"},
        "peripherals": [],
    }
    profile.update(extra)
    return profile


# ---------------------------------------------------------------------------
# D1: registration rules
# ---------------------------------------------------------------------------


def test_duplicate_registration_is_rejected():
    with pytest.raises(SimPlatformError, match="duplicate sim-platform entry point name 'testsim'"):
        discover_sim_platforms(entry_points=[_entry_point(PLATFORM, SPEC_REF), _entry_point(PLATFORM, SPEC_REF)])


def test_sdk_cannot_be_registered(spec_module):
    with pytest.raises(SimPlatformError, match="'sdk' is built in and cannot be registered"):
        discover_sim_platforms(entry_points=[_entry_point(SDK_PLATFORM, SPEC_REF)])


def test_reserved_peripheral_key_cannot_be_registered(spec_module):
    with pytest.raises(SimPlatformError, match="'embedded_sensor' is a reserved non-platform key"):
        discover_sim_platforms(entry_points=[_entry_point("embedded_sensor", SPEC_REF)])


def test_entry_point_name_must_equal_the_platform_name(spec_module):
    spec_module.SPEC = SimPlatform(name="other_name", package="testsim_pkg", embodiment=EMBODIMENT_REF)
    with pytest.raises(SimPlatformError, match="entry point 'testsim' resolved to a SimPlatform named 'other_name'"):
        discover_sim_platforms(entry_points=[_entry_point(PLATFORM, SPEC_REF)])


@pytest.mark.parametrize(
    "value,reason",
    [
        ("testsim_platform", "expected 'module:attribute'"),
        ("no_such_module:SPEC", "failed to import module"),
        ("testsim_platform:NO_SUCH_SPEC", "resolved to a missing attribute"),
    ],
)
def test_bad_entry_point_values_are_rejected(fake_platform, value, reason):
    with pytest.raises(SimPlatformError, match=reason):
        discover_sim_platforms(entry_points=[_entry_point(PLATFORM, value)])


def test_is_physics_selection_is_name_only():
    # the single sim-time policy (D1), decided without the registry
    assert is_physics_selection(False, "gazebo") is False
    assert is_physics_selection(True, None) is False
    assert is_physics_selection(True, "") is False
    assert is_physics_selection(True, SDK_PLATFORM) is False
    assert is_physics_selection(True, "any_name") is True


# ---------------------------------------------------------------------------
# D2: validation accepts a fake name only when registered
# ---------------------------------------------------------------------------


def test_selection_requires_the_fake_platform_to_be_registered():
    with pytest.raises(RuntimeProfileError, match="simulation platform 'testsim' is not installed"):
        resolve_sim_selection(_profile(), "true", PLATFORM, "")
    with pytest.raises(SimPlatformError, match="simulation platform 'testsim' is not installed"):
        build_description(_profile(), use_sim=True, sim_platform=PLATFORM)


def test_registered_fake_platform_selects_and_dispatches(fake_platform):
    assert resolve_sim_selection(_profile(), "true", PLATFORM, "") == (True, PLATFORM)


# ---------------------------------------------------------------------------
# D3: dispatch — runtime_stack_actions returns exactly the embodiment's actions
# ---------------------------------------------------------------------------


def test_runtime_stack_actions_returns_exactly_the_fake_embodiments_actions(fake_platform, tmp_path):
    profile = _profile()
    actions = runtime_stack_actions(
        profile,
        tmp_path / "p.yaml",
        simulated=True,
        sim_platform=PLATFORM,
        sim_world="demo",
        spawn_pose=(1.0, 2.0, 3.0, 0.5),
    )
    assert actions == ["testsim-embodiment-action"]
    (call,) = fake_platform and sys.modules["testsim_platform"].EMBODIMENT.action_calls
    assert call["world"] == "demo"
    assert call["spawn_pose"] == (1.0, 2.0, 3.0, 0.5)


def test_sim_embodiment_actions_returns_none_for_hardware_and_sdk(fake_platform, tmp_path):
    assert sim_embodiment_actions(_profile(), tmp_path / "p.yaml", None) is None
    assert sim_embodiment_actions(_profile(), tmp_path / "p.yaml", "") is None
    assert sim_embodiment_actions(_profile(), tmp_path / "p.yaml", SDK_PLATFORM) is None


def test_render_sim_description_dispatches_to_the_registered_embodiment(fake_platform, tmp_path):
    assert render_sim_description(_profile(), tmp_path / "p.yaml", PLATFORM, simulated=True, world="w") == (
        "<robot name='testsim'/>"
    )
    # hardware and sdk keep the ordinary renderer (no registry access)
    from unittest.mock import patch

    import robot_runtime.launch_support as launch_support

    with patch.object(launch_support, "render_robot_description", return_value="<hardware/>") as render:
        assert render_sim_description(_profile(), tmp_path / "p.yaml", None, simulated=False) == "<hardware/>"
        assert render_sim_description(_profile(), tmp_path / "p.yaml", SDK_PLATFORM, simulated=True) == "<hardware/>"
        assert render.call_count == 2


# ---------------------------------------------------------------------------
# D3: the per-robot embodiment override and its failure mode
# ---------------------------------------------------------------------------


def test_the_profile_overrides_the_platforms_default_embodiment(fake_platform, tmp_path):
    override = FakeEmbodiment()
    module = sys.modules["testsim_platform"]
    module.OVERRIDE = override
    profile = _profile(simulation={PLATFORM: {"embodiment": "testsim_platform:OVERRIDE"}})
    actions = runtime_stack_actions(profile, tmp_path / "p.yaml", simulated=True, sim_platform=PLATFORM)
    assert actions == ["testsim-embodiment-action"]
    assert override.action_calls and not module.EMBODIMENT.action_calls


def test_a_malformed_embodiment_override_fails_naming_the_profile_path(fake_platform, tmp_path):
    profile_path = tmp_path / "p.yaml"
    profile = _profile(simulation={PLATFORM: {"embodiment": "no-colon-reference"}})
    with pytest.raises(SimPlatformError, match=rf"embodiment 'no-colon-reference' of profile {profile_path}"):
        resolve_sim_embodiment(profile, profile_path, fake_platform)


def test_validate_embodiment_reference_checks_the_string_form_only():
    # load-time check (D3): an unregistered platform's override is still a
    # valid declaration; only the 'module:attr' shape is enforced
    validate_embodiment_reference(_profile(simulation={PLATFORM: {"embodiment": "anywhere:Anything"}}))
    with pytest.raises(SimPlatformError, match="must be a 'module:attr' reference"):
        validate_embodiment_reference(_profile(simulation={PLATFORM: {"embodiment": "no-colon"}}))


# ---------------------------------------------------------------------------
# D4: traits — default, platform, and the embodiment override
# ---------------------------------------------------------------------------


def _sensor_profile(**extra) -> dict:
    profile = _profile(
        peripherals=[
            {
                "type": "camera",
                "name": "front",
                "driver": "opencv",
                "width": 640,
                "height": 480,
                "fps": 30,
                "pixel_format": "mjpeg2rgb",
                "frame_id": "camera_front_frame",
                "optical_frame_id": "camera_front_optical_frame",
            },
            {
                "type": "lidar",
                "name": "mid360",
                "driver": "livox_mid360",
                "frame_id": "livox_frame",
                "pointcloud_topic": "/livox/lidar",
                "imu_topic": "/livox/imu",
                "publish_freq": 10.0,
                "xfer_format": 1,
            },
        ]
    )
    profile.update(extra)
    return profile


def test_default_traits_advertise_the_sdk_surface(fake_platform):
    # default traits apply no rewrite: the fake platform's description equals
    # the sdk one for a profile without exclusions or a lidar scan converter
    profile = _sensor_profile()
    fake = build_description(profile, use_sim=True, sim_platform=PLATFORM)
    sdk = build_description(profile, use_sim=True, sim_platform=SDK_PLATFORM)
    fake.pop("sim_platform", None), sdk.pop("sim_platform", None)
    assert fake["interfaces"] == sdk["interfaces"]


def test_platform_traits_rewrite_the_surface(spec_module, fake_platform):
    spec_module.SPEC = SimPlatform(
        name=PLATFORM,
        package="testsim_pkg",
        embodiment=EMBODIMENT_REF,
        sensor_traits=SimSensorTraits(drop_pointcloud=True),
    )
    profile = _sensor_profile(
        peripherals=[{**_sensor_profile()["peripherals"][0], "driver": "realsense", "enable_pointcloud": True}]
    )
    description = build_description(profile, use_sim=True, sim_platform=PLATFORM)
    assert "camera.front.points" not in description["interfaces"], "drop_pointcloud removes the points stream"


def test_an_embodiment_traits_override_changes_description_and_capabilities_identically(fake_platform, monkeypatch):
    # D4: an override that stands in for an extra lidar driver changes the
    # description and the RuntimeStatus perception capabilities identically
    import robot_runtime.interface_description as interface_description

    override = SimSensorTraits(lidar_sim_drivers=("other_driver",))
    profile = _sensor_profile()
    monkeypatch.setattr(interface_description, "effective_sensor_traits", lambda *args: None)
    plain = build_description(profile, use_sim=True, sim_platform=PLATFORM)
    monkeypatch.setattr(interface_description, "effective_sensor_traits", lambda *args: override)
    overridden = build_description(profile, use_sim=True, sim_platform=PLATFORM)

    # the description keeps the MID-360 under the platform's default traits and
    # drops it under the override (the platform then does not stand in for it)
    assert "lidar.mid360.points" in plain["interfaces"]
    assert "lidar.mid360.points" not in overridden["interfaces"]

    # the RuntimeStatus capabilities derive from the same traits: the surface
    # and the advertised endpoints cannot drift apart
    plain_caps = perception_capabilities(profile["peripherals"], traits=None)
    overridden_caps = perception_capabilities(profile["peripherals"], traits=override)
    assert plain_caps["perception.lidar"]["topics"] == ["/livox/lidar", "/livox/imu"]
    assert "perception.lidar" not in overridden_caps


# ---------------------------------------------------------------------------
# D2: reading is not selecting — hardware and sdk never consult the registry
# ---------------------------------------------------------------------------


@pytest.fixture
def raising_discovery(monkeypatch):
    def _raise(*, entry_points=None):
        raise AssertionError("the registry must not be queried by this path")

    monkeypatch.setattr("robot_runtime.sim_platforms.discover_sim_platforms", _raise)
    import robot_runtime.interface_description as interface_description

    monkeypatch.setattr(interface_description, "get_sim_platform", _raise)
    return _raise


def test_an_uninstalled_declared_platform_loads_on_hardware_and_sdk(raising_discovery, tmp_path):
    from robot_runtime.profile import validate_profile

    profile = _sensor_profile(sim_platforms=[SDK_PLATFORM, "nowhere_platform"], sim_platform="nowhere_platform")
    validate_profile(profile, tmp_path / "p.yaml")  # declaration is syntax (D2)
    assert build_description(profile, use_sim=False, sim_platform=None)["execution"] == "physical"
    assert build_description(profile, use_sim=True, sim_platform=SDK_PLATFORM)["sim_platform"] == SDK_PLATFORM


def test_hardware_and_sdk_paths_survive_a_raising_registry(raising_discovery, tmp_path):
    from robot_runtime.profile import load_profile

    profile_path = tmp_path / "p.yaml"
    profile_path.write_text(
        "runtime: {name: test_robot, version: 0.1.0}\n"
        "modes: {initial: idle, idle: {controllers: [], transitions: []}}\n"
        "capabilities: {joint.state: {}}\n"
        "joints: ['1']\n"
        "sim_platforms: [sdk, gazebo, mujoco]\n"
        "sim_platform: gazebo\n",
        encoding="utf-8",
    )
    loaded = load_profile(profile_path)
    assert loaded["sim_platform"] == "gazebo"
    entry = canonicalise_peripheral_entry(
        {"type": "camera", "name": "front", "driver": "opencv", "simulation": {"gazebo": {"enabled": False}}}
    )
    merged = merge_peripherals([entry], [{"type": "camera", "name": "front", "disabled": True}])
    assert merged == []
    hardware = build_description(_sensor_profile(), use_sim=False, sim_platform=None)
    sdk = build_description(_sensor_profile(), use_sim=True, sim_platform=SDK_PLATFORM)
    assert hardware["execution"] == "physical" and sdk["sim_platform"] == SDK_PLATFORM


# ---------------------------------------------------------------------------
# D2: the schema — a fake-platform description round-trips on an empty registry
# ---------------------------------------------------------------------------


def test_a_fake_platform_description_round_trips_and_validates_on_an_empty_registry(fake_platform):
    clear_sim_platform_cache()
    assert discover_sim_platforms(entry_points=[]) == {}
    description = build_description(_sensor_profile(sim_platform=PLATFORM), use_sim=True, sim_platform=PLATFORM)
    assert description["sim_platform"] == PLATFORM
    validate_description(description)  # reading never needs the plugin (D2)
    doc = dict(description)
    doc["digest"] = description_digest(doc)
    validate_description(doc)
    assert doc == description
