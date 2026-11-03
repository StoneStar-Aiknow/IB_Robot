"""The open simulation platform registry (openspec sim-platform-registry D1).

A physics simulation platform is an installed ``ibrobot.sim_platforms`` entry
point resolving to a :class:`SimPlatform` spec. ``sdk`` stays built in and
cannot be registered. Discovery mirrors the repository's existing
``ibrobot.launch_builders`` extension pattern (an ``entry_points=`` injection
seam for tests, duplicate-name rejection, ``module:attr`` validation and
deterministic ordering), and is cached per process.

Declaration is syntax; registration is required only for an effective
selection (D2): profile loading, peripheral canonicalisation and merge never
consult the registry, so a hardware-only deployment loads a profile that
lists ``gazebo``. :func:`is_physics_selection` is the single sim-time policy
— a simulated selection whose platform is not ``sdk`` is physics, decided
from the name alone, without the registry.

This module is the import-light contract core: it imports stdlib modules and
nothing else. The ``module:attr`` references a spec carries (embodiment,
world hook, scene-task transport, trait callables) are resolved lazily by the
callers, never here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points as _discover_entry_points
from typing import Any, Protocol, runtime_checkable

#: The entry-point group every physics platform registers under.
SIM_PLATFORM_ENTRY_POINT_GROUP = "ibrobot.sim_platforms"

#: The built-in vendor SDK virtual bus. It is no physics platform (no world,
#: no simulation clock) and cannot be registered or shadowed.
SDK_PLATFORM = "sdk"

#: A platform name is a lower-case identifier (D2). ``sdk`` always follows
#: the rule too; unknown names are valid declarations — they only need the
#: plugin once they are the effective selection.
SIM_PLATFORM_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def is_valid_sim_platform_name(name: object) -> bool:
    """Whether ``name`` is a syntactically valid platform identifier (D2)."""
    return isinstance(name, str) and bool(SIM_PLATFORM_NAME_PATTERN.match(name))


def is_physics_selection(use_sim: bool, sim_platform: str | None) -> bool:
    """The single sim-time policy (D1), decided without the registry.

    A simulated selection whose platform is not ``sdk`` is a physics
    selection: a simulation clock exists, every provider node and every
    bound consumer uses sim time, peripherals are filtered by
    ``simulation.<name>.enabled`` and the FAST-LIO-derived scan rule applies.
    Hardware and the SDK platform are not physics selections.
    """
    platform = str(sim_platform or "")
    return bool(use_sim) and platform != "" and platform != SDK_PLATFORM


#: Non-platform keys of a peripheral's ``simulation:`` block (D2). Every
#: other key is a platform section and keeps today's checks (a mapping,
#: never null, ``enabled`` a bool).
PERIPHERAL_SIMULATION_RESERVED_KEYS = ("embedded_sensor",)


class SimPlatformError(ValueError):
    """A simulation platform is not installed, or a registration is invalid."""


#: First-party platform names → the package that provides them, for the
#: install hint in "not installed" errors (D2). Unknown names get a generic
#: hint.
_FIRST_PARTY_PACKAGES = {
    "gazebo": "robot_runtime_sim",
    "mujoco": "robot_runtime_sim",
}


def _resolve_reference(reference: str):
    """Resolve a ``module:attr`` reference (importing the module)."""
    module_name, separator, attribute = str(reference).partition(":")
    attribute = attribute.partition("[")[0].strip()
    if not separator or not module_name or not attribute:
        raise SimPlatformError(f"invalid module:attr reference {reference!r}")
    try:
        module = import_module(module_name)
    except Exception as exc:
        raise SimPlatformError(f"reference {reference!r} failed to import module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise SimPlatformError(
            f"reference {reference!r} names a missing attribute {module_name!r}:{attribute}"
        ) from exc


@dataclass(frozen=True)
class SimSensorTraits:
    """The platform's declarative sensor surface rewrite (D4).

    Default traits apply no platform-specific rewrite to the effective
    peripheral surface: no stream removal, no forced encodings, dimensions,
    rates or QoS, and lidars are advertised as declared
    (``lidar_sim_drivers=None``). Gazebo's and MuJoCo's traits reproduce the
    interface description's historical per-platform branches exactly; the
    golden snapshot is the acceptance bar.

    The ``module:attr`` fields (``camera_rate``, ``imu_rate``,
    ``lidar_points_rate``) name pure callables resolved lazily on every use
    (the monkeypatch-seam convention). ``None`` means "no override".
    """

    #: Pointcloud streams the platform cannot produce are not advertised.
    drop_pointcloud: bool = False
    #: The depth streams advertise the rendered camera's dimensions (the
    #: colour dimensions), never the device's depth_* overrides.
    depth_dims_follow_color: bool = False
    #: The depth streams advertise the colour rate, never depth_fps.
    depth_fps_follow_color: bool = False
    #: Stream → encoding the platform always publishes (gazebo: depth
    #: ``32FC1``; mujoco: colour ``rgb8`` and depth ``32FC1``).
    forced_encodings: dict[str, str] = field(default_factory=dict)
    #: The depth image shares the colour intrinsics: no dedicated depth
    #: camera_info interface (gazebo's rgbd_camera publishes none).
    shared_depth_camera_info: bool = False
    #: The platform's camera publishers are reliable.
    camera_qos_reliable: bool = False
    #: ``module:attr`` of ``camera_rate(profile) -> float | None`` — the one
    #: rate every camera stream advertises (mujoco renders one rate).
    camera_rate: str | None = None
    #: Lidar drivers the platform stands in for; ``None`` advertises every
    #: declared lidar as declared. A declared driver the platform does not
    #: simulate is skipped (neither simulated nor advertised).
    lidar_sim_drivers: tuple[str, ...] | None = None
    #: ``module:attr`` of ``points_rate(periph) -> float | None`` overriding
    #: the lidar points rate (mujoco's ``simulation.mujoco.update_rate``).
    lidar_points_rate: str | None = None
    #: Lidar drivers whose integrated IMU the platform simulates; the IMU
    #: interface is advertised for these (and its rate read via ``imu_rate``).
    integrated_imu_drivers: tuple[str, ...] = ()
    #: ``module:attr`` of ``imu_rate(profile, peripherals) -> float | None`` —
    #: the rate the integrated IMU produces.
    imu_rate: str | None = None

    def resolve_camera_rate(self, profile: dict) -> float | None:
        """The effective single camera rate, or None (lazy ``module:attr``)."""
        if self.camera_rate is None:
            return None
        return _resolve_reference(self.camera_rate)(profile)

    def resolve_imu_rate(self, profile: dict, peripherals: list | None) -> float | None:
        """The integrated-IMU rate, or None (lazy ``module:attr``)."""
        if self.imu_rate is None:
            return None
        return _resolve_reference(self.imu_rate)(profile, peripherals)

    def resolve_lidar_points_rate(self, periph: dict) -> float | None:
        """The lidar points-rate override, or None (lazy ``module:attr``)."""
        if self.lidar_points_rate is None:
            return None
        return _resolve_reference(self.lidar_points_rate)(periph)


@dataclass(frozen=True)
class SimWorldStartup:
    """A world hook's result (D5): the actions to start plus the world id.

    ``world`` is the platform hand-off identifier the provider receives as
    ``sim_world`` (the gz world instance name on Gazebo, the scene name on
    MuJoCo).
    """

    actions: list = field(default_factory=list)
    world: str = ""


@runtime_checkable
class SimEmbodiment(Protocol):
    """A platform's robot embodiment (D3): description + launch actions.

    A robot may replace the platform's default embodiment per profile with
    ``simulation.<platform>.embodiment: "module:attr"``; an embodiment then
    needs no ros2_control. Implementations decide whether to reuse the core
    launch helpers (spawners, the facade node builder, absent-frame TFs).
    """

    def render_description(
        self,
        profile: dict,
        profile_path: Any,
        *,
        peripherals: list | None = None,
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        world: str = "",
    ) -> str: ...

    def actions(
        self,
        profile: dict,
        profile_path: Any,
        peripherals_file: str = "",
        instance_id: str = "",
        *,
        robot_description: str | None = None,
        interface_description: dict | None = None,
        world: str = "",
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> list: ...


@dataclass(frozen=True)
class SimPlatform:
    """One registered physics simulation platform (D1).

    The module an entry point names for the spec must stay import-light
    (stdlib plus this module's types): building an interface description
    imports it, so it must never import ``launch``, a simulator package or
    ``sim_models``. Heavy references (the embodiment, the world hook, the
    scene-task transport, the trait callables) are ``module:attr`` strings
    resolved lazily.
    """

    name: str
    #: The ament package to name in "not installed" errors.
    package: str
    #: ``module:attr`` of the default :class:`SimEmbodiment`.
    embodiment: str
    #: ``module:attr`` of the generic-launch world hook
    #: ``start_world(robot_config, *, scene, spawn_pose) -> SimWorldStartup``,
    #: or None: generic launch starts nothing world-side and the default
    #: world resolution applies (D5).
    world: str | None = None
    #: ``module:attr`` of the scene-task transport
    #: ``scene_task_transport(scene, startup, robot_config, parameters) -> list``,
    #: or None: scene tasks are not offered on this platform (D5).
    scene_task: str | None = None
    #: The platform's declarative sensor traits (D4).
    sensor_traits: SimSensorTraits = field(default_factory=SimSensorTraits)

    def __post_init__(self) -> None:
        if not is_valid_sim_platform_name(self.name):
            raise SimPlatformError(
                f"simulation platform name {self.name!r} must match {SIM_PLATFORM_NAME_PATTERN.pattern}"
            )
        if self.name == SDK_PLATFORM:
            raise SimPlatformError(f"{SDK_PLATFORM!r} is built in and cannot be registered as a sim platform")
        for label, reference in (("embodiment", self.embodiment),):
            if not isinstance(reference, str) or not reference:
                raise SimPlatformError(f"simulation platform {self.name!r} needs a 'module:attr' {label}")
        if not isinstance(self.sensor_traits, SimSensorTraits):
            raise SimPlatformError(
                f"simulation platform {self.name!r} sensor_traits must be a SimSensorTraits instance"
            )


_DISCOVERED_PLATFORMS: dict[str, SimPlatform] | None = None


def _entry_point_platforms(entry_points) -> list[tuple[str, str, SimPlatform]]:
    """Resolve declared entry points into validated (name, value, spec) triples."""
    entries = list(entry_points)
    seen: dict[str, str] = {}
    for entry in entries:
        name = getattr(entry, "name", None)
        value = getattr(entry, "value", None)
        if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
            raise SimPlatformError(f"invalid sim-platform entry point {entry!r}; expected name and value strings")
        if name in seen:
            raise SimPlatformError(
                f"duplicate sim-platform entry point name {name!r}: declared twice "
                f"(values {seen[name]!r} and {value!r}); entry point names must be unique"
            )
        if name == SDK_PLATFORM:
            raise SimPlatformError(
                f"sim-platform entry point {SDK_PLATFORM!r} ({value!r}) is invalid: "
                f"{SDK_PLATFORM!r} is built in and cannot be registered"
            )
        if name in PERIPHERAL_SIMULATION_RESERVED_KEYS:
            raise SimPlatformError(
                f"sim-platform entry point {name!r} ({value!r}) is invalid: {name!r} is a reserved "
                "non-platform key of a peripheral 'simulation' block"
            )
        seen[name] = value

    resolved = []
    for entry in sorted(entries, key=lambda entry: (str(entry.name), str(entry.value))):
        name, value = str(entry.name), str(entry.value)
        module_name, separator, attribute = value.partition(":")
        attribute = attribute.partition("[")[0].strip()
        if not separator or not module_name or not attribute:
            raise SimPlatformError(
                f"sim-platform entry point {name!r} has invalid value {value!r}; expected 'module:attribute'"
            )
        try:
            module = import_module(module_name)
        except Exception as exc:
            raise SimPlatformError(
                f"sim-platform entry point {name!r} ({value!r}) failed to import module {module_name!r}: {exc}"
            ) from exc
        try:
            spec = getattr(module, attribute)
        except AttributeError as exc:
            raise SimPlatformError(
                f"sim-platform entry point {name!r} ({value!r}) resolved to a missing attribute "
                f"{module_name!r}:{attribute}"
            ) from exc
        if not isinstance(spec, SimPlatform):
            raise SimPlatformError(
                f"sim-platform entry point {name!r} ({value!r}) must resolve to a SimPlatform, "
                f"got {type(spec).__name__}"
            )
        if name != spec.name:
            raise SimPlatformError(
                f"sim-platform entry point {name!r} resolved to a SimPlatform named {spec.name!r}; "
                "the entry point name must equal the platform name"
            )
        resolved.append((name, value, spec))
    return resolved


def discover_sim_platforms(*, entry_points=None) -> dict[str, SimPlatform]:
    """Every registered physics platform, keyed by name (D1), cached.

    ``entry_points`` is the test injection seam; ``None`` discovers the
    ``ibrobot.sim_platforms`` group through :mod:`importlib.metadata`.
    Registration errors (duplicate names, ``sdk``, unresolvable references)
    are raised as :class:`SimPlatformError` naming the entry point and its
    value. The result is cached per process; tests clear it with
    :func:`clear_sim_platform_cache` or inject ``entry_points=`` (which
    bypasses and does not pollute the cache).
    """
    global _DISCOVERED_PLATFORMS
    if entry_points is None and _DISCOVERED_PLATFORMS is not None:
        return dict(_DISCOVERED_PLATFORMS)
    if entry_points is None:
        entry_points = _discover_entry_points(group=SIM_PLATFORM_ENTRY_POINT_GROUP)
    platforms: dict[str, SimPlatform] = {}
    for name, _value, spec in _entry_point_platforms(entry_points):
        platforms[name] = spec
    if entry_points is None:
        _DISCOVERED_PLATFORMS = dict(platforms)
    return platforms


def clear_sim_platform_cache() -> None:
    """Drop the discovery cache (tests mutate registrations)."""
    global _DISCOVERED_PLATFORMS
    _DISCOVERED_PLATFORMS = None


def registered_sim_platforms() -> dict[str, SimPlatform]:
    """The registered physics platforms (discovered, cached, without ``sdk``)."""
    return discover_sim_platforms()


def sim_platform_names() -> list[str]:
    """Every usable platform name: ``sdk`` plus the registered ones, sorted."""
    return [SDK_PLATFORM, *sorted(discover_sim_platforms())]


def get_sim_platform(name: str) -> SimPlatform:
    """The registered platform ``name``, or :class:`SimPlatformError` (D2).

    The error names the platform, the entry-point group and the package to
    install (first-party names carry their package; anything else gets the
    generic hint), so selecting a platform whose plugin is not installed
    fails before any process starts.
    """
    platform = str(name or "")
    registered = discover_sim_platforms()
    if platform not in registered:
        package = _FIRST_PARTY_PACKAGES.get(platform, f"the package that provides the {platform!r} sim platform")
        raise SimPlatformError(
            f"simulation platform {platform!r} is not installed: no {SIM_PLATFORM_ENTRY_POINT_GROUP!r} entry point "
            f"named {platform!r} (install {package})"
        )
    return registered[platform]


def resolve_sim_embodiment(profile: dict, profile_path: Any, platform: SimPlatform) -> SimEmbodiment:
    """The effective embodiment for ``platform`` on this profile (D3).

    ``profile["simulation"][platform.name]["embodiment"]``, when set, must be
    a ``module:attr`` string and replaces the platform's default embodiment
    (a robot without ros2_control supplies its own). Resolution is lazy, on
    every call (the monkeypatch-seam convention); a failing import or a
    missing attribute raises :class:`SimPlatformError` naming the profile
    path, the platform and the reference.
    """
    override = ((profile.get("simulation") or {}).get(platform.name) or {}).get("embodiment")
    reference = str(override).strip() if override else ""
    if not reference:
        reference = platform.embodiment
    try:
        embodiment = _resolve_reference(reference)
    except SimPlatformError as exc:
        raise SimPlatformError(
            f"sim platform {platform.name!r} embodiment {reference!r} of profile {profile_path} is unresolvable: {exc}"
        ) from exc
    if not isinstance(embodiment, SimEmbodiment):
        raise SimPlatformError(
            f"sim platform {platform.name!r} embodiment {reference!r} of profile {profile_path} "
            f"does not implement the SimEmbodiment protocol"
        )
    return embodiment


def validate_embodiment_reference(profile: dict, source: Any = "<profile>") -> None:
    """Load-time check of a declared embodiment override: string form only (D3).

    Registration and importability are checked when the platform is
    selected, so a hardware-only host still loads a profile that declares
    ``simulation.<platform>.embodiment`` for a platform it has not installed.
    """
    simulation = profile.get("simulation")
    if not isinstance(simulation, dict):
        return
    for key, section in simulation.items():
        if key in PERIPHERAL_SIMULATION_RESERVED_KEYS or not isinstance(section, dict):
            continue
        override = section.get("embodiment")
        if override is None:
            continue
        if not isinstance(override, str) or not override.strip() or ":" not in override:
            raise SimPlatformError(
                f"runtime profile simulation.{key}.embodiment must be a 'module:attr' reference, "
                f"got {override!r}: {source}"
            )
