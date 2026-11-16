"""Every motion endpoint a runtime profile advertises must be one the motion
server actually registers (revision-2 E2E bring-up finding).

so101_motion's motion server registers, per its ``ik_worker_namespaces``:
the global ``/motion/compute_fk``, ``/motion/compute_ik``,
``/motion/move_to_joint`` and ``/motion/move_to_pose``, plus one
``/<namespace>/motion/compute_ik`` per isolated IK worker — and **no**
per-worker FK (the worker loop builds ComputeIk services only,
motion_server.py). A profile capability that advertises a per-worker FK
endpoint therefore advertises a service no component serves: the pick
executor binds it from the live interface description and fails with
SERVICE_UNAVAILABLE at pick start (revision-2 E2E bring-up, 2026-11-01;
maintainer decision: fix the profile, as so101_single_arm).

The registered endpoints are DERIVED from the motion server's own declared
registration function (``motion_server.registered_motion_service_endpoints``,
loaded from the installed program — the same function ``__init__`` registers
through), never a hand-written list. Covers lekiwi_nav_grasp and
so101_single_arm from their shares, plus a negative fixture.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory


def _motion_server_module():
    """The so101_motion server module, for its declared registration function.

    Loaded from the installed program (``lib/so101_motion/motion_server.py``
    — a symlink at the source script under ``--symlink-install``), with the
    source tree as the fallback for an unbuilt checkout. The served endpoints
    must come from the server's own declaration — never a list copied here.
    """
    candidates = [
        Path(get_package_share_directory("so101_motion")).parent.parent / "lib" / "so101_motion" / "motion_server.py"
    ]
    candidates.append(Path(__file__).resolve().parents[4] / "so101" / "so101_motion" / "scripts" / "motion_server.py")
    script = next((path for path in candidates if path.is_file()), None)
    assert script is not None, f"no motion_server.py program found; looked in: {[str(p) for p in candidates]}"
    spec = importlib.util.spec_from_file_location("so101_motion_server_module", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registered_endpoints(worker_namespaces: set[str]) -> set[str]:
    """The endpoints the motion server registers for a worker set, from its
    own declared registration function (the single source)."""
    return set(_motion_server_module().registered_motion_service_endpoints(worker_namespaces))


def _unsupported_motion_endpoints(capabilities: dict) -> dict[str, list[str]]:
    """The motion endpoints a profile's capabilities advertise beyond the
    motion server's registrations for its own worker set."""
    fk = capabilities.get("motion.fk", {}).get("endpoints") or []
    ik = capabilities.get("motion.ik", {}).get("endpoints") or []
    worker_namespaces: set[str] = set()
    for endpoint in ik:
        if endpoint == "/motion/compute_ik":
            worker_namespaces.add("")  # the primary move_group's own namespace
        elif endpoint.startswith("/") and endpoint.endswith("/motion/compute_ik"):
            worker_namespaces.add(endpoint.split("/")[1])
    registered = _registered_endpoints(worker_namespaces)
    unsupported: dict[str, list[str]] = {}
    for capability, endpoints in (("motion.fk", fk), ("motion.ik", ik)):
        missing = [endpoint for endpoint in endpoints if endpoint not in registered]
        if missing:
            unsupported[capability] = missing
    return unsupported


def _capabilities(path: Path) -> dict:
    """The profile's capabilities (lekiwi profiles nest under ``robot``;
    so101 profiles are flat)."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "capabilities" in document:
        return document["capabilities"]
    return document["robot"]["capabilities"]


@pytest.mark.parametrize(
    "package_name, profile_name",
    [
        ("lekiwi_robot", "lekiwi_nav_grasp"),
        ("so101_robot", "so101_single_arm"),
    ],
)
def test_advertised_motion_endpoints_are_served(package_name: str, profile_name: str) -> None:
    capabilities = _capabilities(Path(get_package_share_directory(package_name)) / "profiles" / f"{profile_name}.yaml")
    unsupported = _unsupported_motion_endpoints(capabilities)
    assert not unsupported, (
        f"{profile_name} advertises motion endpoints the motion server never registers "
        f"(per-worker FK has no server): {unsupported}"
    )


def test_the_registered_endpoints_are_exactly_the_global_services_plus_per_worker_ik():
    """The server's own declaration (the ground truth of this audit): the
    three global /motion services from the robot_runtime contract plus the
    primary compute_ik (the ``""`` worker namespace — the primary
    move_group) and one compute_ik per extra worker namespace — no
    per-worker FK anywhere."""
    assert _registered_endpoints({""}) == {
        "/motion/compute_fk",
        "/motion/compute_ik",
        "/motion/move_to_joint",
        "/motion/move_to_pose",
    }
    assert _registered_endpoints({"", "ik_worker_0"}) == {
        "/motion/compute_fk",
        "/motion/compute_ik",
        "/motion/move_to_joint",
        "/motion/move_to_pose",
        "/ik_worker_0/motion/compute_ik",
    }
    assert _registered_endpoints({"", "ik_worker_0", "ik_worker_1"}) == {
        "/motion/compute_fk",
        "/motion/compute_ik",
        "/motion/move_to_joint",
        "/motion/move_to_pose",
        "/ik_worker_0/motion/compute_ik",
        "/ik_worker_1/motion/compute_ik",
    }


def test_negative_a_per_worker_fk_advertisement_is_flagged() -> None:
    capabilities = {
        "motion.fk": {"endpoints": ["/motion/compute_fk", "/ik_worker_0/motion/compute_fk"]},
        "motion.ik": {"endpoints": ["/motion/compute_ik", "/ik_worker_0/motion/compute_ik"]},
    }
    unsupported = _unsupported_motion_endpoints(capabilities)
    assert unsupported["motion.fk"] == ["/ik_worker_0/motion/compute_fk"]
    assert "motion.ik" not in unsupported
