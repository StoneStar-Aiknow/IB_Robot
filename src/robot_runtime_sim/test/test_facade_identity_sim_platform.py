"""The facade identity check across a real physics platform (D6).

Moved from ``robot_runtime/test/test_facade.py``: building the gazebo
description resolves the platform's sensor traits through the registry, so
the plugin must be installed where this runs.
"""

from __future__ import annotations

import pytest


def test_facade_identity_check_covers_sim_platform():
    from robot_runtime.facade_node import check_description_identity
    from robot_runtime.interface_description import build_description

    profile = {
        "runtime": {"name": "facade_test", "version": "0.0.1"},
        "controller_manager": "controller_manager",
        "modes": {"initial": "idle", "idle": {"controllers": [], "transitions": []}},
        "capabilities": {
            "joint.state": {"joint_count": 1, "rate_hz": 50.0},
            "joint.trajectory": {"joint_count": 1},
            "runtime.stop": {"cancel_bound_s": 2.0, "idle_bound_s": 3.0, "torque_off_bound_s": 4.0},
        },
        "joints": ["1"],
        "use_sim": True,
    }
    description = build_description(profile, use_sim=True, sim_platform="sdk")
    check_description_identity(profile, description, True, "sdk")
    gazebo = build_description(profile, use_sim=True, sim_platform="gazebo")
    check_description_identity(profile, gazebo, True, "gazebo")
    with pytest.raises(ValueError, match="does not match"):
        check_description_identity(profile, gazebo, True, "sdk")
    # pre-platform snapshots default from execution and still match the sdk facade
    legacy = build_description(profile, use_sim=True, sim_platform="sdk")
    del legacy["sim_platform"]
    check_description_identity(profile, legacy, True, "sdk")


def test_facade_selection_resolves_a_registered_platform():
    """The explicit-platform success path, needing the installed plugin (D6).

    Moved from ``test_facade.py``: 'gazebo' resolves only where its entry
    point is installed; the core copy keeps the sdk/undeclared/unregistered
    branches that never consult the registry on success.
    """
    from robot_runtime.facade_node import resolve_facade_selection

    profile = {"runtime": {"name": "t", "version": "0"}, "use_sim": False, "sim_platforms": ["sdk", "gazebo"]}
    assert resolve_facade_selection(profile, True, "gazebo") == (True, "gazebo")
