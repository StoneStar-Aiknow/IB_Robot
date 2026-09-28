# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The base command gate, as a pure predicate.

``_commands_permitted`` is the only thing standing between ``/cmd_vel`` and the
wheels, so it is tested directly rather than through a live node: the gate is
pure state, and a ROS context would only add a spin loop around it. The node is
built with ``object.__new__`` for that reason -- no rclpy init, no publishers.
"""

import pytest

from lekiwi_robot.base_node import LeKiwiBaseNode


def make_gate(**overrides) -> LeKiwiBaseNode:
    node = object.__new__(LeKiwiBaseNode)
    node._stop_latched = False
    node._nav_enabled = True
    node._status_seen = True
    node._active_mode = "base_navigation"
    node._base_modes = {"base_navigation"}
    for key, value in overrides.items():
        setattr(node, f"_{key}", value)
    return node


def test_commands_permitted_when_runtime_allows_the_base():
    assert make_gate()._commands_permitted() is True


def test_commands_blocked_before_the_first_runtime_status():
    # The window between node startup and the first RuntimeStatus. Mode and
    # stop latch are both unknown here, and unknown must not mean allowed.
    assert make_gate(status_seen=False)._commands_permitted() is False


def test_commands_blocked_before_status_even_when_navigation_starts_enabled():
    # A profile may set navigation_enabled_on_startup, which would otherwise
    # leave the base drivable during the pre-reconciliation window.
    gate = make_gate(status_seen=False, nav_enabled=True)
    assert gate._commands_permitted() is False


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"stop_latched": True}, "stop latch"),
        ({"nav_enabled": False}, "navigation gate"),
        ({"active_mode": "idle"}, "mode does not allow base commands"),
        ({"active_mode": ""}, "no active mode"),
    ],
)
def test_commands_blocked(overrides, reason):
    assert make_gate(**overrides)._commands_permitted() is False, reason
