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

"""Wheel odometry must be continuous across the encoder wrap.

Wheel-mode joint positions are raw encoder angles in [0, 2*pi): they wrap once
per revolution. Differencing them directly turned every wrap crossing into a
spurious full turn, so the pose jumped once per wheel revolution. The joint
state callback is driven directly, without a ROS context, with a wheel sequence
that crosses the wrap point while the base drives straight forward.
"""

import math
import threading
import types

import lekiwi_sdk_py as lk

from lekiwi_robot.base_node import LeKiwiBaseNode

WHEELS = ["7", "8", "9"]


def make_node() -> tuple[LeKiwiBaseNode, list[float]]:
    node = object.__new__(LeKiwiBaseNode)
    node._lock = threading.Lock()
    node._wheel_joints = WHEELS
    node._geometry = lk.BaseGeometry()
    node._geometry.wheel_radius = 0.05
    node._geometry.base_radius = 0.125
    node._geometry.max_wheel_radps = 10.0
    node._pose = lk.BasePose()
    node._last_wheel_positions = None
    node._last_wheel_velocities = [0.0, 0.0, 0.0]
    xs: list[float] = []
    node._publish_odom = lambda stamp: xs.append(node._pose.x)
    return node, xs


def joint_state(positions: list[float]):
    return types.SimpleNamespace(
        name=WHEELS,
        position=[p % (2.0 * math.pi) for p in positions],
        velocity=[],
        header=types.SimpleNamespace(stamp=None),
    )


def test_odometry_is_continuous_across_the_encoder_wrap():
    node, xs = make_node()
    step = lk.body_to_wheel_velocities(0.2, 0.0, 0.0, node._geometry)
    # Start just below the wrap point, then 3 s at 50 Hz: every wheel crosses it.
    angles = [2.0 * math.pi - 0.02] * 3
    for _ in range(151):
        node._on_joint_states(joint_state(angles))
        angles = [a + w * 0.02 for a, w in zip(angles, step, strict=True)]

    steps = [b - a for a, b in zip(xs, xs[1:], strict=False)]
    assert all(0.0 < d < 0.01 for d in steps), f"odometry jumped: {[d for d in steps if not 0.0 < d < 0.01]}"
    assert math.isclose(xs[-1], 0.6, abs_tol=1e-6), xs[-1]
