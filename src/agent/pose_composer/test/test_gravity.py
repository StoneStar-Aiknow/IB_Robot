"""Static load is gravity, independent of the contact solver's reaction force."""

from types import SimpleNamespace

import mujoco
import pytest
from aimdk_robot.pose_backend import safety


@pytest.mark.parametrize("ground_z", [0.0, 0.96])
def test_gravity_share_matches_analytic_pendulum_with_or_without_contact(monkeypatch, ground_z):
    m = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <geom type="plane" size="2 2 .1" pos="0 0 {ground_z}"/>
      <body pos="0 0 1">
        <joint name="left_shoulder_pitch_joint" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".1" pos=".5 0 0" mass="1"/>
      </body>
    </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    if ground_z:
        assert d.ncon > 0
        assert abs(d.qfrc_inverse[0]) > 100
    robot = SimpleNamespace(model=m, arm_joints=("left_shoulder_pitch_joint",))
    monkeypatch.setattr(safety, "load", lambda: robot)
    # A one-kg mass at a half-metre horizontal lever: mgL = 4.905 Nm,
    # divided by this joint's 36 Nm rated effort. Keep the 30% limit intact.
    assert safety.gravity_share(SimpleNamespace(data=d)) == pytest.approx(4.905 / 36)
