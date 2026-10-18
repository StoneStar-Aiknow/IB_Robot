import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from manipulation_execution.imitation_retargeting.x2.kinematics import MjcfKinematics

MODEL_XML = (Path(__file__).resolve().parents[1] / "manipulation_execution"
             / "imitation_retargeting" / "x2" / "model" / "x2_ultra.xml")
SMALL_GOLDEN = Path(__file__).resolve().parent / "data" / "x2_fk_golden_small.npz"


@pytest.fixture(scope="module")
def golden():
    path = Path(os.environ.get("X2_FK_GOLDEN", SMALL_GOLDEN))
    data = np.load(path)
    assert data["poses"].shape[0] >= 40
    return data


def test_fk_matches_mujoco_golden(golden):
    assert hashlib.sha256(MODEL_XML.read_bytes()).hexdigest() == str(golden["model_xml_sha256"])
    kin = MjcfKinematics(MODEL_XML)
    joints = [str(n) for n in golden["joint_names"]]
    bodies = [str(n) for n in golden["body_names"]]
    assert np.array_equal(np.array([kin.joint_range(n) for n in joints]), golden["limits_rad"])
    for i, q in enumerate(golden["poses"]):
        kin.forward(dict(zip(joints, q)))
        poses = [kin.body_pose(b) for b in bodies]
        np.testing.assert_allclose([kin.anchor(n) for n in joints], golden["anchor"][i], rtol=0, atol=1e-9)
        np.testing.assert_allclose([kin.axis(n) for n in joints], golden["axis"][i], rtol=0, atol=1e-9)
        np.testing.assert_allclose([p for p, _ in poses], golden["body_pos"][i], rtol=0, atol=1e-9)
        np.testing.assert_allclose([r for _, r in poses], golden["body_mat"][i], rtol=0, atol=1e-9)


def test_unsupported_orientation_attribute_raises(tmp_path):
    xml = tmp_path / "bad.xml"
    xml.write_text(
        "<mujoco><compiler angle='radian'/><worldbody>"
        "<body name='a' axisangle='0 0 1 0.5'><joint name='j' axis='0 0 1'/></body>"
        "</worldbody></mujoco>")
    with pytest.raises(ValueError, match="axisangle"):
        MjcfKinematics(xml)
