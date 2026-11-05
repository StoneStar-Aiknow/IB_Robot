"""Workstation-only comparison against the model from which the artifact was exported."""

import json
import os

import mujoco
import numpy as np
import pytest
from aimdk_robot.pose_backend import kin, model, safety

from pose_composer.compiled_geometry import CompiledGeometry, separated


def test_compiled_fk_gravity_and_clearance_match_reference():
    path = os.environ.get("IBROBOT_FIXED_LIBRARY")
    if not path:
        pytest.skip("set IBROBOT_FIXED_LIBRARY")
    with open(path) as stream:
        compiled = CompiledGeometry(json.load(stream)["body_geometry"])
    robot = model.load()
    m = robot.model
    ids = sorted({g for a, b, *_ in safety.pairs() for g in (a, b)})
    rng = np.random.default_rng(123)
    for _ in range(8):
        q = {name: value + float(rng.uniform(-0.08, 0.08)) for name, value in robot.rest.items()}
        reference = kin.fk(q)
        shapes, torques = compiled.forward(q)
        for shape, index in zip(shapes, ids, strict=True):
            np.testing.assert_allclose(shape.center, reference.data.geom_xpos[index], atol=1e-8)
        for name in robot.arm_joints:
            assert torques[name] == pytest.approx(reference.data.qfrc_bias[m.joint(name).dofadr[0]], abs=1e-7)
        for a, b, margin, _ in compiled.data["pairs"]:
            if separated(shapes[a], shapes[b], margin):
                distance = mujoco.mj_geomDistance(m, reference.data, ids[a], ids[b], 1.0, np.zeros(6))
                assert distance >= margin - 1e-7
