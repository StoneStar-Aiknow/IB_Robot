from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.planner import LeftArmPlanner

torch = pytest.importorskip("torch")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

# A palm "under a table edge": the slab x >= 0.25, -0.02 <= z <= 0 is solid.
START = np.array([0.3, 0.2, -0.05, 0.0, 0.0, 0.0, 0.0])
END = np.array([0.4, 0.2, 0.05, 0.0, 0.0, 0.0, 0.0])


def palm_poses(positions):
    """Toy kinematics: joints 0-2 are the palm position, 3-5 its rotation vector."""
    q = np.asarray(positions, dtype=float).reshape(-1, 7)
    transforms = np.repeat(np.eye(4)[None], len(q), axis=0)
    transforms[:, :3, :3] = Rotation.from_rotvec(q[:, 3:6]).as_matrix()
    transforms[:, :3, 3] = q[:, :3]
    return transforms


def valid(positions):
    q = np.asarray(positions).reshape(-1, 7)
    return ~((q[:, 0] >= 0.25) & (q[:, 2] >= -0.02) & (q[:, 2] <= 0.0))


def planner(valid_positions=valid, limit=1.0):
    result = object.__new__(LeftArmPlanner)
    result._palm_poses = palm_poses
    result.valid_positions = valid_positions
    limits = torch.tensor([[-limit] * 7, [limit] * 7])
    result.motion = SimpleNamespace(
        kinematics=SimpleNamespace(
            config=SimpleNamespace(kinematics_config=SimpleNamespace(joint_limits=SimpleNamespace(position=limits)))
        )
    )
    return result


def test_cartesian_paths_translate_palm_at_fixed_orientation_and_flag_unreachable():
    origin = np.array([0.1, 0.2, 0.3, 0.2, -0.1, 0.3, 0.0])
    offsets = np.array([(0.0, 0.0, 0.1), (-0.1, 0.0, 0.0), (0.0, 0.0, 0.9)])
    chains, ok = planner()._cartesian_paths(np.stack([origin] * 3), offsets, 5, lambda: None)
    assert chains.shape == (3, 6, 7)
    assert ok.tolist() == [True, True, False]  # the last one runs into the joint limit
    for chain, offset in zip(chains[:2], offsets[:2], strict=False):
        poses = palm_poses(chain)
        assert np.allclose(poses[0, :3, 3], origin[:3])
        assert np.allclose(poses[-1, :3, 3], origin[:3] + offset, atol=1e-3)
        assert np.allclose(poses[:, :3, :3], poses[0, :3, :3], atol=1e-3)
