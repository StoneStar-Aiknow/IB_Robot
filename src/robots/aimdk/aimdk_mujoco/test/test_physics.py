"""Numerical regressions use a tiny generated model, not private X2 assets."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from aimdk_mujoco import physics


@pytest.fixture
def sim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    joints = "".join(
        f'<body pos="0 0 .01"><joint name="{name}"/><geom size=".01" mass=".01"/></body>'
        for name in physics.JOINT_NAMES
    )
    motors = "".join(f'<motor name="motor_{name}" joint="{name}"/>' for name in physics.JOINT_NAMES)
    xml = (
        '<mujoco><option gravity="0 0 0"/><worldbody><body name="pelvis">'
        '<freejoint name="floating_base_joint"/>'
        '<inertial pos=".02 0 .01" mass="5" diaginertia="1 1 1"/>'
        '<site name="imu"/>' + joints + "</body></worldbody><actuator>" + motors + "</actuator>"
        '<sensor><gyro name="body-angular-velocity" site="imu"/></sensor></mujoco>'
    )
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    (asset_dir / "scene.xml").write_text(xml)
    policy = asset_dir / "policy.onnx"
    policy.write_bytes(b"test-session-stub")
    (asset_dir / "assets.json").write_text(
        json.dumps({"files": {"scene.xml": physics.digest(asset_dir / "scene.xml")}})
    )
    monkeypatch.setattr(physics, "POLICY_SHA256", physics.digest(policy))
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="obs", shape=[1, 500])]
    session.get_outputs.return_value = [SimpleNamespace(name="actions", shape=[1, 29])]
    session.run.return_value = [np.zeros((1, 29), dtype=np.float32)]
    monkeypatch.setattr(physics.ort, "InferenceSession", Mock(return_value=session))
    config = tmp_path / "controller.yaml"
    config.write_text("controller: mc_rl_velocity\npolicy: policy.onnx\n")
    return physics.X2Simulation(asset_dir, config)


def test_observation_contract_preserves_command_order_and_float32(sim):
    command = (0.5, 0.3, 0.2)
    sim.step(command)
    assert sim.data.time == pytest.approx(0.52)
    obs = sim.session.run.call_args.args[1]["obs"]
    assert obs.shape == (1, 500)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[0, 42:45], command)


@pytest.fixture
def bmimic_sim(sim, tmp_path):
    config = tmp_path / "dance.yaml"
    config.write_text(
        "controller: bmimic\nmotions:\n  test:\n    policy: policy.onnx\n    start_count: 1\n    end_count: 3\n"
    )
    session = physics.ort.InferenceSession.return_value
    session.get_inputs.return_value = [
        SimpleNamespace(name="time_step", shape=[1, 1], type="tensor(float)"),
        SimpleNamespace(name="obs", shape=[1, 151], type="tensor(float)"),
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name=name, shape=[1, 29], type="tensor(float)")
        for name in ("joint_vel", "actions", "joint_pos")
    ]
    names = list(reversed([name for name in physics.JOINT_NAMES if not name.startswith("head_")]))
    metadata = {
        "observation_names": "command,projected_gravity,base_ang_vel,joint_pos,joint_vel,actions",
        "command_names": "motion",
        "joint_names": ",".join(names),
        "default_joint_pos": ",".join(["0.1"] * 29),
        "joint_stiffness": ",".join(["80"] * 29),
        "joint_damping": ",".join(["3"] * 29),
        "action_scale": ",".join(["0.25"] * 29),
        "time_step_total": "4",
    }
    session.get_modelmeta.return_value = SimpleNamespace(custom_metadata_map=metadata)

    def run(outputs, _feeds):
        return [np.zeros((1, 29), dtype=np.float32) for _ in outputs]

    session.run.side_effect = run
    configured = physics.X2Simulation(sim.asset_dir, config, "test")
    configured.model.dof_armature[configured.qvel_indices] = 0.1
    configured.config_path = config
    return configured


def test_bmimic_handoff_rejects_large_pose_without_mutating_live_state(bmimic_sim, monkeypatch):
    candidate = bmimic_sim.prepare_policy(bmimic_sim.config_path, "test")
    bmimic_sim.pause_execution()
    monkeypatch.setattr(bmimic_sim, "is_standing", lambda: True)
    bmimic_sim.data.qpos[bmimic_sim.qpos_indices] = candidate.default
    bmimic_sim.data.qvel[:] = 0
    bmimic_sim.data.qpos[bmimic_sim.qpos_indices[0]] += 0.51
    positions, velocities = bmimic_sim.data.qpos.copy(), bmimic_sim.data.qvel.copy()
    now = bmimic_sim.data.time
    with pytest.raises(ValueError, match="handoff"):
        bmimic_sim.activate_policy(candidate, transition=True)
    assert bmimic_sim.session is None and bmimic_sim.selection is None
    assert not bmimic_sim.execution_running and bmimic_sim.data.time == now
    np.testing.assert_array_equal(bmimic_sim.data.qpos, positions)
    np.testing.assert_array_equal(bmimic_sim.data.qvel, velocities)
    assert candidate.motion.frame == 1 and not candidate.motion.finished
