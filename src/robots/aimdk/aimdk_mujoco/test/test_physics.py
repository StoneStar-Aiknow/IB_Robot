"""Numerical regressions use a tiny generated model, not private X2 assets."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import yaml
from aimdk_mujoco import physics
from aimdk_mujoco.bmimic import BMimic
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, RuntimeException


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


def test_loads_policy_from_assets_only_on_first_step(sim, tmp_path):
    assert sim.session is None
    physics.ort.InferenceSession.assert_not_called()
    sim.step()
    assert sim.session is physics.ort.InferenceSession.return_value
    assert physics.ort.InferenceSession.call_args.args[0] == str(tmp_path / "assets/policy.onnx")
    manifest = json.loads((tmp_path / "assets/assets.json").read_text())
    assert set(manifest["files"]) == {"scene.xml"}


def test_shipped_controller_selects_asset_policy(sim):
    config = Path(__file__).resolve().parents[1] / "config" / "mc_rl_velocity.yaml"
    configured = physics.X2Simulation(sim.asset_dir, config)
    assert configured.session is None
    configured.step()
    assert physics.ort.InferenceSession.call_args.args[0] == str(sim.policy_path)


def test_rejects_changed_policy_on_first_use(sim, tmp_path):
    policy = tmp_path / "assets/policy.onnx"
    policy.write_bytes(b"changed weights")
    physics.ort.InferenceSession.reset_mock()
    with pytest.raises(RuntimeError, match="Unsupported walking policy"):
        sim.step()
    physics.ort.InferenceSession.assert_not_called()
    assert sim.fault


def test_without_policy_model_preview_pauses_and_can_load_added_weights(sim):
    weights = sim.policy_path.read_bytes()
    sim.policy_path.unlink()
    preview = physics.X2Simulation(sim.policy_path.parent, sim.policy_path.parent.parent / "controller.yaml")
    before = preview.data.time
    preview.step()
    assert preview.data.time == before
    physics.ort.InferenceSession.assert_not_called()
    with pytest.raises(RuntimeError, match="place authorized weights"):
        preview.step((0.2, 0.0, 0.0))
    assert preview.fault is None
    assert preview.data.time == before
    preview.policy_path.write_bytes(weights)
    preview.step()
    assert preview.session is not None
    assert preview.data.time == pytest.approx(before + 0.02)


def test_omitted_controller_never_loads_existing_policy(sim):
    preview = physics.X2Simulation(sim.asset_dir)
    before = preview.data.time
    preview.step()
    assert preview.data.time == before
    physics.ort.InferenceSession.assert_not_called()
    with pytest.raises(RuntimeError, match="select controller_config"):
        preview.step((0.2, 0, 0))
    assert preview.fault is None


def test_config_selects_nested_policy_without_renaming(sim, tmp_path):
    target = sim.asset_dir / "rl_models/walker.onnx"
    target.parent.mkdir()
    sim.policy_path.rename(target)
    config = tmp_path / "controller.yaml"
    config.write_text("controller: mc_rl_velocity\npolicy: rl_models/walker.onnx\n")
    configured = physics.X2Simulation(sim.asset_dir, config)
    configured.step()
    assert physics.ort.InferenceSession.call_args.args[0] == str(target)


@pytest.mark.parametrize(
    "config",
    [
        "[]",
        "controller: mc_rl_velocity",
        "controller: foundationwalk\npolicy: a.onnx",
        "controller: mc_rl_velocity\npolicy: /tmp/a.onnx",
        "controller: mc_rl_velocity\npolicy: ../a.onnx",
        "controller: mc_rl_velocity\npolicy: null",
        "controller: mc_rl_velocity\npolicy: .",
        "controller: mc_rl_velocity\npolicy: a.onnx\nunknown: 1",
    ],
)
def test_invalid_controller_config_is_rejected_before_inference(sim, tmp_path, config):
    path = tmp_path / "invalid.yaml"
    path.write_text(config)
    with pytest.raises(ValueError):
        physics.X2Simulation(sim.asset_dir, path)
    physics.ort.InferenceSession.assert_not_called()


def test_policy_symlink_escape_is_rejected_on_load(sim, tmp_path):
    outside = tmp_path / "outside.onnx"
    sim.policy_path.rename(outside)
    sim.policy_path.symlink_to(outside)
    with pytest.raises(RuntimeError, match="within assets"):
        sim.step()
    physics.ort.InferenceSession.assert_not_called()


def test_history_is_oldest_first_per_term_not_frame_major():
    history = physics.ObservationHistory()
    history.append([np.full(dim, index + 1) for index, dim in enumerate(physics.TERM_DIMS)])
    result = history.append([np.full(dim, index + 11) for index, dim in enumerate(physics.TERM_DIMS)])
    offset = 0
    for index, dim in enumerate(physics.TERM_DIMS):
        term = result[0, offset : offset + dim * 5].reshape(5, dim)
        np.testing.assert_array_equal(term[:3], 0)
        np.testing.assert_array_equal(term[3], index + 1)
        np.testing.assert_array_equal(term[4], index + 11)
        offset += dim * 5


@pytest.mark.parametrize("command", [(-0.3, -0.3, -0.2), (0.5, 0.3, 0.2)])
def test_inclusive_velocity_limits_survive_observation_float32(sim, command):
    sim.step(command)
    assert sim.data.time == pytest.approx(0.52)
    obs = sim.session.run.call_args.args[1]["obs"]
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[0, 42:45], command)


@pytest.mark.parametrize("command", [(np.nan, 0, 0), (0.50001, 0, 0), (0, 0, 0.20001), (0, 0)])
def test_invalid_command_never_advances_physics(sim, command):
    before = sim.data.time
    with pytest.raises(ValueError):
        sim.step(command)
    assert sim.data.time == before


@pytest.mark.parametrize("exception", [Fail, InvalidArgument, RuntimeException])
def test_native_inference_failure_latches_without_advancing(sim, exception):
    sim.step()
    before = sim.data.time
    sim.session.run.side_effect = exception("injected inference failure")
    with pytest.raises(RuntimeError, match="Policy inference failed"):
        sim.step()
    sim.session.run.side_effect = None
    with pytest.raises(RuntimeError, match="Policy inference failed"):
        sim.step()
    assert sim.data.time == before


@pytest.mark.parametrize("velocity", [np.nan, 1e20])
def test_numerical_failure_cannot_silently_reset(sim, velocity):
    before = sim.data.time
    sim.data.qvel[sim.qvel_indices[0]] = velocity
    with pytest.raises(RuntimeError, match="Nonfinite|numerical instability"):
        sim._integrate(sim.default, 20)
    assert sim.fault
    assert sim.data.time >= before
    with pytest.raises(RuntimeError):
        sim.step()


def test_ground_truth_velocity_is_at_pelvis_origin_not_center_of_mass(sim):
    sim.data.qvel[:] = 0
    sim.data.qvel[4] = 0.3
    mujoco.mj_forward(sim.model, sim.data)
    truth = sim.ground_truth()
    np.testing.assert_allclose(truth["linear_velocity_world"], 0, atol=1e-10)
    np.testing.assert_allclose(truth["angular_velocity_body"], [0, 0.3, 0], atol=1e-10)


def test_fall_fault_requires_explicit_restart(sim):
    sim.data.qpos[2] = 0.1
    mujoco.mj_forward(sim.model, sim.data)
    before = sim.data.time
    with pytest.raises(RuntimeError, match="fell"):
        sim.step()
    sim.data.qpos[2] = 0.7
    mujoco.mj_forward(sim.model, sim.data)
    with pytest.raises(RuntimeError, match="fell"):
        sim.step()
    assert sim.data.time == before


@pytest.fixture
def dance(sim, tmp_path):
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

    def run(outputs, feeds):
        frame = feeds["time_step"][0, 0]
        return [
            np.full((1, 29), {"joint_pos": frame, "joint_vel": frame + 10, "actions": 0.2}[key], np.float32)
            for key in outputs
        ]

    session.run.side_effect = run
    configured = physics.X2Simulation(sim.asset_dir, config, "test")
    # Tiny test spheres need finite rotor inertia for the metadata's X2 gains.
    configured.model.dof_armature[configured.qvel_indices] = 0.1
    return configured


def test_bmimic_requires_explicit_start_and_pauses_without_rewind(dance, monkeypatch):
    before = dance.data.time
    dance.step()
    assert dance.data.time == before
    physics.ort.InferenceSession.assert_not_called()
    with pytest.raises(ValueError, match="velocity"):
        dance.step((0.2, 0, 0))
    assert dance.fault is None
    dance.start_motion()
    monkeypatch.setattr(dance, "_integrate", Mock())
    dance.step()
    assert dance.motion.frame == 2
    np.testing.assert_array_equal(dance.kp[dance.motion.indices], 80)
    target, substeps = dance._integrate.call_args.args
    assert substeps == 20
    np.testing.assert_allclose(target[dance.motion.indices], 0.15)
    np.testing.assert_array_equal(target[[11, 16]], 0)
    dance.pause_motion()
    dance.step()
    assert dance.policy_steps == 1
    assert dance.motion.frame == 2
    dance.start_motion()
    dance.step()
    dance.step()
    assert dance.policy_steps == 3
    assert dance.motion.finished
    assert not dance.motion_running
    assert "not active balance" in dance.preview_message
    calls = dance.session.run.call_count
    dance.step()
    assert dance.session.run.call_count == calls
    with pytest.raises(ValueError, match="restart"):
        dance.start_motion()


def test_bmimic_uses_current_reference_order_and_unstacked_observation(dance, monkeypatch):
    dance.start_motion()
    monkeypatch.setattr(dance, "_integrate", Mock())
    dance.step()
    obs = dance.session.run.call_args.args[1]["obs"]
    assert obs.shape == (1, 151) and obs.dtype == np.float32
    np.testing.assert_array_equal(obs[0, :29], 1)
    np.testing.assert_array_equal(obs[0, 29:58], 11)
    np.testing.assert_allclose(obs[0, 58:61], [0, 0, -1])
    np.testing.assert_allclose(obs[0, 64:93], dance.data.qpos[dance.qpos_indices][dance.motion.indices] - 0.1)
    np.testing.assert_array_equal(obs[0, 122:], 0)
    dance.step()
    obs = dance.session.run.call_args.args[1]["obs"]
    np.testing.assert_array_equal(obs[0, :29], 2)
    np.testing.assert_allclose(obs[0, 122:], 0.2)


@pytest.mark.parametrize("bad_output", ["reference", "action", "exception"])
def test_bmimic_invalid_inference_latches_before_physics(dance, monkeypatch, bad_output):
    dance.start_motion()
    integrate = Mock()
    monkeypatch.setattr(dance, "_integrate", integrate)
    original = physics.ort.InferenceSession.return_value.run.side_effect

    def run(outputs, feeds):
        if bad_output == "exception":
            raise Fail("injected")
        values = original(outputs, feeds)
        if (outputs == ["actions"]) == (bad_output == "action"):
            values[0][0, 0] = np.nan
        return values

    physics.ort.InferenceSession.return_value.run.side_effect = run
    with pytest.raises(RuntimeError, match="BMIMIC inference failed"):
        dance.step()
    assert dance.fault
    integrate.assert_not_called()
    with pytest.raises(RuntimeError):
        dance.start_motion()


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_names", "command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions"),
        ("joint_names", "wrong"),
        ("default_joint_pos", ",".join(["nan"] * 29)),
        ("action_scale", ",".join(["-1"] * 29)),
        ("time_step_total", "3"),
    ],
)
def test_bmimic_metadata_mismatch_latches_before_inference(dance, field, value):
    session = physics.ort.InferenceSession.return_value
    session.get_modelmeta.return_value.custom_metadata_map[field] = value
    dance.start_motion()
    with pytest.raises(RuntimeError, match="Policy loading failed"):
        dance.step()
    session.run.assert_not_called()


def test_bmimic_wrong_shape_rejected(dance):
    dance.start_motion()
    session = physics.ort.InferenceSession.return_value
    session.get_inputs.return_value[1].shape = [1, 154]
    with pytest.raises(RuntimeError, match="151"):
        dance.step()
    session.run.assert_not_called()


def test_bmimic_missing_weights_not_started_and_no_hash_gate(dance):
    dance.policy_path.unlink()
    with pytest.raises(ValueError, match="authorized weights"):
        dance.start_motion()
    assert not dance.motion_running
    dance.policy_path.write_bytes(b"replacement weights with compatible session")
    dance.start_motion()
    dance.step()
    assert dance.session is not None and dance.policy_steps == 1


def test_bmimic_scales_and_clips_without_velocity_tracking(dance):
    session = physics.ort.InferenceSession.return_value
    controller = BMimic(session, dance.joint_names, 1, 3)
    original = session.run.side_effect

    def run(outputs, feeds):
        if outputs == ["actions"]:
            return [np.full((1, 29), 200, dtype=np.float32)]
        return original(outputs, feeds)

    session.run.side_effect = run
    result = controller.target(np.ones(31), np.full(31, 40), np.array([0, 0, -1]), np.array([4, 8, 12]))
    np.testing.assert_allclose(result, 25.1)
    obs = session.run.call_args.args[1]["obs"][0]
    np.testing.assert_allclose(obs[61:64], [1, 2, 3])
    np.testing.assert_allclose(obs[64:93], 0.9)
    np.testing.assert_allclose(obs[93:122], 2)


def test_bmimic_pause_freezes_actual_physics_and_resumes_without_reset(dance):
    dance.start_motion()
    dance.step()
    time = dance.data.time
    positions = dance.data.qpos.copy()
    velocities = dance.data.qvel.copy()
    previous_action = dance.motion.last_action.copy()
    assert np.any(velocities)
    dance.pause_motion()
    count = dance.session.run.call_count
    for _ in range(5):
        dance.step()
    assert dance.data.time == time
    np.testing.assert_array_equal(dance.data.qpos, positions)
    np.testing.assert_array_equal(dance.data.qvel, velocities)
    np.testing.assert_array_equal(dance.motion.last_action, previous_action)
    assert dance.session.run.call_count == count
    assert dance.motion.frame == 2
    dance.start_motion()
    dance.step()
    assert dance.data.time == pytest.approx(time + 0.02)
    assert dance.motion.frame == 3
    np.testing.assert_allclose(dance.session.run.call_args.args[1]["obs"][0, 122:], previous_action)


def test_bmimic_single_frame_phase(dance):
    dance.controller_options["end_count"] = 1
    dance.start_motion()
    dance.step()
    assert dance.policy_steps == 1
    assert dance.motion.finished and dance.motion.frame == 1
    assert not dance.motion_running


@pytest.mark.parametrize(
    "replacement",
    [
        "start_count: true\nend_count: 3",
        "start_count: 1.5\nend_count: 3",
        "start_count: -1\nend_count: 3",
        "start_count: 3\nend_count: 1",
        "start_count: 1\nend_count: null",
        "start_count: 1\nend_count: 3\nextra: 1",
    ],
)
def test_bmimic_rejects_invalid_phase_schema(dance, tmp_path, replacement):
    config = tmp_path / "invalid-dance.yaml"
    config.write_text(
        "controller: bmimic\nmotions:\n  test:\n    policy: policy.onnx\n"
        + "".join(f"    {line}\n" for line in replacement.splitlines())
    )
    with pytest.raises(ValueError):
        physics.X2Simulation(dance.asset_dir, config, "test")
    physics.ort.InferenceSession.assert_not_called()


def test_bmimic_missing_weights_can_be_added_before_explicit_start(dance):
    weights = dance.policy_path.read_bytes()
    dance.policy_path.unlink()
    dance.step()
    assert dance.session is None
    dance.policy_path.write_bytes(weights)
    dance.step()
    assert dance.session is None
    dance.start_motion()
    dance.step()
    assert dance.policy_steps == 1


@pytest.fixture
def policy_configs(dance, tmp_path, monkeypatch):
    motion_config = tmp_path / "dance.yaml"
    motion_config.write_text(
        yaml.safe_dump(
            {
                "controller": "bmimic",
                "motions": {
                    "test": dict(dance.controller_options),
                    "other": {"policy": "other.onnx", "start_count": 0, "end_count": 2},
                },
            }
        )
    )
    (dance.asset_dir / "other.onnx").write_bytes(b"second motion session stub")
    velocity_policy = dance.asset_dir / "velocity.onnx"
    velocity_policy.write_bytes(dance.policy_path.read_bytes())
    velocity_config = tmp_path / "velocity.yaml"
    velocity_config.write_text("controller: mc_rl_velocity\npolicy: velocity.onnx\n")
    first = physics.ort.InferenceSession.return_value
    second = Mock()
    second.get_inputs.return_value = first.get_inputs.return_value
    second.get_outputs.return_value = first.get_outputs.return_value
    metadata = dict(first.get_modelmeta.return_value.custom_metadata_map)
    metadata.update(
        default_joint_pos=",".join(["-0.1"] * 29),
        joint_stiffness=",".join(["60"] * 29),
        joint_damping=",".join(["2"] * 29),
    )
    second.get_modelmeta.return_value = SimpleNamespace(custom_metadata_map=metadata)
    second.run.side_effect = first.run.side_effect
    velocity = Mock()
    velocity.get_inputs.return_value = [SimpleNamespace(name="obs", shape=[1, 500])]
    velocity.get_outputs.return_value = [SimpleNamespace(name="actions", shape=[1, 29])]
    velocity.run.side_effect = lambda *_: [np.full((1, 29), 0.2, dtype=np.float32)]
    sessions = {"policy.onnx": first, "other.onnx": second, "velocity.onnx": velocity}
    monkeypatch.setattr(
        physics.ort.InferenceSession, "side_effect", lambda path, *_args, **_kwargs: sessions[Path(path).name]
    )
    return motion_config, velocity_config


def test_select_policy_switches_and_resets_without_replacing_physics_or_rewinding_time(dance, policy_configs):
    motion_config, velocity_config = policy_configs
    model, data = dance.model, dance.data
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    initial_gains = [value.copy() for value in (dance.kp, dance.kd, dance.default)]
    for config, name, policy, gains in (
        (motion_config, "test", "policy.onnx", (80, 3, 0.1)),
        (motion_config, "other", "other.onnx", (60, 2, -0.1)),
        (velocity_config, None, "velocity.onnx", None),
        (motion_config, "test", "policy.onnx", (80, 3, 0.1)),
    ):
        dance.history.append([np.ones(dim) for dim in physics.TERM_DIMS])
        dance.last_action[:] = 0.4
        now = data.time
        dance.select_policy(config, name, reset=True)
        assert dance.model is model and dance.data is data
        assert data.time == now
        np.testing.assert_array_equal(data.qpos, initial_qpos)
        np.testing.assert_array_equal(data.qvel, initial_qvel)
        assert dance.selection == (str(config.resolve()), name)
        assert dance.policy_path.name == policy
        assert dance.session is not None
        assert dance.policy_steps == 0 and dance.execution_running
        np.testing.assert_array_equal(dance.last_action, 0)
        for term in dance.history.terms:
            np.testing.assert_array_equal(term, 0)
        if name is None:
            assert dance.controller == "mc_rl_velocity"
            assert dance.motion is None and not dance.motion_running
            for actual, expected in zip((dance.kp, dance.kd, dance.default), initial_gains, strict=True):
                np.testing.assert_array_equal(actual, expected)
        else:
            assert dance.controller == "bmimic" and dance.motion_running
            assert dance.motion.frame == (1 if name == "test" else 0)
            assert not dance.motion.finished
            np.testing.assert_array_equal(dance.motion.last_action, 0)
            for actual, expected, initial in zip(
                (dance.kp, dance.kd, dance.default), gains, initial_gains, strict=True
            ):
                np.testing.assert_allclose(actual[dance.motion.indices], expected)
                np.testing.assert_array_equal(actual[[11, 16]], initial[[11, 16]])
        assert dance.fault is None
        dance.step()
        assert data.time == pytest.approx(now + 0.02)
        assert dance.policy_steps == 1
        if name is not None:
            assert np.any(data.qvel)


@pytest.mark.parametrize(
    "failure,exception,message",
    [
        ("config", ValueError, "mapping"),
        ("missing_config", FileNotFoundError, "missing.yaml"),
        ("missing_policy", FileNotFoundError, "other.onnx"),
        ("joint_names", ValueError, "29 X2 non-head joints"),
        ("action_scale", ValueError, "action_scale"),
        ("time_step_total", ValueError, "phase"),
        ("shape", ValueError, "151"),
    ],
)
def test_select_policy_failure_is_transactional_and_does_not_latch(dance, policy_configs, failure, exception, message):
    config, _ = policy_configs
    dance.select_policy(config, "test", reset=True)
    dance.step()
    dance.pause_execution()
    dance.history.append([np.ones(dim) for dim in physics.TERM_DIMS])
    dance.last_action[:] = 0.3
    session = physics.ort.InferenceSession(str(dance.asset_dir / "other.onnx"))
    if failure == "config":
        config = config.parent / "invalid.yaml"
        config.write_text("[]")
    elif failure == "missing_config":
        config = config.parent / "missing.yaml"
    elif failure == "missing_policy":
        (dance.asset_dir / "other.onnx").unlink()
    elif failure == "shape":
        session.get_inputs.return_value = [SimpleNamespace(name="obs", shape=[1, 154], type="tensor(float)")]
    else:
        session.get_modelmeta.return_value.custom_metadata_map[failure] = {
            "joint_names": "wrong",
            "action_scale": ",".join(["-1"] * 29),
            "time_step_total": "2",
        }[failure]
    objects = {
        key: getattr(dance, key) for key in ("model", "data", "session", "motion", "history", "controller_options")
    }
    values = {
        key: getattr(dance, key)
        for key in ("controller", "selection", "policy_path", "policy_steps", "execution_running", "motion_running")
    }
    arrays = [
        value.copy()
        for value in (dance.data.qpos, dance.data.qvel, dance.kp, dance.kd, dance.default, dance.last_action)
    ]
    history = [term.copy() for term in dance.history.terms]
    frame, action, now = dance.motion.frame, dance.motion.last_action.copy(), dance.data.time
    with pytest.raises(exception, match=message):
        dance.select_policy(config, "other", reset=True)
    for key, value in objects.items():
        assert getattr(dance, key) is value
    for key, value in values.items():
        assert getattr(dance, key) == value
    for actual, expected in zip(
        (dance.data.qpos, dance.data.qvel, dance.kp, dance.kd, dance.default, dance.last_action), arrays, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(dance.history.terms, history, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert dance.motion.frame == frame and not dance.motion.finished
    np.testing.assert_array_equal(dance.motion.last_action, action)
    assert dance.data.time == now and dance.fault is None
    session.run.assert_not_called()
    dance.select_policy(policy_configs[0], "test", reset=False)
    dance.step()
    assert dance.data.time == pytest.approx(now + 0.02)


@pytest.mark.parametrize("controller", ["bmimic", "mc_rl_velocity"])
def test_pause_execution_and_same_selection_resume_preserve_state(dance, policy_configs, controller):
    motion_config, velocity_config = policy_configs
    config, name = (motion_config, "test") if controller == "bmimic" else (velocity_config, None)
    dance.select_policy(config, name, reset=True)
    for _ in range(1 if name else 6):
        dance.step()
    history = dance.history
    terms = [term.copy() for term in history.terms]
    action = dance.last_action.copy()
    motion = dance.motion
    motion_action = motion.last_action.copy() if motion else None
    frame = motion.frame if motion else None
    session, steps, now = dance.session, dance.policy_steps, dance.data.time
    positions, velocities = dance.data.qpos.copy(), dance.data.qvel.copy()
    assert np.any(velocities)
    dance.pause_execution()
    assert not dance.execution_running and not dance.motion_running
    calls = session.run.call_count
    loads = physics.ort.InferenceSession.call_count
    for _ in range(5):
        dance.step()
    for other_config, other_name in (
        (motion_config, "other"),
        (velocity_config, None) if name else (motion_config, "test"),
    ):
        with pytest.raises(ValueError, match="same unfinished policy"):
            dance.select_policy(other_config, other_name, reset=False)
        assert not dance.execution_running and not dance.motion_running
    assert session.run.call_count == calls
    dance.select_policy(config, name, reset=False)
    assert physics.ort.InferenceSession.call_count == loads
    assert dance.execution_running and dance.motion_running == bool(name)
    assert dance.session is session and dance.history is history and dance.motion is motion
    assert dance.policy_steps == steps and dance.data.time == now
    np.testing.assert_array_equal(dance.data.qpos, positions)
    np.testing.assert_array_equal(dance.data.qvel, velocities)
    np.testing.assert_array_equal(dance.last_action, action)
    for actual, expected in zip(history.terms, terms, strict=True):
        np.testing.assert_array_equal(actual, expected)
    if motion:
        assert motion.frame == frame
        np.testing.assert_array_equal(motion.last_action, motion_action)
    dance.step()
    assert dance.data.time == pytest.approx(now + 0.02)
    assert dance.policy_steps == steps + 1
    obs = session.run.call_args.args[1]["obs"][0]
    if motion:
        assert motion.frame == frame + 1
        np.testing.assert_array_equal(obs[122:], motion_action)
    else:
        np.testing.assert_array_equal(obs[-29:], action)


def test_select_policy_without_reset_requires_an_existing_selection(dance, policy_configs):
    before = dance.data.time
    with pytest.raises(ValueError, match="same unfinished policy"):
        dance.select_policy(policy_configs[0], "test", reset=False)
    assert dance.selection is None and dance.session is None
    assert dance.data.time == before and dance.fault is None
    physics.ort.InferenceSession.assert_not_called()


@pytest.mark.parametrize("controller", ["bmimic", "mc_rl_velocity"])
def test_activate_policy_transition_preserves_physics_and_reinitializes_policy(dance, policy_configs, controller):
    motion_config, velocity_config = policy_configs
    dance.select_policy(motion_config, "test", reset=True)
    dance.step()
    old_motion, old_history, old_session = dance.motion, dance.history, dance.session
    old_motion.finished = True
    old_motion.last_action[:] = 0.3
    dance.history.append([np.ones(dim) for dim in physics.TERM_DIMS])
    dance.last_action[:] = 0.4
    dance.selection_finished = True
    dance.pause_execution()
    dance.data.qpos[2] = 0.6
    dance.data.qpos[dance.qpos_indices] += 0.01
    dance.data.qvel[:] = 0.01
    mujoco.mj_forward(dance.model, dance.data)
    assert dance.is_standing()
    config, name = (motion_config, "other") if controller == "bmimic" else (velocity_config, None)
    candidate = dance.prepare_policy(config, name)
    if controller == "bmimic":
        dance.data.qpos[dance.qpos_indices] = candidate.default + 0.01
        mujoco.mj_forward(dance.model, dance.data)
        assert dance.is_standing()
    model, data, now = dance.model, dance.data, dance.data.time
    positions, velocities = data.qpos.copy(), data.qvel.copy()
    calls = candidate.session.run.call_count

    dance.activate_policy(candidate, transition=True)

    assert dance.model is model and dance.data is data
    assert data.time == now
    np.testing.assert_array_equal(data.qpos, positions)
    np.testing.assert_array_equal(data.qvel, velocities)
    assert dance.controller == controller
    assert dance.selection == (str(config.resolve()), name)
    assert dance.session is candidate.session and dance.session is not old_session
    assert dance.policy_path == candidate.policy_path
    assert dance.controller_options == candidate.controller_options
    assert dance.history is not old_history
    for term in dance.history.terms:
        np.testing.assert_array_equal(term, 0)
    np.testing.assert_array_equal(dance.last_action, 0)
    assert dance.policy_steps == 0 and not dance.selection_finished
    assert dance.execution_running and dance.motion_running == bool(name)
    assert dance.fault is None
    assert dance.session.run.call_count == calls + (2 if name else 0)
    for actual, expected in zip(
        (dance.kp, dance.kd, dance.default), (candidate.kp, candidate.kd, candidate.default), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    if name:
        assert dance.motion is candidate.motion and dance.motion is not old_motion
        assert dance.motion.frame == 0 and not dance.motion.finished
        np.testing.assert_array_equal(dance.motion.last_action, 0)
        np.testing.assert_allclose(dance.kp[dance.motion.indices], 60)
    else:
        assert dance.motion is None
        assert dance.input is candidate.input and dance.output is candidate.output
        for actual, expected in zip((dance.kp, dance.kd, dance.default), dance._initial_gains, strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "controller,field,value",
    [
        ("mc_rl_velocity", "height", 0.499),
        ("mc_rl_velocity", "tilt_rad", np.deg2rad(25.01)),
        ("mc_rl_velocity", "linear_velocity_world", [0.43, 0.43, 0]),
        ("mc_rl_velocity", "angular_velocity_body", [1.42, 1.42, 0]),
        ("bmimic", "height", 0.549),
        ("bmimic", "tilt_rad", np.deg2rad(8.01)),
        ("bmimic", "linear_velocity_world", [0.04, 0.04, 0]),
        ("bmimic", "angular_velocity_body", [0.11, 0.11, 0]),
        ("bmimic", "joint_velocity", -0.501),
    ],
)
def test_activate_policy_transition_rejects_unsafe_state_without_changes(
    dance, policy_configs, monkeypatch, controller, field, value
):
    motion_config, velocity_config = policy_configs
    dance.select_policy(motion_config, "test", reset=True)
    dance.pause_execution()
    config, name = (motion_config, "other") if controller == "bmimic" else (velocity_config, None)
    candidate = dance.prepare_policy(config, name)
    truth = dict(height=0.6, tilt_rad=0, linear_velocity_world=[0, 0, 0], angular_velocity_body=[0, 0, 0])
    if field == "joint_velocity":
        dance.data.qvel[dance.qvel_indices[0]] = value
    else:
        truth[field] = value
    monkeypatch.setattr(dance, "ground_truth", lambda: truth)
    session, motion, history, selection = dance.session, dance.motion, dance.history, dance.selection
    model, data, now = dance.model, dance.data, dance.data.time
    positions, velocities = data.qpos.copy(), data.qvel.copy()
    message = "standing recovery" if name else "balance handoff limits"

    with pytest.raises(ValueError, match=message):
        dance.activate_policy(candidate, transition=True)

    assert dance.model is model and dance.data is data and data.time == now
    np.testing.assert_array_equal(data.qpos, positions)
    np.testing.assert_array_equal(data.qvel, velocities)
    assert dance.session is session and dance.motion is motion and dance.history is history
    assert dance.selection == selection and dance.policy_steps == 0
    assert not dance.execution_running and not dance.motion_running
    assert dance.fault is None
    candidate.session.run.assert_not_called()


@pytest.mark.parametrize("controller", ["bmimic", "mc_rl_velocity"])
@pytest.mark.parametrize("state", ["qpos", "qvel"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_activate_policy_transition_rejects_nonfinite_physics(dance, policy_configs, controller, state, value):
    motion_config, velocity_config = policy_configs
    config, name = (motion_config, "other") if controller == "bmimic" else (velocity_config, None)
    candidate = dance.prepare_policy(config, name)
    getattr(dance.data, state)[-1] = value
    with pytest.raises(ValueError, match="Nonfinite state"):
        dance.activate_policy(candidate, transition=True)
    assert dance.session is None and dance.selection is None and dance.fault is None
    np.testing.assert_equal(getattr(dance.data, state)[-1], value)
    candidate.session.run.assert_not_called()


@pytest.mark.parametrize("reason", ["pose", "target", "torque"])
def test_dance_handoff_rejects_pose_or_initial_torque_without_mutating_live_state(
    dance, policy_configs, monkeypatch, reason
):
    candidate = dance.prepare_policy(policy_configs[0], "other")
    dance.pause_execution()
    monkeypatch.setattr(dance, "is_standing", lambda: True)
    dance.data.qpos[dance.qpos_indices] = candidate.default
    dance.data.qvel[:] = 0
    if reason == "pose":
        dance.data.qpos[dance.qpos_indices[0]] += 0.51
    else:

        def target(motion, *_args):
            return motion.default + (0.51 if reason == "target" else 0.4)

        monkeypatch.setattr(BMimic, "target", target)
        if reason == "torque":
            candidate.kp[:] = 210
    positions, velocities = dance.data.qpos.copy(), dance.data.qvel.copy()
    now = dance.data.time
    with pytest.raises(ValueError, match="handoff"):
        dance.activate_policy(candidate, transition=True)
    assert dance.session is None and dance.selection is None and dance.fault is None
    assert not dance.execution_running and dance.data.time == now
    np.testing.assert_array_equal(dance.data.qpos, positions)
    np.testing.assert_array_equal(dance.data.qvel, velocities)
    np.testing.assert_array_equal(candidate.motion.last_action, 0)
    assert candidate.motion.frame == 0 and not candidate.motion.finished


def test_activate_policy_transition_accepts_balance_limits_without_requiring_standing(
    dance, policy_configs, monkeypatch
):
    candidate = dance.prepare_policy(policy_configs[1], None)
    truth = dict(
        height=0.5,
        tilt_rad=np.deg2rad(25),
        linear_velocity_world=[0, 0.6, 0],
        angular_velocity_body=[0, 0, 2.0],
    )
    monkeypatch.setattr(dance, "ground_truth", lambda: truth)
    assert not dance.is_standing()
    dance.activate_policy(candidate, transition=True)
    assert dance.controller == "mc_rl_velocity" and dance.execution_running
    assert dance.session is candidate.session and dance.fault is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("height", 0.549),
        ("tilt_rad", np.deg2rad(8.01)),
        ("linear_velocity_world", [0.04, 0.04, 0]),
        ("angular_velocity_body", [0.11, 0.11, 0]),
        ("joint_velocity", -0.501),
    ]
    + [
        (field, value)
        for field in ("height", "tilt_rad", "linear_velocity_world", "angular_velocity_body", "joint_velocity")
        for value in (np.nan, np.inf, -np.inf)
    ],
)
def test_is_standing_requires_all_finite_standing_criteria(sim, monkeypatch, field, value):
    truth = dict(
        height=0.55, tilt_rad=np.deg2rad(8), linear_velocity_world=[0.05, 0, 0], angular_velocity_body=[0, 0.15, 0]
    )
    sim.data.qvel[sim.qvel_indices] = 0.5
    monkeypatch.setattr(sim, "ground_truth", lambda: truth)
    assert sim.is_standing()
    if field == "joint_velocity":
        sim.data.qvel[sim.qvel_indices[-1]] = value
    else:
        if isinstance(truth[field], list) and np.isscalar(value):
            value = [value, 0, 0]
        truth[field] = value
    assert not sim.is_standing()


def test_finished_selection_requires_reset_to_replay(dance, policy_configs):
    config, _ = policy_configs
    initial_qpos, initial_qvel = dance.data.qpos.copy(), dance.data.qvel.copy()
    dance.select_policy(config, "test", reset=True)
    for _ in range(3):
        dance.step()
    motion, session, now = dance.motion, dance.session, dance.data.time
    assert motion.finished and not dance.motion_running
    positions, velocities = dance.data.qpos.copy(), dance.data.qvel.copy()
    calls = session.run.call_count
    with pytest.raises(ValueError, match="same unfinished policy"):
        dance.select_policy(config, "test", reset=False)
    dance.step()
    assert dance.motion is motion and motion.finished
    assert dance.policy_steps == 3 and dance.data.time == now
    assert session.run.call_count == calls
    np.testing.assert_array_equal(dance.data.qpos, positions)
    np.testing.assert_array_equal(dance.data.qvel, velocities)
    dance.select_policy(config, "test", reset=True)
    assert dance.motion is not motion and not dance.motion.finished
    assert dance.motion.frame == 1 and dance.motion_running
    assert dance.policy_steps == 0 and dance.data.time == now
    np.testing.assert_array_equal(dance.data.qpos, initial_qpos)
    np.testing.assert_array_equal(dance.data.qvel, initial_qvel)
    np.testing.assert_array_equal(dance.motion.last_action, 0)
    dance.step()
    assert dance.data.time == pytest.approx(now + 0.02)
    assert dance.motion.frame == 2 and dance.policy_steps == 1


@pytest.mark.parametrize("reset", [False, True])
def test_select_policy_cannot_clear_a_latched_fault(dance, policy_configs, reset):
    config, _ = policy_configs
    dance.select_policy(config, "test", reset=True)
    dance.session.run.side_effect = Fail("injected inference failure")
    with pytest.raises(RuntimeError, match="BMIMIC inference failed"):
        dance.step()
    fault, motion, session = dance.fault, dance.motion, dance.session
    now, positions, velocities = dance.data.time, dance.data.qpos.copy(), dance.data.qvel.copy()
    loads = physics.ort.InferenceSession.call_count
    with pytest.raises(RuntimeError, match="BMIMIC inference failed"):
        dance.select_policy(config, "test", reset=reset)
    assert dance.fault == fault and dance.motion is motion and dance.session is session
    assert dance.data.time == now
    np.testing.assert_array_equal(dance.data.qpos, positions)
    np.testing.assert_array_equal(dance.data.qvel, velocities)
    assert physics.ort.InferenceSession.call_count == loads


def test_bmimic_verified_frame_count_for_older_models(dance):
    session = physics.ort.InferenceSession.return_value
    metadata = session.get_modelmeta.return_value.custom_metadata_map
    del metadata["time_step_total"]
    with pytest.raises(ValueError, match="verified frame_count"):
        BMimic(session, dance.joint_names, 1, 3)
    controller = BMimic(session, dance.joint_names, 1, 3, frame_count=4)
    assert controller.frame == 1 and controller.end == 3
    with pytest.raises(ValueError, match="phase"):
        BMimic(session, dance.joint_names, 1, 4, frame_count=4)
    metadata["time_step_total"] = "5"
    with pytest.raises(ValueError, match="disagrees"):
        BMimic(session, dance.joint_names, 1, 3, frame_count=4)


def test_bmimic_frame_count_config_loads_old_model(dance, tmp_path):
    session = physics.ort.InferenceSession.return_value
    del session.get_modelmeta.return_value.custom_metadata_map["time_step_total"]
    config = tmp_path / "old-model.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "controller": "bmimic",
                "motions": {
                    "test": dict(dance.controller_options, frame_count=4),
                },
            }
        )
    )
    configured = physics.X2Simulation(dance.asset_dir, config, "test")
    configured.model.dof_armature[configured.qvel_indices] = 0.1
    configured.start_motion()
    configured.step()
    assert configured.policy_steps == 1
    assert configured.motion.frame == 2


@pytest.mark.parametrize("frame_count", [True, 0, -1, 3.5, "4", None])
def test_bmimic_frame_count_config_rejects_invalid_values(dance, tmp_path, frame_count):
    config = tmp_path / "invalid-frames.yaml"
    options = dict(dance.controller_options, frame_count=frame_count)
    config.write_text(yaml.safe_dump({"controller": "bmimic", "motions": {"test": options}}))
    with pytest.raises(ValueError, match="frame_count"):
        physics.X2Simulation(dance.asset_dir, config, "test")
    physics.ort.InferenceSession.assert_not_called()


@pytest.mark.parametrize(
    "name,policy,start,end,frames",
    [
        ("kuailechongbai", "kuailechongbai1-Full2.onnx", 1, 2820, None),
        ("fan_dance", "policy_FanDance.onnx", 0, 1170, 1787),
        ("niuniu_dance", "policy_niuniu_Dance.onnx", 0, 1159, 1160),
        ("charleston", "policy_dance_ces_full300.onnx", 0, 2517, 6572),
        ("dingding", "dingding_v5_gc.onnx", 0, 1550, None),
        ("kuailechongbai_video", "kuailechongbai_fromvideo_inpainted.onnx", 1, 2820, None),
    ],
)
def test_shipped_motion_configs_use_shared_controller_and_asset_relative_paths(sim, name, policy, start, end, frames):
    config = Path(__file__).resolve().parents[1] / "config" / "bmimic.yaml"
    configured = physics.X2Simulation(sim.asset_dir, config, name)
    assert configured.controller == "bmimic"
    assert configured.policy_path == sim.asset_dir / policy
    assert configured.controller_options["start_count"] == start
    assert configured.controller_options["end_count"] == end
    assert configured.controller_options.get("frame_count") == frames
    configured.step()
    assert not configured.motion_running and configured.session is None
    physics.ort.InferenceSession.assert_not_called()


@pytest.mark.parametrize("motion", [None, "unknown", ""])
def test_merged_config_requires_explicit_known_motion(sim, motion):
    config = Path(__file__).resolve().parents[1] / "config" / "bmimic.yaml"
    with pytest.raises(ValueError, match="Select a BMIMIC motion explicitly"):
        physics.X2Simulation(sim.asset_dir, config, motion)
    physics.ort.InferenceSession.assert_not_called()


def test_motion_selection_requires_matching_controller(sim, tmp_path):
    with pytest.raises(ValueError, match="requires controller_config"):
        physics.X2Simulation(sim.asset_dir, motion="fan_dance")
    with pytest.raises(ValueError, match="requires a BMIMIC"):
        physics.X2Simulation(sim.asset_dir, tmp_path / "controller.yaml", "fan_dance")


@pytest.mark.parametrize(
    "entry",
    [
        None,
        [],
        {"policy": "../outside.onnx", "start_count": 0, "end_count": 1},
        {"policy": "policy.onnx", "start_count": 0, "end_count": 1, "sha256": "old"},
    ],
)
def test_merged_config_rejects_invalid_selected_entries(sim, tmp_path, entry):
    config = tmp_path / "invalid-catalog.yaml"
    config.write_text(yaml.safe_dump({"controller": "bmimic", "motions": {"test": entry}}))
    with pytest.raises(ValueError):
        physics.X2Simulation(sim.asset_dir, config, "test")
    physics.ort.InferenceSession.assert_not_called()
