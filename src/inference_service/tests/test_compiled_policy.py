from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from inference_service.core.compiled_policy import (
    ACTCompiledAdapter,
    CompiledPolicyWrapper,
    OMRuntimeSession,
    PI05CompiledAdapter,
    PI05OMRuntimeSession,
    PI05RuntimeInputs,
    SD3403RuntimeSession,
    create_compiled_model_adapter,
    create_runtime_session,
    load_compiled_manifest,
    resolve_om_model_path,
    resolve_pi05_om_paths,
)


class FakeRuntimeSession:
    def __init__(self, output=None):
        self.output = output if output is not None else np.zeros((1, 2, 6), dtype=np.float32)
        self.loaded = None
        self.inputs = None

    def load(self, policy_path, config, device):
        self.loaded = (policy_path, config, device)

    def execute(self, inputs):
        self.inputs = inputs
        return [self.output]

    def release(self):
        pass


def _act_config(**updates):
    config = {
        "type": "act",
        "chunk_size": 2,
        "input_features": {
            "observation.state": {"shape": [3]},
            "observation.images.side": {"shape": [3, 4, 5]},
            "observation.images.gripper": {"shape": [3, 4, 5]},
        },
        "output_features": {"action": {"shape": [6]}},
    }
    config.update(updates)
    return config


def _write_policy(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_manifest(tmp_path, manifest):
    (tmp_path / "config.om.json").write_text(json.dumps(manifest), encoding="utf-8")


def _pi05_config(**updates):
    config = {
        "type": "pi05",
        "chunk_size": 50,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.language.tokens": {"shape": [48]},
            "observation.language.attention_mask": {"shape": [48]},
        },
        "output_features": {"action": {"shape": [6]}},
    }
    config.update(updates)
    return config


def test_adapter_selection_from_config_type():
    adapter = create_compiled_model_adapter(_act_config(), "rknn")

    assert isinstance(adapter, ACTCompiledAdapter)
    assert adapter.policy_type == "act"
    assert adapter.uses_action_chunking is True

    pi05_adapter = create_compiled_model_adapter(_pi05_config(), "ascend_om")
    assert isinstance(pi05_adapter, PI05CompiledAdapter)
    assert pi05_adapter.policy_type == "pi05"
    assert pi05_adapter.uses_action_chunking is True


def test_adapter_rejects_missing_and_unsupported_type():
    with pytest.raises(ValueError, match="missing required type"):
        create_compiled_model_adapter({"input_features": {}}, "rknn")

    with pytest.raises(ValueError, match="does not support policy type"):
        create_compiled_model_adapter({"type": "diffusion"}, "rknn")

    with pytest.raises(ValueError, match="does not support PI05"):
        create_compiled_model_adapter(_pi05_config(), "rknn")


def test_act_input_mapping_uses_declared_order_and_camera_names():
    adapter = ACTCompiledAdapter.from_config(_act_config(), "rknn")

    inputs = adapter.prepare_inputs(
        {
            "observation.state": torch.full((3,), 1.0),
            "observation.images.side": torch.full((3, 4, 5), 2.0),
            "observation.images.gripper": torch.full((3, 4, 5), 3.0),
        }
    )

    assert [arr.shape for arr in inputs] == [(1, 3), (1, 3, 4, 5), (1, 3, 4, 5)]
    assert float(inputs[0][0, 0]) == 1.0
    assert float(inputs[1][0, 0, 0, 0]) == 2.0
    assert float(inputs[2][0, 0, 0, 0]) == 3.0


def test_act_input_mapping_rejects_missing_tensor():
    adapter = ACTCompiledAdapter.from_config(_act_config(), "rknn")

    with pytest.raises(KeyError, match="observation.images.gripper"):
        adapter.prepare_inputs(
            {
                "observation.state": torch.ones(3),
                "observation.images.side": torch.ones(3, 4, 5),
            }
        )


def test_act_decodes_om_action_chunk():
    adapter = ACTCompiledAdapter.from_config(_act_config(), "ascend_om")
    action = adapter.decode_outputs([np.arange(12, dtype=np.float32)], torch.device("cpu"))

    assert action.shape == (2, 6)


def test_act_decodes_sd3403_direct_action_shape():
    adapter = ACTCompiledAdapter.from_config(_act_config(chunk_size=1), "ascend_om_3403")
    action = adapter.decode_outputs([np.zeros((1, 100, 6), dtype=np.float32)], torch.device("cpu"))

    assert action.shape == (100, 6)
    assert adapter.get_chunk_size() == 100


def test_act_decodes_sd3403_direct_action_updates_chunk_size():
    adapter = ACTCompiledAdapter.from_config(_act_config(chunk_size=1), "ascend_om_3403")
    action = adapter.decode_outputs([np.zeros((1, 2, 6), dtype=np.float32)], torch.device("cpu"))

    assert action.shape == (2, 6)
    assert adapter.get_chunk_size() == 2


def test_act_decodes_sd3403_direct_action_from_readonly_buffer():
    adapter = ACTCompiledAdapter.from_config(_act_config(chunk_size=1), "ascend_om_3403")
    source = np.frombuffer(np.zeros((1, 100, 6), dtype=np.float32).tobytes(), dtype=np.float32).reshape(1, 100, 6)

    assert not source.flags.writeable

    action = adapter.decode_outputs([source], torch.device("cpu"))

    assert action.shape == (100, 6)
    assert action.data_ptr() != source.__array_interface__["data"][0]


def test_pi05_adapter_prepares_runtime_inputs_and_slices_padding():
    adapter = PI05CompiledAdapter.from_config(_pi05_config(), "ascend_om")

    inputs = adapter.prepare_inputs(
        {
            "observation.images.front": torch.full((1, 3, 224, 224), 1.0),
            "observation.language.tokens": torch.arange(48).reshape(1, 48),
            "observation.language.attention_mask": torch.ones(1, 48, dtype=torch.bool),
            "_noise": torch.zeros(1, 50, 32),
        }
    )

    assert isinstance(inputs, PI05RuntimeInputs)
    assert inputs.images[0].shape == (1, 3, 224, 224)
    assert inputs.tokens.dtype == np.int64
    assert inputs.masks.dtype == np.bool_
    assert inputs.noise.shape == (1, 50, 32)

    action = adapter.decode_outputs(torch.zeros(1, 50, 32), torch.device("cpu"))

    assert action.shape == (50, 6)
    assert adapter.get_chunk_size() == 50


def test_compiled_wrapper_reports_metadata_and_runtime_device(tmp_path):
    _write_policy(tmp_path, _act_config(input_features={"observation.state": {"shape": [3]}}))
    runtime = FakeRuntimeSession(output=np.arange(12, dtype=np.float32))
    wrapper = CompiledPolicyWrapper("rknn", runtime_session=runtime)
    device = torch.device("cpu")

    wrapper.load(str(tmp_path), device)
    action = wrapper.infer({"observation.state": torch.ones(3)})

    assert runtime.loaded[0] == str(tmp_path)
    assert runtime.loaded[1]["type"] == "act"
    assert runtime.loaded[2] == device
    assert runtime.inputs[0].shape == (1, 3)
    assert action.shape == (2, 6)
    assert wrapper.policy_type == "act"
    assert wrapper.backend_type == "rknn"
    assert wrapper.uses_action_chunking is True


def test_compiled_wrapper_passes_backend_config_to_sd3403_adapter(tmp_path):
    _write_policy(
        tmp_path,
        _act_config(
            input_features={"observation.state": {"shape": [3]}},
        ),
    )
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "execution": ["policy", "worker"],
            "backend_config": {
                "action_output": {"index": 1, "layout": "direct"},
            },
        },
    )
    runtime = FakeRuntimeSession(output=np.zeros((1, 2, 6), dtype=np.float32))
    wrapper = CompiledPolicyWrapper("ascend_om_3403", runtime_session=runtime)

    wrapper.load(str(tmp_path), torch.device("cpu"))
    action = wrapper.infer({"observation.state": torch.ones(3)})

    assert action.shape == (2, 6)
    assert runtime.loaded[1]["_compiled_backend_config"]["action_output"]["index"] == 1


def test_compiled_wrapper_requires_config_json(tmp_path):
    wrapper = CompiledPolicyWrapper("rknn", runtime_session=FakeRuntimeSession())

    with pytest.raises(FileNotFoundError, match="config.json"):
        wrapper.load(str(tmp_path), torch.device("cpu"))


def test_runtime_dependencies_import_lazily():
    before = set(sys.modules)

    OMRuntimeSession()
    SD3403RuntimeSession()
    PI05OMRuntimeSession()

    imported = set(sys.modules) - before
    assert "acl" not in imported
    assert "rknnlite.api" not in imported


def test_ascend_runtime_session_selects_pi05_from_config():
    assert isinstance(create_runtime_session("ascend_om", _act_config()), OMRuntimeSession)
    assert isinstance(create_runtime_session("ascend_om", _pi05_config()), PI05OMRuntimeSession)


def test_manifest_resolves_single_om_policy_role(tmp_path):
    model = tmp_path / "om" / "act.om"
    model.parent.mkdir()
    model.write_bytes(b"om")
    _write_policy(tmp_path, _act_config())
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om",
            "artifact_dir": "om",
            "artifacts": {"policy": "act.om"},
            "execution": ["policy"],
        },
    )

    manifest = load_compiled_manifest(str(tmp_path), "ascend_om", "act")

    assert resolve_om_model_path(str(tmp_path), _act_config(), manifest) == model.resolve()
    assert manifest.backend_config == {}


def test_manifest_parses_backend_config(tmp_path):
    model = tmp_path / "act.om"
    worker = tmp_path / "main"
    model.write_bytes(b"om")
    worker.write_bytes(b"")
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "execution": ["policy", "worker"],
            "backend_config": {
                "action_output": {"index": 1, "layout": "direct"},
            },
        },
    )

    manifest = load_compiled_manifest(str(tmp_path), "ascend_om_3403", "act")

    assert manifest.backend_config["action_output"]["index"] == 1


def test_manifest_rejects_non_object_backend_config(tmp_path):
    model = tmp_path / "act.om"
    worker = tmp_path / "main"
    model.write_bytes(b"om")
    worker.write_bytes(b"")
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "backend_config": ["invalid"],
        },
    )

    with pytest.raises(ValueError, match="backend_config must be a JSON object"):
        load_compiled_manifest(str(tmp_path), "ascend_om_3403", "act")


def test_manifest_rejects_unknown_sd3403_output_layout(tmp_path):
    model = tmp_path / "act.om"
    worker = tmp_path / "main"
    model.write_bytes(b"om")
    worker.write_bytes(b"")
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "backend_config": {"action_output": {"index": 1, "layout": "packed"}},
        },
    )

    with pytest.raises(ValueError, match="backend_config.action_output.layout"):
        load_compiled_manifest(str(tmp_path), "ascend_om_3403", "act")


def test_manifest_resolves_pi05_roles_and_execution(tmp_path):
    vlm = tmp_path / "om" / "vlm.om"
    ae = tmp_path / "om" / "action_expert.om"
    vlm.parent.mkdir()
    vlm.write_bytes(b"vlm")
    ae.write_bytes(b"ae")
    _write_policy(tmp_path, _pi05_config())
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "pi05",
            "backend": "ascend_om",
            "artifact_dir": "om",
            "artifacts": {
                "vlm": "vlm.om",
                "action_expert": "action_expert.om",
            },
            "execution": ["vlm", "action_expert"],
        },
    )

    manifest = load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")

    assert resolve_pi05_om_paths(str(tmp_path), _pi05_config(), manifest) == (vlm.resolve(), ae.resolve())


def test_manifest_rejects_wrong_backend_policy_and_execution(tmp_path):
    model = tmp_path / "model.om"
    model.write_bytes(b"om")
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "pi05",
            "backend": "rknn",
            "artifacts": {"policy": "model.om"},
        },
    )
    with pytest.raises(ValueError, match="does not match requested backend"):
        load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")

    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om",
            "artifacts": {"policy": "model.om"},
        },
    )
    with pytest.raises(ValueError, match="does not match config type"):
        load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")

    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "pi05",
            "backend": "ascend_om",
            "artifacts": {"vlm": "model.om", "action_expert": "model.om"},
            "execution": ["action_expert", "vlm"],
        },
    )
    manifest = load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")
    with pytest.raises(ValueError, match="execution must be"):
        resolve_pi05_om_paths(str(tmp_path), _pi05_config(), manifest)


def test_manifest_required_for_om_resolution(tmp_path):
    _write_policy(tmp_path, _act_config())

    with pytest.raises(FileNotFoundError, match="config.om.json"):
        resolve_om_model_path(str(tmp_path), _act_config())


def test_pi05_runtime_builds_prefix_mask_and_forwards():
    class FakeModel:
        prefix_seq_len = 52

        def __init__(self):
            self.forward_args = None

        def forward(self, images, tokens, masks, prefix_mask, noise=None):
            self.forward_args = (images, tokens, masks, prefix_mask, noise)
            return torch.zeros(1, 50, 32)

    session = PI05OMRuntimeSession()
    session._model = FakeModel()
    inputs = PI05RuntimeInputs(
        images=[np.ones((1, 3, 224, 224), dtype=np.float32)],
        tokens=np.ones((1, 48), dtype=np.int64),
        masks=np.ones((1, 48), dtype=np.bool_),
        noise=np.zeros((1, 50, 32), dtype=np.float32),
    )

    output = session.execute(inputs)

    _, _, _, prefix_mask, noise = session._model.forward_args
    assert output.shape == (1, 50, 32)
    assert prefix_mask.shape == (1, 1, 52, 52)
    assert noise is inputs.noise


def test_sd3403_runtime_uses_worker_public_array_api():
    class FakeWorker:
        def __init__(self):
            self.inputs = None
            self.closed = False

        def execute_arrays(self, inputs):
            self.inputs = inputs
            return np.arange(16, dtype=np.float32)

        def close(self):
            self.closed = True

    session = SD3403RuntimeSession()
    session._worker = FakeWorker()
    inputs = [np.ones((1, 3), dtype=np.float32)]

    outputs = session.execute(inputs)

    assert session._worker.inputs is inputs
    assert outputs[0].shape == (16,)


def test_sd3403_runtime_passes_config_to_worker(monkeypatch, tmp_path):
    model = tmp_path / "om" / "act.om"
    worker = tmp_path / "om" / "main"
    model.parent.mkdir()
    model.write_bytes(b"om")
    worker.write_bytes(b"")
    worker.chmod(0o755)
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifact_dir": "om",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "execution": ["policy", "worker"],
            "backend_config": {
                "action_output": {"index": 3, "layout": "direct"},
                "image_height": 360,
                "image_width": 640,
                "perf_enabled": False,
                "perf_log_every": 5,
                "graceful_close_timeout": 2.5,
                "force_close": False,
            },
        },
    )
    config = _act_config(
        output_features={"action": {"shape": [9]}},
    )
    calls = {}

    class FakePolicy:
        def __init__(self, worker_path, model_path, **kwargs):
            calls["worker_path"] = worker_path
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "inference_service.core.ascend_om.ACTWrapper_3403.ACT3403Policy",
        FakePolicy,
    )

    session = SD3403RuntimeSession()
    session.load(str(tmp_path), config, torch.device("cpu"))

    assert calls["worker_path"] == str(worker.resolve())
    assert calls["model_path"] == str(model.resolve())
    assert calls["kwargs"]["action_dim"] == 9
    assert calls["kwargs"]["action_output_index"] == 3
    assert calls["kwargs"]["image_height"] == 360
    assert calls["kwargs"]["image_width"] == 640
    assert calls["kwargs"]["perf_enabled"] is False
    assert calls["kwargs"]["perf_log_every"] == 5
    assert calls["kwargs"]["graceful_close_timeout"] == 2.5
    assert calls["kwargs"]["force_close"] is False


def test_sd3403_runtime_keeps_legacy_config_fallback(monkeypatch, tmp_path):
    model = tmp_path / "om" / "act.om"
    worker = tmp_path / "om" / "main"
    model.parent.mkdir()
    model.write_bytes(b"om")
    worker.write_bytes(b"")
    worker.chmod(0o755)
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifact_dir": "om",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "execution": ["policy", "worker"],
        },
    )
    config = _act_config(
        output_features={"action": {"shape": [7]}},
        sd3403_action_output_index=4,
        sd3403_image_height=300,
        sd3403_image_width=500,
        sd3403_perf_enabled=False,
        sd3403_perf_log_every=4,
        sd3403_graceful_close_timeout=1.5,
        sd3403_force_close=False,
    )
    calls = {}

    class FakePolicy:
        def __init__(self, worker_path, model_path, **kwargs):
            del worker_path, model_path
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "inference_service.core.ascend_om.ACTWrapper_3403.ACT3403Policy",
        FakePolicy,
    )

    session = SD3403RuntimeSession()
    session.load(str(tmp_path), config, torch.device("cpu"))

    assert calls["kwargs"]["action_dim"] == 7
    assert calls["kwargs"]["action_output_index"] == 4
    assert calls["kwargs"]["image_height"] == 300
    assert calls["kwargs"]["image_width"] == 500
    assert calls["kwargs"]["perf_enabled"] is False
    assert calls["kwargs"]["perf_log_every"] == 4
    assert calls["kwargs"]["graceful_close_timeout"] == 1.5
    assert calls["kwargs"]["force_close"] is False


def test_sd3403_runtime_image_dims_from_input_features(monkeypatch, tmp_path):
    """Image height/width must be derived from config.json input_features.<image>.shape,
    taking precedence over any backend_config.image_height/image_width override."""
    model = tmp_path / "om" / "act.om"
    worker = tmp_path / "om" / "main"
    model.parent.mkdir()
    model.write_bytes(b"om")
    worker.write_bytes(b"")
    worker.chmod(0o755)
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifact_dir": "om",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "execution": ["policy", "worker"],
            # backend_config sets a DIFFERENT resolution; input_features must win.
            "backend_config": {"image_height": 240, "image_width": 320},
        },
    )
    config = _act_config(
        input_features={
            "observation.state": {"shape": [6]},
            "observation.images.top": {"shape": [3, 480, 640]},
            "observation.images.wrist": {"shape": [3, 480, 640]},
        },
    )
    calls = {}

    class FakePolicy:
        def __init__(self, worker_path, model_path, **kwargs):
            del worker_path, model_path
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "inference_service.core.ascend_om.ACTWrapper_3403.ACT3403Policy",
        FakePolicy,
    )

    session = SD3403RuntimeSession()
    session.load(str(tmp_path), config, torch.device("cpu"))

    # Derived from input_features shape [3, 480, 640], NOT backend_config 240x320.
    assert calls["kwargs"]["image_height"] == 480
    assert calls["kwargs"]["image_width"] == 640


def test_sd3403_runtime_accepts_flat_backend_config_aliases(monkeypatch, tmp_path):
    model = tmp_path / "om" / "act.om"
    worker = tmp_path / "om" / "main"
    model.parent.mkdir()
    model.write_bytes(b"om")
    worker.write_bytes(b"")
    worker.chmod(0o755)
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om_3403",
            "artifact_dir": "om",
            "artifacts": {"policy": "act.om", "worker": "main"},
            "execution": ["policy", "worker"],
            "backend_config": {
                "action_output_index": 5,
            },
        },
    )
    config = _act_config(
        output_features={"action": {"shape": [7]}},
        sd3403_action_output_index=3,
    )
    calls = {}

    class FakePolicy:
        def __init__(self, worker_path, model_path, **kwargs):
            del worker_path, model_path
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "inference_service.core.ascend_om.ACTWrapper_3403.ACT3403Policy",
        FakePolicy,
    )

    session = SD3403RuntimeSession()
    session.load(str(tmp_path), config, torch.device("cpu"))

    assert calls["kwargs"]["action_output_index"] == 5
