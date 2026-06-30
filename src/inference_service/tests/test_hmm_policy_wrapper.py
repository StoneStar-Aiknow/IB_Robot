from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch

from inference_service.core.compiled_policy import (
    HMM_MANIFEST_BASENAME,
    HMMRuntimeSession,
    create_runtime_session,
    resolve_hmm_model_path,
)
from inference_service.core.hmm.policy_wrapper import (
    HMMPolicyWrapper,
    create_hmm_policy_wrapper,
)


def _act_config(**updates):
    config = {
        "type": "act",
        "chunk_size": 100,
        "input_features": {
            "observation.images.wrist": {"shape": [3, 4, 5]},
            "observation.state": {"shape": [6]},
            "observation.images.top": {"shape": [3, 4, 5]},
        },
        "output_features": {"action": {"shape": [6]}},
    }
    config.update(updates)
    return config


def _write_policy(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.hmm").write_bytes(b"hmm")


class _FakeOutput:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)

    def astype(self, dtype):
        self._data = self._data.astype(dtype)
        return self

    def numpy(self):
        return self._data


class _FakeModule:
    """Minimal stand-in for a tcim_lite runtime module (name-based I/O)."""

    def __init__(self, input_names, output_name, output_data):
        self._input_names = input_names
        self._output_name = output_name
        self._output_data = output_data
        self.set_inputs: list[tuple[str, np.ndarray]] = []
        self.ran = False

    def get_num_inputs(self):
        return len(self._input_names)

    def get_input_name(self, idx):
        return self._input_names[idx]

    def get_input_info(self, name):
        raise AssertionError("get_input_info should not be called by HMMRuntimeSession.execute")

    def set_input(self, name, data):
        self.set_inputs.append((name, np.asarray(data)))

    def run(self):
        self.ran = True

    def sync(self):
        pass

    def get_num_outputs(self):
        return 1

    def get_output_name(self, idx):
        assert idx == 0
        return self._output_name

    def get_output(self, name):
        assert name == self._output_name
        return _FakeOutput(self._output_data)


def _install_fake_tcim_lite(monkeypatch, module):
    fake_runtime = types.ModuleType("tcim_lite.runtime")
    fake_runtime.load = lambda path: module
    fake_api = types.ModuleType("tcim_lite")
    fake_api.runtime = fake_runtime
    monkeypatch.setitem(sys.modules, "tcim_lite", fake_api)
    monkeypatch.setitem(sys.modules, "tcim_lite.runtime", fake_runtime)


def test_create_hmm_policy_wrapper_rejects_other_devices():
    with pytest.raises(ValueError, match="Unsupported HMM"):
        create_hmm_policy_wrapper("cuda")


def test_create_runtime_session_builds_hmm_session():
    session = create_runtime_session("hmm", {"type": "act"})
    assert isinstance(session, HMMRuntimeSession)


def test_resolve_hmm_model_path_prefers_env(tmp_path, monkeypatch):
    _write_policy(tmp_path, _act_config())
    external = tmp_path / "external.hmm"
    external.write_bytes(b"hmm")
    monkeypatch.setenv("HMM_MODEL_PATH", str(external))
    resolved = resolve_hmm_model_path(str(tmp_path))
    assert resolved == external.resolve()


def test_resolve_hmm_model_path_directory_convention(tmp_path):
    _write_policy(tmp_path, _act_config())
    resolved = resolve_hmm_model_path(str(tmp_path))
    assert resolved.name == "model.hmm"


def test_resolve_hmm_model_path_missing(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_act_config()), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="HMM model file not found"):
        resolve_hmm_model_path(str(tmp_path))


def test_hmm_runtime_session_load_sets_input_names(tmp_path, monkeypatch):
    _write_policy(tmp_path, _act_config())
    module = _FakeModule(
        ["observation.images.wrist", "observation.state", "observation.images.top"],
        "action",
        np.zeros((1, 100, 6), dtype=np.float32),
    )
    _install_fake_tcim_lite(monkeypatch, module)

    session = HMMRuntimeSession()
    session.load(str(tmp_path), _act_config(), torch.device("cpu"))
    assert session._input_names == [
        "observation.images.wrist",
        "observation.state",
        "observation.images.top",
    ]


def test_hmm_runtime_session_execute_sets_inputs_by_name_and_runs(tmp_path, monkeypatch):
    _write_policy(tmp_path, _act_config())
    module = _FakeModule(
        ["observation.images.wrist", "observation.state", "observation.images.top"],
        "action",
        np.arange(1 * 100 * 6, dtype=np.float32).reshape(1, 100, 6),
    )
    _install_fake_tcim_lite(monkeypatch, module)

    session = HMMRuntimeSession()
    session.load(str(tmp_path), _act_config(), torch.device("cpu"))

    inputs = [
        np.full((1, 3, 4, 5), 1.0, dtype=np.float32),
        np.full((1, 6), 2.0, dtype=np.float32),
        np.full((1, 3, 4, 5), 3.0, dtype=np.float32),
    ]
    outputs = session.execute(inputs)

    assert module.ran
    assert [name for name, _ in module.set_inputs] == [
        "observation.images.wrist",
        "observation.state",
        "observation.images.top",
    ]
    assert float(module.set_inputs[1][1][0, 0]) == 2.0
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 100, 6)
    assert outputs[0].dtype == np.float32


def test_hmm_runtime_session_execute_validates_input_count(tmp_path, monkeypatch):
    _write_policy(tmp_path, _act_config())
    module = _FakeModule(["a", "b"], "action", np.zeros((1, 2, 6), dtype=np.float32))
    _install_fake_tcim_lite(monkeypatch, module)

    session = HMMRuntimeSession()
    session.load(str(tmp_path), _act_config(), torch.device("cpu"))
    with pytest.raises(RuntimeError, match="expected 2 inputs, got 1"):
        session.execute([np.zeros((1, 6), dtype=np.float32)])


def test_hmm_runtime_session_not_loaded(tmp_path):
    session = HMMRuntimeSession()
    with pytest.raises(RuntimeError, match="not loaded"):
        session.execute([np.zeros((1, 6), dtype=np.float32)])


def test_hmm_wrapper_requires_config_json(tmp_path):
    (tmp_path / "model.hmm").write_bytes(b"hmm")
    wrapper = HMMPolicyWrapper()

    with pytest.raises(FileNotFoundError, match="config.json"):
        wrapper.load(str(tmp_path), torch.device("cpu"))


def test_hmm_wrapper_end_to_end_preserves_input_feature_order(tmp_path, monkeypatch):
    config = _act_config()
    _write_policy(tmp_path, config)

    module = _FakeModule(
        ["observation.images.wrist", "observation.state", "observation.images.top"],
        "action",
        np.zeros((1, 100, 6), dtype=np.float32),
    )
    _install_fake_tcim_lite(monkeypatch, module)

    wrapper = HMMPolicyWrapper()
    wrapper.load(str(tmp_path), torch.device("cpu"))
    assert wrapper.policy_type == "act"
    assert wrapper.backend_type == "hmm"
    assert wrapper.uses_action_chunking is True
    assert wrapper.get_chunk_size() == 100

    batch = {
        "observation.images.wrist": torch.full((3, 4, 5), 1.0),
        "observation.state": torch.full((6,), 2.0),
        "observation.images.top": torch.full((3, 4, 5), 3.0),
    }
    action = wrapper.infer(batch)

    # ACT adapter expands single image state dims to (1, *) and feeds inputs in
    # config input_features order (state/image keys only).
    assert [name for name, _ in module.set_inputs] == [
        "observation.images.wrist",
        "observation.state",
        "observation.images.top",
    ]
    # Output decoded to (chunk_size, action_dim) on the runtime device.
    assert tuple(action.shape) == (100, 6)


def test_hmm_manifest_basename_is_isolated_from_om():
    assert HMM_MANIFEST_BASENAME == "config.hmm.json"
    from inference_service.core.compiled_policy import COMPILED_MANIFEST_BASENAME

    assert COMPILED_MANIFEST_BASENAME == "config.om.json"
