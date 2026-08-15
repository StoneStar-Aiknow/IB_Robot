from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.backends.errors import BackendInferenceError
from inference_service.generic_runtime import NamedTensorRequest
from voice_tts_service.errors import BackendLoadError
from voice_tts_service.zipvoice_onnx_adapter import ZipVoiceOnnxSession


class _Tokenizer:
    pad_id = 0

    @staticmethod
    def text_to_tokens(text):
        return list(text)

    @staticmethod
    def tokens_to_ids(tokens):
        return [1] * len(tokens)

    @staticmethod
    def chunk_tokens(tokens, max_tokens):
        return [tokens[index : index + max_tokens] for index in range(0, len(tokens), max_tokens)]


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros((1, 240), dtype=np.float32)


class _Torch:
    @staticmethod
    def from_numpy(value):
        return _Tensor(value)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _Vocos:
    def __call__(self, value):
        return value


class _OrtInput:
    def __init__(self, name):
        self.name = name


class _OrtOutput:
    def __init__(self, name):
        self.name = name


class _OrtSession:
    """Minimal onnxruntime InferenceSession mock.

    The text encoder receives 4 inputs and the flow decoder receives 5 inputs;
    the mock dispatches on input count to return the right output shape.
    """

    def __init__(self, num_inputs, output_name, velocity_fn=None):
        self._inputs = [_OrtInput(f"input_{i}") for i in range(num_inputs)]
        self._outputs = [_OrtOutput(output_name)]
        self._velocity_fn = velocity_fn or (lambda inputs: np.zeros_like(inputs["input_1"]))

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={"feat_dim": "100"})

    def run(self, output_names, inputs):
        if len(inputs) == 4:
            num_tokens = int(np.asarray(inputs["input_0"]).shape[1])
            frames = 302 + num_tokens
            return [np.zeros((1, frames, 100), dtype=np.float32)]
        if len(inputs) == 5:
            return [self._velocity_fn(inputs)]
        raise AssertionError(f"unexpected run inputs: {sorted(inputs)}")


def _onnx_session(*, velocity_fn=None, config=None):
    session = object.__new__(ZipVoiceOnnxSession)
    session._root = None
    session._prompt_profile_name = "default"
    session._config = config or {
        "text_capacity": 256,
        "num_steps": 8,
        "t_shift": 0.5,
        "sample_rate": 24000,
        "cross_fade_sec": 0.1,
        "speed": 1.0,
        "guidance_scale": 3.0,
        "feature_scale": 0.1,
        "seed": 42,
    }
    session._tokenizer = _Tokenizer()
    session._prompt = SimpleNamespace(
        tokens=np.zeros((1, 29), dtype=np.int64),
        features=np.zeros((1, 302, 100), dtype=np.float32),
        frame_count=302,
    )
    session._torch = _Torch()
    session._vocos = _Vocos()
    session._text_encoder = _OrtSession(4, "text_condition")
    session._fm_decoder = _OrtSession(5, "velocity", velocity_fn=velocity_fn)
    session._feat_dim = 100
    return session


def _request(text="机器人", *, prompt=False):
    return NamedTensorRequest(
        "tts-test",
        {
            "tts.text": np.frombuffer(text.encode(), dtype=np.uint8),
            "tts.prompt_audio": np.ones(1, dtype=np.float32) if prompt else np.empty(0, dtype=np.float32),
            "tts.prompt_sample_rate": np.asarray(24000 if prompt else 0, dtype=np.int64),
            "tts.prompt_text": np.frombuffer("参考".encode(), dtype=np.uint8)
            if prompt
            else np.empty(0, dtype=np.uint8),
        },
    )


def test_onnx_session_runs_eight_flow_steps_and_returns_pcm():
    session = _onnx_session()

    outputs = session._execute(_request())

    assert outputs["tts.audio"].shape == (240,)
    assert outputs["tts.audio"].dtype == np.float32


def test_onnx_session_explicitly_rejects_request_prompt():
    with pytest.raises(BackendInferenceError, match="fixed prompt profile") as error:
        _onnx_session()._execute(_request(prompt=True))
    assert error.value.code == "unsupported_prompt"


def test_onnx_session_rejects_unloaded_assets():
    session = _onnx_session()
    session._text_encoder = None

    with pytest.raises(BackendInferenceError, match="assets are not loaded"):
        session._execute(_request())


def test_onnx_session_rejects_invalid_utf8_text():
    session = _onnx_session()
    bad_request = NamedTensorRequest(
        "tts-test",
        {
            "tts.text": np.array([0xFF, 0xFE], dtype=np.uint8),
            "tts.prompt_audio": np.empty(0, dtype=np.float32),
            "tts.prompt_sample_rate": np.asarray(0, dtype=np.int64),
            "tts.prompt_text": np.empty(0, dtype=np.uint8),
        },
    )

    with pytest.raises(BackendInferenceError, match="not valid UTF-8") as error:
        session._execute(bad_request)
    assert error.value.code == "invalid_text"


def test_onnx_session_detects_nan_velocity():
    def nan_velocity(inputs):
        velocity = np.full_like(inputs["input_1"], np.nan, dtype=np.float32)
        return velocity

    session = _onnx_session(velocity_fn=nan_velocity)

    with pytest.raises(BackendInferenceError, match="NaN or Inf"):
        session._execute(_request())


def test_load_json_rejects_non_object(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(BackendLoadError, match="must be a JSON object"):
        ZipVoiceOnnxSession._load_json(bad_path)


def test_load_json_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(BackendLoadError, match="failed to read"):
        ZipVoiceOnnxSession._load_json(missing_path)


def test_runtime_version_reports_onnxruntime():
    session = _onnx_session()
    version = session.runtime_version

    assert "onnxruntime" in version
