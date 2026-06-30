#!/usr/bin/env python3
"""Tests for SD3403 ACT worker protocol parsing."""

from __future__ import annotations

import io
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from inference_service.core.ascend_om import ACTWrapper_3403 as act3403
from inference_service.core.ascend_om.ACTWrapper_3403 import (
    DEFAULT_ACTION_DIM,
    DIM_STRUCT,
    LEGACY_WORKER_ENV_KEYS,
    OUTPUT_ENTRY_STRUCT,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    RESPONSE_HEADER_STRUCT,
    WORKER_ELEM_FLOAT32,
    WORKER_STATUS_OK,
    ACT3403Policy,
    decode_sd3403_action_array,
)

ACTION_CHUNK_SIZE = 100


class _FakeProcess:
    def __init__(self, stdout: bytes):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO()
        self.terminated = False
        self.killed = False
        self.returncode = None
        self.wait_timeouts = []
        self.pid = 12345

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True


class _NonExitingProcess(_FakeProcess):
    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired("worker", timeout)


def _response_frame(outputs, *, request_id=1, prefix=b"", status=WORKER_STATUS_OK, error_code=0, error_msg=""):
    error_msg_bytes = error_msg.encode("utf-8")
    frame = bytearray(prefix)
    frame.extend(
        RESPONSE_HEADER_STRUCT.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            status,
            request_id,
            len(outputs),
            1234,
            error_code,
            len(error_msg_bytes),
        )
    )
    for output_index, array in outputs:
        payload = np.ascontiguousarray(array, dtype=np.float32)
        frame.extend(
            OUTPUT_ENTRY_STRUCT.pack(
                output_index,
                WORKER_ELEM_FLOAT32,
                payload.size,
                payload.nbytes,
                payload.ndim,
                0,
            )
        )
        for dim in payload.shape:
            frame.extend(DIM_STRUCT.pack(dim))
        frame.extend(memoryview(payload).cast("B"))
    frame.extend(error_msg_bytes)
    return bytes(frame)


def _malformed_output_response(*, elem_count: int, byte_size: int, dim_count: int) -> bytes:
    frame = bytearray(
        RESPONSE_HEADER_STRUCT.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            WORKER_STATUS_OK,
            1,
            1,
            1234,
            0,
            0,
        )
    )
    frame.extend(
        OUTPUT_ENTRY_STRUCT.pack(
            1,
            WORKER_ELEM_FLOAT32,
            elem_count,
            byte_size,
            dim_count,
            0,
        )
    )
    frame.extend(b"\0" * byte_size)
    return bytes(frame)


def _policy_for_response(frame: bytes) -> ACT3403Policy:
    policy = ACT3403Policy.__new__(ACT3403Policy)
    policy._process = _FakeProcess(frame)
    policy._stderr_tail = deque()
    policy._stderr_thread = None
    policy._action_dim = DEFAULT_ACTION_DIM
    policy._action_output_index = 1
    policy._graceful_close_timeout = 5.0
    policy._force_close = True
    return policy


def _policy_for_image_normalize(image_height, image_width) -> ACT3403Policy:
    """Build a policy instance for _normalize_image_tensor tests without starting a worker."""
    policy = ACT3403Policy.__new__(ACT3403Policy)
    policy._process = None  # avoid __del__ noise during GC (no worker started)
    policy._image_height = image_height
    policy._image_width = image_width
    policy._resize_warned = False
    return policy


def test_normalize_image_tensor_no_resize_when_dims_none():
    """When image_height/width are None, the tensor passes through unchanged
    (the resolution is the model's input contract, derived from config.json)."""
    import torch

    policy = _policy_for_image_normalize(None, None)
    image = torch.zeros((1, 3, 480, 640), dtype=torch.float32)
    out = policy._normalize_image_tensor(image, "observation.images.top")
    assert out.shape == (1, 3, 480, 640)


def test_normalize_image_tensor_resizes_when_dims_specified():
    """When image_height/width are explicitly set, inputs are resized to that target."""
    import torch

    policy = _policy_for_image_normalize(240, 320)
    image = torch.zeros((1, 3, 480, 640), dtype=torch.float32)
    out = policy._normalize_image_tensor(image, "observation.images.top")
    assert out.shape == (1, 3, 240, 320)


def test_normalize_image_tensor_passes_through_matching_size():
    """When dims specified and input already matches, no resize occurs."""
    import torch

    policy = _policy_for_image_normalize(480, 640)
    image = torch.ones((1, 3, 480, 640), dtype=torch.float32)
    out = policy._normalize_image_tensor(image, "observation.images.top")
    assert out.shape == (1, 3, 480, 640)


def test_read_response_syncs_past_stdout_prefix_and_selects_action_shape():
    hidden = np.zeros((1, 100, 2048), dtype=np.float32)
    action = np.arange(ACTION_CHUNK_SIZE * DEFAULT_ACTION_DIM, dtype=np.float32).reshape(
        1,
        ACTION_CHUNK_SIZE,
        DEFAULT_ACTION_DIM,
    )
    frame = _response_frame(
        [(0, hidden), (1, action)],
        request_id=7,
        prefix=b"load sys ....OK!\nload svp_npu ....OK!\n",
    )
    policy = _policy_for_response(frame)

    data, latency_us = policy._read_response(7)

    assert latency_us == 1234
    assert data.shape == (1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM)
    assert float(data[0, ACTION_CHUNK_SIZE - 1, DEFAULT_ACTION_DIM - 1]) == float(action.size - 1)


def test_read_response_accepts_single_direct_action_output():
    action = np.arange(ACTION_CHUNK_SIZE * DEFAULT_ACTION_DIM, dtype=np.float32).reshape(
        1,
        ACTION_CHUNK_SIZE,
        DEFAULT_ACTION_DIM,
    )
    frame = _response_frame([(1, action)], request_id=8)
    policy = _policy_for_response(frame)

    data, latency_us = policy._read_response(8)

    assert latency_us == 1234
    assert data.shape == (1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM)
    assert float(data[0, ACTION_CHUNK_SIZE - 1, DEFAULT_ACTION_DIM - 1]) == float(action.size - 1)


def test_read_response_prefers_index_one_action_over_other_6d_output():
    state_like = np.full((1, 1, DEFAULT_ACTION_DIM), -1.0, dtype=np.float32)
    action = np.arange(ACTION_CHUNK_SIZE * DEFAULT_ACTION_DIM, dtype=np.float32).reshape(
        1,
        ACTION_CHUNK_SIZE,
        DEFAULT_ACTION_DIM,
    )
    frame = _response_frame([(0, state_like), (1, action)], request_id=9)
    policy = _policy_for_response(frame)

    data, _latency_us = policy._read_response(9)

    assert data.shape == (1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM)
    assert float(data[0, ACTION_CHUNK_SIZE - 1, DEFAULT_ACTION_DIM - 1]) == float(action.size - 1)


def test_read_response_accepts_configured_action_output_index():
    action = np.arange(ACTION_CHUNK_SIZE * DEFAULT_ACTION_DIM, dtype=np.float32).reshape(
        1,
        ACTION_CHUNK_SIZE,
        DEFAULT_ACTION_DIM,
    )
    frame = _response_frame([(2, action)], request_id=11)
    policy = _policy_for_response(frame)
    policy._action_output_index = 2

    data, _latency_us = policy._read_response(11)

    assert data.shape == (1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM)
    assert float(data[0, ACTION_CHUNK_SIZE - 1, DEFAULT_ACTION_DIM - 1]) == float(action.size - 1)


def test_read_response_raises_worker_error_message():
    frame = _response_frame([], request_id=12, status=1, error_code=7, error_msg="extract output failed")
    policy = _policy_for_response(frame)

    with pytest.raises(RuntimeError, match="extract output failed"):
        policy._read_response(12)


def test_decode_sd3403_action_array_accepts_direct_action_shape():
    direct = decode_sd3403_action_array(np.zeros((1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM), dtype=np.float32))

    assert direct.shape == (1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM)
    assert direct.flags.writeable


def test_decode_sd3403_action_array_accepts_configured_action_dim():
    action_dim = 9
    direct = decode_sd3403_action_array(np.zeros((1, 12, action_dim), dtype=np.float32), action_dim=action_dim)

    assert direct.shape == (1, 12, action_dim)


def test_decode_sd3403_action_array_returns_writable_copy_for_worker_buffer():
    raw = np.frombuffer(
        np.arange(ACTION_CHUNK_SIZE * DEFAULT_ACTION_DIM, dtype=np.float32).tobytes(),
        dtype=np.float32,
    ).reshape(1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM)

    decoded = decode_sd3403_action_array(raw)

    assert decoded.shape == (1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM)
    assert decoded.flags.writeable


def test_decode_sd3403_action_array_rejects_bad_action_dim():
    with pytest.raises(ValueError, match="action_dim"):
        decode_sd3403_action_array(np.zeros((1, 1, DEFAULT_ACTION_DIM), dtype=np.float32), action_dim=0)


def test_decode_sd3403_action_array_rejects_unexpected_last_dim():
    # The worker returns (1, chunk, action_dim); a different last dim is an error.
    with pytest.raises(RuntimeError, match="unexpected action tensor shape"):
        decode_sd3403_action_array(np.zeros((1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM + 2), dtype=np.float32))


def test_read_response_rejects_bad_request_id():
    frame = _response_frame(
        [(1, np.zeros((1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM), dtype=np.float32))],
        request_id=3,
    )
    policy = _policy_for_response(frame)

    with pytest.raises(RuntimeError, match="mismatched response id"):
        policy._read_response(4)


def test_read_response_rejects_oversized_stdout_prefix(monkeypatch):
    monkeypatch.setattr(act3403, "MAX_RESPONSE_PREFIX_BYTES", 8)
    frame = _response_frame([(1, np.zeros((1, ACTION_CHUNK_SIZE, DEFAULT_ACTION_DIM), dtype=np.float32))])
    policy = _policy_for_response(b"x" * 12 + frame)

    with pytest.raises(RuntimeError, match="response header was not found"):
        policy._read_response(1)


def test_read_response_rejects_unreasonable_output_dim_count():
    frame = _malformed_output_response(
        elem_count=1,
        byte_size=4,
        dim_count=act3403.MAX_WORKER_OUTPUT_DIM_COUNT + 1,
    )
    policy = _policy_for_response(frame)

    with pytest.raises(RuntimeError, match="unreasonable response output"):
        policy._read_response(1)


def test_read_response_rejects_output_byte_size_mismatch():
    frame = _malformed_output_response(elem_count=2, byte_size=4, dim_count=0)
    policy = _policy_for_response(frame)

    with pytest.raises(RuntimeError, match="byte size does not match"):
        policy._read_response(1)


def test_close_prefers_stdin_eof_over_force_close():
    policy = _policy_for_response(b"")
    process = policy._process

    policy.close()

    assert process.terminated is False
    assert process.killed is False
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert policy._process is None


def test_close_detaches_state_when_worker_ignores_eof():
    policy = _policy_for_response(b"")
    process = _NonExitingProcess(b"")
    policy._process = process
    policy._force_close = False

    policy.close()

    assert process.terminated is False
    assert process.killed is False
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert policy._process is None
    assert policy._stderr_thread is None


def test_close_uses_configured_graceful_timeout():
    policy = _policy_for_response(b"")
    policy._graceful_close_timeout = 1.25
    process = policy._process

    policy.close()

    assert process.wait_timeouts == [1.25]
    assert policy._process is None


def test_close_force_close_uses_configured_flag():
    policy = _policy_for_response(b"")
    process = _NonExitingProcess(b"")
    policy._process = process

    policy.close()

    assert process.terminated is True
    assert process.killed is True
    assert policy._process is None


def test_policy_starts_worker_with_model_cli_arg(monkeypatch, caplog, capfd, tmp_path):
    worker = tmp_path / "main"
    model = tmp_path / "model.om"
    worker.write_bytes(b"")
    model.write_bytes(b"om")
    worker.chmod(0o755)
    for key in LEGACY_WORKER_ENV_KEYS:
        monkeypatch.setenv(key, f"legacy-{key}")

    popen_call = {}

    class _StartedProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    def fake_popen(args, **kwargs):
        popen_call["args"] = args
        popen_call["env"] = kwargs.get("env", {})
        return _StartedProcess()

    monkeypatch.setattr(
        "inference_service.core.ascend_om.ACTWrapper_3403.subprocess.Popen",
        fake_popen,
    )

    policy = ACT3403Policy(str(worker), str(model))
    policy.close()

    assert popen_call["args"] == [
        str(Path(worker).resolve()),
        "--model",
        str(Path(model).resolve()),
    ]
    # The legacy-env warning is emitted via LOGGER.warning. Different pytest
    # environments route logging output differently (Python logging handler vs
    # raw stderr fd), so collect from both caplog and capfd and accept either.
    fd_text = capfd.readouterr().err
    warning_text = caplog.text or fd_text
    for key in LEGACY_WORKER_ENV_KEYS:
        assert key not in popen_call["env"]
        assert key in warning_text
