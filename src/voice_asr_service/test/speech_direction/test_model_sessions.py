"""Tests for speech-direction ModelSession protocol adapters."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2]
_WORKSPACE_SRC = _SRC.parent
for package_root in (_SRC, _WORKSPACE_SRC / "inference_manifest", _WORKSPACE_SRC / "inference_service"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from inference_manifest import load_inference_manifest  # noqa: E402
from inference_service.backends import RuntimeContext  # noqa: E402
from inference_service.model_sessions import MODEL_SESSION_BUILDER_REGISTRY, StatefulAscendOmModelSession  # noqa: E402
from voice_asr_service.model_session_builders import register_speech_direction_session_builder  # noqa: E402
from voice_asr_service.speech_direction.model_sessions import SpeechDirectionRoleRunner  # noqa: E402


class _Execution:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, role, values):
        self.calls.append((role, values))
        if role == "fullsubnet_fb":
            return {"host.fullsubnet.fb_features": np.zeros((4, 2, 257), dtype=np.float32)}
        return {"host.fullsubnet.sb_mask": np.zeros((1028, 2, 2), dtype=np.float32)}


class _Session:
    def __init__(self) -> None:
        self.execution_calls = 0
        self.execute_role_calls = []
        self.last_execution = None

    @contextmanager
    def execution(self, request):
        self.execution_calls += 1
        self.last_execution = _Execution()
        yield self.last_execution

    def execute_role(self, role, values, request):
        self.execute_role_calls.append((role, values, request))
        return {"host.silero.prob": np.array([[0.75]], dtype=np.float32)}


def test_fullsubnet_roles_share_one_session_execution_scope() -> None:
    session = _Session()
    runner = SpeechDirectionRoleRunner(session, context=None)

    with runner.execution_scope():
        fb = runner.run_fb(np.zeros((4, 2, 257), dtype=np.float32))
        sb = runner.run_sb(np.zeros((1028, 2, 32), dtype=np.float32))

    assert fb.shape == (4, 2, 257)
    assert sb.shape == (1028, 2, 2)
    assert session.execution_calls == 1
    assert [call[0] for call in session.last_execution.calls] == ["fullsubnet_fb", "fullsubnet_sb"]
    assert session.execute_role_calls == []


def test_role_runner_rejects_nested_execution_scopes() -> None:
    runner = SpeechDirectionRoleRunner(_Session(), context=None)

    with runner.execution_scope(), pytest.raises(RuntimeError, match="already active"), runner.execution_scope():
        pass


def test_silero_inference_uses_standalone_session_execution() -> None:
    session = _Session()
    runner = SpeechDirectionRoleRunner(session, context=None)

    probability = runner.infer(np.zeros((1, 576), dtype=np.float32))

    assert probability == pytest.approx(0.75)
    assert session.execution_calls == 0
    assert [call[0] for call in session.execute_role_calls] == ["silero_vad"]


@pytest.mark.parametrize("deployment_name", ["ascend_310p_fullsubnet", "ascend_310p_silero"])
def test_checked_in_speech_manifest_selects_generic_stateful_session(tmp_path, deployment_name) -> None:
    config_root = _SRC / "config"
    manifest = json.loads((config_root / "inference_manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    adapter_dir = tmp_path / "assets"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.json").write_bytes((config_root / "assets" / "adapter.json").read_bytes())
    for deployment in manifest["deployments"].values():
        for artifact in deployment["artifacts"].values():
            path = tmp_path / artifact["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mock-om")

    context = RuntimeContext(load_inference_manifest(tmp_path, deployment_name), {"device_id": 0})
    register_speech_direction_session_builder()
    session = MODEL_SESSION_BUILDER_REGISTRY.create(context)

    assert isinstance(session, StatefulAscendOmModelSession)
