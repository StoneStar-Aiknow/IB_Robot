"""Global must start even when every model composition/plugin is unavailable."""

import subprocess
import sys

import pytest


def test_global_starts_without_model_composition_or_business_plugins():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith((
            "inference_service.runtime_composition", "inference_service.pipeline.factory",
            "perception_service", "voice_tts_service", "voice_asr_service",
            "inference_service.policy_session_service_node", "inference_service.scheduler.global_scheduler_core",
            "inference_service.scheduler.pipeline_registration", "inference_service.scheduler.policy_registration_ros",
        )):
            raise AssertionError("Global imported model dependency: " + fullname)

sys.meta_path.insert(0, NoModels())
import rclpy
from rclpy.parameter import Parameter
from inference_service.global_runtime_scheduler_node import GlobalRuntimeSchedulerNode

rclpy.init()
for name, value in (("profile_min_samples", 20), ("profile_max_age_days", 0), ("status_stale_timeout_ns", 0)):
    try:
        GlobalRuntimeSchedulerNode(parameter_overrides=[Parameter(name, value=value)])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid production scheduler parameter accepted: " + name)
node = GlobalRuntimeSchedulerNode(parameter_overrides=[
    Parameter("runtime_resources_json", value='{"cpu":{"physical_device":"local/cpu"}}'),
])
try:
    assert node._runtime_endpoint is not None
finally:
    node.destroy_node()
    rclpy.shutdown()
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_global_main_shuts_down_context_after_constructor_failure(monkeypatch):
    from inference_service import global_runtime_scheduler_node as module

    events = []
    monkeypatch.setattr(module.rclpy, "init", lambda **kwargs: events.append("init"))
    monkeypatch.setattr(module.rclpy, "try_shutdown", lambda: events.append("shutdown"))

    def fail():
        raise ValueError("invalid endpoint")

    monkeypatch.setattr(module, "GlobalRuntimeSchedulerNode", fail)
    with pytest.raises(ValueError, match="invalid endpoint"):
        module.main()
    assert events == ["init", "shutdown"]
