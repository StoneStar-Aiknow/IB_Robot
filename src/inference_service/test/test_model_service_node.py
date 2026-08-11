from types import SimpleNamespace

from inference_service.model_service_node import ModelServiceNode
from inference_service.model_service_plugin import ModelServiceError, PluginRuntimeStatus


class _Response(SimpleNamespace):
    success = False
    message = ""
    error_code = ""
    inference_time_ms = 0.0
    model = None


def _host(plugin):
    return SimpleNamespace(
        plugin=plugin,
        failure_reason="",
        runtime_info=lambda: "runtime-info",
    )


def test_request_error_does_not_poison_runtime_health():
    class Plugin:
        def runtime_status(self):
            return PluginRuntimeStatus(state="ready", ready=True)

        def handle(self, _request, _response):
            raise ModelServiceError("invalid request", error_code="INVALID_TEXT")

    host = _host(Plugin())
    response = ModelServiceNode._dispatch(host, object(), _Response())

    assert not response.success
    assert response.error_code == "INVALID_TEXT"
    assert response.model == "runtime-info"
    assert host.failure_reason == ""


def test_successful_request_after_request_error_stays_available():
    class Plugin:
        calls = 0

        def runtime_status(self):
            return PluginRuntimeStatus(state="ready", ready=True)

        def handle(self, _request, _response):
            self.calls += 1
            if self.calls == 1:
                raise ModelServiceError("unsupported prompt", error_code="UNSUPPORTED_PROMPT")
            return "ok"

    host = _host(Plugin())

    first = ModelServiceNode._dispatch(host, object(), _Response())
    second = ModelServiceNode._dispatch(host, object(), _Response())

    assert not first.success
    assert first.error_code == "UNSUPPORTED_PROMPT"
    assert second.success
    assert second.message == "ok"
    assert host.failure_reason == ""
