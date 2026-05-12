from perception_service.model_timeout import (
    DEFAULT_MODEL_OUTPUT_IDLE_TIMEOUT_SEC,
    resolve_model_output_idle_timeout,
)


def test_resolve_model_output_idle_timeout_uses_configured_value():
    assert resolve_model_output_idle_timeout(configured_timeout_sec=8.0) == 8.0


def test_resolve_model_output_idle_timeout_uses_larger_value():
    assert resolve_model_output_idle_timeout(configured_timeout_sec=180.0) == 180.0


def test_resolve_model_output_idle_timeout_uses_override_exactly():
    assert (
        resolve_model_output_idle_timeout(
            configured_timeout_sec=120.0,
            override_timeout_sec=20.0,
        )
        == 20.0
    )


def test_resolve_model_output_idle_timeout_falls_back_to_default():
    assert resolve_model_output_idle_timeout(configured_timeout_sec=0.0) == DEFAULT_MODEL_OUTPUT_IDLE_TIMEOUT_SEC
