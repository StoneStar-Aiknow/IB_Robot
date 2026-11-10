"""Public launch projection for processes that execute local models."""

from inference_manifest.runtime_target import DEFAULT_RUNTIME_ENDPOINT
from robot_config.runtime_scheduling import scheduling_available
from robot_config.runtime_target_config import runtime_scheduler_block

# Model hosts keep control transport alive while draining requests and SDK
# cleanup (50 s). Leave room for final receipts before launch forces exit.
MODEL_HOST_SHUTDOWN_PARAMETERS = {"sigterm_timeout": "65.0", "sigkill_timeout": "5.0"}


def model_runtime_connection_parameters(robot_config):
    """Keep control-plane parameter names and defaults in the composition layer."""
    block = runtime_scheduler_block(robot_config)
    if not scheduling_available(robot_config):
        return {"runtime_scheduler_enabled": False}
    return {
        "runtime_scheduler_enabled": True,
        "runtime_scheduler_endpoint": block.get("runtime_endpoint", DEFAULT_RUNTIME_ENDPOINT),
        "runtime_initialization_timeout": float(block.get("startup_readiness_timeout", 60.0)),
    }
