"""Scheduling primitives, with product APIs loaded only by product consumers."""

from importlib import import_module

__all__ = [
    "DeadlineReservation",
    "DeadlineReservationTable",
    "Certainty",
    "DownstreamOperationContext",
    "OperationState",
    "OperationIdentity",
    "OperationKind",
    "OperationRegistry",
    "BindingDecision",
    "BindingState",
    "GlobalSchedulerCore",
    "GlobalSessionState",
    "PipelineBinding",
    "PipelineCandidate",
    "SchedulerError",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = (
        "inference_service.scheduler.global_scheduler_core"
        if name
        in {
            "BindingDecision",
            "BindingState",
            "GlobalSchedulerCore",
            "GlobalSessionState",
            "PipelineBinding",
            "PipelineCandidate",
            "SchedulerError",
        }
        else "inference_service.unified_runtime.operations"
    )
    if name in {"DeadlineReservation", "DeadlineReservationTable"}:
        module = "inference_service.scheduler.deadline_reservations"
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
