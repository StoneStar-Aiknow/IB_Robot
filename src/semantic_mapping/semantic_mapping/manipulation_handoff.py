"""Fresh target confirmation before invoking the existing grasp planner."""

from dataclasses import dataclass

from .association import SemanticTrack


@dataclass(frozen=True)
class ManipulationHandoffResult:
    success: bool
    message: str
    confirmation: object | None = None
    grasp_plan: object | None = None


def handoff_to_manipulation(track: SemanticTrack, confirm_target, plan_grasp) -> ManipulationHandoffResult:
    if track.state != "observed":
        return ManipulationHandoffResult(False, f"object state {track.state} is not manipulation-ready")
    confirmation = confirm_target(track)
    if confirmation is None or not confirmation.success:
        message = "fresh target confirmation failed"
        if confirmation is not None and getattr(confirmation, "message", ""):
            message = confirmation.message
        return ManipulationHandoffResult(False, message, confirmation=confirmation)
    plan = plan_grasp(confirmation)
    if plan is None or not plan.success:
        message = "grasp planning failed"
        if plan is not None and getattr(plan, "message", ""):
            message = plan.message
        return ManipulationHandoffResult(False, message, confirmation=confirmation, grasp_plan=plan)
    return ManipulationHandoffResult(True, "fresh confirmation and grasp planning succeeded", confirmation, plan)
