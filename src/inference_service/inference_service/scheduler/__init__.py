"""Global inference scheduler control-plane primitives."""

from inference_service.scheduler.deadline_reservations import DeadlineReservation, DeadlineReservationTable

__all__ = ["DeadlineReservation", "DeadlineReservationTable"]
