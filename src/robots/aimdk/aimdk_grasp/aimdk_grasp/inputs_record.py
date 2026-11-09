"""Diagnostic records of grasp/place inputs, kept for offline replay when a directory is configured.

IBROBOT_X2_PLAN_INPUTS_DIR names the directory (planner_service, place_request,
grasp_head, place_verify write there). A record is pickled on a daemon thread,
so the request does not wait for the file; a failure to save is printed and
never changes the request's outcome.
"""

import os
import pickle
import sys
import threading
import time
from pathlib import Path

INPUTS_DIR_ENV = "IBROBOT_X2_PLAN_INPUTS_DIR"


def inputs_directory() -> Path | None:
    directory = os.environ.get(INPUTS_DIR_ENV)
    return Path(directory) if directory else None


def save_record(directory: Path, prefix: str, record: dict) -> threading.Thread:
    """Pickle ``record`` as ``<prefix>_<ns>.pkl`` in ``directory`` on a thread; renamed into place when complete.

    Everything in ``record`` must no longer be changed by its producer.
    """

    def write():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{prefix}_{time.time_ns()}.pkl"
            partial = path.with_suffix(".tmp")
            with open(partial, "wb") as stream:
                pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
            partial.rename(path)
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never change the outcome
            print(f"{prefix} record not saved: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=write, name=f"{prefix}-record", daemon=True)
    thread.start()
    return thread


def observation_fields(observation) -> dict:
    """The plain arrays and transforms of an Observation, without its live robot-state objects.

    The image-time feedback is kept (robot self filter and hand covers need it).
    """
    robot_state = getattr(observation, "robot_state", None)
    live = getattr(robot_state, "live", None)
    registered = getattr(live, "observation", None)
    return dict(
        rgb=getattr(observation, "rgb", None),
        depth_m=observation.depth_m,
        intrinsics=observation.intrinsics,
        frame_id=getattr(observation, "frame_id", None),
        stamp_ns=getattr(observation, "stamp_ns", None),
        base_frame=getattr(observation, "base_frame", None),
        base_from_camera=observation.base_from_camera,
        image_feedback=getattr(registered, "image_feedback", None),
        report=dict(getattr(observation, "report", None) or {}),
    )
