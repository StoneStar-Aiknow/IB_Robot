"""Read a PEAR batch clip directory as a source animation.

The fitter writes one directory per clip and splits the fit across three files
with its own names:

    pear_decoded_outputs.npz   global_orient [N,1,3,3], body_pose [N,21,3,3], ...
    pear_ehm_joints.npz        joints [N,145,3]
    frame_records.json         frame_id, timestamp_ns, bbox, person_id per frame

`SourceAnimation` wants all five of its arrays in one npz under different
names, so nothing downstream can open a clip directory as it stands. This
module is the whole difference: it renames, it drops `global_orient`'s length-1
axis, and it lifts the per-frame records out of JSON. It fits no angles and
changes no numbers.

Two things it also carries, which the fitter records and the retargeting would
otherwise lose:

* `bbox_xyxy`, in DECODED frame coordinates -- several of these clips are
  phone video with a rotation flag, so the stored stream is 1280x720 while the
  frame the boxes were measured on is 720x1280. Anything drawing on top of the
  footage must decode first, as OpenCV and ffmpeg both do, and must not use the
  stream dimensions ffprobe reports.
* `detector_ran`, and the count of frames whose box fell back to the whole
  frame. A fallback box means the detector lost the subject on that frame and
  the fit ran on whatever was in the picture -- clip01 of the first batch has
  20 such frames, and they contain no person at all. Those frames produce a
  perfectly well-formed skeleton that is not anybody's pose. They are reported,
  never silently dropped: which frames to trust is the operator's call.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DECODED_FILE = "pear_decoded_outputs.npz"
JOINTS_FILE = "pear_ehm_joints.npz"
RECORDS_FILE = "frame_records.json"
CLIP_FILES: tuple[str, ...] = (DECODED_FILE, JOINTS_FILE, RECORDS_FILE)


def is_clip_directory(path: Path | str) -> bool:
    """True when `path` is a directory holding all three files of one clip."""
    path = Path(path)
    return path.is_dir() and all((path / name).is_file() for name in CLIP_FILES)


def _records(directory: Path) -> list[dict]:
    document = json.loads((directory / RECORDS_FILE).read_text())
    if not isinstance(document, dict) or "records" not in document:
        raise ValueError(f"{directory / RECORDS_FILE} has no 'records' list")
    records = document["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{directory / RECORDS_FILE} has an empty 'records' list")
    return records


def meta(directory: Path | str) -> dict:
    """The clip's own `meta` block: fps, fit mode, detector interval and misses."""
    directory = Path(directory)
    document = json.loads((directory / RECORDS_FILE).read_text())
    return dict(document.get("meta") or {})


def _squeeze_orient(global_orient: np.ndarray) -> np.ndarray:
    """[N,1,3,3] -> [N,3,3], and pass [N,3,3] through unchanged.

    Squeezing blindly would also silently accept [1,N,3,3] on a one-frame clip,
    which is a different array with the same shape. Check the axis instead.
    """
    if global_orient.ndim == 4:
        if global_orient.shape[1] != 1:
            raise ValueError(f"global_orient has shape {global_orient.shape}; "
                             f"expected [N,1,3,3] or [N,3,3]")
        return global_orient[:, 0]
    if global_orient.ndim == 3:
        return global_orient
    raise ValueError(f"global_orient has shape {global_orient.shape}; "
                     f"expected [N,1,3,3] or [N,3,3]")


def load_clip(directory: Path | str) -> dict[str, np.ndarray]:
    """One PEAR clip directory -> the arrays `SourceAnimation` expects, renamed."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    missing = [name for name in CLIP_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"{directory} is not a PEAR clip directory; missing {missing}")

    with np.load(directory / DECODED_FILE, allow_pickle=False) as decoded:
        for key in ("global_orient", "body_pose"):
            if key not in decoded:
                raise ValueError(f"{DECODED_FILE} is missing {key!r}; "
                                 f"present: {sorted(decoded.files)}")
        global_orient = _squeeze_orient(np.asarray(decoded["global_orient"], dtype=np.float64))
        body_pose = np.asarray(decoded["body_pose"], dtype=np.float64)

    with np.load(directory / JOINTS_FILE, allow_pickle=False) as joints_file:
        if "joints" not in joints_file:
            raise ValueError(f"{JOINTS_FILE} is missing 'joints'; "
                             f"present: {sorted(joints_file.files)}")
        points = np.asarray(joints_file["joints"], dtype=np.float64)

    records = _records(directory)
    frames = len(records)
    counts = {DECODED_FILE: len(body_pose), JOINTS_FILE: len(points), RECORDS_FILE: frames}
    if len(set(counts.values())) != 1:
        raise ValueError(f"the three files of {directory.name} disagree on frame count: {counts}")

    def column(key: str, dtype):
        try:
            return np.array([record[key] for record in records], dtype=dtype)
        except KeyError as error:
            raise ValueError(f"{RECORDS_FILE} records have no {key!r} field") from error

    return {
        "frame_id": column("frame_id", np.int64),
        "timestamp_ns": column("timestamp_ns", np.int64),
        "ehm_joints_3d_model": points,
        "body_pose_matrix": body_pose,
        "global_orient_matrix": global_orient,
        "bbox_xyxy": column("bbox", np.float64),
        "person_id": column("person_id", np.int64),
        "detector_ran": np.array([bool(r.get("detector_ran", False)) for r in records]),
    }


def fallback_bbox_frames(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Frames whose box is the whole picture -- i.e. the detector found nobody.

    The fitter does not flag these; it just writes the frame rectangle and runs
    anyway. Recovering them is a heuristic, and deliberately a conservative one:
    a box pinned to the origin AND as large as the largest box in the clip. A
    real detection can be large, but it is not pinned to (0,0) to sub-pixel
    precision as well.
    """
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.ndim != 2 or bbox.shape[1] != 4:
        raise ValueError(f"bbox_xyxy must be [N,4], got {bbox.shape}")
    at_origin = (bbox[:, 0] <= 0.5) & (bbox[:, 1] <= 0.5)
    full = (bbox[:, 2] >= bbox[:, 2].max() - 0.5) & (bbox[:, 3] >= bbox[:, 3].max() - 0.5)
    return np.flatnonzero(at_origin & full)


def stale_bbox_runs(bbox_xyxy: np.ndarray) -> list[tuple[int, int]]:
    """Spans over which the box never moved, as (first_frame, length).

    The fitter runs the detector every `detector_interval` frames and holds the
    last box in between, so a run of 5 is the design. A much longer run means
    the detector kept failing and the crop is being taken from where she WAS --
    the fit still converges, on a badly framed picture.
    """
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    changed = np.flatnonzero(np.any(np.diff(bbox, axis=0) != 0.0, axis=1)) + 1
    starts = np.concatenate([[0], changed])
    ends = np.concatenate([changed, [len(bbox)]])
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def quality(clip: dict[str, np.ndarray]) -> dict:
    """What is worth knowing about a clip before trusting its angles."""
    bbox = clip["bbox_xyxy"]
    fallback = fallback_bbox_frames(bbox)
    runs = stale_bbox_runs(bbox)
    longest = max(runs, key=lambda run: run[1]) if runs else (0, 0)
    points = clip["ehm_joints_3d_model"][:, :22]
    step = np.linalg.norm(np.diff(points, axis=0), axis=-1).max(axis=1) if len(points) > 1 \
        else np.zeros(0)
    return {
        "frames": int(len(clip["frame_id"])),
        "fallback_bbox_frames": fallback.tolist(),
        "detector_ran_count": int(np.count_nonzero(clip["detector_ran"])),
        "distinct_bbox": int(len(np.unique(bbox, axis=0))),
        "longest_stale_bbox_run": int(longest[1]),
        "longest_stale_bbox_start": int(longest[0]),
        "step_median_m": float(np.median(step)) if step.size else 0.0,
        "step_p99_m": float(np.percentile(step, 99)) if step.size else 0.0,
        "step_max_m": float(step.max()) if step.size else 0.0,
        "steps_over_10cm": int(np.count_nonzero(step > 0.10)),
    }


def write_source_animation(directory: Path | str, output: Path | str) -> Path:
    """Convert one clip directory into a single canonical source-animation npz."""
    directory = Path(directory)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **load_clip(directory))
    return output
