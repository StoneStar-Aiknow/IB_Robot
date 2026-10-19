"""The X2 streaming retargeter: the runtime's FollowMeStream behind ``RetargetStream``.

Everything that maps and smooths is the vendored runtime code in ``_vendor``
(video2x2_runtime 0717710, see ``_vendor/SOURCES.md``); this module only adapts
its inputs and its round:

* a PEAR frame arrives as ``smplx_pose_raw``; it is decoded and posed on the
  neutral SMPL-X skeleton here, which is what the runtime's PEAR node did before
  publishing ``SourceFrame`` fields;
* a round is the runtime's forever session (``live_s=0``): park on the neutral
  pose, cross-fade to the person, follow. It never ends on its own;
* mirroring is on by default (her right arm drives the robot's left), as in the
  runtime's follow-me;
* the runtime's own rate limiter only counts here (``limit_rate=False``); the
  live node caps the published command against the robot's measured bands;
* with ``separate_process=True`` the mapping (``retarget``/``reset_input``) runs
  in a child process. On duck one frame costs ~36 ms of the GIL, and a worker
  thread holding it starved the 50 Hz timer down to ~17 Hz.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Sequence
from multiprocessing.connection import Connection

import numpy as np
# The vendored code imports these lazily, the first output sample's included: on duck
# that first import stalled the 50 Hz tick ~2.3 s and the subscription lost ~40 frames.
import scipy.interpolate  # noqa: F401
import scipy.optimize  # noqa: F401
import scipy.signal  # noqa: F401
import scipy.sparse  # noqa: F401
import scipy.spatial.transform  # noqa: F401

from ..rotations import decode_rotation_6d, encode_rotation_6d
from ._vendor import followme_core as core
from ._vendor.neutral_fk import NeutralFK

POSE_DIM = 52 * 6
# The branch guard's longest window, in frames. Push cost grows with the window
# (about 0.2 ms per frame of it on the dev machine) and the runtime's 1200 would
# take 230 ms a frame against a 50 ms frame budget. 200 is 10 s of a wrist held
# past the Euler cut; beyond that the vendor truncates and notes it.
MAX_ANCHOR_FRAMES = 200
# Notes kept across retargeter rebuilds and per-frame truncation notes.
NOTES_KEPT = 10
# Status keys that describe the mapping; with a child process they come from there.
_MAPPING_KEYS = ("branch_guard_trips", "branch_guard_worst", "retargeted", "input_resets")
_CHILD_START_TIMEOUT_S = 120.0


def _session() -> core.Session:
    return core.Session(live_s=0.0)


class X2FollowMe:
    """``RetargetStream`` for the X2 arms, in ``ARM_JOINT_NAMES`` (arm_pos) order."""

    def __init__(self, *, base_pose: Sequence[float], latency_s: float = core.DEFAULT_LATENCY_S,
                 lookahead_s: float = core.DEFAULT_LOOKAHEAD_S, mirror: bool = True,
                 separate_process: bool = False):
        self._vendor = core._import_vendor()
        self.joint_names = tuple(self._vendor["ARM_JOINT_NAMES"])
        base = np.asarray(base_pose, dtype=np.float64)
        if base.shape != (len(self.joint_names),) or not np.isfinite(base).all():
            raise ValueError(f"base_pose must be {len(self.joint_names)} finite radians")
        self._fk = NeutralFK()
        self._stream = core.FollowMeStream(
            vendor=self._vendor, session=_session(), latency_s=latency_s, lookahead_s=lookahead_s,
            base_pose=base, limit_rate=False, mirror=mirror)
        self._stream.retargeter = self._new_retargeter(self._stream.retargeter)
        self.retarget_count = 0
        self.input_resets = 0
        # Guard counts of the retargeters that reset_input replaced.
        self._guard_trips = 0
        self._guard_worst = 0
        self._dry_run()
        self._child = None
        if separate_process:
            self._child = _MappingProcess(dict(base_pose=base.tolist(), latency_s=latency_s,
                                               lookahead_s=lookahead_s, mirror=mirror))

    def _dry_run(self) -> None:
        """Pay the first-call costs now and run the mirror's FK self-check once.

        The per-frame self-check is off in the vendored core; here the vendor's
        own ``mirrored(..., kinematics=)`` poses the robot both ways on one
        frame and raises past its 20 mm limit, which only a wrong sign table
        reaches. The pose is a T-pose with both elbows bent 1.2 rad about z,
        mirror-symmetric; it clips no channel (checked when this was written).
        """
        c, s = np.cos(1.2), np.sin(1.2)
        rotations = np.tile(np.eye(3), (52, 1, 1))
        rotations[18] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # left elbow
        rotations[19] = rotations[18].T  # right elbow
        frame = self._frame(0.0, 0, encode_rotation_6d(rotations).reshape(-1))
        retargeter = self._stream.retargeter
        trajectory = self._vendor["retarget"](
            retargeter._animation([frame]), kinematics=retargeter.kinematics,
            ledger=retargeter.ledger, readout=self._vendor["READOUT_B"], verify=False)
        if self._stream.mirror:
            trajectory.mirrored(retargeter.ledger, kinematics=retargeter.kinematics)

    def _frame(self, t_camera: float, frame_id: int, smplx_pose_raw) -> core.SourceFrame:
        raw = np.asarray(smplx_pose_raw, dtype=np.float64).reshape(-1)
        if raw.shape != (POSE_DIM,) or not np.isfinite(raw).all():
            raise ValueError(f"smplx_pose_raw must be {POSE_DIM} finite values, got {raw.shape}")
        rotations = decode_rotation_6d(raw)
        if np.abs(np.linalg.det(rotations) - 1.0).max() > 1e-3:
            raise ValueError("smplx_pose_raw holds a degenerate 6D rotation")
        # float32, as the runtime's PEAR node published them.
        rotations = rotations.astype(np.float32)
        decoded = {
            "global_orient": rotations[0:1],
            "body_pose": rotations[1:22],
            "left_hand_pose": rotations[22:37],
            "right_hand_pose": rotations[37:52],
        }
        return core.SourceFrame(t_camera=float(t_camera), frame_id=int(frame_id), points=self._fk(decoded),
                                body_pose=rotations[1:22], global_orient=rotations[0])

    def retarget(self, t_camera: float, frame_id: int, smplx_pose_raw) -> np.ndarray:
        if self._child is not None:
            return self._child.retarget(t_camera, frame_id, smplx_pose_raw)
        retargeter = self._stream.retargeter
        pose = retargeter.push(self._frame(t_camera, frame_id, smplx_pose_raw))
        del retargeter.notes[:-NOTES_KEPT]
        self.retarget_count += 1
        return pose

    def push_pose(self, t_camera: float, pose) -> None:
        self._stream.push_pose(float(t_camera), pose)

    def reset_input(self) -> None:
        if self._child is not None:
            self._child.reset_input()
            return
        # The branch guard follows one person's wrist unwrap; across a gap there is
        # no continuity to keep. The despiker's velocity statistics are the arm's,
        # not the person's, and the runtime keeps them across rounds too.
        # Its counts and notes carry over: the status reports the whole session.
        old = self._stream.retargeter
        self._guard_trips += old.guard.trips
        self._guard_worst = max(self._guard_worst, old.guard.worst_branch)
        self._stream.retargeter = self._new_retargeter(old)
        self._stream.retargeter.notes.extend(old.notes[-NOTES_KEPT:])
        self.input_resets += 1

    def _new_retargeter(self, old: core.StreamRetargeter) -> core.StreamRetargeter:
        return core.StreamRetargeter(self._vendor, old.kinematics, old.ledger,
                                     max_anchor_frames=MAX_ANCHOR_FRAMES, mirror=old.mirror)

    def start_round(self) -> None:
        self._stream.reset(_session())

    def poll(self, now: float) -> list[np.ndarray]:
        # The output clock starts on the poll that opens the gate, so later samples fall
        # due exactly on the caller's tick phase: a tick that runs a little early finds
        # none and the next finds two. Once started, read half a step ahead so jitter
        # under half a period still yields one sample per tick.
        lead = 0.5 * self._stream.session.dt if self._stream._t0 is not None else 0.0
        return [sample.pose for sample in self._stream.poll(float(now) + lead)]

    def status(self) -> dict:
        stream = self._stream
        guard = stream.retargeter.guard
        status = {
            "phase": stream.session.phase(stream._k)[0] if stream._t0 is not None else "filling",
            "buffered": len(stream._buf),
            "starved_frames": stream.starved_frames,
            "stretched_frames": stream.stretched_frames,
            "offset_samples": stream.offset_samples,
            "despike_flagged": int(stream.despiker.flagged.sum()),
            "branch_guard_trips": self._guard_trips + guard.trips,
            "branch_guard_worst": max(self._guard_worst, guard.worst_branch),
            "retargeted": self.retarget_count,
            "input_resets": self.input_resets,
            "notes": (stream.notes + stream.retargeter.notes)[-3:],
        }
        if self._child is not None:
            mapping = self._child.mapping_status
            status.update({key: mapping[key] for key in _MAPPING_KEYS})
            status["notes"] = (stream.notes + mapping["notes"])[-3:]
        return status

    def close(self) -> None:
        if self._child is not None:
            self._child.close()


class _MappingProcess:
    """An in-process ``X2FollowMe`` in a child interpreter, driven over a socket pair.

    Only ``retarget`` waits for an answer (on the worker thread, without the GIL);
    ``reset_input`` is fire-and-forget, and the socket keeps it ahead of the next frame.
    The child is a fresh ``python3 -c``, not multiprocessing's spawn, which would
    re-run the parent's main script.
    """

    def __init__(self, params: dict):
        ours, theirs = socket.socketpair()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        code = f"from {__name__} import _serve_mapping; _serve_mapping({theirs.fileno()}, {json.dumps(params)!r})"
        self._process = subprocess.Popen([sys.executable, "-c", code], pass_fds=(theirs.fileno(),), env=env)
        theirs.close()
        self._conn = Connection(ours.detach())
        if not self._conn.poll(_CHILD_START_TIMEOUT_S):
            self.close()
            raise RuntimeError("the mapping process did not start")
        kind, payload = self._conn.recv()
        if kind != "ready":
            self.close()
            raise RuntimeError(f"the mapping process failed to start: {payload}")
        self.mapping_status = payload

    def retarget(self, t_camera, frame_id, smplx_pose_raw) -> np.ndarray:
        raw = np.asarray(smplx_pose_raw, dtype=np.float64)
        try:
            self._conn.send(("frame", float(t_camera), int(frame_id), raw))
            kind, payload, mapping = self._conn.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"the mapping process is gone (exit code {self._process.poll()})") from exc
        self.mapping_status = mapping
        if kind == "error":
            raise ValueError(payload)
        return payload

    def reset_input(self) -> None:
        try:
            self._conn.send(("reset",))
        except OSError:
            pass  # the next retarget reports the dead child

    def close(self) -> None:
        try:
            self._conn.send(("close",))
        except OSError:
            pass
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._conn.close()


def _mapping_status(stream: X2FollowMe) -> dict:
    status = stream.status()
    return {**{key: status[key] for key in _MAPPING_KEYS}, "notes": stream._stream.retargeter.notes[-3:]}


def _serve_mapping(fd: int, params_json: str) -> None:
    # The parent decides when to stop: Ctrl-C reaches the whole process group.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    conn = Connection(fd)
    params = json.loads(params_json)
    try:
        stream = X2FollowMe(**params)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which raises it
        conn.send(("failed", repr(exc)))
        return
    conn.send(("ready", _mapping_status(stream)))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message[0] == "close":
            return
        if message[0] == "reset":
            stream.reset_input()
            continue
        _, t_camera, frame_id, raw = message
        try:
            reply = ("pose", stream.retarget(t_camera, frame_id, raw))
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end the child
            reply = ("error", repr(exc)[:200])
        conn.send((*reply, _mapping_status(stream)))


def create(*, base_pose: Sequence[float], **params) -> X2FollowMe:
    return X2FollowMe(base_pose=base_pose, **params)
