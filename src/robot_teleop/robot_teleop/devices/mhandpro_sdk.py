"""ctypes bindings for the external x86_64 mHandPro SDK."""

from __future__ import annotations

import platform
import time
from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, c_bool, c_float, c_int
from pathlib import Path

NODES_HAND = 20
PC_FINGERS_VIRTUAL = 5

WS_GEO = 0
CONNECTED_NONE = 0
CONNECTED_RIGHT_GLOVE = 1
CONNECTED_LEFT_GLOVE = 2
CONNECTED_BOTH_GLOVES = 3
TREMOR_06 = 6
CM_APOSE = 0
CM_PPOSE = 1
CS_UNSTART = 0
CS_INPOSE = 1
CS_SUCCEEDED = 2
CS_FAILED = 3


class GloveMocapData(Structure):
    _fields_ = [
        ("glove", c_int),
        ("isUpdate", c_bool),
        ("frameIndex", c_int),
        ("devicePower", c_float),
        ("frequency", c_int),
        ("sensorState", (c_int * NODES_HAND)),
        ("gyr", (c_float * 3) * NODES_HAND),
        ("acc", (c_float * 3) * NODES_HAND),
        ("velocity", (c_float * 3) * NODES_HAND),
        ("position", (c_float * 3) * NODES_HAND),
        ("quaternion", (c_float * 4) * NODES_HAND),
    ]


class GloveMocapDataWithVirtual(Structure):
    _fields_ = [
        *GloveMocapData._fields_,
        ("positionVirtual", (c_float * 3) * PC_FINGERS_VIRTUAL),
    ]


class CalibrationProgress(Structure):
    _fields_ = [("state", c_int), ("progress", c_float)]


GLOVE_DATA_CALLBACK = CFUNCTYPE(None, GloveMocapData, GloveMocapData)
GLOVE_DATA_VIRTUAL_CALLBACK = CFUNCTYPE(None, GloveMocapDataWithVirtual, GloveMocapDataWithVirtual)
GLOVE_BREAK_CALLBACK = CFUNCTYPE(None, c_int)


INITIAL_POSITION_RHAND = [
    [0.748, 0.0, 1.597],
    [0.782, 0.042, 1.6],
    [0.817, 0.077, 1.6],
    [0.842, 0.101, 1.6],
    [0.792, 0.026, 1.604],
    [0.862, 0.04, 1.603],
    [0.912, 0.04, 1.601],
    [0.939, 0.04, 1.599],
    [0.794, 0.01, 1.604],
    [0.864, 0.014, 1.603],
    [0.917, 0.014, 1.6],
    [0.951, 0.014, 1.597],
    [0.793, -0.001, 1.605],
    [0.856, -0.008, 1.604],
    [0.903, -0.008, 1.6],
    [0.935, -0.008, 1.597],
    [0.791, -0.016, 1.604],
    [0.847, -0.031, 1.603],
    [0.884, -0.031, 1.601],
    [0.908, -0.031, 1.6],
]

INITIAL_POSITION_LHAND = [[-x, y, z] for x, y, z in INITIAL_POSITION_RHAND]


def _to_c_float_60(data):
    if len(data) != NODES_HAND or any(len(position) != 3 for position in data):
        raise ValueError("Initial hand positions must contain twenty xyz triples")
    array = (c_float * (NODES_HAND * 3))()
    for index, position in enumerate(data):
        for axis, value in enumerate(position):
            array[index * 3 + axis] = float(value)
    return array


def mocap_data_to_list(glove_data):
    return [[float(glove_data.position[index][axis]) for axis in range(3)] for index in range(NODES_HAND)]


def mocap_quaternions_to_list(glove_data):
    """Return the vendor node orientations in documented wxyz order."""
    return [[float(glove_data.quaternion[index][axis]) for axis in range(4)] for index in range(NODES_HAND)]


def mocap_virtual_positions_to_list(glove_data):
    """Return the five SDK-computed fingertip positions in thumb-to-pinky order."""
    return [
        [float(glove_data.positionVirtual[index][axis]) for axis in range(3)] for index in range(PC_FINGERS_VIRTUAL)
    ]


def mocap_sensor_states_to_list(glove_data):
    return [int(glove_data.sensorState[index]) for index in range(NODES_HAND)]


def mocap_vectors_to_list(glove_data, field_name):
    """Return one vendor per-node xyz vector field without changing its units."""
    values = getattr(glove_data, field_name)
    return [[float(values[index][axis]) for axis in range(3)] for index in range(NODES_HAND)]


def connection_includes_sides(state: int, sides) -> bool:
    requested = set(sides)
    if not requested or not requested.issubset({"left", "right"}):
        raise ValueError("sides must contain left and/or right")
    return all(connection_includes_side(state, side) for side in requested)


def connection_satisfies_policy(state: int, sides, failure_policy: str) -> bool:
    """Return whether a connection state satisfies the configured multi-glove policy."""
    requested = tuple(dict.fromkeys(sides))
    if failure_policy == "require_all":
        return connection_includes_sides(state, requested)
    if failure_policy == "allow_available":
        if not requested or not set(requested).issubset({"left", "right"}):
            raise ValueError("sides must contain left and/or right")
        return any(connection_includes_side(state, side) for side in requested)
    raise ValueError("failure_policy must be require_all or allow_available")


def connection_includes_side(state: int, side: str) -> bool:
    if side == "right":
        return state in (CONNECTED_RIGHT_GLOVE, CONNECTED_BOTH_GLOVES)
    if side == "left":
        return state in (CONNECTED_LEFT_GLOVE, CONNECTED_BOTH_GLOVES)
    raise ValueError("side must be 'left' or 'right'")


class MHandProSDK:
    """Own one mHandPro C SDK instance and retain callback references."""

    def __init__(self, lib_path: str):
        machine = platform.machine().lower()
        if machine not in ("x86_64", "amd64"):
            raise RuntimeError(f"mHandPro SDK supports x86_64 only, current architecture is {machine}")
        if not lib_path:
            raise ValueError("mHandPro lib_path must be configured explicitly")
        resolved_path = Path(lib_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"mHandPro SDK library not found: {resolved_path}")

        self.lib_path = resolved_path
        self._lib = CDLL(str(resolved_path))
        self._bind_api()
        self._connected = False
        self._callbacks = []
        self._init_rhand = _to_c_float_60(INITIAL_POSITION_RHAND)
        self._init_lhand = _to_c_float_60(INITIAL_POSITION_LHAND)

    def _bind_api(self) -> None:
        self._lib.Initial.argtypes = [c_int, POINTER(c_float), POINTER(c_float)]
        self._lib.Initial.restype = None
        self._lib.Connect.argtypes = []
        self._lib.Connect.restype = c_int
        self._lib.DisConnect.argtypes = []
        self._lib.DisConnect.restype = None
        self._lib.SetHandDimension.argtypes = [c_bool]
        self._lib.SetHandDimension.restype = None
        self._lib.SetTremor.argtypes = [c_int, c_int]
        self._lib.SetTremor.restype = None
        self._lib.SetGloveDataCallBackFunc.argtypes = [GLOVE_DATA_CALLBACK]
        self._lib.SetGloveDataCallBackFunc.restype = None
        self._lib.SetGloveDataWithVirtualCallBackFunc.argtypes = [GLOVE_DATA_VIRTUAL_CALLBACK]
        self._lib.SetGloveDataWithVirtualCallBackFunc.restype = None
        self._lib.SetGloveBreakCallBackFunc.argtypes = [GLOVE_BREAK_CALLBACK]
        self._lib.SetGloveBreakCallBackFunc.restype = None
        self._lib.StartCalibration.argtypes = [c_int, POINTER(c_float)]
        self._lib.StartCalibration.restype = None
        self._lib.GetCalibrationProgress.argtypes = []
        self._lib.GetCalibrationProgress.restype = CalibrationProgress

    def initial(self, world_space=WS_GEO, rhand_pos=None, lhand_pos=None) -> None:
        if rhand_pos is not None:
            self._init_rhand = _to_c_float_60(rhand_pos)
        if lhand_pos is not None:
            self._init_lhand = _to_c_float_60(lhand_pos)
        self._lib.Initial(world_space, self._init_rhand, self._init_lhand)

    def set_break_callback(self, callback) -> None:
        wrapped = GLOVE_BREAK_CALLBACK(callback)
        self._lib.SetGloveBreakCallBackFunc(wrapped)
        self._callbacks.append(wrapped)

    def set_data_callback(self, callback) -> None:
        wrapped = GLOVE_DATA_CALLBACK(callback)
        self._lib.SetGloveDataCallBackFunc(wrapped)
        self._callbacks.append(wrapped)

    def set_data_with_virtual_callback(self, callback) -> None:
        wrapped = GLOVE_DATA_VIRTUAL_CALLBACK(callback)
        self._lib.SetGloveDataWithVirtualCallBackFunc(wrapped)
        self._callbacks.append(wrapped)

    def connect(self) -> int:
        state = int(self._lib.Connect())
        self._connected = state != CONNECTED_NONE
        return state

    def disconnect(self) -> None:
        if self._connected:
            self._lib.DisConnect()
        self._connected = False

    def set_hand_dimension(self, is_3d=True) -> None:
        self._lib.SetHandDimension(bool(is_3d))

    def set_tremor(self, tremor_r=TREMOR_06, tremor_l=TREMOR_06) -> None:
        self._lib.SetTremor(int(tremor_r), int(tremor_l))

    def start_calibration(self, mode=CM_PPOSE, timeout=30.0) -> tuple[int, float]:
        quaternion = (c_float * 4)(0.0, 0.0, 0.0, 0.0)
        self._lib.StartCalibration(int(mode), quaternion)
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            progress = self._lib.GetCalibrationProgress()
            state = int(progress.state)
            if state in (CS_SUCCEEDED, CS_FAILED):
                return state, float(progress.progress)
            time.sleep(0.05)
        return CS_UNSTART, 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected
