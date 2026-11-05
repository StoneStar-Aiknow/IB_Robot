#!/usr/bin/env python3
"""Check the actual PASID-returning SVM ABI; no model load or kernel launch."""

import argparse
import ctypes
import json
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pid", choices=("zero", "self"), default="self")
parser.add_argument("--pasid", type=int, default=-1)
args = parser.parse_args()


class ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_int),
        ("ttbr", ctypes.c_ulonglong),
        ("tcr", ctypes.c_ulonglong),
        ("pasid", ctypes.c_int),
        ("flags", ctypes.c_uint),
    ]


libc = ctypes.CDLL(None, use_errno=True)
libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
libc.ioctl.restype = ctypes.c_int
info = ProcessInfo(pid=os.getpid() if args.pid == "self" else 0, pasid=args.pasid)
fd = os.open("/dev/svm0", os.O_RDWR)
try:
    result = libc.ioctl(fd, 0xFFFF, ctypes.byref(info))
    error = ctypes.get_errno()
finally:
    os.close(fd)
valid = result == 0 and info.pasid >= 0 and info.ttbr != 0 and info.tcr != 0
print(
    json.dumps(
        {
            "valid": valid,
            "result": result,
            "errno": error,
            "after": {name: getattr(info, name) for name, _ in info._fields_},
        },
        indent=2,
    )
)
raise SystemExit(0 if valid else 1)
