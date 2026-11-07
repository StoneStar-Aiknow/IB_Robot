import ctypes as C
import os
from pathlib import Path


class PeriodicControl:
    def __init__(self, path, create=False):
        self.lib = C.CDLL(
            str(Path(os.environ["RTINFER_PREFIX"]) / "lib" / "periodic_control_bridge.so"), use_errno=True
        )
        for name in ("periodic_shared_create", "periodic_shared_open"):
            getattr(self.lib, name).restype = C.c_void_p
        self.lib.periodic_shared_create.argtypes = [C.c_char_p]
        self.lib.periodic_shared_open.argtypes = [C.c_char_p, C.c_int]
        self.lib.periodic_shared_arm.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64]
        self.lib.periodic_shared_release.argtypes = [C.c_void_p, C.c_uint64]
        self.lib.periodic_shared_disarm.argtypes = [C.c_void_p]
        self.lib.periodic_shared_close.argtypes = [C.c_void_p]
        self.handle = (
            self.lib.periodic_shared_create(str(path).encode())
            if create
            else self.lib.periodic_shared_open(str(path).encode(), 1)
        )
        if not self.handle:
            raise OSError(C.get_errno(), "periodic control mapping")

    def arm(self, epoch, period, budget):
        if not self.handle:
            raise RuntimeError("closed periodic control")
        rc = self.lib.periodic_shared_arm(self.handle, epoch, period, budget)
        if rc:
            raise OSError(-rc, "arm periodic control")

    def release(self, ns):
        if not self.handle:
            raise RuntimeError("closed periodic control")
        rc = self.lib.periodic_shared_release(self.handle, ns)
        if rc:
            raise OSError(-rc, "publish periodic release")

    def disarm(self):
        if self.handle:
            self.lib.periodic_shared_disarm(self.handle)

    def close(self):
        if self.handle:
            self.lib.periodic_shared_close(self.handle)
            self.handle = None
