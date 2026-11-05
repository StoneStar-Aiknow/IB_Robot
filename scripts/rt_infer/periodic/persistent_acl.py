import ctypes as C
import os
from pathlib import Path

import numpy as np


class PersistentSession:
    def __init__(self, path, expected):
        self.lib = C.CDLL(str(Path(os.environ["RTINFER_PREFIX"]) / "lib" / "persistent_acl_bridge.so"))
        self.lib.persistent_error.restype = C.c_char_p
        self.lib.persistent_create.argtypes = [C.c_char_p]
        self.lib.persistent_create.restype = C.c_void_p
        self.lib.persistent_run.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_size_t]
        self.lib.persistent_output_count.argtypes = [C.c_void_p]
        self.lib.persistent_output_count.restype = C.c_size_t
        self.lib.persistent_output.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
        self.lib.persistent_output.restype = C.c_void_p
        self.lib.persistent_destroy.argtypes = [C.c_void_p]
        self.handle = self.lib.persistent_create(path.encode())
        if not self.handle:
            raise RuntimeError(self.lib.persistent_error().decode())
        self.views = []
        try:
            assert len(expected) == self.lib.persistent_output_count(self.handle)
            for i, e in enumerate(expected):
                size = C.c_size_t()
                addr = self.lib.persistent_output(self.handle, i, C.byref(size))
                assert e.dtype == np.float32 and size.value == e.nbytes
                self.views.append(
                    np.ctypeslib.as_array((C.c_float * (size.value // 4)).from_address(addr)).reshape(e.shape)
                )
        except BaseException:
            self.close()
            raise
        self.busy = False

    def infer(self, inputs):
        if self.busy:
            raise RuntimeError("PersistentSession is non-reentrant")
        self.busy = True
        try:
            arrays = [np.ascontiguousarray(x) for x in inputs]
            ptrs = (C.c_void_p * len(arrays))(*(x.ctypes.data for x in arrays))
            sizes = (C.c_size_t * len(arrays))(*(x.nbytes for x in arrays))
            if self.lib.persistent_run(self.handle, ptrs, sizes, len(arrays)):
                raise RuntimeError(self.lib.persistent_error().decode())
            return self.views
        finally:
            self.busy = False

    def close(self):
        if self.handle:
            self.lib.persistent_destroy(self.handle)
            self.handle = None
