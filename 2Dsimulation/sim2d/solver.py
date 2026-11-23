"""NVIDIA cuDSS complex sparse direct solve for the two field models."""

import ctypes as ct
import os
import sys
import warnings
from pathlib import Path
import numpy as np


class CuDSS:
    """No CPU factorization fallback; all matrix buffers are caller-owned GPU arrays."""

    def __init__(self, A, rhs, library_dir):
        import cupy as cp

        self.cp = cp
        cp.cuda.device.get_cublas_handle()
        libdir = Path(library_dir)
        self.dll_path = libdir / "cudss64_0.dll"
        self.dll_dirs = [os.add_dll_directory(str(libdir))]
        self.lib = ct.WinDLL(str(self.dll_path))
        self._bind()
        self.handle = ct.c_void_p()
        self.config = ct.c_void_p()
        self.data = ct.c_void_p()
        self.matrix = ct.c_void_p()
        self.bmatrix = ct.c_void_p()
        self.xmatrix = ct.c_void_p()
        self.A = A
        self.B = cp.array(rhs, order="F", copy=True)
        self.X = cp.empty_like(self.B, order="F")
        try:
            self.call("cudssCreate", ct.byref(self.handle))
            self.call("cudssSetStream", self.handle, ct.c_void_p(cp.cuda.get_current_stream().ptr))
            self.call("cudssConfigCreate", ct.byref(self.config))
            self.call("cudssDataCreate", self.handle, ct.byref(self.data))
            # Constants are pinned to the cuDSS 0.8.0.10 headers.
            for param, value in ((0, 4), (3, 0), (5, 5), (11, 0), (15, 0)):
                self.set_config(param, value, ct.c_int)
            for param, value in ((6, 2e-13), (9, 0.0)):
                self.set_config(param, value, ct.c_double)
            self.set_config(20, 0, ct.c_int)
            n = int(A.shape[0])
            self.call(
                "cudssMatrixCreateCsr",
                ct.byref(self.matrix),
                n,
                n,
                int(A.nnz),
                ct.c_void_p(A.indptr.data.ptr),
                None,
                ct.c_void_p(A.indices.data.ptr),
                ct.c_void_p(A.data.data.ptr),
                10,
                10,
                5,
                0,
                0,
                0,
            )
            for matrix, array in ((self.bmatrix, self.B), (self.xmatrix, self.X)):
                self.call(
                    "cudssMatrixCreateDn",
                    ct.byref(matrix),
                    n,
                    int(array.shape[1]),
                    n,
                    ct.c_void_p(array.data.ptr),
                    5,
                    0,
                )
        except BaseException:
            self.close()
            raise

    def _bind(self):
        p, i, z, q = ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_int64
        signatures = {
            "cudssCreate": [ct.POINTER(p)],
            "cudssDestroy": [p],
            "cudssConfigCreate": [ct.POINTER(p)],
            "cudssConfigDestroy": [p],
            "cudssDataCreate": [p, ct.POINTER(p)],
            "cudssDataDestroy": [p, p],
            "cudssSetStream": [p, p],
            "cudssConfigSet": [p, i, p, z],
            "cudssMatrixCreateCsr": [ct.POINTER(p), q, q, q, p, p, p, p, i, i, i, i, i, i],
            "cudssMatrixCreateDn": [ct.POINTER(p), q, q, q, p, i, i],
            "cudssMatrixDestroy": [p],
            "cudssExecute": [p, i, p, p, p, p, p],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, ct.c_int

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name}: cuDSS status {status}")

    def set_config(self, param, value, kind):
        v = kind(value)
        self.call("cudssConfigSet", self.config, param, ct.byref(v), ct.sizeof(v))

    def close(self):
        if hasattr(self, "handle") and self.handle.value:
            self.cp.cuda.get_current_stream().synchronize()
            for name in ("matrix", "bmatrix", "xmatrix"):
                obj = getattr(self, name, None)
                if obj is not None and obj.value:
                    self.call("cudssMatrixDestroy", obj)
                    obj.value = None
            if self.data.value:
                self.call("cudssDataDestroy", self.handle, self.data)
                self.data.value = None
            if self.config.value:
                self.call("cudssConfigDestroy", self.config)
                self.config.value = None
            self.call("cudssDestroy", self.handle)
            self.handle.value = None
        self.A = self.B = self.X = None
        for directory in getattr(self, "dll_dirs", []):
            directory.close()
        self.dll_dirs = []


class Factorization:
    """Factor the full natural-boundary system or eliminate the specified zero-boundary DOFs."""

    def __init__(self, matrix, boundary=None):
        import cupy as cp
        import cupyx.scipy.sparse as csp

        self.size = matrix.shape[0]
        self.interior = slice(None) if boundary is None else np.flatnonzero(~boundary)
        interior = matrix.tocsr() if boundary is None else matrix[self.interior][:, self.interior].tocsr()
        interior.sort_indices()
        device_matrix = csp.csr_matrix(interior, dtype=cp.complex128)
        rhs = cp.zeros((interior.shape[0], 1), dtype=cp.complex128)
        self.native = CuDSS(device_matrix, rhs, Path(sys.prefix) / "Library" / "bin")
        try:
            self.phase(3)  # symbolic analysis and ordering
            self.phase(4)  # numerical factorization
        except BaseException:
            self.close()
            raise

    def phase(self, number):
        b = self.native
        status = b.lib.cudssExecute(b.handle, number, b.config, b.data, b.matrix, b.xmatrix, b.bmatrix)
        if number == 1008 and status == 7:
            warnings.warn("cuDSS returned its solution with iterative-refinement status 7.", RuntimeWarning)
        elif status:
            raise RuntimeError(f"cuDSS phase {number} returned status {status}")
        b.cp.cuda.get_current_stream().synchronize()

    def solve(self, rhs):
        b = self.native
        b.B[:, 0] = b.cp.asarray(rhs[self.interior], dtype=b.cp.complex128)
        self.phase(1008)
        result = np.zeros(self.size, dtype=np.complex128)
        result[self.interior] = b.cp.asnumpy(b.X[:, 0])
        return result

    def close(self):
        self.native.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
