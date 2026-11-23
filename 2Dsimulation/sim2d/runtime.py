"""Keep the compiler and plotting caches beside this run."""

import os
import sys
import tempfile
from pathlib import Path


def set_runtime(root):
    root = Path(root).resolve()
    for key, name in (
        ("TEMP", "tmp"),
        ("TMP", "tmp"),
        ("TMPDIR", "tmp"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("CUPY_CACHE_DIR", "cupy"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("XDG_CACHE_HOME", "xdg"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("PYTHONPYCACHEPREFIX", "pycache"),
    ):
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    tempfile.tempdir = str(root / "tmp")
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    sys.dont_write_bytecode = True
