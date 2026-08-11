"""Import the compiled CUDA Graph test binding staged by setup.py."""

import sys
from pathlib import Path


_TEST_LIB_DIR = Path(__file__).resolve().parents[3] / "libs" / "test"
if str(_TEST_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_LIB_DIR))

try:
    from libtest_cuda_graph_runner import CudaGraphRunner
except ImportError as exc:
    raise ImportError(
        f"libtest_cuda_graph_runner.so not found under {_TEST_LIB_DIR}; "
        "run `python setup.py build_ext --inplace` before pytest"
    ) from exc


__all__ = ["CudaGraphRunner"]
