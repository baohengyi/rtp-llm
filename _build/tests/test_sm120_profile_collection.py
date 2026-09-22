"""The SM120 manifest must collect its original linear profiling method."""

import ast
import os
from pathlib import Path
import subprocess
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "rtp_llm/models_py/modules/factory/linear/impl/cuda/test"


def test_sm120_linear_profile_is_collected_with_the_real_ci_mark_expression(tmp_path):
    classes = []
    for filename, names in (
        ("fp8_linear_test.py", {"CudaFp8LinearTestBase", "CudaFp8GEMMLinearTestBase"}),
        ("fp8_deepgemm_linear_sm120_test.py", {"OnlineFp8LoaderTestBase", "CudaFp8DeepGEMMLinearSM120Test"}),
    ):
        tree = ast.parse((TESTS / filename).read_text())
        classes.extend(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names)
    assert len(classes) == 4
    # Collect exact class ASTs without importing the native extension. No GPU
    # fixture or method is executed; the stand-ins only satisfy class attributes.
    (tmp_path / "test_collection.py").write_text(
        "from __future__ import annotations\nimport unittest\nimport pytest\n"
        "CudaFp8DeepGEMMLinear = object\nCudaFp8GEMMLinear = object\n"
        + "\n\n".join(ast.unparse(c) for c in classes) + "\n"
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    profile = config["tool"]["rtp_llm"]["pytest_ci"]["profiles"]["py_ut_cuda13_sm120"]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", "-c", "/dev/null",
         "--confcutdir", str(tmp_path), "-m", profile["markexpr"],
         "test_collection.py::CudaFp8DeepGEMMLinearSM120Test"],
        cwd=tmp_path, env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = [line.split("::CudaFp8DeepGEMMLinearSM120Test::", 1)[1]
             for line in result.stdout.splitlines()
             if "::CudaFp8DeepGEMMLinearSM120Test::" in line]
    assert "test_profile_cuda_fp8_deepgemm_linear" in nodes, result.stdout + result.stderr
    assert len(nodes) == 29
    assert profile["expected_count"] == 184
