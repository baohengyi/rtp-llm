"""The repair diagnostic must leave the full OSS acceptance scope intact."""

from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def test_repair_profile_keeps_complete_modules_and_strict_report_requirements():
    root = Path(__file__).resolve().parents[2]
    profiles = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
    full = profiles["py_ut_oss_sm8x"]
    repair = profiles["py_ut_oss_sm8x_repair"]
    assert full["expected_count"] == 3053
    assert repair["expected_count"] == 118
    assert repair["paths"] == [
        "rtp_llm/test/test_build_packaging_contract.py",
        "rtp_llm/server/server_args/test/server_args_test.py",
        "rtp_llm/models_py/triton_kernels/moe/test/test_ep_scatter.py",
    ]
    assert repair["markexpr"] == full["markexpr"]
    for profile in (full, repair):
        assert profile["forbid_skips"] is True
        assert profile["minimum_count"] == 1
        assert profile["gpu_type"] == "A10"
    for path in repair["paths"]:
        assert (root / path).is_file()
        assert any(path == parent or path.startswith(parent.rstrip("/") + "/")
                   for parent in full["paths"])
