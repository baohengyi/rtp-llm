"""Exercise installed pytest entry points without the source checkout on sys.path.

This tests setuptools' Python payload, not native extension or dependency
installation. The CI wheel check separately exercises the real platform wheel.
"""

import ast
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from setuptools import Distribution, find_namespace_packages
from setuptools.command.build_py import build_py

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def test_installed_plugins_and_complete_backend_registry_module(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    find = config["tool"]["setuptools"]["packages"]["find"]
    payload = tmp_path / "installed"
    dist = Distribution({
        "name": "rtp-llm",
        "version": "0.0.0",
        "packages": find_namespace_packages(str(root), include=find["include"]),
        "package_dir": {"": str(root)},
        "script_name": str(root / "setup.py"),
    })
    command = build_py(dist)
    command.ensure_finalized()
    command.build_lib = str(payload)
    command.run()
    metadata = payload / "rtp_llm-0.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: rtp-llm\nVersion: 0.0.0\n")
    plugins = config["project"]["entry-points"]["pytest11"]
    (metadata / "entry_points.txt").write_text(
        "[pytest11]\n" + "".join(f"{name} = {module}\n" for name, module in plugins.items())
    )
    original = root / "rtp_llm/utils/test/backend_registry_test.py"
    installed = payload / original.relative_to(root)
    assert installed.read_bytes() == original.read_bytes()
    expected = {node.name for node in ast.walk(ast.parse(original.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
    assert len(expected) == 13
    work = tmp_path / "empty-cwd"
    work.mkdir()
    (work / "pytest.ini").write_text("[pytest]\n")
    report = tmp_path / "installed-pytest.xml"
    script = """
import pathlib, sys
payload = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(payload))
import pytest
class Evidence:
    def pytest_sessionstart(self, session):
        for name in ('remote-gpu', 'rtp-ci-profile', 'smoke-runs-per-test'):
            plugin = session.config.pluginmanager.get_plugin(name)
            assert plugin is not None, name
            assert pathlib.Path(plugin.__file__).resolve().is_relative_to(payload)
sys.exit(pytest.main([sys.argv[2], '--import-mode=importlib', '-q',
                      '-c', 'pytest.ini', '--confcutdir=' + str(payload),
                      '--junitxml=' + sys.argv[3]], plugins=[Evidence()]))
"""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "PYTEST", "RTP_"))}
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(payload), str(installed), str(report)],
        cwd=work, env=env, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert len(cases) == len(expected)
    assert {case.get("name") for case in cases} == expected
    assert not any(case.find(tag) is not None for case in cases
                   for tag in ("failure", "error", "skipped"))
