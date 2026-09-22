"""Collection failures must not launch remote work without a test loop."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("broken,collect_only", [(True, False), (False, True), (False, False)])
def test_remote_dispatch_respects_real_pytest_collection(tmp_path, broken, collect_only):
    tree = ast.parse((ROOT / "rtp_llm/test/remote_tests/plugin.py").read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and any(isinstance(m, ast.FunctionDef) and m.name == "pytest_collection_modifyitems" for m in n.body))
    method = next(m for m in owner.body if isinstance(m, ast.FunctionDef)
                  and m.name == "pytest_collection_modifyitems")
    # Execute the actual hook under a real pytest collection lifecycle. The
    # dispatch boundary is recorded locally instead of contacting a scheduler.
    wrapper = ast.ClassDef(name="Probe", bases=[], keywords=[], body=[method], decorator_list=[])
    hook = ast.unparse(ast.fix_missing_locations(wrapper))
    (tmp_path / "conftest.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "class RemoteDispatchMode: SESSION = 'session'\n" + hook + "\n"
        "def dispatch(self, items): Path('submitted').write_text(str(len(items)))\n"
        "Probe._per_test_collection_modifyitems = dispatch\n"
        "def pytest_configure(config):\n"
        "    probe = Probe(); probe.mode = 'per_test'; probe.config = config\n"
        "    config.pluginmanager.register(probe)\n"
    )
    (tmp_path / "test_valid.py").write_text("def test_valid(): assert 2 + 2 == 4\n")
    if broken:
        (tmp_path / "test_broken.py").write_text("raise RuntimeError('collection sentinel')\n")
    args = [sys.executable, "-m", "pytest", "-q", "-c", "/dev/null", "--confcutdir", str(tmp_path), "--junitxml=original.xml"]
    if collect_only:
        args += ["--collect-only"]
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (2 if broken else 0), result.stdout + result.stderr
    assert (tmp_path / "submitted").exists() == (not broken and not collect_only)
    xml = ET.parse(tmp_path / "original.xml").getroot()
    if broken:
        assert len(list(xml.iter("error"))) == 1
        assert "collection sentinel" in ET.tostring(xml, encoding="unicode")
    elif not collect_only:
        assert len(list(xml.iter("testcase"))) == 1
        assert not list(xml.iter("error")) and not list(xml.iter("failure"))


def test_cuda13_smoke_platform_marker_is_registered():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert "L20D_TEST" in {m.split(":", 1)[0].split("(", 1)[0] for m in markers}
