"""Native pytest must preserve application stream replacement and keep logs."""

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stream", ["stderr", "stdout"])
def test_logging_preserves_application_stream_and_junit_evidence(tmp_path, stream):
    options = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    lines = ["[pytest]"]
    for key in ("log_cli", "log_cli_level", "log_level", "junit_logging"):
        if key in options:
            value = options[key]
            lines.append(f"{key} = {str(value).lower() if isinstance(value, bool) else value}")
    (tmp_path / "pytest.ini").write_text("\n".join(lines) + "\n")
    (tmp_path / "test_stream.py").write_text(
        "import argparse, io, logging, sys\n"
        "from unittest.mock import patch\n"
        "import pytest\n\n"
        "def test_stream():\n"
        "    parser = argparse.ArgumentParser()\n"
        f"    with patch('sys.{stream}', new_callable=io.StringIO) as output:\n"
        "        logging.info('native-info-evidence')\n"
        f"        assert sys.{stream} is output\n"
        + (
            "        with pytest.raises(SystemExit) as error:\n"
            "            parser.parse_args(['--removed', '1'])\n"
            "        assert error.value.code == 2\n"
            "        assert 'unrecognized arguments: --removed 1' in output.getvalue()\n"
            if stream == "stderr" else
            "        print('native-application-output')\n"
            "        assert output.getvalue() == 'native-application-output\\n'\n"
        )
    )
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_stream.py", "--junitxml=report.xml"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ET.parse(tmp_path / "report.xml").getroot().findall(".//testcase")
    assert len(cases) == 1
    assert not any(cases[0].find(tag) is not None for tag in ("failure", "error", "skipped"))
    assert "native-info-evidence" in "".join(cases[0].itertext())
