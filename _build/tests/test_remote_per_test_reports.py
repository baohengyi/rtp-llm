"""Exercise worker XML generation and controller JUnit without GPU dispatch."""

import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from rtp_llm.test.remote_tests import plugin as remote
from rtp_llm.test.remote_tests.remote_exec_rtp import RemoteRuntimeConfig
from rtp_llm.test.remote_tests.remote_timeout_policy import select_remote_timeout_policy

pytest_plugins = ["pytester"]


def worker_command(root, test_file):
    item = SimpleNamespace(
        fspath=str(test_file),
        nodeid="test_remote.py::test_case",
        get_closest_marker=lambda name: (
            SimpleNamespace(kwargs={"type": "H20", "count": 1})
            if name == "gpu"
            else None
        ),
    )
    plugin = object.__new__(remote.RemoteREAPIPlugin)
    plugin.rootdir = root
    plugin.config = SimpleNamespace(option=SimpleNamespace(markexpr=""))
    plugin._collect_outputs = False
    plugin.timeout_policy = select_remote_timeout_policy(
        "smoke_h20_internal", per_test=True
    )
    runtime = RemoteRuntimeConfig(
        ignore_args=[], env_vars={}, platform_properties={}, remote_setup_prefix=""
    )
    command = plugin._build_command(item, runtime)
    subprocess.run(["bash", "-n", "-c", command[2]], check=True)
    inner = shlex.split(command[2])[-1]
    subprocess.run(["bash", "-n", "-c", inner], check=True)
    assert inner.index("rm -f bazel-testlogs/pytest/remote_per_test.xml") < inner.index(
        "python -m pytest"
    )
    assert "--junitxml=bazel-testlogs/pytest/remote_per_test.xml" in inner
    assert "-o junit_duration_report=total" in inner
    assert inner.index("echo EXIT_CODE=$ec") < inner.index("echo '<<<JUNIT_XML>>>'")
    assert inner.rstrip().endswith("exit $ec")
    return inner


@pytest.mark.parametrize(
    "scenario",
    [
        "pass",
        "reapi_failure",
        "exit_mismatch",
        "missing_exit",
        "missing_xml",
        "malformed_xml",
        "zero_cases",
        "wrong_identity",
        "missing_time",
        "nan_time",
        "negative_time",
        "xml_failure",
        "xml_error",
        "skip_forbidden",
        "skip_allowed",
        "cached_local",
        "cached_reapi",
        "extra_identity",
        "internal_source",
    ],
)
def test_worker_evidence_survives_controller_junit(pytester, monkeypatch, scenario):
    root = Path(remote.__file__).resolve().parents[3]
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(root), str(pytester.path)]))
    pytester.makeini("[pytest]")
    test_file = pytester.makepyfile(test_remote="""
        import time
        def test_case():
            time.sleep(0.025)
            assert 2 + 2 == 4
    """)
    nodeid = "test_remote.py::test_case"
    if scenario == "internal_source":
        internal = pytester.path.parent / "internal_source"
        case_dir = internal / pytester.path.name
        case_dir.mkdir(parents=True)
        test_file = test_file.rename(case_dir / test_file.name)
        (pytester.path / "internal_source").symlink_to(
            internal, target_is_directory=True
        )
        test_file = pytester.path / "internal_source" / case_dir.name / test_file.name
        nodeid = f"internal_source/{case_dir.name}/test_remote.py::test_case"
    shell = worker_command(pytester.path, test_file)
    body = shell.split("<< '_PER_TEST_JUNIT_PY_'\n", 1)[1].split(
        "\n_PER_TEST_JUNIT_PY_", 1
    )[0]
    pytester.makepyfile(rtp_remote_per_test_junit_plugin=body)
    worker = pytester.runpytest_subprocess(
        "--noconftest",
        "-p",
        "rtp_remote_per_test_junit_plugin",
        "-q",
        "--junitxml=worker.xml",
        "-o",
        "junit_duration_report=total",
        nodeid,
    )
    worker.assert_outcomes(passed=1)
    worker_xml = ET.parse(pytester.path / "worker.xml").getroot()
    tc = worker_xml.find(".//testcase")
    assert remote._testcase_nodeid(tc) == nodeid
    actual_duration = float(tc.get("time"))
    assert actual_duration >= 0.025
    expected_outcome = "passed"
    if scenario == "zero_cases":
        worker_xml.find("testsuite").remove(tc)
    elif scenario == "wrong_identity":
        tc.find("properties/property").set("value", "test_remote.py::other_case")
    elif scenario == "extra_identity":
        ET.SubElement(
            worker_xml.find("testsuite"), "testcase", name="unexpected", time="1"
        )
    elif scenario == "missing_time":
        tc.attrib.pop("time")
    elif scenario in {"nan_time", "negative_time"}:
        tc.set("time", "nan" if scenario == "nan_time" else "-1")
    elif scenario in {"xml_failure", "xml_error"}:
        ET.SubElement(tc, scenario[4:], message="original worker assertion").text = (
            "assert 1 == 2"
        )
    elif scenario in {"skip_forbidden", "skip_allowed"}:
        ET.SubElement(tc, "skipped", message="worker skip")
    xml = ET.tostring(worker_xml, encoding="unicode")
    if scenario == "malformed_xml":
        xml = "<testsuites>"
    stdout = "EXIT_CODE=1\n" if scenario == "exit_mismatch" else "EXIT_CODE=0\n"
    if scenario == "missing_exit":
        stdout = ""
    if scenario != "missing_xml":
        stdout += f"<<<JUNIT_XML>>>\n{xml}\n<<<END_JUNIT_XML>>>\n"
    result = SimpleNamespace(
        exit_code=1 if scenario == "reapi_failure" else 0,
        stdout_raw=stdout.encode(),
        stderr_raw=b"",
        stdout_digest=None,
        stderr_digest=None,
        worker_host_ip="worker-fixture",
        metadata_worker="fixture",
        stream_stdout_path=None,
        stream_stderr_path=None,
        cached_result=scenario == "cached_reapi",
    )
    calls, stored = [], []

    class Controller:
        def pytest_runtest_protocol(self, item, nextitem):
            plugin = object.__new__(remote.RemoteREAPIPlugin)
            plugin.rootdir = pytester.path
            plugin.config = item.config
            plugin.config._rtp_ci_forbid_skips = scenario != "skip_allowed"
            plugin._collect_outputs = False
            plugin._cached_items = {}
            plugin._test_cache = object()
            plugin._cache_manifest = {}
            plugin.executor = SimpleNamespace(reapi_targets_combined="test fixture")
            plugin._store_test_result_in_cache = lambda *args: stored.append(args)
            plugin._report_per_test(item, result, cached=scenario == "cached_local")
            calls.append(item.nodeid)
            return True

    if scenario == "skip_allowed":
        expected_outcome = "skipped"
    elif scenario not in {"pass", "cached_local", "cached_reapi", "internal_source"}:
        expected_outcome = "failed"
    controller = pytester.runpytest(
        "--noconftest",
        "-q",
        "--junitxml=controller.xml",
        nodeid,
        plugins=[Controller()],
    )
    controller.assert_outcomes(**{expected_outcome: 1})
    assert len(calls) == 1
    cases = ET.parse(pytester.path / "controller.xml").findall(".//testcase")
    assert len(cases) == 1
    props = {
        p.get("name"): p.get("value") for p in cases[0].findall("properties/property")
    }
    assert props["nodeid"] == nodeid
    assert props["remote_cached"] == str(scenario.startswith("cached_")).lower()
    assert props["remote_worker"] == "worker-fixture"
    if scenario in {
        "pass",
        "cached_local",
        "cached_reapi",
        "xml_failure",
        "xml_error",
        "internal_source",
    }:
        assert float(cases[0].get("time")) == actual_duration
        originals = list(
            (pytester.path / "bazel-testlogs/pytest/remote_junit").glob("*.worker.xml")
        )
        assert len(originals) == 1
        assert originals[0].read_text() == xml
    assert len(stored) == int(
        expected_outcome == "passed" and scenario != "cached_local"
    )
    if scenario in {"xml_failure", "xml_error"}:
        assert "original worker assertion" in cases[0].find("failure").text
