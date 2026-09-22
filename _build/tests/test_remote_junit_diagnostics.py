"""Report diagnostics must not impersonate test execution or hide failures."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from rtp_llm.test.remote_tests.executor import ExecutionResult
from rtp_llm.test.remote_tests.junit_merge import merge_reports
from rtp_llm.test.remote_tests.plugin import RemoteREAPIPlugin


def _controller(tmp_path):
    plugin = RemoteREAPIPlugin.__new__(RemoteREAPIPlugin)
    plugin.rootdir = tmp_path
    plugin._session_cached_entries = {}
    plugin._test_cache = None
    plugin._cache_manifest = None
    plugin._collect_outputs = False
    plugin.ci_profile = None
    plugin.config = SimpleNamespace()
    plugin.executor = SimpleNamespace(reapi_targets_combined="test executor")
    return plugin


@pytest.mark.parametrize("bad", [None, "<broken", "<testsuite/>",
                                  "<testsuite><testcase name='failed'><failure/></testcase></testsuite>",
                                  "<testsuite><testcase name='skipped'><skipped/></testcase></testsuite>"])
def test_integrity_error_preserves_real_case_count_and_failure(tmp_path, bad):
    good = tmp_path / "good.xml"
    good.write_text("<testsuite><testcase name='passed' time='0.1'/></testsuite>")
    other = tmp_path / "other.xml"
    if bad is not None:
        other.write_text(bad)
    output = tmp_path / "merged.xml"
    with pytest.raises(ValueError):
        merge_reports([good, other], output, required=[good, other], forbid_skips=True)
    controller = _controller(tmp_path)
    xml, actual, replayed = controller._merge_session_junit(output.read_text())
    expected = 2 if bad and "testcase" in bad else 1
    assert actual == expected
    assert replayed == 0
    root = ET.fromstring(xml)
    assert len(list(root.iter("testcase"))) == expected + 1
    assert root.find(".//testcase[@name='complete_execution_reports']/error") is not None
    result = ExecutionResult(exit_code=0, stdout_raw=(
        "EXIT_CODE=0\n<<<JUNIT_XML>>>" + xml + "<<<END_JUNIT_XML>>>"
    ).encode())
    assert controller._parse_remote_output(result, None) != 0


def test_real_case_named_like_diagnostic_still_counts(tmp_path):
    xml = """<testsuites><testsuite name="report-integrity">
      <testcase name="complete_execution_reports"><error/></testcase>
    </testsuite></testsuites>"""
    _, actual, _ = _controller(tmp_path)._merge_session_junit(xml)
    assert actual == 1


@pytest.mark.parametrize("outcome", ["failure", "error"])
def test_junit_failure_cannot_be_overridden_by_zero_worker_exit(tmp_path, outcome):
    xml = f"<testsuites><testsuite><testcase name='real'><{outcome}/></testcase></testsuite></testsuites>"
    result = ExecutionResult(exit_code=0, stdout_raw=(
        "EXIT_CODE=0\n<<<JUNIT_XML>>>" + xml + "<<<END_JUNIT_XML>>>"
    ).encode())
    assert _controller(tmp_path)._parse_remote_output(result, None) != 0
