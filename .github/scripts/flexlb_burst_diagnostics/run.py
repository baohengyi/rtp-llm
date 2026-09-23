"""Diagnose the original W burst failure on Linux; no acceptance credit."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source, evidence = map(lambda arg: Path(arg).resolve(), sys.argv[1:])
    evidence.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    baseline = here.parent / "flexlb_burst_qualification"
    manifest = json.loads((baseline / "source-manifest.json").read_text())
    diagnostic = json.loads((here / "instrumentation-manifest.json").read_text())
    proof = {
        "scope": diagnostic["scope"],
        "baseline_scope": manifest["scope"],
        "logical_cpu_count": os.cpu_count(),
        "candidate_tree": manifest["candidate_tree"],
        "base_commit": manifest["base_commit"],
        "host": platform.platform(),
        "machine": platform.machine(),
        "commands": [],
        "accepted": False,
    }
    env = os.environ.copy()
    for key in ("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS", "MAVEN_OPTS", "MAVEN_ARGS"):
        env.pop(key, None)

    def run(name, command, cwd=source):
        record = {"name": name, "argv": command, "cwd": str(cwd)}
        proof["commands"].append(record)
        start = time.monotonic()
        with (evidence / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)
        return result.returncode

    def require(name, command, cwd=source):
        if run(name, command, cwd):
            raise RuntimeError(name + " failed; original log retained")

    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Qualification requires Linux x86_64")
        actual_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if actual_ref != manifest["base_commit"]:
            raise RuntimeError("Unexpected source revision")
        patch = baseline / "production-compatibility.patch"
        if sha(patch) != manifest["patch_sha256"]:
            raise RuntimeError("Source patch digest mismatch")
        require("patch-check", ["git", "apply", "--check", str(patch)])
        require("patch-apply", ["git", "apply", str(patch)])
        verified = []
        for item in manifest["inputs"]:
            path = source / item["path"]
            if sha(path) != item["sha256"] or path.stat().st_size != item["size"]:
                raise RuntimeError("Source differs from candidate: " + item["path"])
            verified.append(item)
        (evidence / "verified-source-inputs.json").write_text(json.dumps(verified, indent=2) + "\n")
        proof["verified_source_files"] = len(verified)
        require("java-version", ["java", "-version"])
        require("maven-version", ["mvn", "-version"])
        flexlb = source / "rtp_llm/flexlb"
        mock = flexlb / "flexlb-api"
        # This only prepares real production artifacts. No Maven-test coverage is credited.
        require("production-build", [
            "mvn", "-B", "-ntp", "-pl", "flexlb-api", "-am", "install",
            "-Dmaven.test.skip=true", "-Derror-prone.skip=true", "-Dautoconfig.skip=true",
        ], flexlb)
        require("test-dependencies", [
            "mvn", "-B", "-ntp", "dependency:build-classpath", "-DincludeScope=test",
            "-Dmdep.outputFile=target/qualification-classpath.txt",
        ], mock)
        require("junit-download", [
            "mvn", "-B", "-ntp", "dependency:copy",
            "-Dartifact=org.junit.platform:junit-platform-console-standalone:1.8.2",
            "-DoutputDirectory=" + str(evidence / "tools"),
        ], mock)
        junit = evidence / "tools/junit-platform-console-standalone-1.8.2.jar"
        if sha(junit) != manifest["junit_console_sha256"]:
            raise RuntimeError("JUnit console differs from local qualification launcher")
        classes = evidence / "test-classes"
        classes.mkdir()
        dependencies = (mock / "target/qualification-classpath.txt").read_text().strip().split(os.pathsep)
        # Resolve project code exclusively from this build's production output directories.
        dependencies = [p for p in dependencies if "/org/flexlb/" not in p]
        production = sorted(flexlb.glob("*/target/classes"))
        resources = sorted(flexlb.glob("*/src/test/resources"))
        classpath = os.pathsep.join(map(str, [classes, *production, *resources, *dependencies, junit]))
        proof["classpath"] = classpath.split(os.pathsep)
        require("test-compile", [
            "javac", "-cp", classpath, "-d", str(classes),
            *[str(source / path) for path in manifest["test_sources"]],
        ], mock)
        # Baseline production and all eleven original test/helper sources have
        # compiled. Apply only explicit diagnostic production overrides now.
        instrumentation = here / "instrumentation.patch"
        if sha(instrumentation) != diagnostic["instrumentation_sha256"]:
            raise RuntimeError("Instrumentation patch digest mismatch")
        for item in diagnostic["sources"]:
            path = source / item["path"]
            expected = item["baseline_sha256"]
            if (expected is None and path.exists()) or (expected is not None and sha(path) != expected):
                raise RuntimeError("Instrumentation baseline mismatch: " + item["path"])
        require("instrumentation-check", ["git", "apply", "--check", str(instrumentation)])
        require("instrumentation-apply", ["git", "apply", str(instrumentation)])
        for item in diagnostic["sources"]:
            path = source / item["path"]
            if sha(path) != item["sha256"] or path.stat().st_size != item["size"]:
                raise RuntimeError("Instrumentation output mismatch: " + item["path"])
        for item in verified:
            if item["path"] in manifest["test_sources"] and sha(source / item["path"]) != item["sha256"]:
                raise RuntimeError("Original test/helper changed")
        proof["diagnostic_sources"] = diagnostic["sources"]
        diagnostic_classes = evidence / "diagnostic-classes"
        diagnostic_classes.mkdir()
        require("instrumentation-compile", [
            "javac", "-proc:none", "-cp", classpath, "-d", str(diagnostic_classes),
            *[str(source / item["path"]) for item in diagnostic["sources"]],
        ], mock)
        classpath = str(diagnostic_classes) + os.pathsep + classpath
        proof["classpath"] = classpath.split(os.pathsep)
        pom = ET.parse(flexlb / "pom.xml")
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}
        profile = next(p for p in pom.findall("./m:profiles/m:profile", ns) if p.findtext("m:id", namespaces=ns) == "api-performance-regression")
        plugins = profile.findall("./m:build/m:plugins/m:plugin", ns)
        surefire = next(p for p in plugins if p.findtext("m:artifactId", namespaces=ns) == "maven-surefire-plugin")
        configuration = surefire.find("./m:executions/m:execution/m:configuration", ns)
        heap = profile.findtext("m:properties/m:flexlb.perf.heap", namespaces=ns)
        if heap != "2g":
            raise RuntimeError("Original performance profile heap changed")
        flags = shlex.split(configuration.findtext("m:argLine", namespaces=ns).replace("${flexlb.perf.heap}", heap))
        proof["jvm_flags"] = flags
        if any("flexlb.perf" in f for f in flags):
            raise RuntimeError("Unexpected performance property override")
        diagnostic_flags = [
            "-Droute.stage.probe.output=" + str(evidence / "measurements.json"),
            "-XX:StartFlightRecording=filename=" + str(evidence / "route-submit.jfr") + ",settings=profile,dumponexit=true",
        ]
        proof["diagnostic_jvm_flags"] = diagnostic_flags
        junit_dir = evidence / "junit"
        code = run("test", [
            "java", *flags, *diagnostic_flags, "-Dsurefire.test.class.path=" + classpath,
            "-jar", str(junit), "--class-path", classpath,
            "--select-method", manifest["test_class"] + "#" + manifest["expected_methods"][0],
            "--reports-dir", str(junit_dir), "--disable-ansi-colors", "--details", "tree", "--fail-if-no-tests",
        ], mock)
        cases = []
        for xml in sorted(junit_dir.glob("TEST-*.xml")):
            for case in ET.parse(xml).getroot().iter("testcase"):
                cases.append({
                    "classname": case.attrib["classname"], "name": case.attrib["name"],
                    "time": case.attrib["time"], "report": xml.name, "report_sha256": sha(xml),
                    "status": next((tag for tag in ("failure", "error", "skipped") if case.find(tag) is not None), "pass"),
                })
        proof["actual_cases"] = cases
        expected = {(manifest["test_class"], m + "()") for m in manifest["expected_methods"]}
        actual = {(c["classname"], c["name"]) for c in cases}
        proof["original_test_passed"] = code == 0 and len(cases) == 1 and actual == expected and all(c["status"] == "pass" for c in cases)
        proof["accepted"] = False
        proof["W_runtime_credit"] = 0
        if not (evidence / "measurements.json").is_file() or not (evidence / "route-submit.jfr").is_file():
            raise RuntimeError("Diagnostic measurements or JFR missing")
        if not proof["original_test_passed"]:
            raise RuntimeError("Original burst method did not pass its one exact identity without failures/errors/skips")
    except Exception as error:
        proof["error"] = str(error)
        raise
    finally:
        (evidence / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
