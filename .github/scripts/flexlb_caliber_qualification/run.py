"""Run the complete, unchanged decode-caliber class against a pinned source snapshot."""

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
    manifest = json.loads((here / "source-manifest.json").read_text())
    proof = {
        "scope": manifest["scope"],
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
        patch = here / "production-compatibility.patch"
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
        mock = flexlb / "flexlb-mock-engine"
        # This only prepares real production artifacts. No Maven-test coverage is credited.
        require("production-build", [
            "mvn", "-B", "-ntp", "-pl", "flexlb-mock-engine", "-am", "install",
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
            str(source / manifest["test_source"]), str(source / manifest["helper_source"]),
        ], mock)
        pom = ET.parse(flexlb / "pom.xml")
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}
        plugins = pom.findall("./m:build/m:pluginManagement/m:plugins/m:plugin", ns)
        surefire = next(p for p in plugins if p.findtext("m:artifactId", namespaces=ns) == "maven-surefire-plugin")
        flags = shlex.split(surefire.findtext("m:configuration/m:argLine", namespaces=ns))
        if any("flexlb.perf" in f for f in flags):
            raise RuntimeError("Unexpected performance property override")
        junit_dir = evidence / "junit"
        code = run("test", [
            "java", *flags, "-Dsurefire.test.class.path=" + classpath,
            "-jar", str(junit), "--class-path", classpath,
            "--select-class", "org.flexlb.mockengine.ProductionCaliberDecodeTest",
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
        expected = {("org.flexlb.mockengine.ProductionCaliberDecodeTest", m + "()") for m in manifest["expected_methods"]}
        actual = {(c["classname"], c["name"]) for c in cases}
        proof["accepted"] = code == 0 and len(cases) == 8 and actual == expected and all(c["status"] == "pass" for c in cases)
        if not proof["accepted"]:
            raise RuntimeError("Original class did not pass all eight identities without failures/errors/skips")
    except Exception as error:
        proof["error"] = str(error)
        raise
    finally:
        (evidence / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
