"""Qualify a candidate in fresh Linux venvs; never alter project dependencies."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

from prepare_decord_candidate import build, UPSTREAM_SHA256


def run(root, name, command, *, cwd, env):
    with (root / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; see original log")


def main(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    proof = {"accepted": False, "RTP_LLM_acceptance": False,
             "revision": os.environ.get("GITHUB_SHA"), "platform": platform.platform(),
             "glibc": platform.libc_ver(), "python": sys.version,
             "scope": "CPU Linux decord candidate only; no RTP-LLM, MORI or GPU acceptance"}
    (root / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    scripts = Path(__file__).resolve().parent
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTHON", "PYTEST", "PIP_", "RTP_", "UV_"))
           and k not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "VIRTUAL_ENV")}
    empty = root / "empty-cwd"
    empty.mkdir()
    with urllib.request.urlopen("https://pypi.org/pypi/decord/0.6.0/json", timeout=60) as response:
        metadata_bytes = response.read()
    (root / "pypi-source.json").write_bytes(metadata_bytes)
    files = [item for item in json.loads(metadata_bytes)["urls"]
             if item["digests"]["sha256"] == UPSTREAM_SHA256]
    assert len(files) == 1
    source = root / files[0]["filename"]
    with urllib.request.urlopen(files[0]["url"], timeout=120) as response:
        source.write_bytes(response.read())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == UPSTREAM_SHA256
    candidate_proof = build(source, root / "candidate")
    shutil.copyfile(root / "candidate/candidate-integrity.json", root / "candidate-integrity.json")
    runtime = root / "runtime"
    run(root, "create-venv", [sys.executable, "-I", "-m", "venv", str(runtime)], cwd=empty, env=env)
    python = runtime / "bin/python"
    run(root, "pip-bootstrap", [str(python), "-I", "-m", "pip", "install", "--no-cache-dir",
                                "pip==26.2.1"], cwd=empty, env=env)
    run(root, "pip-install", [str(python), "-I", "-m", "pip", "install", "--no-cache-dir",
                              "--report", str(root / "pip-install.json"),
                              candidate_proof["candidate"], "pytest"], cwd=empty, env=env)
    run(root, "pip-check", [str(python), "-I", "-m", "pip", "check"], cwd=empty, env=env)
    original_test = scripts / "test_decord_runtime.py"
    test = empty / original_test.name
    shutil.copyfile(original_test, test)
    assert test.read_bytes() == original_test.read_bytes()
    (empty / "pytest.ini").write_text("[pytest]\n")
    env["DECORD_QUALIFICATION_ROOT"] = str(root)
    report = root / "test.xml"
    run(root, "pytest", [str(python), "-I", "-m", "pytest", str(test), "-c", str(empty / "pytest.ini"),
                         "--confcutdir=" + str(empty), "--import-mode=importlib", "-v",
                         "--junitxml=" + str(report)], cwd=empty, env=env)
    expected = {n.name for n in ast.parse(test.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert len(expected) == len(cases) == 6
    assert {c.get("name") for c in cases} == expected
    assert not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
    proof.update(accepted=True, actual=6, failures=0, errors=0, skips=0,
                 candidate_sha256=candidate_proof["candidate_sha256"],
                 source_sha256=UPSTREAM_SHA256,
                 test_source_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
                 runtime_python=str(python), identities=sorted(expected))
    (root / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output.resolve())
