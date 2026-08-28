import ast
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.version import Version
from setuptools import find_namespace_packages

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Hosts that serve internal-only / non-publicly-reproducible build artifacts. Direct wheel pins
# from these must never reach a publicly published wheel's install metadata. Kept deliberately
# narrow (see the review decision): the OSS `rtp-opensource.*` bucket and `download.pytorch.org`
# are public and allowed.
INTERNAL_ONLY_HOST_MARKERS = ("sinian-metrics-platform",)

# Platform extras whose stack is merged into install_requires of a PUBLICLY published wheel.
# `rocm` is intentionally excluded: its wheel is not published to public indexes, so its
# internal `sinian-metrics-platform` pins never ship to external users. If ROCm wheels ever
# become publicly published, add "rocm" here and relocate those pins to the internal overlay.
PUBLIC_PLATFORM_EXTRAS = ("cuda12", "cuda12_arm", "cuda12_9")


def _oss_optional_extras() -> dict:
    """Load [project.optional-dependencies] from the OSS extras file that ships in this repo."""
    extras_file = PROJECT_ROOT / "_build" / "oss_optional_extras.toml"
    with open(extras_file, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("optional-dependencies", {})


def _requirement_url(req: str) -> str:
    """Return the direct-reference URL of a `name @ url` requirement, else ''."""
    parts = req.split(" @ ", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def _is_local_path_reference(url: str) -> bool:
    """True if a direct reference points at a local path rather than a remote artifact."""
    if not url:
        return False
    if url.startswith("file:"):
        return True
    # Absolute or relative filesystem paths (never valid for a published wheel).
    return url.startswith(("/", "./", "../"))


def _load_platform_module():
    """Load _build/platform.py in isolation (stdlib-only, no side effects)."""
    spec = importlib.util.spec_from_file_location(
        "_rtp_build_platform_under_test", PROJECT_ROOT / "_build" / "platform.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "_rtp_llm_setup_under_test", PROJECT_ROOT / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(os.environ, {"RTP_BAZEL_CONFIG": "--config=cuda12_9"}, clear=False):
        spec.loader.exec_module(module)
    return module


class BuildPackagingContractTest(TestCase):
    def test_transformers5_uses_compatible_xgrammar_metadata(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        requirements = {
            requirement.name: requirement
            for requirement in map(
                Requirement,
                pyproject["tool"]["rtp-llm"]["base-dependencies"],
            )
        }
        transformers_version = next(
            Version(specifier.version)
            for specifier in requirements["transformers"].specifier
            if specifier.operator == "=="
        )
        xgrammar_version = next(
            Version(specifier.version)
            for specifier in requirements["xgrammar"].specifier
            if specifier.operator == "=="
        )

        if transformers_version.major >= 5:
            self.assertGreaterEqual(
                xgrammar_version,
                Version("0.2.6rc1"),
                "xgrammar 0.2.5 metadata requires transformers<5",
            )

    def test_xgrammar_platform_extras_use_compatible_tvm_ffi(self):
        xgrammar_minimum_tvm_ffi = Version("0.1.10")
        for extra_name in ("cuda12", "cuda12_9"):
            extra_requirements = {
                requirement.name: requirement
                for requirement in map(Requirement, _oss_optional_extras()[extra_name])
            }
            tvm_ffi_version = next(
                Version(specifier.version)
                for specifier in extra_requirements["apache-tvm-ffi"].specifier
                if specifier.operator == "=="
            )
            self.assertGreaterEqual(
                tvm_ffi_version,
                xgrammar_minimum_tvm_ffi,
                f"{extra_name} must satisfy xgrammar's apache-tvm-ffi requirement",
            )

    def test_cub_compat_does_not_require_cuda_header_on_rocm(self):
        compat_header = (PROJECT_ROOT / "3rdparty/cub_compat.h").read_text()

        self.assertNotIn("#include <cub/version.cuh>", compat_header)
        self.assertIn("defined(CUB_VERSION)", compat_header)

    def test_dash_sc_protos_are_part_of_python_build_outputs(self):
        setup_module = _load_setup_module()

        expected = {
            "rtp_llm/dash_sc/proto/model_config_pb2.py",
            "rtp_llm/dash_sc/proto/model_config_pb2_grpc.py",
            "rtp_llm/dash_sc/proto/predict_v2_pb2.py",
            "rtp_llm/dash_sc/proto/predict_v2_pb2_grpc.py",
            "rtp_llm/dash_sc/proto/__init__.py",
        }
        self.assertEqual(set(setup_module.DASH_SC_PROTO_OUTPUTS), expected)
        self.assertTrue(expected.issubset(set(setup_module.PROTO_OUTPUTS)))

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            proto_dir = project_root / "rtp_llm/dash_sc/proto"
            generator = proto_dir / "create_grpc_proto.py"
            with patch.object(setup_module.subprocess, "run") as run:
                setup_module._generate_dash_sc_proto_files(project_root)
            run.assert_called_once_with(
                [setup_module.sys.executable, str(generator), str(proto_dir)],
                check=True,
            )

    def test_core_build_stages_grammar_tokenizer_binding(self):
        setup_module = _load_setup_module()

        self.assertIn(
            (
                "core",
                "//:th_grammar_tokenizer_info",
                ("libth_grammar_tokenizer_info.so",),
            ),
            setup_module._CORE_BAZEL_STAGED_OUTPUTS,
        )

    def test_stubgen_preloads_native_modules_in_runtime_order(self):
        setup_module = _load_setup_module()

        self.assertEqual(
            setup_module._pybind_stubgen_preload("libth_transformer"),
            "rtp_llm.ops.ensure_engine_ops_loaded(); ",
        )
        self.assertEqual(
            setup_module._pybind_stubgen_preload("librtp_compute_ops"),
            "rtp_llm.ops.ensure_compute_ops_loaded(); ",
        )
        self.assertEqual(
            setup_module._pybind_stubgen_preload("libth_transformer_config"),
            "",
        )

    def _bazel_cmd_prefix(self, setup_module, scope=None):
        env = {"XDG_CACHE_HOME": "/tmp/rtp-llm-test-cache"}
        if scope is not None:
            env["RTP_BAZEL_CACHE_SCOPE"] = scope
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                setup_module,
                "parse_bazel_config",
                return_value=["--config=cuda12_9"],
            ),
            patch.object(setup_module, "_get_remote_bazel_args", return_value=[]),
            patch.object(setup_module, "_get_local_jobs_args", return_value=[]),
        ):
            if scope is None:
                os.environ.pop("RTP_BAZEL_CACHE_SCOPE", None)
            return setup_module._get_bazel_cmd_prefix("cuda12_9")

    def test_bazel_cache_scope_defaults_to_platform_cache(self):
        setup_module = _load_setup_module()

        cmd, build_args = self._bazel_cmd_prefix(setup_module)

        self.assertEqual(
            cmd,
            [
                "bazelisk",
                "--output_user_root=/tmp/rtp-llm-test-cache/bazel_cuda12_9_cache",
            ],
        )
        self.assertEqual(build_args, ["--config=cuda12_9"])

    def test_bazel_cache_scope_isolates_cpp_ut_cache(self):
        setup_module = _load_setup_module()

        cmd, build_args = self._bazel_cmd_prefix(setup_module, scope="cpp_ut")

        self.assertEqual(
            cmd,
            [
                "bazelisk",
                "--output_user_root=/tmp/rtp-llm-test-cache/bazel_cuda12_9_cpp_ut_cache",
            ],
        )
        self.assertEqual(build_args, ["--config=cuda12_9"])

    def test_bazel_cache_scope_rejects_unsafe_path_content(self):
        setup_module = _load_setup_module()

        with self.assertRaisesRegex(ValueError, "RTP_BAZEL_CACHE_SCOPE"):
            self._bazel_cmd_prefix(setup_module, scope="../../cpp-ut")

    def test_remote_bazel_tests_allow_for_gpu_lock_queueing(self):
        setup_module = _load_setup_module()

        with patch.object(setup_module, "is_remote_enabled", return_value=True):
            args = setup_module._with_default_remote_test_timeout(
                ["--config=cuda12_9"]
            )

        self.assertEqual(args, ["--config=cuda12_9", "--test_timeout=900"])

    def test_explicit_remote_test_timeout_is_preserved(self):
        setup_module = _load_setup_module()

        with patch.object(setup_module, "is_remote_enabled", return_value=True):
            args = setup_module._with_default_remote_test_timeout(
                ["--config=cuda12_9", "--test_timeout=1200"]
            )

        self.assertEqual(args, ["--config=cuda12_9", "--test_timeout=1200"])

    def test_local_bazel_tests_keep_native_timeout(self):
        setup_module = _load_setup_module()

        with patch.object(setup_module, "is_remote_enabled", return_value=False):
            args = setup_module._with_default_remote_test_timeout(
                ["--config=cuda12_9"]
            )

        self.assertEqual(args, ["--config=cuda12_9"])

    def test_build_cleanup_removes_only_stale_test_artifacts(self):
        setup_module = _load_setup_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            output_root = tmp_path / "bazel_cuda12_9_cache"
            testlogs = (
                output_root
                / "install-hash"
                / "execroot"
                / "rtp_llm"
                / "bazel-out"
                / "k8-opt"
                / "testlogs"
            )
            testlogs.mkdir(parents=True)
            (testlogs / "old-test.xml").write_text("old", encoding="utf-8")
            keep = output_root / "install-hash" / "action-cache"
            keep.parent.mkdir(parents=True, exist_ok=True)
            keep.write_text("keep", encoding="utf-8")

            project_root.mkdir()
            (project_root / "bazel-testlogs").symlink_to(testlogs)
            remote_logs = project_root / ".pytest_cache" / "remote_stream_logs"
            remote_logs.mkdir(parents=True)
            (remote_logs / "old.log").write_text("old", encoding="utf-8")

            setup_module._clean_stale_test_artifacts(
                project_root,
                ["bazelisk", f"--output_user_root={output_root}"],
            )

            self.assertFalse(testlogs.exists())
            self.assertFalse((project_root / "bazel-testlogs").exists())
            self.assertFalse(remote_logs.exists())
            self.assertEqual(keep.read_text(encoding="utf-8"), "keep")

    def test_cuda129_stages_cuda_graph_pytest_binding_outside_wheel_glob(self):
        setup_module = _load_setup_module()

        staged = setup_module._selected_bazel_staged_outputs(
            "cuda12_9", ["--config=cuda12_9"]
        )
        runner = [
            entry
            for entry in staged
            if entry[1] == "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner"
        ]

        self.assertEqual(
            runner,
            [
                (
                    "test",
                    "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner",
                    (
                        (
                            "libtest_cuda_graph_runner.so",
                            "test/libtest_cuda_graph_runner.so",
                        ),
                    ),
                )
            ],
        )

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        excluded = pyproject["tool"]["setuptools"]["exclude-package-data"][
            "rtp_llm"
        ]
        self.assertIn("libs/test/*", excluded)

    def test_cuda129_stages_pywrapped_model_pytest_binding(self):
        setup_module = _load_setup_module()

        staged = setup_module._selected_bazel_staged_outputs(
            "cuda12_9", ["--config=cuda12_9"]
        )
        self.assertIn(
            (
                "test",
                "//rtp_llm/cpp/models/test:th_pywrapped_model_cache_store_integration_test",
                (
                    (
                        "libth_pywrapped_model_cache_store_integration_test.so",
                        "test/libth_pywrapped_model_cache_store_integration_test.so",
                    ),
                ),
            ),
            staged,
        )

    def test_pywrapped_model_integration_test_is_h20_pytest_only(self):
        build_file = PROJECT_ROOT / "rtp_llm/cpp/models/test/BUILD"
        if not build_file.exists():
            self.skipTest("source-only Bazel BUILD file is not staged")
        build_text = build_file.read_text(encoding="utf-8")
        target_block = re.search(
            r'py_test\(\s*name = "pywrapped_model_cache_store_integration_test",'
            r'.*?\n\)',
            build_text,
            re.S,
        )
        self.assertIsNotNone(target_block)
        self.assertIn('tags = ["manual"]', target_block.group(0))

        test_file = (
            PROJECT_ROOT
            / "rtp_llm/cpp/models/test/pywrapped_model_cache_store_integration_test.py"
        )
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
        test_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PyWrappedModelCacheStoreIntegrationTest"
        )
        decorators = {ast.unparse(node) for node in test_class.decorator_list}
        self.assertIn("pytest.mark.H20", decorators)
        test_text = test_file.read_text(encoding="utf-8")
        self.assertIn("subprocess.run(", test_text)
        self.assertIn('"--native-scenario"', test_text)
        self.assertIn("_run_native_scenario_isolated", test_text)
        self.assertNotIn("ensure_compute_ops_loaded()", test_text)

        binding_file = (
            PROJECT_ROOT
            / "rtp_llm/cpp/models/test/PyWrappedModelCacheStoreIntegrationTest.cc"
        )
        binding_text = binding_file.read_text(encoding="utf-8")
        self.assertIn("registerPyOpDefs(m)", binding_text)
        self.assertNotIn('py::module_::import("librtp_compute_ops")', binding_text)
        self.assertIn(
            "std::make_shared<StoreContext>(shared_from_this())", binding_text
        )
        self.assertIn(
            "store_context->store(request_block_buffers, timeout_ms)", binding_text
        )

        build_text = build_file.read_text(encoding="utf-8")
        self.assertIn(
            '"//rtp_llm/models_py/bindings/core:exec_ops_test_lib"', build_text
        )

    def test_config_pickle_test_is_h20_pytest_only(self):
        build_file = PROJECT_ROOT / "rtp_llm/cpp/pybind/BUILD"
        if not build_file.exists():
            self.skipTest("source-only Bazel BUILD file is not staged")
        build_text = build_file.read_text(encoding="utf-8")
        target_block = re.search(
            r'py_test\(\s*name = "config_pickle_test",.*?\n\)',
            build_text,
            re.S,
        )
        self.assertIsNotNone(target_block)
        self.assertIn('tags = ["manual"]', target_block.group(0))

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        self.assertIn(
            "rtp_llm/cpp/pybind",
            pyproject["tool"]["pytest"]["ini_options"]["testpaths"],
        )

        test_file = PROJECT_ROOT / "rtp_llm/cpp/pybind/config_pickle_test.py"
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
        test_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "GrammarConfigPickleTest"
        )
        decorators = {ast.unparse(node) for node in test_class.decorator_list}
        self.assertIn("pytest.mark.H20", decorators)

    def test_cuda_graph_runner_defers_native_import_until_execution(self):
        module_path = (
            PROJECT_ROOT / "rtp_llm/cpp/cuda_graph/tests/cuda_graph_test_runner.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_rtp_cuda_graph_test_runner_contract", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None

        with patch.dict("sys.modules", {"libtest_cuda_graph_runner": None}):
            spec.loader.exec_module(module)
            with self.assertRaisesRegex(
                ImportError, "libtest_cuda_graph_runner.so not found"
            ):
                module.CudaGraphRunner()

    def test_cuda_graph_unittest_fixtures_do_not_initialize_during_collection(self):
        for filename, class_name in (
            ("cuda_graph_decode_padding.py", "TestCudaGraphDecodePadding"),
            ("cuda_graph_prefill.py", "TestCudaGraphPrefill"),
        ):
            path = PROJECT_ROOT / "rtp_llm/cpp/cuda_graph/tests" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
            test_class = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            methods = {
                node.name for node in test_class.body if isinstance(node, ast.FunctionDef)
            }
            self.assertIn("setUp", methods)
            self.assertNotIn("__init__", methods)

    def test_non_cuda129_builds_do_not_stage_cuda_graph_pytest_binding(self):
        setup_module = _load_setup_module()

        staged = setup_module._selected_bazel_staged_outputs(
            "rocm", ["--config=rocm"]
        )

        self.assertNotIn(
            "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner",
            [entry[1] for entry in staged],
        )

    def test_dynamic_version_uses_release_version(self):
        setup_module = _load_setup_module()
        release_text = (PROJECT_ROOT / "rtp_llm" / "release_version.py").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r'^RELEASE_VERSION\s*=\s*["\']([^"\']+)["\']', release_text, re.M
        )
        assert match is not None
        expected = match.group(1)

        self.assertEqual(setup_module.get_release_version(), expected)
        self.assertEqual(setup_module.get_version_with_platform(), f"{expected}+cu129")

    def test_public_platform_extras_have_no_internal_only_wheel_pins(self):
        """Publicly published wheels must not carry internal-only wheel sources in their metadata.

        setup.get_all_dependencies() merges the auto-detected platform's extras into
        install_requires, so any internal-only direct wheel pin in a public platform stack would
        leak the internal source into the published wheel's install metadata. Assert none of the
        public platform extras reference an internal-only host.
        """
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                url = _requirement_url(req)
                if any(marker in url for marker in INTERNAL_ONLY_HOST_MARKERS):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders,
            [],
            "Public platform extras must not pin internal-only wheels; move these to the "
            f"internal overlay (internal_source/pyproject_internal.toml):\n{offenders}",
        )

    def test_public_platform_extras_have_no_local_path_dependencies(self):
        """Public platform extras must not reference local filesystem paths (non-reproducible)."""
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                if _is_local_path_reference(_requirement_url(req)):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders, [], f"Public platform extras must not use local paths:\n{offenders}"
        )

    def test_public_platform_extras_use_https_for_direct_wheels(self):
        """Direct wheel pins in public platform extras must be fetched over HTTPS, not plaintext."""
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                url = _requirement_url(req)
                if url.startswith("http://"):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders, [], f"Public platform extras must use https:// wheel URLs:\n{offenders}"
        )

    def test_cuda129_rtp_kernel_pin_contains_sm120_cubins(self):
        """Keep the SM120-capable rtp-kernel build across packaging migrations."""
        public_deps = _oss_optional_extras()["cuda12_9"]
        public_pins = [req for req in public_deps if req.startswith("rtp_kernel @ ")]
        self.assertEqual(len(public_pins), 1)
        self.assertIn("/rtp_kernel_260612/", public_pins[0])

        internal_overlay = PROJECT_ROOT / "internal_source" / "pyproject_internal.toml"
        if internal_overlay.exists():
            with open(internal_overlay, "rb") as f:
                internal_extras = tomllib.load(f)["project"]["optional-dependencies"]
            internal_pins = [
                req
                for req in internal_extras["cuda12_9"]
                if req.startswith("rtp_kernel @ ")
            ]
            self.assertEqual(len(internal_pins), 1)
            self.assertIn("/rtp_kernel_260612/", internal_pins[0])

    def test_pytest_testpaths_all_exist(self):
        """Every configured pytest testpath must exist.

        A stale/typo'd testpath (e.g. rtp_llm/models/multimodal/test vs the real
        rtp_llm/multimodal/test) is silently dropped by pytest, so those tests never run in CI.
        Assert each path resolves so such drift fails loudly at contract-test time instead.
        """
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        testpaths = pyproject["tool"]["pytest"]["ini_options"]["testpaths"]
        missing = [p for p in testpaths if not (PROJECT_ROOT / p).exists()]
        self.assertEqual(
            missing, [], f"pyproject testpaths point at non-existent directories: {missing}"
        )

    def test_py_ut_amd_profile_collects_only_rocm_roots(self):
        """ROCm collection must not import CUDA-only modules before -m filtering."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profile = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"][
            "py_ut_amd"
        ]
        expected_paths = [
            "rtp_llm/models_py/modules/base/rocm/test/",
            "rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/",
            "rtp_llm/models_py/modules/factory/linear/impl/rocm/test/",
            "rtp_llm/utils/test/jit_cache_smoke_test.py",
        ]
        self.assertEqual(profile["paths"], expected_paths)
        self.assertTrue(all((PROJECT_ROOT / path).exists() for path in expected_paths))

    def test_non_sm100_py_ut_profiles_ignore_dsv4(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profiles = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
        dsv4_paths = ["rtp_llm/test/dsv4", "rtp_llm/models_py/modules/dsv4"]
        self.assertEqual(profiles["py_ut_sm8x"]["ignore_paths"], dsv4_paths)
        self.assertEqual(profiles["py_ut_sm9x"]["ignore_paths"], dsv4_paths)
        self.assertNotIn("ignore_paths", profiles["py_ut_sm100_arm"])

    def test_deepgemm_optional_symbol_does_not_block_available_symbols(self):
        wrapper_path = (
            PROJECT_ROOT
            / "rtp_llm"
            / "models_py"
            / "kernels"
            / "cuda"
            / "deepgemm_wrapper.py"
        )
        tree = ast.parse(wrapper_path.read_text())
        init_func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_lazy_init_deep_gemm"
        )
        namespace = {
            "List": list,
            "has_deep_gemm": lambda: True,
            "_prepare_deep_gemm_jit_env": lambda: None,
            "resolve_symbol": lambda module, new, old: getattr(
                module, new, getattr(module, old, None)
            ),
            "_deep_gemm_impl_new_map": {
                "available": "available_impl",
                "optional": "optional_impl",
            },
            "_deep_gemm_impl_old_map": {
                "available": "available_impl",
                "optional": "optional_impl",
            },
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[init_func], type_ignores=[])
                ),
                str(wrapper_path),
                "exec",
            ),
            namespace,
        )

        available_impl = object()
        fake_deep_gemm = SimpleNamespace(available_impl=available_impl)
        with patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}):
            namespace["_lazy_init_deep_gemm"](["available", "optional"])

        self.assertIs(namespace["_available_impl"], available_impl)
        self.assertIsNone(namespace.get("_optional_impl"))

    def test_rocm_wheel_version_matches_dependency_abi(self):
        """The rocm wheel version suffix must track the ROCm ABI the rocm extras are built for.

        get_version_with_platform() stamps every OSS ROCm wheel with this suffix, so a stale value
        (e.g. rocm62 while the deps/toolchain moved to ROCm 7.2) makes cache/publish/rollback pick
        the wrong binary stack. Derive the ABI from the suffix and assert the rocm extras' wheels
        actually reference it — and that no wheel references a different ROCm ABI.
        """
        platform_module = _load_platform_module()
        suffix = platform_module.PLATFORM_CONFIG_VERSIONS.get("rocm", "")
        m = re.fullmatch(r"rocm(\d)(\d+)", suffix)
        self.assertIsNotNone(m, f"unexpected rocm version suffix {suffix!r}")
        expected_abi = f"{m.group(1)}.{m.group(2)}"  # rocm72 -> "7.2"

        rocm_reqs = _oss_optional_extras().get("rocm", [])
        rocm_abis = set(re.findall(r"rocm(\d+\.\d+)", " ".join(rocm_reqs)))
        self.assertIn(
            expected_abi,
            rocm_abis,
            f"rocm suffix {suffix!r} (ABI {expected_abi}) not found in rocm extras "
            f"wheel URLs (found ABIs: {sorted(rocm_abis)})",
        )
        stale = rocm_abis - {expected_abi}
        self.assertEqual(
            stale, set(), f"rocm extras reference ROCm ABIs {sorted(stale)} != suffix {expected_abi}"
        )

    def test_pytest_entry_points_are_packaged_with_tests(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        packages = set(
            find_namespace_packages(
                where=str(PROJECT_ROOT), include=["rtp_llm", "rtp_llm.*"]
            )
        )

        self.assertIn("rtp_llm.test.remote_tests", packages)
        self.assertIn("rtp_llm.test.smoke_framework", packages)
        find_cfg = pyproject["tool"]["setuptools"]["packages"]["find"]
        self.assertNotIn("exclude", find_cfg)

        entry_points = pyproject["project"]["entry-points"]["pytest11"]
        for target in entry_points.values():
            module_name = target.split(":", 1)[0]
            module_path = PROJECT_ROOT / (module_name.replace(".", "/") + ".py")
            self.assertTrue(module_path.exists(), module_name)

        package_data = pyproject["tool"]["setuptools"]["package-data"]["rtp_llm"]
        self.assertIn("test/**/*.proto", package_data)
        self.assertIn("test/**/*.json", package_data)
        self.assertIn("models_py/**/data/*.json", package_data)
