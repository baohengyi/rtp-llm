import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch


_IMPL_DIR = Path(__file__).resolve().parents[1]
_MODULE_PATH = _IMPL_DIR / "aiter_compat.py"
_SPEC = importlib.util.spec_from_file_location(
    "rtp_aiter_compat_test_module", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_JIT_MODULE_PATH = Path(__file__).resolve().parents[6] / "utils" / "aiter_compat.py"
_JIT_SPEC = importlib.util.spec_from_file_location(
    "rtp_aiter_jit_compat_test_module", _JIT_MODULE_PATH
)
assert _JIT_SPEC is not None and _JIT_SPEC.loader is not None
_JIT_MODULE = importlib.util.module_from_spec(_JIT_SPEC)
_JIT_SPEC.loader.exec_module(_JIT_MODULE)


class AiterCompatTest(TestCase):
    def test_affected_gfx942_build_uses_ck_sorting(self):
        with patch.dict(os.environ, {}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx942:sramecc+:xnack-",
                aiter_version="0.1.17.dev79+g2570b35f9.d20260623",
            )

            self.assertTrue(changed)
            self.assertEqual(os.environ["AITER_USE_CK_MOE_SORTING"], "1")

        cpp_extension = SimpleNamespace(
            _get_pybind11_abi_build_flags=lambda: ["torch-abi"]
        )
        def operation(value):
            self.assertEqual(
                cpp_extension._get_pybind11_abi_build_flags(),
                list(_JIT_MODULE._BUNDLED_CORE_PYBIND_ABI_FLAGS),
            )
            return value

        result = _JIT_MODULE.call_aiter_with_bundled_core_abi(
            operation,
            7,
            aiter_version="0.1.21.dev80+g987203ba5.d20260825",
            cpp_extension_module=cpp_extension,
        )
        self.assertEqual(result, 7)
        self.assertEqual(
            cpp_extension._get_pybind11_abi_build_flags(), ["torch-abi"]
        )

        runtime_cpp_extension = SimpleNamespace(
            _get_pybind11_abi_build_flags=lambda: ["runtime-torch-abi"]
        )
        runtime_compile = lambda: None
        runtime_compile.__module__ = "cpp_extension"
        aiter_package = ModuleType("aiter")
        aiter_package.__path__ = []
        jit_package = ModuleType("aiter.jit")
        jit_package.__path__ = []
        aiter_core = ModuleType("aiter.jit.core")
        aiter_core._jit_compile = runtime_compile
        jit_package.core = aiter_core
        aiter_package.jit = jit_package

        with patch.dict(
            sys.modules,
            {
                "aiter": aiter_package,
                "aiter.jit": jit_package,
                "aiter.jit.core": aiter_core,
                "cpp_extension": runtime_cpp_extension,
            },
        ):
            result = _JIT_MODULE.call_aiter_with_bundled_core_abi(
                lambda: runtime_cpp_extension._get_pybind11_abi_build_flags(),
                aiter_version="0.1.21.dev80+g987203ba5.d20260825",
            )

        self.assertEqual(
            result,
            list(_JIT_MODULE._BUNDLED_CORE_PYBIND_ABI_FLAGS),
        )
        self.assertEqual(
            runtime_cpp_extension._get_pybind11_abi_build_flags(),
            ["runtime-torch-abi"],
        )

    def test_gfx950_keeps_opus_default(self):
        with patch.dict(os.environ, {}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx950",
                aiter_version="0.1.17.dev79+g2570b35f9.d20260623",
            )

            self.assertFalse(changed)
            self.assertNotIn("AITER_USE_CK_MOE_SORTING", os.environ)

    def test_newer_aiter_build_keeps_its_default(self):
        with patch.dict(os.environ, {}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx942",
                aiter_version="0.1.18.dev1+gabcdef.d20260801",
            )

            self.assertFalse(changed)
            self.assertNotIn("AITER_USE_CK_MOE_SORTING", os.environ)

        original_flags = lambda: ["torch-abi"]
        cpp_extension = SimpleNamespace(
            _get_pybind11_abi_build_flags=original_flags
        )
        result = _JIT_MODULE.call_aiter_with_bundled_core_abi(
            lambda: "unchanged",
            aiter_version="0.1.22.dev1+gabcdef.d20260901",
            cpp_extension_module=cpp_extension,
        )
        self.assertEqual(result, "unchanged")
        self.assertIs(
            cpp_extension._get_pybind11_abi_build_flags, original_flags
        )

    def test_explicit_user_choice_is_preserved(self):
        with patch.dict(os.environ, {"AITER_USE_CK_MOE_SORTING": "0"}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx942",
                aiter_version="0.1.17.dev79+g2570b35f9.d20260623",
            )

            self.assertFalse(changed)
            self.assertEqual(os.environ["AITER_USE_CK_MOE_SORTING"], "0")

    def test_executor_configures_sorting_before_importing_aiter(self):
        source = (_IMPL_DIR / "executors" / "rocm_moe.py").read_text()

        self.assertLess(
            source.index("configure_aiter_moe_sorting()"),
            source.index('importlib.import_module("aiter")'),
        )


if __name__ == "__main__":
    main()
