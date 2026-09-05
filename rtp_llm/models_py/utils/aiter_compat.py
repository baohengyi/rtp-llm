"""Compatibility setup for ROCm AITer kernels."""

import sys
from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from typing import Any, Optional


_JIT_PYBIND_ABI_BROKEN_VERSION_PREFIX = "0.1.21.dev80+g987203ba5"
_JIT_BUILD_LOCK = RLock()


def _installed_aiter_version() -> str:
    try:
        return version("aiter")
    except PackageNotFoundError:
        return ""


def _aiter_runtime_cpp_extension_module() -> Any:
    from aiter.jit import core as aiter_core

    # AITer prepends jit/utils to sys.path and imports this as the top-level
    # ``cpp_extension`` module. Importing it by package name creates a second
    # module object, so resolve the builder that core actually calls.
    return sys.modules[aiter_core._jit_compile.__module__]


def call_aiter_with_bundled_core_abi(
    operation: Any,
    *args: Any,
    aiter_version: Optional[str] = None,
    cpp_extension_module: Any = None,
    **kwargs: Any,
) -> Any:
    """Run an affected JIT build with the bundled core module's pybind ABI.

    The affected wheel builds ``module_aiter_core`` with pybind11's default ABI
    namespace. Its JIT builder overrides that namespace with PyTorch's pybind11
    ABI constants, so a freshly compiled module cannot recognize the
    ``aiter_tensor_t`` instances created by the bundled core module.
    """
    resolved_version = (
        aiter_version if aiter_version is not None else _installed_aiter_version()
    )
    if not resolved_version.startswith(_JIT_PYBIND_ABI_BROKEN_VERSION_PREFIX):
        return operation(*args, **kwargs)

    if cpp_extension_module is None:
        cpp_extension_module = _aiter_runtime_cpp_extension_module()

    with _JIT_BUILD_LOCK:
        original_flags = cpp_extension_module._get_pybind11_abi_build_flags
        cpp_extension_module._get_pybind11_abi_build_flags = lambda: []
        try:
            return operation(*args, **kwargs)
        finally:
            cpp_extension_module._get_pybind11_abi_build_flags = original_flags
