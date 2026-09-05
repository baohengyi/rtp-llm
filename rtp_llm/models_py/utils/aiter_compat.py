"""Compatibility setup for ROCm AITer kernels."""

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


def call_aiter_with_bundled_core_abi(
    operation: Any,
    *args: Any,
    aiter_version: Optional[str] = None,
    cpp_extension_module: Any = None,
    jit_core_module: Any = None,
    **kwargs: Any,
) -> Any:
    """Rebuild affected ops with the bundled core module's pybind ABI.

    The affected wheel builds ``module_aiter_core`` with pybind11's default ABI
    namespace, while some bundled op modules use PyTorch's pybind11 ABI
    constants. Force those ops through AITer's built-in one-time rebuild path
    without the constants so they recognize the ``aiter_tensor_t`` instances
    created by the bundled core module.
    """
    resolved_version = (
        aiter_version if aiter_version is not None else _installed_aiter_version()
    )
    if not resolved_version.startswith(_JIT_PYBIND_ABI_BROKEN_VERSION_PREFIX):
        return operation(*args, **kwargs)

    if cpp_extension_module is None:
        from aiter.jit.utils import cpp_extension as cpp_extension_module
    if jit_core_module is None:
        from aiter.jit import core as jit_core_module

    with _JIT_BUILD_LOCK:
        original_flags = cpp_extension_module._get_pybind11_abi_build_flags
        original_rebuild = jit_core_module.AITER_REBUILD
        cpp_extension_module._get_pybind11_abi_build_flags = lambda: []
        jit_core_module.AITER_REBUILD = True
        try:
            return operation(*args, **kwargs)
        finally:
            jit_core_module.AITER_REBUILD = original_rebuild
            cpp_extension_module._get_pybind11_abi_build_flags = original_flags
