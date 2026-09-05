"""Compatibility setup for ROCm AITer kernels."""

import mmap
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from typing import Any, Optional


_JIT_PYBIND_ABI_BROKEN_VERSION_PREFIX = "0.1.21.dev80+g987203ba5"
_JIT_BUILD_LOCK = RLock()
_BUNDLED_CORE_PYBIND_ABI_FLAGS = (
    '-DPYBIND11_COMPILER_TYPE=\\"_system\\"',
    '-DPYBIND11_STDLIB=\\"\\"',
    '-DPYBIND11_BUILD_ABI=\\"_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1\\"',
    # AITer's in-memory versioner does not hash the ABI flags returned by its
    # builder helper. Keep a marker in the Ninja command so a worker reusing a
    # JIT directory cannot retain an object from the old PyTorch-ABI build.
    "-DRTP_AITER_BUNDLED_CORE_ABI=1",
)
_PYBIND_INTERNALS_ID_RE = re.compile(rb"__pybind11_internals_v[^\x00]+__")


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


def _report_pybind_internals_ids() -> None:
    """Print bounded ABI diagnostics only when an affected JIT call fails."""
    try:
        from aiter.jit import core as aiter_core

        core_path = Path(aiter_core.get_module("module_aiter_core").__file__)
        candidates = [core_path]
        candidates.extend(
            sorted(
                core_path.parent.glob("module_*.so"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:8]
        )
        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            with path.open("rb") as binary:
                with mmap.mmap(binary.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    ids = sorted(
                        match.decode("ascii", errors="replace")
                        for match in set(_PYBIND_INTERNALS_ID_RE.findall(mapped))
                    )
            if ids:
                print(
                    f"[rtp-aiter-compat] {path.name}: {', '.join(ids)}",
                    file=sys.stderr,
                )
    except Exception as error:  # noqa: BLE001
        print(
            f"[rtp-aiter-compat] unable to inspect pybind ABI ids: {error}",
            file=sys.stderr,
        )


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
    ``aiter_tensor_t`` instances created by the bundled core module. Spell out
    the bundled namespace instead of relying on compiler defaults, and include
    a command marker so reused Ninja directories rebuild their objects.
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
        cpp_extension_module._get_pybind11_abi_build_flags = lambda: list(
            _BUNDLED_CORE_PYBIND_ABI_FLAGS
        )
        try:
            return operation(*args, **kwargs)
        except TypeError:
            _report_pybind_internals_ids()
            raise
        finally:
            cpp_extension_module._get_pybind11_abi_build_flags = original_flags
