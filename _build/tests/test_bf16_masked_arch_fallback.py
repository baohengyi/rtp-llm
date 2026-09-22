"""Exercise the real wrapper with CPU tensors; GPU executor tests stay intact."""

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "rtp_llm/models_py/kernels/cuda/deepgemm_wrapper.py"


def wrapper(implementation):
    source = ast.parse(WRAPPER.read_text())
    function = next(
        n for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "m_grouped_bf16_gemm_nt_masked"
    )
    module = ast.Module(body=[function], type_ignores=[])
    env = {"torch": torch, "_m_grouped_bf16_gemm_nt_masked_impl": implementation}
    exec(compile(module, str(WRAPPER), "exec"), env)
    return env[function.name]


def unsupported(*_):
    raise RuntimeError("Assertion error (csrc/apis/gemm.hpp:462): Unsupported architecture")


def test_unsupported_architecture_preserves_valid_products_and_padding():
    a = torch.arange(4 * 4 * 3, dtype=torch.bfloat16).reshape(4, 4, 3)
    b = torch.arange(4 * 2 * 3, dtype=torch.bfloat16).reshape(4, 2, 3)
    mask = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
    out = torch.full((4, 4, 2), -17, dtype=torch.bfloat16)
    expected = out.clone()
    for group in range(4):
        for row in range(int(mask[group])):
            # Independent scalar dot products, exact after BF16 rounding.
            for col in range(2):
                expected[group, row, col] = sum(
                    float(a[group, row, k]) * float(b[group, col, k]) for k in range(3)
                )
    assert wrapper(unsupported)(a, b, out, mask, 2) is None
    assert torch.equal(out, expected)


def test_supported_implementation_receives_original_arguments():
    seen = []
    args = (object(), object(), object(), object(), 19, "mn")
    wrapper(lambda *a: seen.append(a))(*args)
    assert seen == [args]


def test_unrelated_runtime_error_is_not_replaced_by_a_fallback():
    error = RuntimeError("CUDA illegal memory access")
    def fail(*_):
        raise error
    with pytest.raises(RuntimeError) as caught:
        wrapper(fail)(object(), object(), object(), object(), 3)
    assert caught.value is error
