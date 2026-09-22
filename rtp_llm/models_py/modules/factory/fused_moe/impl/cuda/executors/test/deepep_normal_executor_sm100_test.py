import platform
import unittest

import pytest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepep_normal_executor_test import (
    DeepGemmHybridExecutorTestBase as _DeepGemmHybridExecutorTestBase,
)

pytestmark = [pytest.mark.gpu(type="SM100_ARM")]


class DeepGemmHybridExecutorTestBase(
    _DeepGemmHybridExecutorTestBase, unittest.TestCase
):
    # Preserve the historical GB200 collection identity after the shared base
    # became a mixin. Its inherited assertions also cover the added edge cases.
    pass


class DeepGemmHybridExecutorSM100Test(
    DeepGemmHybridExecutorTestBase, unittest.TestCase
):
    def test_sm100_arm(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())
        self.assertTrue("aarch64" in platform.machine())


if __name__ == "__main__":
    unittest.main()
