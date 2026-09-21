"""Mainline CUDA 13 SM120 coverage, executed by the Native pytest runner."""

import importlib
import os
import unittest

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {'qwen3_bf16_sm120_cuda13': {'smoke_args': '--role_type PDFUSION --act_type BF16 --warm_up 0 '
                                    '--tp_size 1 --world_size 1 '
                                    '--frontend_server_count 1 --max_seq_len 8192 '
                                    '--seq_size_per_block 64 --concurrency_limit 4 '
                                    '--enable_cuda_graph 1 --decode_capture_config '
                                    "'1,2' --reuse_cache 1 --enable_device_cache 1 "
                                    '--enable_memory_cache 0 --enable_remote_cache 0',
                      'gpu_type': 'RTX_5000_PRO_CU13',
                      'platform': 'cuda',
                      'markers': ['smoke', 'cuda', 'RTX_5000_PRO_CU13'],
                      'timeout': 600,
                      'custom_module': 'sm120_qwen3_test'},
 'generation_prefill_cuda_graph_sm120_cuda13': {'task_info': 'data/model/qwen25/q_r_generation_prefill_cuda_graph_sm120.json',
                                         'smoke_args': '--act_type BF16 --warm_up 1 '
                                                       '--seq_size_per_block 64 '
                                                       '--concurrency_limit 5 '
                                                       '--max_context_batch_size 5 '
                                                       '--enable_cuda_graph 1 '
                                                       "--decode_capture_config '1' "
                                                       '--generation_prefill_cuda_graph_max_requests '
                                                       '5 '
                                                       '--generation_prefill_capture_config '
                                                       "'64,256'",
                                         'gpu_type': 'RTX_5000_PRO_CU13',
                                         'platform': 'cuda',
                                         'markers': ['smoke',
                                                     'cuda',
                                                     'RTX_5000_PRO_CU13'],
                                         'timeout': 600},
 'embedding_bert_sm120_cuda13': {'task_info': 'data/model/bert/q_r.json',
                          'smoke_args': '--seq_size_per_block 16 --act_type FP16',
                          'gpu_type': 'RTX_5000_PRO_CU13',
                          'platform': 'cuda',
                          'markers': ['smoke', 'cuda', 'RTX_5000_PRO_CU13'],
                          'timeout': 600},
 'dense_fp8pb_dynamic_sm120_cuda13': {'task_info': 'data/model/qwen3/q_r_1_7b_fp8pb_tp2_sm120.json',
                               'envs': ['LOAD_PYTHON_MODEL=1'],
                               'smoke_args': '--quantization FP8_PER_BLOCK --act_type '
                                             'BF16 --warm_up 0 --tp_size 2 '
                                             '--world_size 2 --seq_size_per_block 2048 '
                                             '--max_seq_len 16384 '
                                             '--reserver_runtime_mem_mb 16005 '
                                             '--concurrency_limit 4 '
                                             '--enable_cuda_graph 1 '
                                             "--decode_capture_config '1,2' "
                                             '--reuse_cache 0',
                               'gpu_type': 'RTX_5000_PRO_CU13',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'RTX_5000_PRO_CU13'],
                               'timeout': 600},
 'dense_fp8pb_reuse_cache_tp2_sm120_cuda13': {'smoke_args': '--quantization FP8_PER_BLOCK '
                                                     '--act_type BF16 --warm_up 0 '
                                                     '--tp_size 2 --world_size 2 '
                                                     '--seq_size_per_block 2048 '
                                                     '--max_seq_len 16384 '
                                                     '--reserver_runtime_mem_mb 16005 '
                                                     '--concurrency_limit 4 '
                                                     '--enable_cuda_graph 1 '
                                                     "--decode_capture_config '1,2' "
                                                     '--reuse_cache 1 '
                                                     '--enable_device_cache 1 '
                                                     '--enable_memory_cache 0 '
                                                     '--enable_remote_cache 0',
                                       'gpu_type': 'RTX_5000_PRO_CU13',
                                       'platform': 'cuda',
                                       'markers': ['smoke',
                                                   'cuda',
                                                   'RTX_5000_PRO_CU13'],
                                       'timeout': 600,
                                       'custom_module': 'sm120_reuse_cache_test'},
 'moe_fp8pb_tp2_sm120_cuda13': {'smoke_args': '--moe_strategy auto --quantization '
                                       'FP8_PER_BLOCK --warm_up 0 --act_type BF16 '
                                       '--tp_size 2 --world_size 2 '
                                       '--reserver_runtime_mem_mb 16005 '
                                       '--seq_size_per_block 2048 --concurrency_limit '
                                       '64 --enable_cuda_graph 1 '
                                       "--decode_capture_config '1,2'",
                         'gpu_type': 'RTX_5000_PRO_CU13',
                         'platform': 'cuda',
                         'markers': ['smoke', 'cuda', 'RTX_5000_PRO_CU13'],
                         'timeout': 600,
                         'custom_module': 'sm120_moe_test'}}

_test_params = build_smoke_params(
    pytest, {"smoke_sm120_cuda13": SMOKE_CASES}, composite_suites={}
)


@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_sm120_cuda13(test_name, test_config, monkeypatch, tmp_path):
    custom_module = test_config.get("custom_module")
    if custom_module is None:
        run_smoke_test(test_name, test_config)
        return
    monkeypatch.setenv("SMOKE_ARGS", test_config["smoke_args"])
    monkeypatch.setenv(
        "TEST_UNDECLARED_OUTPUTS_DIR",
        os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", str(tmp_path)),
    )
    module = importlib.import_module(f"rtp_llm.test.smoke.{custom_module}")
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    assert suite.countTestCases() == 1, f"Unexpected custom smoke inventory: {custom_module}"
    result = unittest.TestResult()
    suite.run(result)
    assert result.testsRun == 1, f"Custom smoke did not execute: {custom_module}"
    assert not result.skipped, result.skipped
    assert not result.errors, result.errors
    assert not result.failures, result.failures
