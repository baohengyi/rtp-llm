"""Pytest entry for smoke suite ``smoke_cuda_remote_cache``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {'remote_cache_basic': {'sleep_time_qr': 10,
                        'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 '
                                      '--seq_size_per_block 8 --enable_remote_cache '
                                      'true --enable_device_cache 1 '
                                      '--enable_memory_cache 0 --enable_disk_cache 0 '
                                      '--test_block_num 500 '
                                      '--block_tree_device_evict_low_watermark_ratio '
                                      '0.001 '
                                      '--block_tree_device_evict_high_watermark_ratio '
                                      '0.002',
                        'task_info': 'data/model/qwen25/q_r_l20_remote_cache.json',
                        'gpu_type': 'L20',
                        'platform': 'cuda',
                        'markers': ['smoke', 'cuda', 'L20'],
                        'timeout': 600,
                        'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG']},
 'remote_cache_basic_async': {'sleep_time_qr': 20,
                              'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type '
                                            'FP16 --seq_size_per_block 8 '
                                            '--enable_remote_cache true '
                                            '--enable_device_cache 1 '
                                            '--enable_memory_cache 0 '
                                            '--enable_disk_cache 0 --test_block_num '
                                            '500 '
                                            '--block_tree_device_evict_low_watermark_ratio '
                                            '0.001 '
                                            '--block_tree_device_evict_high_watermark_ratio '
                                            '0.002',
                              'task_info': 'data/model/qwen25/q_r_l20_remote_cache.json',
                              'gpu_type': 'L20',
                              'platform': 'cuda',
                              'markers': ['smoke', 'cuda', 'L20'],
                              'timeout': 600,
                              'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG']},
 'remote_cache_kill': {'kill_remote': True,
                       'sleep_time_qr': 10,
                       'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 '
                                     '--seq_size_per_block 8 --enable_remote_cache '
                                     'true --enable_device_cache 1 '
                                     '--enable_memory_cache 0 --enable_disk_cache 0 '
                                     '--test_block_num 500 '
                                     '--block_tree_device_evict_low_watermark_ratio '
                                     '0.001 '
                                     '--block_tree_device_evict_high_watermark_ratio '
                                     '0.002',
                       'task_info': 'data/model/qwen25/q_r_l20_remote_cache_kill_remote.json',
                       'gpu_type': 'L20',
                       'platform': 'cuda',
                       'markers': ['smoke', 'cuda', 'L20'],
                       'timeout': 600,
                       'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG']},
 'remote_cache_tp2': {'sleep_time_qr': 20,
                      'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type FP16 '
                                    '--seq_size_per_block 8 --tp_size 2 '
                                    '--enable_remote_cache true --kvcm_put_timeout_ms '
                                    '12000 --kvcm_get_timeout_ms 12000 '
                                    '--kvcm_get_broadcast_timeout 15000 '
                                    '--kvcm_put_broadcast_timeout 15000 '
                                    '--enable_device_cache 1 --enable_memory_cache 0 '
                                    '--enable_disk_cache 0 --test_block_num 500 '
                                    '--block_tree_device_evict_low_watermark_ratio '
                                    '0.001 '
                                    '--block_tree_device_evict_high_watermark_ratio '
                                    '0.002',
                      'task_info': 'data/model/qwen25/q_r_l20_remote_cache_tpsize_2.json',
                      'gpu_type': 'L20',
                      'platform': 'cuda',
                      'markers': ['smoke', 'cuda', 'L20'],
                      'timeout': 600,
                      'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG']},
 'remote_cache_pd': {'sleep_time_qr': 20,
                     'smoke_args': {'prefill': '--warm_up 0  --reuse_cache 1 '
                                               '--role_type PREFILL --act_type FP16 '
                                               '--seq_size_per_block 8 '
                                               '--enable_remote_cache true '
                                               '--kvcm_put_timeout_ms 12000 '
                                               '--kvcm_get_timeout_ms 12000 '
                                               '--kvcm_get_broadcast_timeout 15000 '
                                               '--kvcm_put_broadcast_timeout 15000 '
                                               '--enable_device_cache 1 '
                                               '--enable_memory_cache 0 '
                                               '--enable_disk_cache 0 --test_block_num '
                                               '500 '
                                               '--block_tree_device_evict_low_watermark_ratio '
                                               '0.001 '
                                               '--block_tree_device_evict_high_watermark_ratio '
                                               '0.002',
                                    'decode': '--warm_up 0  --reuse_cache 1 '
                                              '--role_type DECODE --act_type FP16 '
                                              '--seq_size_per_block 8 '
                                              '--enable_remote_cache true '
                                              '--kvcm_put_timeout_ms 12000 '
                                              '--kvcm_get_timeout_ms 12000 '
                                              '--kvcm_get_broadcast_timeout 15000 '
                                              '--kvcm_put_broadcast_timeout 15000 '
                                              '--enable_device_cache 1 '
                                              '--enable_memory_cache 0 '
                                              '--enable_disk_cache 0 --test_block_num '
                                              '500 '
                                              '--block_tree_device_evict_low_watermark_ratio '
                                              '0.001 '
                                              '--block_tree_device_evict_high_watermark_ratio '
                                              '0.002'},
                     'task_info': 'data/model/qwen25/q_r_l20_remote_cache_pd_sep.json',
                     'gpu_type': 'L20',
                     'platform': 'cuda',
                     'markers': ['smoke', 'cuda', 'L20'],
                     'timeout': 600,
                     'envs': ['SEQ_SIZE_PER_BLOCK=8', 'KVCM_LOG_LEVEL=DEBUG']},
 'remote_cache_match_fail': {'sleep_time_qr': 10,
                             'smoke_args': '--warm_up 0 --reuse_cache 1 --act_type '
                                           'FP16 --seq_size_per_block 8 '
                                           '--enable_remote_cache true '
                                           '--enable_device_cache 1 '
                                           '--enable_memory_cache 0 '
                                           '--enable_disk_cache 0 --test_block_num 500 '
                                           '--block_tree_device_evict_low_watermark_ratio '
                                           '0.001 '
                                           '--block_tree_device_evict_high_watermark_ratio '
                                           '0.002',
                             'task_info': 'data/model/qwen25/q_r_l20_remote_cache_match_failure.json',
                             'gpu_type': 'L20',
                             'platform': 'cuda',
                             'markers': ['smoke', 'cuda', 'L20'],
                             'timeout': 600,
                             'envs': ['SEQ_SIZE_PER_BLOCK=8',
                                      'KVCM_LOG_LEVEL=DEBUG',
                                      'ENABLE_DEBUG_SERVICE=TRUE',
                                      'TEST_MATCH_FAILURE=1']},
 'remote_cache_write_start_fail': {'sleep_time_qr': 10,
                                   'smoke_args': '--warm_up 0 --reuse_cache 1 '
                                                 '--act_type FP16 --seq_size_per_block '
                                                 '8 --enable_remote_cache true '
                                                 '--enable_device_cache 1 '
                                                 '--enable_memory_cache 0 '
                                                 '--enable_disk_cache 0 '
                                                 '--test_block_num 500 '
                                                 '--block_tree_device_evict_low_watermark_ratio '
                                                 '0.001 '
                                                 '--block_tree_device_evict_high_watermark_ratio '
                                                 '0.002',
                                   'task_info': 'data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json',
                                   'gpu_type': 'L20',
                                   'platform': 'cuda',
                                   'markers': ['smoke', 'cuda', 'L20'],
                                   'timeout': 600,
                                   'envs': ['SEQ_SIZE_PER_BLOCK=8',
                                            'KVCM_LOG_LEVEL=DEBUG',
                                            'ENABLE_DEBUG_SERVICE=TRUE',
                                            'TEST_START_WRITE_FAILURE=1']},
 'remote_cache_write_finish_fail': {'sleep_time_qr': 10,
                                    'smoke_args': '--warm_up 0 --reuse_cache 1 '
                                                  '--act_type FP16 '
                                                  '--seq_size_per_block 8 '
                                                  '--enable_remote_cache true '
                                                  '--enable_device_cache 1 '
                                                  '--enable_memory_cache 0 '
                                                  '--enable_disk_cache 0 '
                                                  '--test_block_num 500 '
                                                  '--block_tree_device_evict_low_watermark_ratio '
                                                  '0.001 '
                                                  '--block_tree_device_evict_high_watermark_ratio '
                                                  '0.002',
                                    'task_info': 'data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json',
                                    'gpu_type': 'L20',
                                    'platform': 'cuda',
                                    'markers': ['smoke', 'cuda', 'L20'],
                                    'timeout': 600,
                                    'envs': ['SEQ_SIZE_PER_BLOCK=8',
                                             'KVCM_LOG_LEVEL=DEBUG',
                                             'ENABLE_DEBUG_SERVICE=TRUE',
                                             'TEST_FINISH_WRITE_FAILURE=1']},
 'remote_cache_edge': {'sleep_time_qr': 20,
                       'smoke_args': '--warm_up 0  --reuse_cache 1 --act_type FP16 '
                                     '--seq_size_per_block 4 --enable_remote_cache '
                                     'true --enable_device_cache 1 '
                                     '--enable_memory_cache 0 --enable_disk_cache 0 '
                                     '--test_block_num 500 '
                                     '--block_tree_device_evict_low_watermark_ratio '
                                     '0.001 '
                                     '--block_tree_device_evict_high_watermark_ratio '
                                     '0.002',
                       'task_info': 'data/model/qwen25/q_r_l20_cache_edge_case_1_remote_cache.json',
                       'gpu_type': 'L20',
                       'platform': 'cuda',
                       'markers': ['smoke', 'cuda', 'L20'],
                       'timeout': 600,
                       'envs': ['SEQ_SIZE_PER_BLOCK=4', 'KVCM_LOG_LEVEL=DEBUG']}}

SUITE_NAME = "smoke_cuda_remote_cache"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_cuda_remote_cache(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
