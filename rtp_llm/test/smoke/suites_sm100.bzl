load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")


def sm100_suites():
    native.test_suite(
        name = "smoke_sm100_dense",
        tests = [
            smoke_test(
                name = "dense_tp1_sm100",
                task_info = "data/model/qwen3/q_r_l20a_fp4_tp1_py.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 1 --world_size 1 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "dense_tp2_sm100",
                task_info = "data/model/qwen3/q_r_l20a_fp4_tp2_py.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "fp8_attention_sm100",
                task_info = "data/model/qwen3/q_r_block_fp8_sm100_arm.json",
                smoke_args = "--act_type BF16 --seq_size_per_block 64 --fp8_kv_cache 1 --reserver_runtime_mem_mb 178125 --warm_up 0",
                gpu_type = ["SM100_ARM"],
            ),
        ],
    )

    native.test_suite(
        name = "smoke_sm100_moe",
        tests = [
            smoke_test(
                name = "moe_deepep_normal_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_py_tp2_sm100.json",
                smoke_args = "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
                smoke_args = "--decode_capture_config '1,2' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "moe_nvfp4_deepep_ll_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_ll_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "moe_nvfp4_deepep_normal_dp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_normal_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "moe_nvfp4_deepep_normal_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_normal_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "moe_nvfp4_online_deepep_ll_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_nvfp4_online_py_tp2_ll_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1 --quantization MODELOPT_FP4",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "moe_nvfp4_online_deepep_normal_dp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_nvfp4_online_py_dp2_normal_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0 --quantization MODELOPT_FP4",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
                task_info = "data/model/qwen35/q_r_35b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
                smoke_args = "--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 22000 --seq_size_per_block 2048 --concurrency_limit 64 --kernel_seq_size_per_block 64 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type = ["SM100_ARM"],
            ),
            smoke_test(
                name = "next_moe_nvfp4_cudagraph_tp2_sm100",
                task_info = "data/model/qwen35/q_r_35b_nvfp4_py_tp2_cg_sm100_arm.json",
                smoke_args = "--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --seq_size_per_block 2048 --concurrency_limit 64 --kernel_seq_size_per_block 64",
                gpu_type = ["SM100_ARM"],
            ),
        ],
    )

    native.test_suite(
        name = "smoke_sm100_eval",
        tests = [
            smoke_test(
                name = "qwen3_moe_tau2_bench_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_tau2_bench_tp2_sm100.json",
                smoke_args = "--warm_up 0 --act_type BF16 --max_seq_len 16384 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
                gpu_type = ["SM100_ARM"],
                data = [
                    "data/model/qwen3_moe/passing_tasks.json",
                    "data/model/qwen3_moe/run_tau2_bench.py",
                ],
            ),
        ],
    )
