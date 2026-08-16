import ast
from pathlib import Path


RTP_LLM_DIR = Path(__file__).resolve().parents[1]
ATTENTION_DIR = RTP_LLM_DIR / "models_py/modules/factory/attention"


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_fused_rope_calls_support_old_arm_and_new_x86_rtp_kernel_signatures():
    tree = ast.parse((RTP_LLM_DIR / "ops/fused_rope_kvcache_op.py").read_text())
    source = ast.unparse(tree)
    assert "'position_ids' in inspect.signature(prefill_fused_rope_kvcache).parameters" in source
    assert "else 'cp_position_ids'" in source
    assert "'cu_seqlens' in inspect.signature(decode_fused_rope_kvcache).parameters" in source

    prefill_prepare = _find_method(
        _find_class(tree, "FusedRopeKVCachePrefillOpBase"), "prepare"
    )
    prepare_source = ast.unparse(prefill_prepare)
    assert "_PREFILL_POSITION_IDS_ARG == 'position_ids'" in prepare_source
    assert "attn_inputs.context_parallel_info.prefill_shuffle_indices" in prepare_source

    prefill_forward = _find_method(
        _find_class(tree, "FusedRopeKVCachePrefillOpBase"), "_forward"
    )
    prefill_call = next(
        node
        for node in ast.walk(prefill_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prefill_fused_rope_kvcache"
    )
    assert any(keyword.arg is None for keyword in prefill_call.keywords)

    decode_forward = _find_method(
        _find_class(tree, "FusedRopeKVCacheDecodeOp"), "forward"
    )
    decode_source = ast.unparse(decode_forward)
    assert "if _DECODE_HAS_CU_SEQLENS" in decode_source
    assert "decode_args.extend([params.position_ids, params.sequence_lengths])" in decode_source
    assert "decode_args.append(params.sequence_lengths)" in decode_source
    call = next(
        node
        for node in ast.walk(decode_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_fused_rope_kvcache"
    )
    assert [ast.unparse(arg) for arg in call.args] == ["*decode_args"]
    assert any(keyword.arg == "tokens_per_block" for keyword in call.keywords)


def test_paged_prefill_backends_require_kv_cache():
    native_tree = ast.parse(
        (ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text()
    )
    native_support = _find_method(
        _find_class(native_tree, "PyFlashinferPagedPrefillImpl"), "support"
    )
    native_source = ast.unparse(native_support)
    assert "attn_inputs.kv_cache_kernel_block_id is not None" in native_source
    assert "attn_inputs.kv_cache_kernel_block_id.numel() > 0" in native_source

    trt_tree = ast.parse((ATTENTION_DIR / "cuda_impl/trt.py").read_text())
    trt_support = _find_method(
        _find_class(trt_tree, "TRTLLMFMHAv2PagedPrefillOp"), "support"
    )
    trt_source = ast.unparse(trt_support)
    assert "attn_inputs.kv_cache_kernel_block_id is not None" in trt_source
    assert "attn_inputs.kv_cache_kernel_block_id.numel() > 0" in trt_source


def test_ragged_prefill_uses_flashinfer_cuda_graph_buffers():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    ragged_op = _find_class(tree, "PyFlashinferPrefillAttnOp")
    init_source = ast.unparse(_find_method(ragged_op, "__init__"))
    prepare_source = ast.unparse(_find_method(ragged_op, "prepare"))

    assert "use_cuda_graph=self.enable_cuda_graph" in init_source
    assert "qo_indptr_buf=qo_indptr" in init_source
    assert "kv_indptr_buf=kv_indptr" in init_source
    assert "causal=self.is_causal" in prepare_source
    assert "forbid_realloc" in prepare_source

    ragged_impl = _find_class(tree, "PyFlashinferPrefillImpl")
    support_cuda_graph = _find_method(ragged_impl, "support_cuda_graph")
    assert isinstance(support_cuda_graph.body[0], ast.Return)
    assert support_cuda_graph.body[0].value.value is True


def test_paged_cuda_graph_plan_matches_padded_query_buffer():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    paged_op = _find_class(tree, "PyFlashinferPrefillPagedAttnOp")
    init_source = ast.unparse(_find_method(paged_op, "__init__"))
    prepare_source = ast.unparse(
        _find_method(paged_op, "prepare")
    )

    assert "self.is_causal = attn_configs.is_causal" in init_source
    assert "causal=self.is_causal" in prepare_source
    assert "qo_indptr = self.qo_indptr" in prepare_source
    assert "offsets + attn_inputs.input_lengths" not in prepare_source


def test_native_prefill_keeps_rope_without_writing_dummy_cache():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    base_impl = _find_class(tree, "PyFlashinferPrefillImplBase")
    forward_source = ast.unparse(_find_method(base_impl, "forward"))

    assert "if self.need_rope_kv_cache" in forward_source
    assert "if self.need_rope_kv_cache and kv_cache is not None" in forward_source
    assert "self.kv_cache_write_op.forward(key, value, kv_cache)" in forward_source


def test_embedding_cuda_graph_does_not_fabricate_kv_cache():
    runner_source = (RTP_LLM_DIR / "cpp/cuda_graph/cuda_graph_runner.cc").read_text()
    compact_source = "".join(runner_source.split())

    assert "if (!kv_cache_group_tags_.empty())" in runner_source
    assert "kv_cache_kernel_block_id_device=torch::empty({0}" in compact_source
    assert "kv_cache_kernel_block_id=torch::empty({0}" in compact_source


def test_decode_cuda_graph_passes_cache_group_tags_to_capture():
    source = (
        RTP_LLM_DIR / "cpp/cuda_graph/tests/cuda_graph_decode_padding.py"
    ).read_text()

    assert "self.kv_cache.group_tags" in source


def test_qwen35_sm100_cuda_graph_cases_use_scratch_loader():
    tree = ast.parse(
        (RTP_LLM_DIR / "test/smoke/suites/test_smoke_sm100_moe.py").read_text()
    )
    cases = ast.literal_eval(
        next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SMOKE_CASES"
                for target in node.targets
            )
        )
    )

    for name in (
        "next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
        "next_moe_nvfp4_cudagraph_tp2_sm100",
    ):
        assert "--load_method scratch" in cases[name]["smoke_args"]
        assert "envs" not in cases[name]


def test_sm100_head_dim_256_cuda_graph_skips_trtllm_gen_decode():
    tree = ast.parse(
        (ATTENTION_DIR / "cuda_impl/trtllm_gen.py").read_text()
    )
    support_source = ast.unparse(
        _find_method(_find_class(tree, "FlashInferTRTLLMDecodeOp"), "support")
    )

    assert "not attention_inputs.is_prefill" in support_source
    assert "attention_inputs.is_cuda_graph" in support_source
    assert "self.head_dim == 256" in support_source


def test_sm100_head_dim_256_cuda_graph_skips_xqa_decode_backends():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/xqa.py").read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_reject_sm100_head_dim_256_cuda_graph"
    )
    helper_source = ast.unparse(helper)

    assert "torch.cuda.get_device_capability()[0] == 10" in helper_source
    assert "attn_inputs.is_cuda_graph" in helper_source
    assert "attn_configs.size_per_head == 256" in helper_source

    for class_name in ("XQAImpl", "XQADecodeImpl"):
        support_source = ast.unparse(
            _find_method(_find_class(tree, class_name), "support")
        )
        assert "_reject_sm100_head_dim_256_cuda_graph" in support_source

    registry_source = (ATTENTION_DIR / "__init__.py").read_text()
    assert registry_source.index("DECODE_MHA_IMPS.append(XQAImpl)") < registry_source.index(
        "DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)"
    )


def test_qwen3_standalone_goldens_match_python_native_h20_outputs():
    source = (
        RTP_LLM_DIR / "models_py/standalone/test/qwen3_test.py"
    ).read_text()

    assert (
        "\u4f60\u597d\uff01\u6211\u662f\u4f60\u7684\u865a\u62df\u52a9\u624b\uff0c\u4e00\u4e2a\u4e13\u6ce8\u4e8e\u5e2e\u52a9\u4f60\u89e3\u51b3\u95ee\u9898\u548c\u63d0\u4f9b\u652f\u6301\u7684AI\u52a9\u624b"
        in source
    )
    assert (
        "\u4f60\u597d\uff01\u6211\u662f\u5c0f\u660e\u54e5\uff0c\u5f88\u9ad8\u5174\u89c1\u5230\u4f60\uff01"
        in source
    )
    assert (
        "3.9 \u548c 3.11 \u4e2d\uff0c**3.9 \u5927\u4e8e 3.11**\u3002"
        in source
    )
    assert "3.9\u6bd43.11111111\u5927" in source


def test_qwen_loader_detects_qwen3_vl_text_tower_prefix():
    tree = ast.parse((RTP_LLM_DIR / "models/qwen_v2.py").read_text())
    process_meta = _find_method(_find_class(tree, "QWenV2Weight"), "_process_meta")
    source = ast.unparse(process_meta)

    assert "self._contains(weight_keys, 'model.language_model.')" in source
    assert "self.prefix = 'model.language_model.'" in source
    assert "self.model_prefix = ''" in source


def test_xqa_backends_reject_sm120():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/xqa.py").read_text())

    for class_name in ("XQAImpl", "XQADecodeImpl"):
        support_source = ast.unparse(
            _find_method(_find_class(tree, class_name), "support")
        )
        assert "torch.cuda.get_device_capability()[0] == 12" in support_source
        assert "return False" in support_source


def test_missing_py_flashinfer_decode_does_not_break_attention_registration():
    tree = ast.parse((ATTENTION_DIR / "__init__.py").read_text())
    source = ast.unparse(tree)

    assert "except ImportError as e:" in source
    assert "PyFlashinferDecodeImpl = None" in source
    assert "if PyFlashinferDecodeImpl is not None:" in source
    assert "DECODE_MHA_IMPS.append(PyFlashinferDecodeImpl)" in source


def test_py_flashinfer_module_exports_all_registered_implementations():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    export = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )

    assert ast.literal_eval(export.value) == [
        "PyFlashinferPagedPrefillImpl",
        "PyFlashinferHybridPrefillImpl",
        "PyFlashinferPrefillImpl",
        "PyFlashinferDecodeImpl",
    ]


def test_attention_compatibility_names_survive_remote_source_cache_rollout():
    trt_tree = ast.parse((ATTENTION_DIR / "cuda_impl/trt.py").read_text())
    assignments = {
        target.id: ast.unparse(node.value)
        for node in trt_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["TRTPagedMHAImpl"] == "FlashInferTRTLLMFMHAv2PagedPrefillImpl"
    assert assignments["TRTMHAImpl"] == "FlashInferTRTLLMFMHAv2PrefillImpl"

    xqa_tree = ast.parse((ATTENTION_DIR / "cuda_impl/xqa.py").read_text())
    prepare_source = ast.unparse(
        _find_method(_find_class(xqa_tree, "XQAImpl"), "prepare_cuda_graph")
    )
    assert "update_attention_params" in prepare_source
    assert "getattr(self.fmha_impl, 'update', None)" in prepare_source
    assert "update_kv_cache_offset" in prepare_source
