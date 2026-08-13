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


def test_decode_fused_rope_call_matches_rtp_kernel_signature():
    tree = ast.parse((RTP_LLM_DIR / "ops/fused_rope_kvcache_op.py").read_text())
    forward = _find_method(_find_class(tree, "FusedRopeKVCacheDecodeOp"), "forward")
    call = next(
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_fused_rope_kvcache"
    )

    assert [ast.unparse(arg) for arg in call.args] == [
        "qkv",
        "params.position_ids",
        "params.sequence_lengths",
        "params.sequence_lengths.size(0)",
        "self.attn_configs.head_num",
        "self.attn_configs.kv_head_num",
        "self.attn_configs.size_per_head",
        "kv_cache.kv_cache_base",
        "params.kv_cache_offset",
    ]
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


def test_native_prefill_keeps_rope_without_writing_dummy_cache():
    tree = ast.parse((ATTENTION_DIR / "cuda_impl/py_flashinfer_mha.py").read_text())
    base_impl = _find_class(tree, "PyFlashinferPrefillImplBase")
    forward_source = ast.unparse(_find_method(base_impl, "forward"))

    assert "if self.need_rope_kv_cache" in forward_source
    assert "if kv_cache is not None" in forward_source
    assert "self.kv_cache_write_op.forward(key, value, kv_cache)" in forward_source


def test_embedding_cuda_graph_does_not_fabricate_kv_cache():
    runner_source = (RTP_LLM_DIR / "cpp/cuda_graph/cuda_graph_runner.cc").read_text()
    compact_source = "".join(runner_source.split())

    assert "if (!kv_cache_group_tags_.empty())" in runner_source
    assert "kv_cache_kernel_block_id_device=torch::empty({0}" in compact_source
    assert "kv_cache_kernel_block_id=torch::empty({0}" in compact_source


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
        "PyFlashinferPrefillImpl",
        "PyFlashinferDecodeImpl",
    ]
