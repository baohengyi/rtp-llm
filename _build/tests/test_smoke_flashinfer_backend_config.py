"""Exercise smoke CLI bindings and backend exclusions without loading CUDA."""

import argparse
import ast
import logging
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    "dense_fp8_prequant_flashinfer_no_cudagraph",
    "dense_fp8_prequant_flashinfer_cudagraph",
)


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)


def smoke_cases():
    tree = ast.parse((ROOT / "rtp_llm/test/smoke/suites/test_smoke_h20_dense.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SMOKE_CASES" for t in n.targets))


def parse_smoke_args(text):
    ns = dict(argparse=argparse, logging=logging, os=os, sys=sys, warnings=warnings)
    definitions("rtp_llm/server/server_args/server_args.py",
                {"ConfigBinding", "EnvArgumentGroup", "EnvArgumentParser"}, ns)
    definitions("rtp_llm/server/server_args/util.py", {"str2bool"}, ns)
    definitions("rtp_llm/server/server_args/fmha_group_args.py",
                {"_DeprecatedFmhaFlag", "init_fmha_group_args"}, ns)
    parser = ns["EnvArgumentParser"]()
    config = SimpleNamespace()
    parser.set_root_config(config)
    ns["init_fmha_group_args"](parser, config)
    for name in ("act_type", "reserver_runtime_mem_mb", "tp_size", "warm_up",
                 "seq_size_per_block", "enable_cuda_graph", "decode_capture_config"):
        parser.add_argument("--" + name)
    parser.parse_args(shlex.split(text))
    return config


def backend_registry():
    result = {}
    for name in ("trt", "trtllm_gen", "xqa", "py_flashinfer_mha"):
        tree = ast.parse((ROOT / f"rtp_llm/models_py/modules/factory/attention/cuda_impl/{name}.py").read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            for node in cls.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NAME" for t in node.targets):
                    result[cls.name] = type(cls.name, (), {"NAME": ast.literal_eval(node.value)})
    return result


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prefill", [True, False])
def test_smoke_keeps_flashinfer_and_excludes_the_old_disabled_backends(case, prefill, monkeypatch):
    # The real argument parser must reject stale/unknown options. Its bindings
    # then drive the real resolver; class metadata comes from production files.
    monkeypatch.setattr(os, "environ", {})
    config = parse_smoke_args(smoke_cases()[case]["smoke_args"])
    registry = backend_registry()
    excluded = {"FlashInferTRTLLMPrefillImpl", "FlashInferTRTLLMSpecDecodeImpl",
                "FlashInferTRTLLMDecodeImpl", "FlashInferTRTLLMFMHAv2PrefillImpl",
                "FlashInferTRTLLMFMHAv2PagedPrefillImpl", "XQAImpl", "XQADecodeImpl"}
    allowed = ["PyFlashinferPagedPrefillImpl", "PyFlashinferPrefillImpl"] if prefill else ["PyFlashinferDecodeImpl"]
    impls = [registry[n] for n in sorted(excluded) + allowed]
    ns = {"PREFILL_MHA_IMPS": impls, "DECODE_MHA_IMPS": impls,
          "PREFILL_MLA_IMPS": [], "DECODE_MLA_IMPS": []}
    definitions("rtp_llm/models_py/modules/factory/attention/attn_factory.py",
                {"_get_effective_backends", "_expand_flashinfer_alias", "_get_blocked_backends",
                 "_blocklist_known_names", "_select_attn_impls", "_is_fmha_impl_disabled_legacy"}, ns)
    selected = [impl.__name__ for impl in ns["_select_attn_impls"](impls, config, prefill)
                if not ns["_is_fmha_impl_disabled_legacy"](impl, config)]
    assert selected == allowed


def test_bazel_and_python_smoke_entrypoints_keep_identical_arguments():
    tree = ast.parse((ROOT / "rtp_llm/test/smoke/suites_h20_oss.bzl").read_text())
    base = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "flashinfer_decode_base_args" for t in n.targets))
    cases = smoke_cases()
    assert shlex.split(cases[CASES[0]]["smoke_args"]) == shlex.split(base + " --enable_cuda_graph 0")
    assert shlex.split(cases[CASES[1]]["smoke_args"]) == shlex.split(base + " --enable_cuda_graph 1 --decode_capture_config '2'")
