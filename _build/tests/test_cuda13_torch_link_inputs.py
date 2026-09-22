"""CUDA 13 wheel dependencies must be explicit inputs to native links."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CUDA13_WHEEL_LIBS = {
    "nvidia/cu13/lib/libcublasLt.so.13",
    "nvidia/cu13/lib/libcupti.so.13",
    "nvidia/cu13/lib/libcufft.so.12",
    "nvidia/nccl/lib/libnccl.so.2",
}


def torch_sources(platform):
    rules = {}

    def select(branches):
        if platform in branches:
            return branches[platform]
        if platform.startswith("@//:using_cuda") and "@//:using_cuda" in branches:
            return branches["@//:using_cuda"]
        return branches.get("//conditions:default", [])

    exec(
        compile((ROOT / "BUILD.pytorch").read_text(), "BUILD.pytorch", "exec"),
        {
            "config_setting": lambda **_: None,
            "cc_library": lambda **rule: rules.update({rule["name"]: rule}),
            "glob": lambda *_, **__: [],
            "select": select,
        },
    )
    return set(rules["torch"]["srcs"])


@pytest.mark.parametrize("platform", ["@//:using_cuda13_x86", "@//:using_cuda13_arm"])
def test_cuda13_links_declare_required_wheel_sonames(platform):
    # These are DT_NEEDED dependencies of the installed torch/cublas libraries.
    # Passing only host -L paths does not place them in a remote link action.
    assert CUDA13_WHEEL_LIBS <= torch_sources(platform)


@pytest.mark.parametrize("platform", ["@//:using_cuda", "@//:using_rocm", "cpu"])
def test_other_platforms_do_not_require_cuda13_wheels(platform):
    assert not (CUDA13_WHEEL_LIBS & torch_sources(platform))
