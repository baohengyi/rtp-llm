load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test(
    name = "combo_position_ids_validation_test",
    srcs = ["combo_position_ids_validation_test.cc"],
    copts = copts(),
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
    deps = [
        "//rtp_llm/cpp/cuda_graph:combo_position_ids_validation",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

cc_library(
    name = "test_cuda_graph_runner_libs",
    srcs = [
        "cuda_graph_test_runner.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_binary(
    name = "test_cuda_graph_runner",
    deps = [
        ":test_cuda_graph_runner_libs",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

# The Python drivers are collected by pytest's H20 profile. Python packages are
# installed by pyproject/uv and are intentionally no longer modeled as Bazel
# dependencies. Keep only the compiled binding here for setup.py to stage.
