load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("@rtp_llm//bazel:defs.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_test(
    name = "zigzag_token_layout_test",
    srcs = ["ZigzagTokenLayoutTest.cc"],
    deps = [
        "//rtp_llm/cpp/models:context_parallel_layout",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "th_context_parallel_py_wrapper_test_lib",
    srcs = [
        "zigzag_processor_py_wrapper_test.cc",
    ],
    hdrs = [
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/models:context_parallel",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/core:op_data",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "th_context_parallel_py_wrapper_test",
    deps = [":th_context_parallel_py_wrapper_test_lib"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)
