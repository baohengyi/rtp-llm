# The standalone Python package is built and tested through pip/setuptools.
# This file keeps only Bazel-visible C++ runtime data and test helpers.

py_library(
    name = "auto_model",
    srcs = ["auto_model.py"],
    deps = [":py_standalone_testlib"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "py_standalone_testlib",
    deps = [
        "//rtp_llm/test/utils:test_util",
    ],
    data = [
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)
