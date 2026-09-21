load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps", "torch_deps")

cc_test = cc_test_wrapper

test_copts = copts()

test_deps = [
    "//rtp_llm/cpp/cache/block_tree_cache",
    "//rtp_llm/cpp/testing:test_log_capture",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + torch_deps() + no_block_copy_link_deps()

cc_test(
    name = "full_prefix_invariant_scanner_test",
    exec_properties = {"gpu": "A10"},
    testonly = True,
    srcs = ["FullPrefixInvariantScannerTest.cc"],
    copts = test_copts,
    tags = ["exclusive-if-local"],
    deps = test_deps + ["//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils"],
    env = {},
)
