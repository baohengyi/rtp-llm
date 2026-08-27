load("//bazel:tf_proto.bzl", "tf_proto_library_cc")

tf_proto_library_cc(
    name = "model_rpc_service",
    srcs = ["model_rpc_service.proto"],
    has_services = True,
    cc_grpc_version = True,
    default_header = True,
)

cc_library(
    name = "grpc_buffer_list_link_fix",
    srcs = ["@grpc//:src/core/lib/iomgr/buffer_list.cc"],
    alwayslink = True,
    copts = ["-DGRPC_LINUX_ERRQUEUE=1"],
    deps = ["@grpc//:grpc_base_c"],
)

cc_library(
    name = "model_rpc_service_cc_proto",
    hdrs = ["model_rpc_service.pb.h",
            "model_rpc_service.grpc.pb.h"],
    srcs = ["model_rpc_service.pb.cc",
            "model_rpc_service.grpc.pb.cc"],
    deps = [
        ":grpc_buffer_list_link_fix",
        ":model_rpc_service_cc_impl",
    ],
    visibility = ["//visibility:public"],
)
