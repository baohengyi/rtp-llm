#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {
namespace {

using namespace kvcm;

KVCacheSpecPtr makeTestMhaSpec(const std::string& tag, uint32_t seq_size_per_block) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = 8;
    attn_config.size_per_head    = 128;
    attn_config.tokens_per_block = seq_size_per_block;

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = rtp_llm::DataType::TYPE_FP16;

    SpecBuildContext ctx;
    ctx.dtype              = rtp_llm::DataType::TYPE_FP16;
    ctx.seq_size_per_block = seq_size_per_block;
    ctx.attn_config        = &attn_config;
    ctx.parallelism_config = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

KVCacheSpecPtr makeTestLinearSpec(const std::string& tag, uint32_t seq_size_per_block) {
    LinearAttentionConfig linear_config;
    linear_config.linear_conv_kernel_dim = 2;
    linear_config.linear_key_head_dim    = 1;
    linear_config.linear_value_head_dim  = 1;
    linear_config.linear_num_key_heads   = 1;
    linear_config.linear_num_value_heads = 1;

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = rtp_llm::DataType::TYPE_FP16;

    SpecBuildContext ctx;
    ctx.dtype                   = rtp_llm::DataType::TYPE_FP16;
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.linear_attention_config = &linear_config;
    ctx.parallelism_config      = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}


StorageBackend::BufferResolver unusedLegacyResolver() {
    return [](int, int, int) { return std::vector<BlockInfo>{}; };
}

// Observe the config produced by the real backend initialization. Do not
// reproduce its location-spec serialization in the test.
ClientWrapper::ConfigMap captureLegacyClientConfig(const CacheConfig& config, int tp_size = 1) {
    std::vector<block_tree_cache_test::DeviceLayerBufferSpec> layers(
        config.layer_num, {config.kvBlockStrideBytesForGroup(0), 0});
    auto pool = block_tree_cache_test::makeDevicePool(layers, 8, "legacy_kvcm_internal");
    std::vector<DeviceBlockPoolPtr> pools(config.groupNums(), pool);
    auto client = std::make_shared<MockClientWrapper>();
    ClientWrapper::ConfigMap captured;
    EXPECT_CALL(*client, init(_, _))
        .WillOnce(Invoke([&](const ClientWrapper::ConfigMap& configs, const kv_cache_manager::InitParams&) {
            captured = configs;
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    ParallelismConfig parallelism;
    parallelism.tp_size = tp_size;
    parallelism.tp_rank = 0;
    parallelism.local_rank = 0;
    // Config generation uses the TP shape but performs no broadcast I/O.
    auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{});
    KVCMStorageBackend backend(config, KVCacheConfig{}, RuntimeConfig{}, parallelism,
                              SpeculativeExecutionConfig{}, broadcast, client);
    EXPECT_TRUE(backend.init(config.topologyPtr(), pools, unusedLegacyResolver()));
    backend.shutdown();
    return captured;
}

class RemoteConnectorInternalTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto mha = makeTestMhaSpec("0", 8);
        cache_config_.block_num = 8;
        cache_config_.layer_num = layer_num_;
        cache_config_.layer_all_num = layer_num_;
        byte_size_per_block_ = mha->block_size_bytes() * layer_num_;
        cache_config_.block_size_bytes = byte_size_per_block_;
        cache_config_.dtype = DataType::TYPE_FP16;
        std::vector<int> layers(layer_num_);
        std::iota(layers.begin(), layers.end(), 0);
        cache_config_.fromGroupedSpecs(
            {mha, makeTestLinearSpec("1", 8), makeTestLinearSpec("2", 8)},
            {layers, layers, layers},
            {CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::LINEAR},
            {"0", "1", "2"});
        cache_config_.setGroupBlockLayout(
            {8, 8, 8}, {mha->block_size_bytes(), mha->block_size_bytes(), mha->block_size_bytes()}, {0, 0, 0});
    }
    CacheConfig cache_config_;
    size_t byte_size_per_block_{0};
    static constexpr int layer_num_ = 10;
};

TEST_F(RemoteConnectorInternalTest, test_genClientConfig) {
    EXPECT_EQ(captureLegacyClientConfig(cache_config_).size(), 1u);
    EXPECT_EQ(captureLegacyClientConfig(cache_config_).size(), 1u);
    autil::EnvGuard biz_name("BIZ_NAME", "test_biz");
    EXPECT_EQ(captureLegacyClientConfig(cache_config_).size(), 1u);
}

TEST_F(RemoteConnectorInternalTest, test_genLocationSpecInfoMapAndGroups) {
    const auto configs = captureLegacyClientConfig(cache_config_, 2);
    ASSERT_EQ(configs.size(), 1u);
    ASSERT_NE(configs.at(""), nullptr);
    const auto& config = *configs.at("");
    ASSERT_NE(config.location_spec_info_map_, nullptr);
    ASSERT_NE(config.location_spec_groups_, nullptr);
    EXPECT_EQ(*config.location_spec_info_map_,
              (KVCMConfig::LocationSpecInfoMap{{"tp0_F0", byte_size_per_block_},
                                             {"tp0_L1", byte_size_per_block_},
                                             {"tp0_L2", byte_size_per_block_},
                                             {"tp1_F0", byte_size_per_block_},
                                             {"tp1_L1", byte_size_per_block_},
                                             {"tp1_L2", byte_size_per_block_}}));
    EXPECT_EQ(*config.location_spec_groups_,
              (KVCMConfig::LocationSpecGroups{{"F0", {"tp0_F0", "tp1_F0"}},
                  {"F0L1L2", {"tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1", "tp0_L2", "tp1_L2"}},
                  {"L1", {"tp0_L1", "tp1_L1"}}, {"L2", {"tp0_L2", "tp1_L2"}}}));

    FullLinearLayerGroupPolicy policy(cache_config_.topology(), unusedLegacyResolver(), {0}, {1, 2}, 1);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(2, groups));
    EXPECT_EQ(groups, *config.location_spec_groups_);
    EXPECT_EQ(policy.location_spec_group_map_,
              (std::unordered_map<uint64_t, std::string>{{0b111, "F0L1L2"}, {0b100, "L2"},
                                                        {0b010, "L1"}, {0b001, "F0"}}));
    ASSERT_EQ(policy.spec_info_map().size(), 6u);
    for (int rank = 0; rank < 2; ++rank) {
        for (int group = 0; group < 3; ++group) {
            const std::string tag = std::to_string(group);
            const std::string name = "tp" + std::to_string(rank) + "_" + (group == 0 ? "F" : "L") + tag;
            const auto& info = policy.spec_info_map().at(name);
            EXPECT_EQ(info.group_id, group);
            EXPECT_EQ(info.tp_rank, rank);
            EXPECT_EQ(info.tag, tag);
        }
    }
}

TEST_F(RemoteConnectorInternalTest, PublishesTagLocalHeterogeneousGroupBlockSizes) {
    const auto per_layer_bytes = byte_size_per_block_ / layer_num_;
    cache_config_.setGroupBlockLayout(
        {8, 8, 8}, {per_layer_bytes, per_layer_bytes / 2, per_layer_bytes}, {0, 0, 0});
    const auto configs = captureLegacyClientConfig(cache_config_);
    ASSERT_EQ(configs.size(), 1u);
    const auto& config = *configs.at("");
    ASSERT_NE(config.location_spec_info_map_, nullptr);
    ASSERT_NE(config.location_spec_groups_, nullptr);
    EXPECT_EQ(config.location_spec_info_map_->at("tp0_F0"), byte_size_per_block_);
    EXPECT_EQ(config.location_spec_info_map_->at("tp0_L1"), byte_size_per_block_ / 2);
    EXPECT_EQ(config.location_spec_info_map_->at("tp0_L2"), byte_size_per_block_);
    EXPECT_EQ(config.location_spec_groups_->at("F0L1L2"),
              (std::vector<std::string>{"tp0_F0", "tp0_L1", "tp0_L2"}));
}

TEST_F(RemoteConnectorInternalTest, test_genLocationSpecGroupsScalesLinearly) {
    constexpr size_t linear_group_count = 19;
    constexpr size_t group_count = linear_group_count + 1;
    CacheConfig config;
    config.block_num = 8;
    config.layer_num = group_count;
    config.layer_all_num = group_count;
    config.dtype = DataType::TYPE_FP16;
    auto full_spec = makeTestMhaSpec("full", 8);
    config.block_size_bytes = full_spec->block_size_bytes();
    std::vector<KVCacheSpecPtr> specs{full_spec};
    std::vector<std::vector<int>> layers{{0}};
    std::vector<CacheGroupType> types{CacheGroupType::FULL};
    std::vector<std::string> tags{"full"};
    std::vector<int32_t> linear_ids;
    for (size_t i = 0; i < linear_group_count; ++i) {
        const auto group_id = static_cast<int32_t>(i + 1);
        const std::string tag = "linear" + std::to_string(i);
        specs.push_back(makeTestLinearSpec(tag, 8));
        layers.push_back({group_id});
        types.push_back(CacheGroupType::LINEAR);
        tags.push_back(tag);
        linear_ids.push_back(group_id);
    }
    config.fromGroupedSpecs(specs, layers, types, tags);
    config.setGroupBlockLayout(std::vector<uint32_t>(group_count, 8),
                              std::vector<size_t>(group_count, full_spec->block_size_bytes()),
                              std::vector<size_t>(group_count, 0));
    const auto configs = captureLegacyClientConfig(config);
    ASSERT_EQ(configs.size(), 1u);
    ASSERT_NE(configs.at("")->location_spec_info_map_, nullptr);
    ASSERT_NE(configs.at("")->location_spec_groups_, nullptr);
    EXPECT_EQ(configs.at("")->location_spec_info_map_->size(), group_count);
    EXPECT_EQ(configs.at("")->location_spec_groups_->size(), group_count + 1);
    FullLinearLayerGroupPolicy policy(config.topology(), unusedLegacyResolver(), {0}, linear_ids, 1);
    ASSERT_TRUE(policy.init());
    GroupPolicy::LocationSpecGroups groups;
    ASSERT_TRUE(policy.buildLocationSpecGroups(1, groups));
    EXPECT_EQ(policy.location_spec_group_map_.size(), group_count + 1);
}

TEST(RemoteConnectorTagIdentityTest, GroupNamesDoNotDependOnNumericGroupOrder) {
    CacheConfig first;
    first.layer_num = first.layer_all_num = 1;
    first.fromGroupedSpecs({makeTestMhaSpec("full", 8), makeTestLinearSpec("linear", 8)},
                           {{0}, {0}}, {CacheGroupType::FULL, CacheGroupType::LINEAR}, {"full", "linear"});
    CacheConfig reversed;
    reversed.layer_num = reversed.layer_all_num = 1;
    reversed.fromGroupedSpecs({makeTestLinearSpec("linear", 8), makeTestMhaSpec("full", 8)},
                              {{0}, {0}}, {CacheGroupType::LINEAR, CacheGroupType::FULL}, {"linear", "full"});
    FullLinearLayerGroupPolicy first_policy(first.topology(), unusedLegacyResolver(), {0}, {1}, 1);
    FullLinearLayerGroupPolicy reversed_policy(reversed.topology(), unusedLegacyResolver(), {1}, {0}, 1);
    ASSERT_TRUE(first_policy.init());
    ASSERT_TRUE(reversed_policy.init());
    auto names_by_tag = [](const GroupPolicy& policy, const CacheTopology& topology) {
        std::map<std::string, std::string> result;
        for (const auto& [id, group] : policy.groups()) {
            result.emplace(topology.groupById(id).tag, group.group_name);
        }
        return result;
    };
    EXPECT_EQ(names_by_tag(first_policy, first.topology()), names_by_tag(reversed_policy, reversed.topology()));
    EXPECT_EQ(names_by_tag(first_policy, first.topology()),
              (std::map<std::string, std::string>{{"full", "Ffull"}, {"linear", "Llinear"}}));
}

TEST(RemoteConnectorTagIdentityTest, FullOnlyPolicyRoutesSameLayerGroupsByTagWithoutHotPathLayoutLookup) {
    for (bool reverse : {false, true}) {
        SCOPED_TRACE(reverse);
        const std::vector<std::string> tags = reverse ? std::vector<std::string>{"full_b", "full_a"} :
                                                       std::vector<std::string>{"full_a", "full_b"};
        CacheConfig config;
        config.layer_num = config.layer_all_num = 1;
        config.fromGroupedSpecs({makeTestMhaSpec(tags[0], 8), makeTestMhaSpec(tags[1], 8)},
                                {{0}, {0}}, {CacheGroupType::FULL, CacheGroupType::FULL}, tags);
        std::vector<std::tuple<int, std::string, int>> requests;
        auto resolver = [&](int layer, int group, int block) {
            const auto& tag = config.topology().groupById(group).tag;
            requests.emplace_back(layer, tag, block);
            BlockInfo info;
            info.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(block + 1));
            info.size_bytes = config.kvBlockStrideBytesForGroup(group) + config.kvScaleStrideBytesForGroup(group);
            return std::vector<BlockInfo>{info};
        };
        // Topology is an input value now; there is no allocator or
        // allLayerCacheBase() lookup in the policy's runtime interface.
        FullLayerGroupPolicy policy(config.topology(), resolver, {0, 1}, {});
        ASSERT_TRUE(policy.init());
        EXPECT_TRUE(requests.empty());
        EXPECT_EQ(policy.groups().at(0).tag, tags[0]);
        EXPECT_EQ(policy.groups().at(1).tag, tags[1]);
        EXPECT_EQ(policy.reachableAggregateMasks(), (std::vector<uint64_t>{0b11}));
        GroupPolicy::LocationSpecGroups groups;
        ASSERT_TRUE(policy.buildLocationSpecGroups(1, groups));
        EXPECT_EQ(policy.spec_info_map().at("tp0_Ffull_b").tag, "full_b");
        kv_cache_manager::BlockBuffers buffers;
        ASSERT_TRUE(policy.genBlockBuffersByTag({"full_b", "full_a"}, {7, 9}, buffers));
        EXPECT_EQ(requests, (std::vector<std::tuple<int, std::string, int>>{
                               {0, "full_b", 7}, {0, "full_a", 9}}));
    }
}

TEST(RemoteConnectorBlockBufferValidationTest, RejectsAllocatorBufferSizeThatDoesNotMatchTopology) {
    CacheConfig config;
    config.layer_num = config.layer_all_num = 1;
    config.fromGroupedSpecs({makeTestMhaSpec("full", 8)}, {{0}}, {CacheGroupType::FULL}, {"full"});
    auto resolver = [&](int layer, int group, int block) {
        EXPECT_EQ(layer, 0);
        EXPECT_EQ(group, 0);
        EXPECT_EQ(block, 7);
        BlockInfo info;
        info.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(block + 1));
        info.size_bytes = config.kvBlockStrideBytesForGroup(0) + 1;
        return std::vector<BlockInfo>{info};
    };
    FullLayerGroupPolicy policy(config.topology(), resolver, {0}, {});
    ASSERT_TRUE(policy.init());
    kv_cache_manager::BlockBuffers buffers;
    EXPECT_FALSE(policy.genBlockBuffersByTag({"full"}, {7}, buffers));
    EXPECT_TRUE(buffers.empty());
}

TEST(RemoteConnectorTopologyInvariantTest, InitializationRejectsMissingTopology) {
    // The new backend accepts topology in init(), rather than obtaining it
    // from the old allocator constructor. Keep invalid topology rejection at
    // the point where the production API first receives that topology.
    KVCMStorageBackend backend(CacheConfig{}, KVCacheConfig{}, RuntimeConfig{}, ParallelismConfig{},
                              SpeculativeExecutionConfig{}, nullptr);
    EXPECT_ANY_THROW(backend.init(nullptr, {}, unusedLegacyResolver()));
    backend.shutdown();
}

}  // namespace
}  // namespace rtp_llm
