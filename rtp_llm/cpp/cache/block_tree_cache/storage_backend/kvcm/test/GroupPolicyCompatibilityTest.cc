#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"

using namespace rtp_llm;
using namespace rtp_llm::kvcm;
using namespace ::testing;
using namespace kv_cache_manager;

namespace rtp_llm {
namespace kvcm {

bool operator==(const GroupPolicy::Group& lhs, const GroupPolicy::Group& rhs) {
    return lhs.is_full == rhs.is_full && lhs.group_name == rhs.group_name
           && lhs.group_name_bithash == rhs.group_name_bithash;
}

bool operator==(const GroupPolicy::SpecInfo& lhs, const GroupPolicy::SpecInfo& rhs) {
    return lhs.group_id == rhs.group_id && lhs.tp_rank == rhs.tp_rank && lhs.tag == rhs.tag;
}

namespace test {

namespace {

KVCacheSpecPtr makeFakeSpec(const std::string& tag) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = 1;
    attn_config.size_per_head    = 1;
    attn_config.tokens_per_block = 1;
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = DataType::TYPE_FP16;
    SpecBuildContext ctx;
    ctx.dtype              = DataType::TYPE_FP16;
    ctx.seq_size_per_block = 1;
    ctx.attn_config        = &attn_config;
    ctx.parallelism_config = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

}  // namespace

class TestTopology {
public:
    TestTopology(const CacheConfig&,
                 const std::vector<int32_t>& full_group_ids,
                 const std::vector<int32_t>& other_group_ids,
                 size_t per_group_layer_num) {
        std::vector<int> layer_group_ids;
        for (int32_t full_group_id : full_group_ids) {
            for (int i = 0; i < per_group_layer_num; i++) {
                layer_group_ids.push_back(full_group_id);
            }
        }
        for (int32_t other_group_id : other_group_ids) {
            for (int i = 0; i < per_group_layer_num; i++) {
                layer_group_ids.push_back(other_group_id);
            }
        }
        if (!layer_group_ids.empty()) {
            const auto  max_gid = *std::max_element(layer_group_ids.begin(), layer_group_ids.end());
            CacheConfig fake_config;
            fake_config.layer_num     = static_cast<uint32_t>(layer_group_ids.size());
            fake_config.layer_all_num = fake_config.layer_num;
            std::vector<KVCacheSpecPtr>   specs;
            std::vector<std::vector<int>> layers_by_group(static_cast<size_t>(max_gid + 1));
            std::vector<CacheGroupType>   types(static_cast<size_t>(max_gid + 1), CacheGroupType::FULL);
            std::vector<std::string>      tags;
            for (int gid = 0; gid <= max_gid; ++gid) {
                tags.push_back(std::to_string(gid));
                specs.push_back(makeFakeSpec(tags.back()));
            }
            for (size_t layer_id = 0; layer_id < layer_group_ids.size(); ++layer_id) {
                layers_by_group[static_cast<size_t>(layer_group_ids[layer_id])].push_back(static_cast<int>(layer_id));
            }
            fake_config.fromGroupedSpecs(specs, layers_by_group, types, tags);
            topology_ = fake_config.topologyPtr();
        }
        if (topology_ == nullptr) {
            // An empty policy selection is invalid even with a valid topology.
            // Keep that negative test focused on policy.init(), since the new
            // topology constructor rejects an empty model before it gets there.
            topology_ = TestTopology(CacheConfig{}, {0}, {}, 1).topology_;
        }
    }
    const CacheTopology& topology() const { return *topology_; }
private:
    std::shared_ptr<const CacheTopology> topology_;
};

MATCHER_P(LocationsEqLocationsView, locations_view, "") {
    const kv_cache_manager::Locations& locations = arg;
    if (locations.size() != locations_view.size()) {
        *result_listener << "locations size mismatch: expected " << locations.size() << ", actual "
                         << locations_view.size();
        return false;
    }
    for (size_t i = 0; i < locations.size(); i++) {
        const auto& location      = locations[i];
        const auto& location_view = locations_view[i];
        if (location.size() != location_view.size()) {
            *result_listener << "location[" << i << "] size mismatch: expected " << location.size() << ", actual "
                             << location_view.size();
            return false;
        }
        for (size_t j = 0; j < location.size(); j++) {
            const auto& spec_unit      = location[j];
            const auto& spec_unit_view = location_view[j];
            if (spec_unit.spec_name != spec_unit_view.spec_name) {
                *result_listener << "location[" << i << "][" << j << "] spec_name mismatch: expected "
                                 << spec_unit.spec_name << ", actual " << spec_unit_view.spec_name;
                return false;
            }
            if (spec_unit.uri != spec_unit_view.uri) {
                *result_listener << "location[" << i << "][" << j << "] uri mismatch: expected " << spec_unit.spec_name
                                 << ", actual " << spec_unit_view.spec_name;
                return false;
            }
        }
    }
    return true;
}

class GroupPolicyTest: public ::testing::Test {
public:
    enum class RemoteConnectorGroupMode {
        RCGM_LAYER_DEFAULT,
        RCGM_ONLY_FULL_LAYER,
        RCGM_FULL_LINEAR_LAYER
    };

    void SetUp() override {
        rtp_llm::initLogger();
    }

    void TearDown() override {}

    void initGroupPolicy(size_t                      tp_size,
                         RemoteConnectorGroupMode    group_mode,
                         size_t                      per_group_layer_num,
                         const std::vector<int32_t>& full_group_ids,
                         const std::vector<int32_t>& other_group_ids                 = {},
                         uint32_t                    linear_attention_write_interval = 0,
                         size_t                      sink_size                       = 0,
                         size_t                      sw_size                         = 0) {
        allocator_ =
            std::make_shared<TestTopology>(config_, full_group_ids, other_group_ids, per_group_layer_num);
        switch (group_mode) {
            case RemoteConnectorGroupMode::RCGM_LAYER_DEFAULT: {
                group_policy_ = std::make_shared<kvcm::DefaultLayerGroupPolicy>(
                    allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids);
                break;
            }
            case RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER: {
                group_policy_ = std::make_shared<kvcm::FullLayerGroupPolicy>(
                    allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids);
                break;
            }
            case RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER: {
                group_policy_ = std::make_shared<kvcm::FullLinearLayerGroupPolicy>(
                    allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids, linear_attention_write_interval);
                break;
            }
        }
        ASSERT_TRUE(group_policy_->init());
        GroupPolicy::LocationSpecGroups location_groups;
        ASSERT_TRUE(group_policy_->buildLocationSpecGroups(static_cast<int>(tp_size), location_groups));
        ASSERT_FALSE(location_groups.empty());
        if (group_mode == RemoteConnectorGroupMode::RCGM_LAYER_DEFAULT) {
            // As in the original test, this generic policy exercises arbitrary
            // subsets; production Full/Linear policies only register reachable masks.
            std::vector<GroupPolicy::Group> groups;
            for (const auto& entry : group_policy_->groups()) {
                groups.push_back(entry.second);
            }
            for (uint64_t subset = 1; subset < (uint64_t{1} << groups.size()); ++subset) {
                uint64_t mask = 0;
                std::string name;
                for (size_t i = 0; i < groups.size(); ++i) {
                    if (subset & (uint64_t{1} << i)) {
                        mask |= groups[i].group_name_bithash;
                        name += groups[i].group_name;
                    }
                }
                group_policy_->location_spec_group_map_[mask] = name;
            }
        }
        RTP_LLM_LOG_INFO("initGroupPolicy debug info\n [%s]", group_policy_->debugString().c_str());
    }

private:
    static StorageBackend::BufferResolver unusedResolver() {
        return [](int, int, int) { return std::vector<BlockInfo>{}; };
    }

    bool needWriteGroups(const std::shared_ptr<KVCacheResource>& resource,
                         std::vector<std::string>& result) {
        // Preserve every old key, group slot and alignment bit; only transpose
        // group-major test data into the new key-major transport request.
        StorageRequest request;
        request.keys = std::make_shared<CacheKeysType>(resource->cache_keys);
        request.handles.resize(resource->cache_keys.size());
        for (size_t group_id = 0; group_id < resource->groupBlocks().size(); ++group_id) {
            const auto& blocks = resource->groupBlocks()[group_id]->blocks();
            for (size_t key_id = 0; key_id < blocks.size(); ++key_id) {
                request.handles.at(key_id).push_back({group_id, blocks[key_id]});
            }
        }
        size_t valid_keys = resource->cache_keys.size();
        if (valid_keys > 0 && !resource->lastBlockAligned()) {
            --valid_keys;
        }
        return group_policy_->getNeedWriteGroups(request, valid_keys, result);
    }

    std::shared_ptr<BlockIds> makeGroupBlockIds(const BlockIndicesType& block_indices) {
        auto result           = std::make_shared<BlockIds>();
        result->assign(block_indices);
        return result;
    }

    kv_cache_manager::Locations genFullLinearLocations(size_t                      tp_size,
                                                       const std::vector<int32_t>& full_group_ids,
                                                       const std::vector<int32_t>& linear_group_ids,
                                                       size_t                      cache_key_size,
                                                       const std::vector<size_t>&  linear_pos_vec) const {
        kv_cache_manager::Locations locations;
        locations.resize(cache_key_size, {});
        for (size_t i = 0; i < cache_key_size; i++) {
            for (auto group_id : full_group_ids) {
                std::string full_group_name = "F" + std::to_string(group_id);
                for (int r = 0; r < tp_size; r++) {
                    std::string uri = "uri_" + full_group_name + "_" + std::to_string(r) + "_" + std::to_string(i);
                    locations[i].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, full_group_name), uri}));
                }
            }
        }
        for (auto pos : linear_pos_vec) {
            for (auto group_id : linear_group_ids) {
                std::string linear_group_name = "L" + std::to_string(group_id);
                for (int r = 0; r < tp_size; r++) {
                    std::string uri = "uri_" + linear_group_name + "_" + std::to_string(r) + "_" + std::to_string(pos);
                    locations[pos].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, linear_group_name), uri}));
                }
            }
        }
        return locations;
    }

    inline std::string genLocationSpecName(int tp_rank, const std::string& group_name) const {
        static std::string location_spec_name("tp");
        return location_spec_name + std::to_string(tp_rank) + "_" + group_name;
    }

    void test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(size_t                      tp_size,
                                                                 const std::vector<int32_t>& full_group_ids,
                                                                 const std::vector<int32_t>& linear_group_ids) {
        {
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 4, {3});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 4, {3});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // only load the last linear block
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 4, {1, 3});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 4, {3});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // only load the last full + linear block
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 4, {1, 2});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 3, {2});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // only load the last full + linear block
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 5, {1, 2});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 3, {2});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // empty linear block
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 4, {});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 0, {});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // edge case : empty locations
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 0, {});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 0, {});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // edge case : one full block + empty linear block
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 1, {});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 0, {});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
        {  // edge case : one full block + one linear block
            Locations     locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 1, {0});
            LocationsView locations_view;
            ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
            auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, linear_group_ids, 1, {0});
            ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
        }
    }

    void test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_2() {
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(false);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3, 4};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7, 21}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11, 22}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1, -1, -1, 7}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1, -1, -1, 11}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, -1, 6, 7}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, -1, 10, 11}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, -1}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, -1}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0L1L2", "F0"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3, 4};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, -1, 7, 21}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, -1, 11, 22}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0L1L2", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3, 4};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, -1, -1, 7, 21}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, -1, -1, 11, 22}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // exist empty block
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1, 5, -1, 7}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1, 9, -1, 11}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0};
            resource->groupBlocks().push_back(makeGroupBlockIds({0}));
            resource->groupBlocks().push_back(makeGroupBlockIds({1}));
            resource->groupBlocks().push_back(makeGroupBlockIds({2}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {};  // all full linear
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0};
            resource->groupBlocks().push_back(makeGroupBlockIds({0}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0"};
            ASSERT_EQ(expected, real);
        }
    }

    void test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_1() {
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3, 4};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, -1, 21}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, -1, 22}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0L1L2", "F0L1L2", "F0L1L2", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0};
            resource->groupBlocks().push_back(makeGroupBlockIds({0}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {};
            ASSERT_EQ(expected, real);
        }
    }

    void test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_0() {
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3, 4};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7, 21}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11, 22}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0", "F0", "F0L1L2"};
            ASSERT_EQ(expected, real);
        }
        {
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0, 1, 2, 3};
            resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, -1}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, -1}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0", "F0", "F0L1L2", "F0"};
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0};
            resource->groupBlocks().push_back(makeGroupBlockIds({0}));
            resource->groupBlocks().push_back(makeGroupBlockIds({4}));
            resource->groupBlocks().push_back(makeGroupBlockIds({8}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {};
            ASSERT_EQ(expected, real);
        }
        {  // edge case
            auto resource        = std::make_shared<KVCacheResource>();
            resource->cache_keys = {0};
            resource->groupBlocks().push_back(makeGroupBlockIds({0}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1}));
            resource->groupBlocks().push_back(makeGroupBlockIds({-1}));
            std::vector<std::string> real;
            resource->setLastBlockAligned(true);
            ASSERT_TRUE(needWriteGroups(resource, real));
            std::vector<std::string> expected = {"F0"};
            ASSERT_EQ(expected, real);
        }
    }

private:
    std::shared_ptr<TestTopology> allocator_;
    std::shared_ptr<GroupPolicy>      group_policy_;
    CacheConfig                       config_;
};

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_success_single_tp) {
    initGroupPolicy(1, RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER, 4, {0}, {1, 2}, 0);
    auto cast_group_policy = std::dynamic_pointer_cast<FullLinearLayerGroupPolicy>(group_policy_);
    ASSERT_EQ(GroupPolicy::GroupIdMap({{0, {true, 0b001, "F0"}}, {1, {false, 0b010, "L1"}}, {2, {false, 0b100, "L2"}}}),
              cast_group_policy->groups_);
    ASSERT_EQ(
        (std::unordered_map<uint64_t, std::string>({{0b111, "F0L1L2"}, {0b001, "F0"}, {0b010, "L1"}, {0b100, "L2"}})),
        cast_group_policy->location_spec_group_map_);
    EXPECT_EQ(cast_group_policy->reachableAggregateMasks(), (std::vector<uint64_t>{0b001, 0b111}));
    ASSERT_EQ(GroupPolicy::SpecInfoMap({{"tp0_F0", {0, 0, "0"}}, {"tp0_L1", {1, 0, "1"}}, {"tp0_L2", {2, 0, "2"}}}),
              cast_group_policy->spec_name_to_info_);
    ASSERT_EQ((std::map<int32_t, std::vector<int>>({{0, {0, 1, 2, 3}}, {1, {4, 5, 6, 7}}, {2, {8, 9, 10, 11}}})),
              cast_group_policy->group_to_layer_ids_);
    ASSERT_EQ(0b001, cast_group_policy->valid_full_bithash_);
    ASSERT_EQ(0b111, cast_group_policy->valid_full_other_bithash_);
    ASSERT_EQ((GroupPolicy::SpecNames{"tp0_F0"}), cast_group_policy->full_spec_names_);
    ASSERT_EQ((GroupPolicy::SpecNames{"tp0_F0", "tp0_L1", "tp0_L2"}), cast_group_policy->full_other_spec_names_);
}

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_success_two_tp) {
    initGroupPolicy(2, RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER, 4, {0}, {1, 2}, 0);
    auto cast_group_policy = std::dynamic_pointer_cast<FullLinearLayerGroupPolicy>(group_policy_);
    ASSERT_EQ(GroupPolicy::GroupIdMap({{0, {true, 0b001, "F0"}}, {1, {false, 0b010, "L1"}}, {2, {false, 0b100, "L2"}}}),
              cast_group_policy->groups_);
    ASSERT_EQ(
        (std::unordered_map<uint64_t, std::string>({{0b111, "F0L1L2"}, {0b001, "F0"}, {0b010, "L1"}, {0b100, "L2"}})),
        cast_group_policy->location_spec_group_map_);
    ASSERT_EQ(GroupPolicy::SpecInfoMap({{"tp0_F0", {0, 0, "0"}},
                                        {"tp0_L1", {1, 0, "1"}},
                                        {"tp0_L2", {2, 0, "2"}},
                                        {"tp1_F0", {0, 1, "0"}},
                                        {"tp1_L1", {1, 1, "1"}},
                                        {"tp1_L2", {2, 1, "2"}}}),
              cast_group_policy->spec_name_to_info_);
    ASSERT_EQ((std::map<int32_t, std::vector<int>>({{0, {0, 1, 2, 3}}, {1, {4, 5, 6, 7}}, {2, {8, 9, 10, 11}}})),
              cast_group_policy->group_to_layer_ids_);
    ASSERT_EQ(0b001, cast_group_policy->valid_full_bithash_);
    ASSERT_EQ(0b111, cast_group_policy->valid_full_other_bithash_);
    ASSERT_EQ((GroupPolicy::SpecNames{"tp0_F0", "tp1_F0"}), cast_group_policy->full_spec_names_);
    ASSERT_EQ((GroupPolicy::SpecNames{"tp0_F0", "tp0_L1", "tp0_L2", "tp1_F0", "tp1_L1", "tp1_L2"}), cast_group_policy->full_other_spec_names_);
}

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_success_two_full_groups) {
    initGroupPolicy(2, RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER, 4, {0, 1}, {2, 3}, 0);
    auto cast_group_policy = std::dynamic_pointer_cast<FullLinearLayerGroupPolicy>(group_policy_);
    ASSERT_EQ(GroupPolicy::GroupIdMap({{0, {true, 0b0001, "F0"}},
                                       {1, {true, 0b0010, "F1"}},
                                       {2, {false, 0b0100, "L2"}},
                                       {3, {false, 0b1000, "L3"}}}),
              cast_group_policy->groups_);
    EXPECT_EQ(
        (std::unordered_map<uint64_t, std::string>(
            {{0b1111, "F0F1L2L3"}, {0b0001, "F0"}, {0b0010, "F1"}, {0b0100, "L2"}, {0b1000, "L3"}, {0b0011, "F0F1"}})),
        cast_group_policy->location_spec_group_map_);
    EXPECT_EQ(cast_group_policy->reachableAggregateMasks(), (std::vector<uint64_t>{0b0011, 0b1111}));
    EXPECT_EQ(GroupPolicy::SpecInfoMap({
                  {"tp0_F0", {0, 0, "0"}},
                  {"tp0_F1", {1, 0, "1"}},
                  {"tp0_L2", {2, 0, "2"}},
                  {"tp0_L3", {3, 0, "3"}},
                  {"tp1_F0", {0, 1, "0"}},
                  {"tp1_F1", {1, 1, "1"}},
                  {"tp1_L2", {2, 1, "2"}},
                  {"tp1_L3", {3, 1, "3"}},
              }),
              cast_group_policy->spec_name_to_info_);
    EXPECT_EQ((std::map<int32_t, std::vector<int>>(
                  {{0, {0, 1, 2, 3}}, {1, {4, 5, 6, 7}}, {2, {8, 9, 10, 11}}, {3, {12, 13, 14, 15}}})),
              cast_group_policy->group_to_layer_ids_);
    EXPECT_EQ(0b0011, cast_group_policy->valid_full_bithash_);
    EXPECT_EQ(0b1111, cast_group_policy->valid_full_other_bithash_);
    EXPECT_EQ((GroupPolicy::SpecNames{"tp0_F0", "tp0_F1", "tp1_F0", "tp1_F1"}), cast_group_policy->full_spec_names_);
    EXPECT_EQ((GroupPolicy::SpecNames{"tp0_F0", "tp0_F1", "tp0_L2", "tp0_L3", "tp1_F0", "tp1_F1", "tp1_L2", "tp1_L3"}), cast_group_policy->full_other_spec_names_);
}

TEST_F(GroupPolicyTest, test_init_DefaultLayerGroupPolicy_fail_for_duplicate_group) {
    std::vector<int32_t> full_group_ids  = {0, 1};
    std::vector<int32_t> other_group_ids = {0, 1};
    allocator_ = std::make_shared<TestTopology>(config_, full_group_ids, other_group_ids, 10);
    group_policy_ =
        std::make_shared<kvcm::DefaultLayerGroupPolicy>(allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids);
    ASSERT_FALSE(group_policy_->init());
}

TEST_F(GroupPolicyTest, test_init_FullLayerGroupPolicy_fail_for_empty_full_group) {
    std::vector<int32_t> full_group_ids;
    std::vector<int32_t> other_group_ids;
    allocator_ = std::make_shared<TestTopology>(config_, full_group_ids, other_group_ids, 10);
    group_policy_ =
        std::make_shared<kvcm::FullLayerGroupPolicy>(allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids);
    ASSERT_FALSE(group_policy_->init());
}

TEST_F(GroupPolicyTest, test_init_FullLayerGroupPolicy_success_for_multiple_full_groups) {
    initGroupPolicy(/*tp_size=*/1,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    /*per_group_layer_num=*/1,
                    /*full_group_ids=*/{0, 1});

    EXPECT_EQ(group_policy_->reachableAggregateMasks(), (std::vector<uint64_t>{0b11}));
    EXPECT_EQ(group_policy_->location_spec_group_map_,
              (std::unordered_map<uint64_t, std::string>{{0b01, "F0"}, {0b10, "F1"}, {0b11, "F0F1"}}));

    auto resource        = std::make_shared<KVCacheResource>();
    resource->cache_keys = {0, 1};
    resource->groupBlocks().push_back(makeGroupBlockIds({10, 11}));
    resource->groupBlocks().push_back(makeGroupBlockIds({20, NULL_BLOCK_IDX}));
    resource->setLastBlockAligned(true);

    std::vector<std::string> need_write_groups;
    ASSERT_TRUE(needWriteGroups(resource, need_write_groups));
    EXPECT_EQ(need_write_groups, (std::vector<std::string>{"F0F1", "F0"}));
}

TEST_F(GroupPolicyTest, test_init_FullLayerGroupPolicy_fail_for_not_empty_other_group) {
    std::vector<int32_t> full_group_ids  = {0};
    std::vector<int32_t> other_group_ids = {1};
    allocator_ = std::make_shared<TestTopology>(config_, full_group_ids, other_group_ids, 10);
    group_policy_ =
        std::make_shared<kvcm::FullLayerGroupPolicy>(allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids);
    ASSERT_FALSE(group_policy_->init());
}

TEST_F(GroupPolicyTest, test_init_FullLinearLayerGroupPolicy_fail_for_not_empty_group) {
    {
        std::vector<int32_t> full_group_ids;
        std::vector<int32_t> other_group_ids = {1};
        allocator_    = std::make_shared<TestTopology>(config_, full_group_ids, other_group_ids, 10);
        group_policy_ = std::make_shared<kvcm::FullLinearLayerGroupPolicy>(
            allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids, 0);
        ASSERT_FALSE(group_policy_->init());
    }
    {
        std::vector<int32_t> full_group_ids = {0};
        std::vector<int32_t> other_group_ids;
        allocator_    = std::make_shared<TestTopology>(config_, full_group_ids, other_group_ids, 10);
        group_policy_ = std::make_shared<kvcm::FullLinearLayerGroupPolicy>(
            allocator_->topology(), unusedResolver(), full_group_ids, other_group_ids, 0);
        ASSERT_FALSE(group_policy_->init());
    }
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_one_tp) {
    size_t               tp_size                         = 1;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(tp_size, full_group_ids, linear_group_ids);
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_two_tp) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(tp_size, full_group_ids, linear_group_ids);
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_two_tp_two_full_group) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0, 1};
    std::vector<int32_t> linear_group_ids                = {2, 3};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedLoadLocations(tp_size, full_group_ids, linear_group_ids);
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_2) {
    size_t               tp_size                         = 1;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 2;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_2();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_2) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 2;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_2();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_1) {
    size_t               tp_size                         = 1;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 1;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_1();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_1) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 1;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_1();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_0) {
    size_t               tp_size                         = 1;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_0();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_0) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_interval_0();
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_fail) {
    size_t               tp_size                         = 1;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    {
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_L1", "uri"}, {"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {  // location size error
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_L1", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {  // not exist full location
        Locations     locations({{{"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {  // invalid spec name
        Locations     locations({{{"not_exist", "uri"}, {"tp0_L1", "uri"}, {"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_fail_two_full_group) {
    size_t               tp_size                         = 1;
    std::vector<int32_t> full_group_ids                  = {0, 1};
    std::vector<int32_t> linear_group_ids                = {2, 3};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    {
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_F1", "uri"}, {"tp0_L2", "uri"}, {"tp0_L3", "uri"}}});
        LocationsView locations_view;
        ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {
        Locations     locations({{{"tp0_F0", "uri"}, {"tp0_L2", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
    {
        Locations     locations({{{"tp0_F0", "uri"}}});
        LocationsView locations_view;
        ASSERT_FALSE(group_policy_->filterNeedLoadLocations(locations, locations_view));
    }
}

TEST_F(GroupPolicyTest, test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_fail) {
    size_t               tp_size                         = 1;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> linear_group_ids                = {1, 2};
    uint32_t             linear_attention_write_interval = 2;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                    4,
                    full_group_ids,
                    linear_group_ids,
                    linear_attention_write_interval);
    {  // incomplete block
        auto resource        = std::make_shared<KVCacheResource>();
        resource->cache_keys = {0, 1, 2, 3, 4};
        resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
        resource->groupBlocks().push_back(makeGroupBlockIds({4, -1, 6, 7, -1}));
        resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11, -1}));
        std::vector<std::string> real;
        resource->setLastBlockAligned(true);
        ASSERT_FALSE(needWriteGroups(resource, real));
    }
    {  // invalid group size
        auto resource        = std::make_shared<KVCacheResource>();
        resource->cache_keys = {0, 1, 2, 3, 4};
        resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
        resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7, 21}));
        std::vector<std::string> real;
        resource->setLastBlockAligned(true);
        ASSERT_FALSE(needWriteGroups(resource, real));
    }
    {  // invalid group size
        auto resource        = std::make_shared<KVCacheResource>();
        resource->cache_keys = {0, 1, 2, 3, 4};
        resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20}));
        resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7, 21}));
        resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11, 22}));
        resource->groupBlocks().push_back(makeGroupBlockIds({12, 13, 14, 15, 23}));
        std::vector<std::string> real;
        resource->setLastBlockAligned(true);
        ASSERT_FALSE(needWriteGroups(resource, real));
    }
}

TEST_F(GroupPolicyTest, test_FullLayerGroupPolicy_filterNeedLoadLocations_success) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> other_group_ids                 = {};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    4,
                    full_group_ids,
                    other_group_ids,
                    linear_attention_write_interval);
    EXPECT_EQ(group_policy_->reachableAggregateMasks(), (std::vector<uint64_t>{0b1}));
    EXPECT_EQ(group_policy_->location_spec_group_map_.size(), 1u);
    {
        Locations     locations = genFullLinearLocations(tp_size, full_group_ids, other_group_ids, 4, {});
        LocationsView locations_view;
        ASSERT_TRUE(group_policy_->filterNeedLoadLocations(locations, locations_view));
        auto expect_locations = genFullLinearLocations(tp_size, full_group_ids, other_group_ids, 4, {});
        ASSERT_THAT(expect_locations, LocationsEqLocationsView(locations_view));
    }
}

TEST_F(GroupPolicyTest, test_FullLayerGroupPolicy_filterNeedWriteGroups_success) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> other_group_ids                 = {};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                    4,
                    full_group_ids,
                    other_group_ids,
                    linear_attention_write_interval);
    {
        auto resource        = std::make_shared<KVCacheResource>();
        resource->cache_keys = {0, 1, 2, 3};
        resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
        resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7}));
        resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11}));
        std::vector<std::string> real;
        resource->setLastBlockAligned(true);
        ASSERT_TRUE(needWriteGroups(resource, real));
        std::vector<std::string> expected = {};
        ASSERT_EQ(expected, real);
    }
}

TEST_F(GroupPolicyTest, test_DefaultLayerGroupPolicy_filterNeedWriteGroups_success) {
    size_t               tp_size                         = 2;
    std::vector<int32_t> full_group_ids                  = {0};
    std::vector<int32_t> other_group_ids                 = {1, 2};
    uint32_t             linear_attention_write_interval = 0;
    initGroupPolicy(tp_size,
                    RemoteConnectorGroupMode::RCGM_LAYER_DEFAULT,
                    4,
                    full_group_ids,
                    other_group_ids,
                    linear_attention_write_interval);
    {
        auto resource        = std::make_shared<KVCacheResource>();
        resource->cache_keys = {0, 1, 2, 3};
        resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
        resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7}));
        resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11}));
        std::vector<std::string> real;
        resource->setLastBlockAligned(true);
        ASSERT_TRUE(needWriteGroups(resource, real));
        std::vector<std::string> expected = {"F0G1G2", "F0G1G2", "F0G1G2", "F0G1G2"};
        ASSERT_EQ(expected, real);
    }
    {
        auto resource        = std::make_shared<KVCacheResource>();
        resource->cache_keys = {0, 1, 2, 3};
        resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3}));
        resource->groupBlocks().push_back(makeGroupBlockIds({4, 5, 6, 7}));
        resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, 10, 11}));
        std::vector<std::string> real;
        resource->setLastBlockAligned(false);
        ASSERT_TRUE(needWriteGroups(resource, real));
        std::vector<std::string> expected = {"F0G1G2", "F0G1G2", "F0G1G2"};
        ASSERT_EQ(expected, real);
    }
    {
        auto resource        = std::make_shared<KVCacheResource>();
        resource->cache_keys = {0, 1, 2, 3, 4, 5};
        resource->groupBlocks().push_back(makeGroupBlockIds({0, 1, 2, 3, 20, -1}));
        resource->groupBlocks().push_back(makeGroupBlockIds({4, -1, 6, 7, -1, 21}));
        resource->groupBlocks().push_back(makeGroupBlockIds({8, 9, -1, 11, -1, 22}));
        std::vector<std::string> real;
        resource->setLastBlockAligned(true);
        ASSERT_TRUE(needWriteGroups(resource, real));
        std::vector<std::string> expected = {"F0G1G2", "F0G2", "F0G1", "F0G1G2", "F0", "G1G2"};
        ASSERT_EQ(expected, real);
    }
}

}  // namespace test
}  // namespace kvcm
}  // namespace rtp_llm
