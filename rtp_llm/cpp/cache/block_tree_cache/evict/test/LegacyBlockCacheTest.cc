#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

// Observe production deletion events so a bulk eviction's actual victim order
// is checked without reproducing the eviction loop in the test.
class LegacyEvictionRecorder: public KVCacheEventPublisher {
public:
    bool start() noexcept override { return true; }
    void stop() noexcept override {}
    PublisherStatus status() const noexcept override { return {PublisherState::READY, 0, 0, 0}; }
    PublishResult tryPublish(KVCacheEvent event) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
        if (event.type == KVCacheEventType::BLOCK_DELETE) {
            deleted_.push_back(event.block_key);
        }
        return PublishResult::ACCEPTED;
    }
    CacheKeysType takeDeleted() {
        std::lock_guard<std::mutex> lock(mutex_);
        CacheKeysType result;
        result.swap(deleted_);
        return result;
    }
    std::vector<KVCacheEvent> events() {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }
private:
    std::mutex mutex_;
    CacheKeysType deleted_;
    std::vector<KVCacheEvent> events_;
};

// The old eviction result exported parent dependencies. The new drop path
// consumes TreeNode links directly. Observe that real link when the production
// delete event fires, before topology cleanup destroys the evicted node.
class LegacyDependencyRecorder: public LegacyEvictionRecorder {
public:
    LegacyDependencyRecorder(BlockTree* tree, CacheKeysType path): tree_(tree), path_(std::move(path)) {}
    PublishResult tryPublish(KVCacheEvent event) noexcept override {
        if (event.type == KVCacheEventType::BLOCK_DELETE && event.block_key == path_.back()) {
            const auto nodes = tree_->findNode(path_);
            if (nodes.size() == path_.size() && nodes.back()->parent != nullptr
                && nodes.back()->parent != tree_->root()) {
                std::lock_guard<std::mutex> lock(mutex_);
                deleted_parent_keys_.push_back(nodes.back()->parent->cache_key);
            }
        }
        return LegacyEvictionRecorder::tryPublish(event);
    }
    CacheKeysType deletedParentKeys() {
        std::lock_guard<std::mutex> lock(mutex_);
        return deleted_parent_keys_;
    }
private:
    BlockTree* tree_;
    CacheKeysType path_;
    std::mutex mutex_;
    CacheKeysType deleted_parent_keys_;
};

class BlockCacheTest: public ::testing::Test {
protected:
    void SetUp() override { makeCache(1); }

    void makeCache(size_t member_count, bool separate_groups = false) {
        cache_.reset();
        pools_.clear();
        std::vector<GroupBase> groups;
        std::vector<size_t> group_ids;
        std::vector<int> publication_groups;
        for (size_t group_id = 0; group_id < member_count; ++group_id) {
            pools_.push_back(block_tree_cache_test::makeStructuralDevicePool(group_id));
            auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
            policy.enable_prefix_reuse = true;
            groups.push_back(block_transfer_engine_test::makeTestGroupBase(
                policy, {static_cast<int>(group_id)}, 1));
            group_ids.push_back(group_id);
            publication_groups.push_back(static_cast<int>(group_id));
        }
        auto topology = block_transfer_engine_test::makeTestTopology(std::move(groups));
        std::vector<GroupSetPtr> group_sets;
        if (separate_groups) {
            for (size_t group_id = 0; group_id < member_count; ++group_id) {
                auto full = std::make_shared<FullGroupSet>(
                    std::vector<DeviceBlockPoolPtr>{pools_[group_id]}, nullptr, nullptr);
                full->initialize(group_id, topology, {group_id});
                group_sets.push_back(std::move(full));
            }
        } else {
            auto full = std::make_shared<FullGroupSet>(pools_, nullptr, nullptr);
            full->initialize(0, topology, group_ids);
            group_sets.push_back(std::move(full));
        }
        cache_ = block_tree_cache_test::makeBlockTreeCacheForTest(std::move(group_sets));
        recorder_ = std::make_shared<LegacyEvictionRecorder>();
        cache_->setEventPublisher(recorder_, publication_groups);
    }

    void put(CacheKeyType key, const BlockIndicesType& blocks, bool resident = false) {
        GroupSetResource resource;
        resource.device_blocks = blocks;
        cache_->insert({key}, {{resource}}, Tier::DEVICE, resident);
    }

    BlockIndicesType match(CacheKeyType key, size_t group_id = 0) {
        auto result = cache_->match({key});
        auto blocks = cache_->matchedBlocksForGroup(group_id, result.matched_device_resources);
        block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
        return blocks;
    }

    int evict(size_t count) {
        const int result = cache_->evictForGroup(0, count);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
        return result;
    }

    size_t cachedBlocksCount() const {
        size_t count = 0;
        for (const auto& [key, node] : cache_->tree()->root()->children) {
            for (const auto& resource : node->group_set_resources) {
                count += resource.device_blocks.size();
            }
        }
        return count;
    }

    std::vector<DeviceBlockPoolPtr> pools_;
    std::shared_ptr<LegacyEvictionRecorder> recorder_;
    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockCacheTest, ConstructorTest) {
    EXPECT_TRUE(cache_->tree()->root()->children.empty());
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(BlockCacheTest, MatchBasicTest) {
    EXPECT_TRUE(match(1).empty());
    put(101, {1});
    ASSERT_EQ(cache_->tree()->size(), 1u);
    EXPECT_EQ(match(101), (BlockIndicesType{1}));
    const auto ref_count = pools_[0]->refCount(1);
    put(101, {1});
    EXPECT_EQ(cache_->tree()->size(), 1u);
    EXPECT_EQ(pools_[0]->refCount(1), ref_count);
    EXPECT_EQ(match(101), (BlockIndicesType{1}));
    EXPECT_TRUE(match(102).empty());
}

TEST_F(BlockCacheTest, PopBasicTest) {
    for (BlockIdxType block = 1; block <= 5; ++block) {
        put(100 + block, {block});
        EXPECT_EQ(cache_->tree()->size(), static_cast<size_t>(block));
    }
    EXPECT_EQ(evict(2), 2);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{101, 102}));
    EXPECT_EQ(cache_->tree()->size(), 3u);
    EXPECT_FALSE(pools_[0]->isAllocated(1));
    EXPECT_FALSE(pools_[0]->isAllocated(2));
    EXPECT_EQ(evict(3), 3);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{103, 104, 105}));
    EXPECT_EQ(cache_->tree()->size(), 0u);
    for (BlockIdxType block = 3; block <= 5; ++block) {
        EXPECT_FALSE(pools_[0]->isAllocated(block));
    }
    EXPECT_EQ(evict(3), 0);
    EXPECT_TRUE(recorder_->takeDeleted().empty());
    EXPECT_EQ(cache_->tree()->size(), 0u);

    // The new pool owns slots. Reallocate the five released slots, retaining
    // the same cache instance for the original empty-to-resident transition.
    const auto recycled = pools_[0]->malloc(5);
    ASSERT_TRUE(recycled.has_value());
    EXPECT_EQ((std::set<BlockIdxType>(recycled->begin(), recycled->end())),
              (std::set<BlockIdxType>{1, 2, 3, 4, 5}));
    put(101, {1}, true);
    EXPECT_EQ(cache_->tree()->size(), 1u);
    EXPECT_EQ(evict(2), 0);
    EXPECT_TRUE(recorder_->takeDeleted().empty());
    EXPECT_EQ(cache_->tree()->size(), 1u);
}

TEST_F(BlockCacheTest, SelectAndEvictEmptyCache) {
    EXPECT_EQ(evict(5), 0);
    EXPECT_TRUE(recorder_->takeDeleted().empty());
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(BlockCacheTest, SelectAndEvictBasic) {
    put(101, {1});
    put(102, {2});
    put(103, {3});
    EXPECT_EQ(cache_->tree()->size(), 3u);
    EXPECT_EQ(evict(2), 2);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{101, 102}));
    EXPECT_EQ(cache_->tree()->size(), 1u);
    EXPECT_TRUE(match(101).empty());
    EXPECT_TRUE(match(102).empty());
    EXPECT_EQ(match(103), (BlockIndicesType{3}));
}

TEST_F(BlockCacheTest, SelectAndEvictMultipleGroups) {
    makeCache(2);
    put(101, {10, 11});
    put(102, {20, 21});
    ASSERT_EQ(cache_->tree()->size(), 2u);
    EXPECT_EQ(cachedBlocksCount(), 4u);
    const auto first = cache_->tree()->findNode({101});
    const auto second = cache_->tree()->findNode({102});
    ASSERT_EQ(first.size(), 1u);
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(first[0]->group_set_resources[0].device_blocks, (BlockIndicesType{10, 11}));
    EXPECT_EQ(second[0]->group_set_resources[0].device_blocks, (BlockIndicesType{20, 21}));
    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{101}));
    EXPECT_FALSE(pools_[0]->isAllocated(10));
    EXPECT_FALSE(pools_[1]->isAllocated(11));
    EXPECT_EQ(cache_->tree()->size(), 1u);
    EXPECT_EQ(cachedBlocksCount(), 2u);
    EXPECT_TRUE(match(101, 0).empty());
    EXPECT_TRUE(match(101, 1).empty());
    EXPECT_EQ(match(102, 0), (BlockIndicesType{20}));
    EXPECT_EQ(match(102, 1), (BlockIndicesType{21}));
}

TEST_F(BlockCacheTest, SelectAndEvictSkipsResident) {
    put(101, {1}, true);
    put(102, {2}, true);
    EXPECT_EQ(cache_->tree()->size(), 2u);
    EXPECT_EQ(evict(5), 0);
    EXPECT_TRUE(recorder_->takeDeleted().empty());
    EXPECT_EQ(cache_->tree()->size(), 2u);
}

TEST_F(BlockCacheTest, SelectAndEvictSkipsKeyWithResidentItem) {
    makeCache(2, true);
    cache_->setEventPublisher(recorder_, {0});
    auto put_group = [&](CacheKeyType key, size_t group_id, BlockIdxType block, bool resident) {
        std::vector<GroupSetResource> resources(2);
        resources[group_id].device_blocks = {block};
        cache_->insert({key}, {resources}, Tier::DEVICE, resident);
    };
    put_group(101, 0, 10, false);
    put_group(101, 1, 11, true);
    put_group(102, 0, 20, false);
    EXPECT_EQ(cachedBlocksCount(), 3u);
    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{102}));
    EXPECT_FALSE(pools_[0]->isAllocated(20));
    EXPECT_TRUE(pools_[0]->isAllocated(10));
    EXPECT_TRUE(pools_[1]->isAllocated(11));
    EXPECT_EQ(cachedBlocksCount(), 2u);
    EXPECT_EQ(match(101, 0), (BlockIndicesType{10}));
}

TEST_F(BlockCacheTest, SelectAndEvictRequestMoreThanAvailable) {
    put(101, {1});
    put(102, {2});
    EXPECT_EQ(evict(100), 2);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{101, 102}));
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(BlockCacheTest, SelectAndEvictLRUOrder) {
    put(101, {1});
    put(102, {2});
    put(103, {3});
    EXPECT_EQ(match(101), (BlockIndicesType{1}));
    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{102}));
    EXPECT_EQ(cache_->tree()->size(), 2u);
}

class SharedBlockCacheTest: public BlockCacheTest {};

TEST_F(SharedBlockCacheTest, EmptyCacheKeepsLegacyVersion) {
    EXPECT_EQ(cache_->getKeySnapshot().version, -1);
}

TEST_F(SharedBlockCacheTest, PublisherIgnoresGroupsThatDoNotParticipateInReuse) {
    makeCache(2, true);
    cache_->setEventPublisher(recorder_, {0});
    std::vector<GroupSetResource> resources(2);
    resources[0].device_blocks = {101};
    cache_->insert({1}, {resources}, Tier::DEVICE);

    auto events = recorder_->events();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].type, KVCacheEventType::BLOCK_ADD);
    EXPECT_EQ(events[0].block_key, 1);
    EXPECT_EQ(cache_->logicalCacheSnapshot().block_keys, (CacheKeysType{1}));

    EXPECT_EQ(evict(1), 1);
    events = recorder_->events();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[1].type, KVCacheEventType::BLOCK_DELETE);
    EXPECT_EQ(events[1].block_key, 1);
    EXPECT_TRUE(cache_->logicalCacheSnapshot().block_keys.empty());
}

TEST_F(SharedBlockCacheTest, MatchGroupTouchesPrefixTreeLeafLru) {
    std::vector<std::vector<GroupSetResource>> prefix(2, std::vector<GroupSetResource>(1));
    prefix[0][0].device_blocks = {101};
    prefix[1][0].device_blocks = {102};
    cache_->insert({1, 2}, prefix, Tier::DEVICE);
    put(3, {103});

    auto matched = cache_->match({1, 2});
    const auto blocks = cache_->matchedBlocksForGroup(0, matched.matched_device_resources);
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, matched.matched_device_resources);
    ASSERT_EQ(blocks, (BlockIndicesType{101, 102}));

    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{3}));
    const auto remaining = cache_->tree()->findNode({1, 2});
    ASSERT_EQ(remaining.size(), 2u);
    EXPECT_EQ(remaining[0]->cache_key, 1);
    EXPECT_EQ(remaining[1]->cache_key, 2);
    EXPECT_TRUE(cache_->tree()->findNode({3}).empty());
    EXPECT_FALSE(pools_[0]->isAllocated(103));
}

TEST_F(SharedBlockCacheTest, ResidentIsStickyAcrossPuts) {
    put(1, {101});
    // An empty incoming resource is the new API's "no replacement block";
    // the existing physical block must survive both metadata-only updates.
    put(1, {}, true);
    put(1, {}, false);

    EXPECT_EQ(evict(1), 0);
    EXPECT_TRUE(recorder_->takeDeleted().empty());
    EXPECT_EQ(match(1), (BlockIndicesType{101}));
    const auto nodes = cache_->tree()->findNode({1});
    ASSERT_EQ(nodes.size(), 1u);
    EXPECT_TRUE(nodes.front()->is_resident);
}

TEST_F(SharedBlockCacheTest, PrefixTreeStopsAtBranchPoint) {
    put(1, {101});
    std::vector<std::vector<GroupSetResource>> left(2, std::vector<GroupSetResource>(1));
    left[0][0].device_blocks = {101};
    left[1][0].device_blocks = {102};
    cache_->insert({1, 2}, left, Tier::DEVICE);
    auto right = left;
    right[1][0].device_blocks = {103};
    cache_->insert({1, 3}, right, Tier::DEVICE);

    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{2}));
    EXPECT_EQ(cache_->tree()->findNode({1, 2}).size(), 1u);
    EXPECT_EQ(cache_->tree()->findNode({1}).size(), 1u);
    EXPECT_EQ(cache_->tree()->findNode({1, 3}).size(), 2u);
    EXPECT_FALSE(pools_[0]->isAllocated(102));
    EXPECT_TRUE(pools_[0]->isAllocated(101));
    EXPECT_TRUE(pools_[0]->isAllocated(103));
}

TEST_F(SharedBlockCacheTest, PrefixTreeEvictsOrphanLeafWithMissingParentDependency) {
    auto dependency_recorder = std::make_shared<LegacyDependencyRecorder>(cache_->tree(), CacheKeysType{1, 2});
    recorder_ = dependency_recorder;
    cache_->setEventPublisher(recorder_, {0});

    // Key 1 exists only as a structural parent; it owns no physical block.
    std::vector<std::vector<GroupSetResource>> path(2, std::vector<GroupSetResource>(1));
    path[1][0].device_blocks = {102};
    cache_->insert({1, 2}, path, Tier::DEVICE);
    const auto inserted = cache_->tree()->findNode({1, 2});
    ASSERT_EQ(inserted.size(), 2u);
    EXPECT_TRUE(inserted[0]->group_set_resources[0].is_empty());
    EXPECT_EQ(inserted[1]->parent, inserted[0]);
    EXPECT_EQ(cache_->getKeySnapshot().keys, (CacheKeysType{2}));

    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{2}));
    EXPECT_EQ(dependency_recorder->deletedParentKeys(), (CacheKeysType{1}));
    EXPECT_FALSE(pools_[0]->isAllocated(102));
    EXPECT_TRUE(cache_->tree()->root()->children.empty());
    EXPECT_EQ(cache_->tree()->size(), 0u);
    EXPECT_TRUE(cache_->getKeySnapshot().keys.empty());
}

TEST_F(SharedBlockCacheTest, PrefixTreeStopsAtResidentParent) {
    auto dependency_recorder = std::make_shared<LegacyDependencyRecorder>(cache_->tree(), CacheKeysType{1, 2});
    recorder_ = dependency_recorder;
    cache_->setEventPublisher(recorder_, {0});

    put(1, {101}, true);
    std::vector<std::vector<GroupSetResource>> path(2, std::vector<GroupSetResource>(1));
    path[0][0].device_blocks = {101};
    path[1][0].device_blocks = {102};
    cache_->insert({1, 2}, path, Tier::DEVICE);
    const auto inserted = cache_->tree()->findNode({1, 2});
    ASSERT_EQ(inserted.size(), 2u);
    EXPECT_TRUE(inserted[0]->is_resident);
    EXPECT_FALSE(inserted[1]->is_resident);
    EXPECT_EQ(inserted[1]->parent, inserted[0]);

    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(recorder_->takeDeleted(), (CacheKeysType{2}));
    EXPECT_EQ(dependency_recorder->deletedParentKeys(), (CacheKeysType{1}));
    EXPECT_EQ(cache_->tree()->findNode({1, 2}).size(), 1u);
    EXPECT_EQ(match(1), (BlockIndicesType{101}));
    EXPECT_TRUE(pools_[0]->isAllocated(101));
    EXPECT_FALSE(pools_[0]->isAllocated(102));
    EXPECT_EQ(cache_->tree()->size(), 1u);
    EXPECT_EQ(evict(1), 0);
    EXPECT_TRUE(recorder_->takeDeleted().empty());
}

// The old memory cache stored completed host blocks directly. Seed that same
// tier in the production tree; no device copy is needed for metadata/LRU tests.
class MemoryBlockCacheTest: public ::testing::Test {
protected:
    void SetUp() override { makeCache(1); }

    void makeCache(size_t payload_bytes, bool enable_disk = false, size_t usable_count = 128) {
        cache_.reset();
        disk_pool_.reset();
        host_pool_ = block_tree_cache_test::makeHostPool(payload_bytes, usable_count);
        ASSERT_NE(host_pool_, nullptr);
        const auto blocks = host_pool_->malloc(usable_count);
        ASSERT_TRUE(blocks.has_value());
        ASSERT_EQ(blocks->front(), 1);
        ASSERT_EQ(blocks->back(), usable_count);
        if (enable_disk) {
            disk_pool_ = block_tree_cache_test::makeDiskPool(
                payload_bytes, 128, std::make_unique<block_tree_cache_test::MemoryDiskBlockIO>());
            ASSERT_NE(disk_pool_, nullptr);
            const auto disk_blocks = disk_pool_->malloc(128);
            ASSERT_TRUE(disk_blocks.has_value());
            ASSERT_EQ(disk_blocks->front(), 1);
            ASSERT_EQ(disk_blocks->back(), 128);
        }
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)},
            host_pool_, disk_pool_);
        BlockTreeCacheConfig config;
        config.enable_host_cache = true;
        config.enable_disk_cache = enable_disk;
        cache_ = block_tree_cache_test::makeBlockTreeCacheForTest({full}, config);
        ASSERT_NE(cache_, nullptr);
    }

    void put(CacheKeyType key, BlockIdxType block) {
        const auto size_before = cache_->tree()->size();
        GroupSetResource resource;
        resource.host_block = block;
        host_pool_->incTreeRef(block, BlockTreeRefType::CACHE);
        ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache_, {key}, {{resource}}));
        EXPECT_EQ(cache_->tree()->size(), size_before + 1);
    }

    bool contains(CacheKeyType key) const {
        const auto path = cache_->tree()->findNode({key});
        return path.size() == 1 && path.front()->group_set_resources[0].hasTier(Tier::HOST);
    }

    int evict(size_t count, Tier tier = Tier::HOST) {
        const auto freed = block_tree_cache_test::BlockTreeCacheTestPeer::reclaimBlocksForTest(
            *cache_, count, tier);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
        return freed;
    }

    void putResident(CacheKeyType key, BlockIdxType block) {
        const auto size_before = cache_->tree()->size();
        GroupSetResource resource;
        resource.host_block = block;
        host_pool_->incTreeRef(block, BlockTreeRefType::CACHE);
        const auto inserted = cache_->tree()->insertNode({key}, {{resource}}, false, true);
        block_tree_cache_test::releaseLowerTierSeedRefs(cache_->groupSets(), {{resource}});
        cache_->evictor_.onInserted(inserted);
        ASSERT_EQ(inserted.accepted_resource_count, 1u);
        EXPECT_EQ(cache_->tree()->size(), size_before + 1);
    }

    BlockIndicesType popHostBlocks(size_t count) {
        BlockIndicesType allocated;
        for (BlockIdxType block = 1; block <= 128; ++block) {
            if (host_pool_->isAllocated(block)) {
                allocated.push_back(block);
            }
        }
        const auto freed = evict(count);
        BlockIndicesType popped;
        for (const auto block : allocated) {
            if (!host_pool_->isAllocated(block)) {
                popped.push_back(block);
            }
        }
        // Observe actual production ownership release, including unexpected
        // victims; do not synthesize a result from the requested count/keys.
        EXPECT_EQ(popped.size(), static_cast<size_t>(freed));
        return popped;
    }

    CacheKeysType hostKeysInMRUOrder() const {
        std::lock_guard<std::mutex> lock(cache_->mutex_);
        const auto* heap = cache_->evictor_.heapFor(0, Tier::HOST);
        EXPECT_NE(heap, nullptr);
        if (heap == nullptr) {
            return {};
        }
        const auto size_before = heap->size();
        std::set<TreeNode*> observed;
        CacheKeysType keys;
        // Read the production LRU comparator through best(), excluding only
        // already observed nodes. Do not sort copied timestamps or mutate the
        // heap to manufacture an order. The legacy accessor returned MRU first.
        while (const auto entry = heap->best([&](TreeNode* node) { return observed.count(node) == 0; })) {
            keys.push_back(entry->cache_key);
            observed.insert(entry->node);
        }
        EXPECT_EQ(heap->size(), size_before);
        EXPECT_EQ(keys.size(), size_before);
        std::reverse(keys.begin(), keys.end());
        return keys;
    }

    std::shared_ptr<HostBlockPool> host_pool_;
    BlockTreeDiskBlockPoolPtr disk_pool_;
    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(MemoryBlockCacheTest, match_ReturnNull_WhenKeyNotFoundAndCacheNonEmpty) {
    makeCache(7000);
    put(700, 70);

    const auto result = cache_->match({999});
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_TRUE(cache_->matchedBlocksForGroup(0, result.matched_device_resources).empty());
    EXPECT_TRUE(contains(700));
    EXPECT_TRUE(host_pool_->isAllocated(70));
    EXPECT_EQ(cache_->tree()->size(), 1u);
}

TEST_F(MemoryBlockCacheTest, empty_ReturnTrue_WhenCacheEmpty) {
    EXPECT_TRUE(cache_->tree()->root()->children.empty());
    EXPECT_EQ(cache_->tree()->size(), 0u);
    EXPECT_FALSE(contains(42));

    const auto result = cache_->match({42});
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_TRUE(cache_->matchedBlocksForGroup(0, result.matched_device_resources).empty());
}

TEST_F(MemoryBlockCacheTest, match_ReturnHit_WhenKeyExistsAndUpdatesRecency) {
    // The old per-entry widths do not affect LRU ordering. Exercise each one
    // with the fixed-width host pool, preserving the original keys and IDs.
    for (size_t payload_bytes = 3000; payload_bytes < 3003; ++payload_bytes) {
        SCOPED_TRACE(payload_bytes);
        makeCache(payload_bytes);
        for (int i = 0; i < 3; ++i) {
            put(500 + i, 40 + i);
        }

        auto result = cache_->match({500});
        auto context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->matchedBlocks(), 1u);
        ASSERT_EQ(context->loadDescs().size(), 1u);
        ASSERT_EQ(context->loadDescs()[0].source_tier, Tier::HOST);
        ASSERT_EQ(context->loadDescs()[0].source_blocks, (BlockIndicesType{40}));

        // Release the real match's transient load ownership before testing
        // recency. Otherwise the matched block could survive because it was
        // still pinned, masking a broken LRU update.
        ASSERT_TRUE(cache_->abortPendingLoad(context));
        block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
        result.async_context.reset();
        context.reset();
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
        ASSERT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        ASSERT_EQ(cache_->evictor_.candidateCount(0, Tier::HOST), 3u);
        ASSERT_EQ(host_pool_->treeRefCount(40), 1u);

        const auto popped = popHostBlocks(1);
        ASSERT_EQ(popped.size(), 1u);
        EXPECT_NE(popped[0], 40);
        EXPECT_EQ(cache_->tree()->size(), 2u);
    }
}

TEST_F(MemoryBlockCacheTest, contains_ReturnTrue_WhenKeyExists) {
    makeCache(9000);
    put(900, 90);
    EXPECT_TRUE(contains(900));
}

TEST_F(MemoryBlockCacheTest, cacheKeys_ReturnsKeysInMRUOrder) {
    makeCache(1);
    for (int i = 1; i <= 3; ++i) {
        put(i, 100 + i);
    }
    const auto keys = hostKeysInMRUOrder();
    ASSERT_EQ(keys.size(), 3u);
    EXPECT_EQ(keys[0], 3);
    EXPECT_EQ(keys[1], 2);
    EXPECT_EQ(keys[2], 1);
    EXPECT_EQ(cache_->tree()->size(), 3u);
}

TEST_F(MemoryBlockCacheTest, cacheKeys_UpdatesOrderAfterMatch) {
    // Keep the original physical block IDs 201..203; the old cache did not
    // own a pool, so this fixture needs enough real host slots for those IDs.
    makeCache(1, false, 256);
    for (int i = 1; i <= 3; ++i) {
        put(i, 200 + i);
    }
    {
        const auto keys = hostKeysInMRUOrder();
        ASSERT_EQ(keys.size(), 3u);
        EXPECT_EQ(keys[0], 3);
        EXPECT_EQ(keys[1], 2);
        EXPECT_EQ(keys[2], 1);
    }
    auto result = cache_->match({1});
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->matchedBlocks(), 1u);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    ASSERT_EQ(context->loadDescs()[0].source_tier, Tier::HOST);
    ASSERT_EQ(context->loadDescs()[0].source_blocks, (BlockIndicesType{201}));
    ASSERT_TRUE(cache_->abortPendingLoad(context));
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
    result.async_context.reset();
    context.reset();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    ASSERT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    ASSERT_EQ(host_pool_->treeRefCount(201), 1u);
    ASSERT_EQ(cache_->evictor_.candidateCount(0, Tier::HOST), 3u);
    const auto keys2 = hostKeysInMRUOrder();
    ASSERT_EQ(keys2.size(), 3u);
    EXPECT_EQ(keys2[0], 1);
    EXPECT_EQ(keys2[1], 3);
    EXPECT_EQ(keys2[2], 2);
    EXPECT_EQ(cache_->tree()->size(), 3u);
}

TEST_F(MemoryBlockCacheTest, contains_ReturnFalse_WhenKeyNotFoundAndCacheNonEmpty) {
    makeCache(9001);
    put(901, 91);
    EXPECT_FALSE(contains(999));
    EXPECT_EQ(cache_->tree()->size(), 1u);
}

TEST_F(MemoryBlockCacheTest, contains_ReturnFalse_WhenKeyNotFoundAndCacheEmpty) {
    EXPECT_FALSE(contains(42));
}

TEST_F(MemoryBlockCacheTest, contains_ReturnFalse_WhenContainsDoesNotUpdateRecency) {
    for (int index = 0; index < 3; ++index) {
        put(1000 + index, 10 + index);
    }
    ASSERT_TRUE(contains(1000));
    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(contains(1000));
    EXPECT_FALSE(host_pool_->isAllocated(10));
    EXPECT_TRUE(contains(1001));
    EXPECT_TRUE(contains(1002));
    EXPECT_TRUE(host_pool_->isAllocated(11));
    EXPECT_TRUE(host_pool_->isAllocated(12));
}

TEST_F(MemoryBlockCacheTest, put_ReturnTrue_WhenInsertNewItem) {
    makeCache(1234);
    put(100, 7);
    EXPECT_FALSE(cache_->tree()->root()->children.empty());
    EXPECT_EQ(cache_->tree()->size(), 1u);
    EXPECT_TRUE(contains(100));

    auto result = cache_->match({100});
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), 1u);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    EXPECT_EQ(context->loadDescs()[0].source_tier, Tier::HOST);
    ASSERT_EQ(context->loadDescs()[0].source_blocks, (BlockIndicesType{7}));
    EXPECT_EQ(host_pool_->blockBuffer(context->loadDescs()[0].source_blocks.front()).payload_bytes, 1234u);
    EXPECT_TRUE(cache_->abortPendingLoad(context));
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
}

TEST_F(MemoryBlockCacheTest, pop_ReturnEmpty_WhenCacheEmpty) {
    EXPECT_EQ(evict(1), 0);
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(MemoryBlockCacheTest, pop_ReturnNonResidentBlocks_WhenSkipResident) {
    // HostBlockPool is fixed-width. Repeat the unchanged resident/count
    // scenario at every payload width from the old per-entry metadata.
    // Neither the old pop predicate nor the new eligibility rule uses bytes.
    for (size_t payload_bytes = 1000; payload_bytes < 1004; ++payload_bytes) {
        SCOPED_TRACE(payload_bytes);
        makeCache(payload_bytes);
        for (int i = 0; i < 4; ++i) {
            if (i == 1) {
                putResident(300 + i, 20 + i);
            } else {
                put(300 + i, 20 + i);
            }
        }
        EXPECT_EQ(cache_->tree()->size(), 4u);
        EXPECT_TRUE(contains(301));
        const auto popped = popHostBlocks(10);
        EXPECT_EQ(popped.size(), 3u);
        EXPECT_EQ(cache_->tree()->size(), 1u);
        EXPECT_TRUE(contains(301));
        for (const auto block : popped) {
            EXPECT_NE(block, 21);
        }
        EXPECT_TRUE(host_pool_->isAllocated(21));
    }
}

TEST_F(MemoryBlockCacheTest, pop_ReturnLimitedBlocks_WhenNumsLessThanSize) {
    for (size_t payload_bytes = 2000; payload_bytes < 2005; ++payload_bytes) {
        SCOPED_TRACE(payload_bytes);
        makeCache(payload_bytes);
        for (int i = 0; i < 5; ++i) {
            put(400 + i, 30 + i);
        }
        EXPECT_EQ(cache_->tree()->size(), 5u);
        const auto popped = popHostBlocks(2);
        EXPECT_EQ(popped.size(), 2u);
        EXPECT_EQ(cache_->tree()->size(), 3u);
        for (const auto block : popped) {
            bool found = false;
            for (int i = 0; i < 5; ++i) {
                if (block == 30 + i) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found);
        }
    }
}

TEST_F(MemoryBlockCacheTest, pop_ReturnEmpty_WhenAllResident) {
    for (size_t payload_bytes = 4000; payload_bytes < 4002; ++payload_bytes) {
        SCOPED_TRACE(payload_bytes);
        makeCache(payload_bytes);
        for (int i = 0; i < 2; ++i) {
            putResident(600 + i, 50 + i);
        }
        EXPECT_EQ(cache_->tree()->size(), 2u);
        const auto popped = popHostBlocks(3);
        EXPECT_TRUE(popped.empty());
        EXPECT_EQ(cache_->tree()->size(), 2u);
        EXPECT_TRUE(host_pool_->isAllocated(50));
        EXPECT_TRUE(host_pool_->isAllocated(51));
    }
}

class PrefixTreeMemoryBlockCacheTest: public MemoryBlockCacheTest {
protected:
    void SetUp() override { makeCache(1024, true); }

    void makeKindCache() {
        cache_.reset();
        host_pool_.reset();
        disk_pool_.reset();
        kind_pools_.clear();
        std::vector<GroupBase> groups;
        for (size_t kind = 0; kind < 2; ++kind) {
            auto pool = block_tree_cache_test::makeHostPool(1024, 128);
            ASSERT_NE(pool, nullptr);
            ASSERT_TRUE(pool->malloc(128).has_value());
            kind_pools_.push_back(std::move(pool));
            auto policy = defaultCacheGroupPolicy(kind == 0 ? CacheGroupType::FULL : CacheGroupType::SWA);
            policy.enable_prefix_reuse = true;
            if (kind == 1) {
                policy.sliding_window_size = 1;
            }
            groups.push_back(block_transfer_engine_test::makeTestGroupBase(policy, {static_cast<int>(kind)}, 1024));
        }
        const auto topology = block_transfer_engine_test::makeTestTopology(std::move(groups));
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)},
            kind_pools_[0], nullptr);
        auto state = std::make_shared<SWAGroupSet>(
            1, 1, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(1)},
            kind_pools_[1], nullptr);
        full->initialize(0, topology, {0});
        state->initialize(1, topology, {1});
        BlockTreeCacheConfig config;
        config.enable_host_cache = true;
        cache_ = block_tree_cache_test::makeBlockTreeCacheForTest({full, state}, config);
        ASSERT_NE(cache_, nullptr);
    }

    bool putKind(const CacheKeysType& path, size_t kind, BlockIdxType block) {
        std::vector<std::vector<GroupSetResource>> resources(path.size(), std::vector<GroupSetResource>(2));
        resources.back()[kind].host_block = block;
        kind_pools_[kind]->incTreeRef(block, BlockTreeRefType::CACHE);
        return block_tree_cache_test::insertGroupSetResources(*cache_, path, resources);
    }

    bool containsKind(const CacheKeysType& keys, size_t kind) const {
        const auto path = cache_->tree()->findNode(keys);
        return path.size() == keys.size() && path.back()->group_set_resources[kind].hasTier(Tier::HOST);
    }

    void expectBothKinds(CacheKeyType key, BlockIdxType compressed, BlockIdxType state) {
        auto context = matchLowerTier(key);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->matchedBlocks(), 1u);
        ASSERT_EQ(context->loadDescs().size(), 2u);
        std::vector<BlockIndicesType> matched(2);
        for (const auto& descriptor : context->loadDescs()) {
            ASSERT_LT(descriptor.group_set_id, matched.size());
            EXPECT_TRUE(matched[descriptor.group_set_id].empty());
            EXPECT_EQ(descriptor.source_tier, Tier::HOST);
            matched[descriptor.group_set_id] = descriptor.source_blocks;
        }
        EXPECT_EQ(matched[0], (BlockIndicesType{compressed}));
        EXPECT_EQ(matched[1], (BlockIndicesType{state}));
        releaseMatch(context);
    }

    void putPath(const CacheKeysType& keys, const BlockIndicesType& blocks, bool resident = false) {
        ASSERT_EQ(keys.size(), blocks.size());
        std::vector<std::vector<GroupSetResource>> resources(keys.size(), std::vector<GroupSetResource>(1));
        for (size_t index = 0; index < keys.size(); ++index) {
            resources[index][0].host_block = blocks[index];
            if (!isNullBlockIdx(blocks[index])) {
                host_pool_->incTreeRef(blocks[index], BlockTreeRefType::CACHE);
            }
        }
        const auto inserted = cache_->tree()->insertNode(keys, resources, false, resident);
        block_tree_cache_test::releaseLowerTierSeedRefs(cache_->groupSets(), resources);
        cache_->evictor_.onInserted(inserted);
        EXPECT_EQ(inserted.accepted_resource_count, 1u);
    }

    std::shared_ptr<LoadAsyncContext> matchLowerTier(CacheKeyType key) {
        auto result = cache_->match({key});
        EXPECT_TRUE(result.matched_device_resources.empty());
        auto context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
        block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
        return context;
    }

    void releaseMatch(std::shared_ptr<LoadAsyncContext>& context) {
        ASSERT_TRUE(cache_->abortPendingLoad(context));
        context.reset();
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    }

    std::vector<std::shared_ptr<HostBlockPool>> kind_pools_;
};

TEST_F(PrefixTreeMemoryBlockCacheTest, ContainsAndMatchAreKindAware) {
    makeKindCache();
    ASSERT_TRUE(putKind({1}, 0, 11));
    ASSERT_TRUE(putKind({1}, 1, 12));
    EXPECT_TRUE(containsKind({1}, 0));
    EXPECT_TRUE(containsKind({1}, 1));
    expectBothKinds(1, 11, 12);
}

TEST_F(PrefixTreeMemoryBlockCacheTest, DuplicateKindDoesNotBlockMissingOtherKind) {
    makeKindCache();
    ASSERT_TRUE(putKind({1}, 0, 11));
    EXPECT_FALSE(putKind({1}, 0, 13));
    ASSERT_TRUE(putKind({1}, 1, 12));
    EXPECT_EQ(cache_->tree()->size(), 1u);
    EXPECT_TRUE(kind_pools_[0]->isAllocated(11));
    EXPECT_TRUE(kind_pools_[1]->isAllocated(12));
    expectBothKinds(1, 11, 12);
}

TEST_F(PrefixTreeMemoryBlockCacheTest, KindLeafAccountingIsIndependent) {
    makeKindCache();
    ASSERT_TRUE(putKind({1}, 0, 11));
    ASSERT_TRUE(putKind({1, 2}, 0, 12));
    ASSERT_TRUE(putKind({1}, 1, 21));
    {
        std::lock_guard<std::mutex> lock(cache_->mutex_);
        EXPECT_TRUE(cache_->evictor_.dropLocked(1, Tier::HOST, true));
    }
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_FALSE(containsKind({1}, 1));
    EXPECT_TRUE(containsKind({1}, 0));
    EXPECT_TRUE(containsKind({1, 2}, 0));
    EXPECT_FALSE(kind_pools_[1]->isAllocated(21));
    EXPECT_TRUE(kind_pools_[0]->isAllocated(11));
    EXPECT_TRUE(kind_pools_[0]->isAllocated(12));
}

TEST_F(PrefixTreeMemoryBlockCacheTest, StatusCacheKeysAreUnorderedAndDeduplicated) {
    makeKindCache();
    ASSERT_TRUE(putKind({1}, 0, 11));
    ASSERT_TRUE(putKind({1}, 1, 12));
    ASSERT_TRUE(putKind({1, 2}, 0, 21));
    ASSERT_TRUE(putKind({1, 2, 3}, 1, 31));

    auto keys = cache_->getKeySnapshot().keys;
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys, (CacheKeysType{1, 2, 3}));
    EXPECT_EQ(keys.size(), cache_->tree()->size());
}

TEST_F(PrefixTreeMemoryBlockCacheTest, EvictionIsPerKindAndStopsAtBranchPoint) {
    putPath({1}, {11});
    putPath({1, 2}, {11, 12});
    putPath({1, 3}, {11, 13});

    EXPECT_EQ(evict(1), 1);
    EXPECT_EQ(cache_->tree()->findNode({1, 2}).size(), 1u);
    EXPECT_TRUE(contains(1));
    const auto right = cache_->tree()->findNode({1, 3});
    ASSERT_EQ(right.size(), 2u);
    EXPECT_TRUE(right.back()->group_set_resources[0].hasTier(Tier::HOST));
    EXPECT_FALSE(host_pool_->isAllocated(12));
    EXPECT_TRUE(host_pool_->isAllocated(11));
    EXPECT_TRUE(host_pool_->isAllocated(13));
}

TEST_F(PrefixTreeMemoryBlockCacheTest, PrefixTreeLinksChildInsertedBeforeParent) {
    // The new tree represents a pending parent as a structural node without a backing.
    putPath({1, 2}, {NULL_BLOCK_IDX, 12});
    EXPECT_FALSE(contains(1));
    putPath({1}, {11});

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(host_pool_->isAllocated(12));
    EXPECT_EQ(cache_->tree()->findNode({1, 2}).size(), 1u);
    EXPECT_TRUE(contains(1));
    EXPECT_TRUE(host_pool_->isAllocated(11));

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(contains(1));
    EXPECT_FALSE(host_pool_->isAllocated(11));
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(PrefixTreeMemoryBlockCacheTest, MultipleOrphanChildrenAttachOnParentInsert) {
    putPath({1, 2}, {NULL_BLOCK_IDX, 12});
    putPath({1, 3}, {NULL_BLOCK_IDX, 13});
    EXPECT_FALSE(contains(1));
    putPath({1}, {11});

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(host_pool_->isAllocated(12));
    EXPECT_EQ(cache_->tree()->findNode({1, 2}).size(), 1u);
    EXPECT_TRUE(contains(1));
    EXPECT_EQ(cache_->tree()->findNode({1, 3}).size(), 2u);
    EXPECT_TRUE(host_pool_->isAllocated(11));
    EXPECT_TRUE(host_pool_->isAllocated(13));

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(host_pool_->isAllocated(13));
    EXPECT_EQ(cache_->tree()->findNode({1, 3}).size(), 1u);
    EXPECT_TRUE(contains(1));
    EXPECT_TRUE(host_pool_->isAllocated(11));

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(contains(1));
    EXPECT_FALSE(host_pool_->isAllocated(11));
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(PrefixTreeMemoryBlockCacheTest, BranchParentBecomesEvictableAfterAllChildrenGone) {
    putPath({1}, {11});
    putPath({1, 2}, {11, 12});
    putPath({1, 3}, {11, 13});

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(host_pool_->isAllocated(12));
    EXPECT_EQ(cache_->tree()->findNode({1, 2}).size(), 1u);
    EXPECT_TRUE(contains(1));
    EXPECT_TRUE(host_pool_->isAllocated(11));
    EXPECT_TRUE(host_pool_->isAllocated(13));

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(host_pool_->isAllocated(13));
    EXPECT_EQ(cache_->tree()->findNode({1, 3}).size(), 1u);
    EXPECT_TRUE(contains(1));
    EXPECT_TRUE(host_pool_->isAllocated(11));

    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(contains(1));
    EXPECT_FALSE(host_pool_->isAllocated(11));
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(PrefixTreeMemoryBlockCacheTest, ResidentItemIsMatchableButNeverEvictable) {
    putPath({1}, {11}, true);
    auto context = matchLowerTier(1);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), 1u);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    EXPECT_EQ(context->loadDescs()[0].source_blocks, (BlockIndicesType{11}));
    releaseMatch(context);

    // Remove the load reservation before checking the independent resident rule.
    const auto path = cache_->tree()->findNode({1});
    ASSERT_EQ(path.size(), 1u);
    EXPECT_EQ(path.front()->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_TRUE(path.front()->is_resident);
    EXPECT_EQ(evict(1), 0);
    EXPECT_TRUE(contains(1));
    EXPECT_TRUE(host_pool_->isAllocated(11));
}

TEST_F(PrefixTreeMemoryBlockCacheTest, InFlightReleaseRestoresEvictability) {
    putPath({1}, {11});
    auto context = matchLowerTier(1);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), 1u);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    EXPECT_EQ(context->loadDescs()[0].source_tier, Tier::HOST);
    EXPECT_EQ(context->loadDescs()[0].source_blocks, (BlockIndicesType{11}));
    EXPECT_EQ(evict(1), 0);
    EXPECT_TRUE(contains(1));
    EXPECT_TRUE(host_pool_->isAllocated(11));

    releaseMatch(context);
    EXPECT_EQ(evict(1), 1);
    EXPECT_FALSE(contains(1));
    EXPECT_FALSE(host_pool_->isAllocated(11));
}

TEST_F(PrefixTreeMemoryBlockCacheTest, DiskBackingMatchesAndEvictsByBacking) {
    GroupSetResource disk_resource;
    disk_resource.disk_block = 7;
    disk_pool_->incTreeRef(7, BlockTreeRefType::CACHE);
    ASSERT_TRUE(block_tree_cache_test::insertGroupSetResources(*cache_, {1}, {{disk_resource}}));
    put(2, 22);

    auto context = matchLowerTier(1);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), 1u);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    EXPECT_EQ(context->loadDescs()[0].source_tier, Tier::DISK);
    EXPECT_EQ(context->loadDescs()[0].source_blocks, (BlockIndicesType{7}));
    EXPECT_EQ(evict(1, Tier::DISK), 0);

    releaseMatch(context);
    // Releasing an in-flight lookup must not retire a still-published backing.
    EXPECT_EQ(disk_pool_->treeRefCount(7), 1u);
    EXPECT_TRUE(disk_pool_->isAllocated(7));
    EXPECT_EQ(evict(1, Tier::DISK), 1);
    EXPECT_TRUE(cache_->tree()->findNode({1}).empty());
    EXPECT_FALSE(disk_pool_->isAllocated(7));
    EXPECT_TRUE(contains(2));
    EXPECT_TRUE(host_pool_->isAllocated(22));

    EXPECT_EQ(evict(1, Tier::HOST), 1);
    EXPECT_FALSE(contains(2));
    EXPECT_FALSE(host_pool_->isAllocated(22));
}

}  // namespace
}  // namespace rtp_llm
