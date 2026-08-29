#include "gtest/gtest.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "rtp_llm/cpp/cache/SharedBlockCache.h"

namespace rtp_llm::test {
namespace {

BlockDependency rootDep(uint32_t ordinal = 0) {
    BlockDependency dep;
    dep.ordinal = ordinal;
    return dep;
}

BlockDependency childDep(CacheKeyType parent, uint32_t ordinal) {
    BlockDependency dep;
    dep.has_parent = true;
    dep.parent_key = parent;
    dep.ordinal    = ordinal;
    return dep;
}

}  // namespace

TEST(SharedBlockCachePerfTest, FlatFallbackLargeLru) {
    constexpr int kItemCount    = 20000;
    constexpr int kTargetStride = 5;
    constexpr int kEvictCount   = 2000;

    SharedBlockCache cache;
    cache.setPrefixTreeEnabled(false);
    for (int i = 0; i < kItemCount; ++i) {
        const auto key         = static_cast<CacheKeyType>(i + 1);
        const auto target_slot = i % kTargetStride == 0 ? static_cast<BlockIdxType>(i + 100001) : NULL_BLOCK_IDX;
        cache.put(key,
                  std::vector<BlockIdxType>{static_cast<BlockIdxType>(i + 1), target_slot},
                  /*is_resident=*/false,
                  SharedBlockCache::kGpuLogicalNamespace,
                  rootDep());
    }

    const auto start   = std::chrono::steady_clock::now();
    const auto evicted = cache.selectAndEvictForGroup(/*group_id=*/1, kEvictCount);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);

    EXPECT_EQ(evicted.evicted_keys.size(), kEvictCount);
    std::cout << "[ PERF ] prefix_tree=off items=" << kItemCount << " evicted=" << evicted.evicted_keys.size()
              << " selection_us=" << elapsed.count() << std::endl;
}

TEST(SharedBlockCachePerfTest, PrefixTreeLongSessionChains) {
    constexpr int kFamilyCount = 16;
    constexpr int kChainDepth  = 512;

    SharedBlockCache cache;
    for (int family = 0; family < kFamilyCount; ++family) {
        CacheKeyType parent_key = 0;
        for (int depth = 0; depth < kChainDepth; ++depth) {
            const auto key         = static_cast<CacheKeyType>(family * kChainDepth + depth + 1);
            const bool target_leaf = family == kFamilyCount - 1 && depth == kChainDepth - 1;
            cache.put(key,
                      std::vector<BlockIdxType>{
                          static_cast<BlockIdxType>(key + 10000),
                          target_leaf ? static_cast<BlockIdxType>(key + 20000) : NULL_BLOCK_IDX},
                      /*is_resident=*/false,
                      SharedBlockCache::kGpuLogicalNamespace,
                      depth == 0 ? rootDep() : childDep(parent_key, static_cast<uint32_t>(depth)));
            parent_key = key;
        }
    }

    const auto start   = std::chrono::steady_clock::now();
    const auto evicted = cache.selectAndEvictForGroup(/*group_id=*/1, /*min_blocks=*/1);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);

    EXPECT_EQ(evicted.evicted_keys.size(), kChainDepth);
    std::cout << "[ PERF ] prefix_tree=on items=" << kFamilyCount * kChainDepth
              << " chains=" << kFamilyCount << " depth=" << kChainDepth
              << " evicted=" << evicted.evicted_keys.size() << " selection_us=" << elapsed.count() << std::endl;
}

}  // namespace rtp_llm::test
