#include "gtest/gtest.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskMountGuard.h"

namespace rtp_llm::test {
namespace {

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/rtp_disk_pool_test_XXXXXX";
        auto path   = ::mkdtemp(tmpl);
        EXPECT_NE(path, nullptr);
        if (path != nullptr) {
            path_ = path;
        }
    }
    ~TempDir() {
        if (path_.empty()) {
            return;
        }
        const auto work_dir = path_ + "/rtp_llm_disk_kv";
        if (auto* dir = ::opendir(work_dir.c_str())) {
            while (auto* entry = ::readdir(dir)) {
                const std::string name(entry->d_name);
                if (name == "." || name == "..") {
                    continue;
                }
                ::unlink((work_dir + "/" + name).c_str());
            }
            ::closedir(dir);
        }
        ::rmdir(work_dir.c_str());
        ::rmdir(path_.c_str());
    }
    const std::string& path() const {
        return path_;
    }

private:
    std::string path_;
};

std::shared_ptr<BlockTreeDiskBlockPoolConfig> makeConfig(const std::string& path, size_t usable_bytes = 3 * 4096) {
    auto config = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    config->pool_type = BlockPoolType::DISK;
    config->pool_name = "complete";
    config->work_dir = path;
    config->local_rank = 0;
    config->world_rank = 0;
    // The tree pool reserves physical block zero. Keep the old usable capacity.
    config->disk_size_bytes = usable_bytes + 4096;
    config->payload_bytes = 1024;
    config->stride_bytes = 4096;
    config->buffered_io = true;
    return config;
}

}  // namespace

TEST(DiskBlockPoolTest, InitPreallocatesFileAndCleansStaleFiles) {
    TempDir temp_dir;
    ASSERT_FALSE(temp_dir.path().empty());

    const auto work_dir = temp_dir.path() + "/rtp_llm_disk_kv";
    ASSERT_EQ(::mkdir(work_dir.c_str(), 0755), 0);
    const auto stale = work_dir + "/rank_stale.kv";
    int        fd    = ::open(stale.c_str(), O_CREAT | O_WRONLY, 0600);
    ASSERT_GE(fd, 0);
    ::close(fd);
    ASSERT_EQ(::access(stale.c_str(), F_OK), 0);

    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));

    BlockTreeDiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());
    EXPECT_EQ(::access(stale.c_str(), F_OK), -1);
    EXPECT_EQ(::access(pool.filePath().c_str(), F_OK), 0);
    EXPECT_NE(pool.filePath().find("disk_block_pool_complete_r0_l0.bin"), std::string::npos);
    EXPECT_EQ(pool.totalBlocksNum(), 3u);
    EXPECT_EQ(pool.freeBlocksNum(), 3u);
}

TEST(DiskBlockPoolTest, InitFailsWhenMountPathDoesNotExist) {
    TempDir temp_dir;
    ASSERT_FALSE(temp_dir.path().empty());

    BlockTreeDiskMountGuard guard;
    EXPECT_FALSE(guard.init(temp_dir.path() + "/missing_mount"));
}

TEST(DiskBlockPoolTest, MountGuardAllowsTwoPoolsOnSameMountWithoutDeletingFirst) {
    TempDir temp_dir;
    ASSERT_FALSE(temp_dir.path().empty());

    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));

    BlockTreeDiskBlockPool complete_pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(complete_pool.init());
    ASSERT_EQ(::access(complete_pool.filePath().c_str(), F_OK), 0);

    auto incomplete_cfg       = makeConfig(guard.workDir(), 6 * 4096);
    incomplete_cfg->pool_name = "incomplete";
    incomplete_cfg->local_rank = 0;
    incomplete_cfg->world_rank = 0;
    BlockTreeDiskBlockPool incomplete_pool(incomplete_cfg);
    ASSERT_TRUE(incomplete_pool.init());

    EXPECT_EQ(::access(complete_pool.filePath().c_str(), F_OK), 0);
    EXPECT_EQ(::access(incomplete_pool.filePath().c_str(), F_OK), 0);
    EXPECT_NE(complete_pool.filePath(), incomplete_pool.filePath());
}

TEST(DiskBlockPoolTest, ReserveCommitAbortAndFreeSlots) {
    TempDir        temp_dir;
    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    BlockTreeDiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    pool.incTreeRef(*slot, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool.freeBlocksNum(), 2u);

    pool.incTreeRef(*slot, BlockTreeRefType::CACHE);
    pool.decTreeRef(*slot, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool.freeBlocksNum(), 2u);
    EXPECT_EQ(pool.availableBlocksNum(), 3u);

    pool.decTreeRef(*slot, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool.freeBlocksNum(), 3u);
}

TEST(DiskBlockPoolTest, RequestRefPreventsReuseUntilReleased) {
    TempDir        temp_dir;
    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    BlockTreeDiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    pool.incTreeRef(*slot, BlockTreeRefType::LOAD);
    pool.incTreeRef(*slot, BlockTreeRefType::CACHE);
    pool.incTreeRef(*slot, BlockTreeRefType::LOAD);

    pool.decTreeRef(*slot, BlockTreeRefType::CACHE);
    pool.decTreeRef(*slot, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool.freeBlocksNum(), 2u);

    pool.decTreeRef(*slot, BlockTreeRefType::LOAD);
    EXPECT_EQ(pool.freeBlocksNum(), 3u);
}

TEST(DiskBlockPoolTest, ReadWriteFullSlot) {
    TempDir        temp_dir;
    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    BlockTreeDiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    pool.incTreeRef(*slot, BlockTreeRefType::LOAD);
    std::vector<unsigned char> write_buf(pool.strideBytes(), 0x5a);
    std::vector<unsigned char> read_buf(pool.strideBytes(), 0);

    ASSERT_EQ(pool.write(*slot, write_buf.data(), write_buf.size()), BlockIOStatus::OK);
    ASSERT_EQ(pool.read(*slot, read_buf.data(), read_buf.size()), BlockIOStatus::OK);
    EXPECT_EQ(read_buf, write_buf);
    EXPECT_EQ(pool.writeBytes(), write_buf.size());
    EXPECT_EQ(pool.readBytes(), read_buf.size());
}

TEST(DiskBlockPoolTest, FullPoolReturnsNullopt) {
    TempDir        temp_dir;
    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    BlockTreeDiskBlockPool pool(makeConfig(guard.workDir(), 2 * 4096));
    ASSERT_TRUE(pool.init());
    ASSERT_TRUE(pool.malloc().has_value());
    ASSERT_TRUE(pool.malloc().has_value());
    EXPECT_FALSE(pool.malloc().has_value());
}

}  // namespace rtp_llm::test
