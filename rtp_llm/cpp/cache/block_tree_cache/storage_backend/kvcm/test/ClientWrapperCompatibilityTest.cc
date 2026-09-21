#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <memory>
#include <shared_mutex>
#include <thread>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MockKVCMClient.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace ::testing;

namespace rtp_llm::kvcm::compatibility {

class ClientWrapperTest: public ::testing::Test {
public:
    void SetUp() override {
        rtp_llm::initLogger();
        auto factory = std::make_unique<MockClientFactory>();
        mock_client_factory_ = factory.get();
        auto subscriber = std::make_unique<MockSubscriber>();
        mock_subscriber_ = subscriber.get();
        auto meta = std::make_unique<kv_cache_manager::MockMetaClient>();
        default_meta_client_ = meta.get();
        auto transfer = std::make_unique<kv_cache_manager::MockTransferClient>(std::make_shared<int>(0));
        EXPECT_CALL(*mock_client_factory_, createSubscriber(true)).WillOnce(Invoke([&](bool) {
            return std::move(subscriber);
        }));
        EXPECT_CALL(*mock_subscriber_, init(std::vector<std::string>{"vip"})).WillOnce(Return(true));
        EXPECT_CALL(*mock_subscriber_, getAddresses(_))
            .WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)));
        EXPECT_CALL(*mock_client_factory_, createMetaClient(_, _)).WillOnce(Invoke([&](const auto&, const auto&) {
            return std::move(meta);
        }));
        static const std::string storage_config = R"({"sdk_backend_configs":[]})";
        EXPECT_CALL(*default_meta_client_, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
        EXPECT_CALL(*mock_client_factory_, createTransferClient(_, _)).WillOnce(Invoke([&](const auto&, const auto&) {
            return std::move(transfer);
        }));
        auto config = std::make_shared<KVCMConfig>(
            true, "vip", 8, "instance_group", "default_instance", std::vector<std::string>{},
            std::make_shared<KVCMConfig::LocationSpecInfoMap>(),
            std::make_shared<MetaChannelConfig>(1, 1000, 100), std::make_shared<SdkWrapperConfig>(),
            std::make_shared<KVCMConfig::LocationSpecGroups>(), ModelDeployment());
        client_wrapper_ = std::make_shared<ClientWrapper>(std::move(factory));
        ASSERT_TRUE(client_wrapper_->init({{"", config}}, {kv_cache_manager::RoleType::HYBRID, &span_, "tp0_Ffull"}));
        ASSERT_TRUE(Mock::VerifyAndClearExpectations(mock_client_factory_));
        ASSERT_TRUE(Mock::VerifyAndClearExpectations(mock_subscriber_));
        ASSERT_TRUE(Mock::VerifyAndClearExpectations(default_meta_client_));
        ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    }

    void TearDown() override {
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
    }

protected:
    std::array<char, 32> registration_{};
    kv_cache_manager::RegistSpan span_{registration_.data(), registration_.size()};
    MockClientFactory* mock_client_factory_{nullptr};
    MockSubscriber* mock_subscriber_{nullptr};
    kv_cache_manager::MockMetaClient* default_meta_client_{nullptr};
    std::shared_ptr<ClientWrapper> client_wrapper_;
    inline static const std::vector<std::string> init_addresses_ = {"init_address"};
};

TEST_F(ClientWrapperTest, test_no_need_reinit) {
    EXPECT_CALL(*mock_client_factory_, createMetaClient(_, _)).Times(0);
    EXPECT_CALL(*mock_subscriber_, getAddresses(_)).WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("default_trace"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "default_trace", "", {}, {}));
}

TEST_F(ClientWrapperTest, test_no_invalid_addresses) {
    EXPECT_CALL(*mock_client_factory_, createMetaClient(_, _)).Times(0);
    const std::vector<std::string> empty_addresses = {};
    EXPECT_CALL(*mock_subscriber_, getAddresses(_)).WillOnce(DoAll(SetArgReferee<0>(empty_addresses), Return(false)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(_, _, _, _)).Times(0);
    ASSERT_FALSE(client_wrapper_->finishWrite("", "", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
}

TEST_F(ClientWrapperTest, test_reinit_with_new_addresses) {
    auto new_default_meta_client     = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_default_meta_client = new_default_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, createMetaClient(_, _))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) {
            return std::move(new_default_meta_client);
        }));
    const std::vector<std::string> new_addresses = {"new_address"};
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(2)
        .WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(new_addresses), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("trace_1"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_1", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    ASSERT_EQ(default_meta_client_, client_wrapper_->meta_client_map_.at("").get());

    // reinit default instance
    EXPECT_CALL(*raw_new_default_meta_client, FinishWrite(Eq("trace_2"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_2", "", {}, {}));
    ASSERT_EQ(new_addresses, client_wrapper_->address_snapshot_);
    ASSERT_EQ(new_addresses, client_wrapper_->config_map_.at("")->addresses_);
    ASSERT_EQ(raw_new_default_meta_client, client_wrapper_->meta_client_map_.at("").get());
}

TEST_F(ClientWrapperTest, test_new_address_create_client_first_fail_second_success) {
    auto new_default_meta_client     = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_default_meta_client = new_default_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, createMetaClient(_, _))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) { return nullptr; }))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) {
            return std::move(new_default_meta_client);
        }));
    const std::vector<std::string> new_addresses = {"new_address"};
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(3)
        .WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(new_addresses), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(new_addresses), Return(true)));
    // init address
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("trace_1"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_1", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    ASSERT_EQ(default_meta_client_, client_wrapper_->meta_client_map_.at("").get());
    // first : new address, but failed to create new meta client
    ASSERT_FALSE(client_wrapper_->finishWrite("", "trace_2", "", {}, {}));
    const std::vector<std::string> empty_addresses = {};
    ASSERT_EQ(empty_addresses, client_wrapper_->address_snapshot_);
    ASSERT_EQ(empty_addresses, client_wrapper_->config_map_.at("")->addresses_);
    // second : new address, succeed to create to new meta client
    EXPECT_CALL(*raw_new_default_meta_client, FinishWrite(Eq("trace_3"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_3", "", {}, {}));
    ASSERT_EQ(new_addresses, client_wrapper_->address_snapshot_);
    ASSERT_EQ(new_addresses, client_wrapper_->config_map_.at("")->addresses_);
    ASSERT_EQ(raw_new_default_meta_client, client_wrapper_->meta_client_map_.at("").get());
}

TEST_F(ClientWrapperTest, test_registration) {
    auto new_default_meta_client     = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_default_meta_client = new_default_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, createMetaClient(_, _))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) {
            return std::move(new_default_meta_client);
        }));
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(3)
        .WillRepeatedly(DoAll(SetArgReferee<0>(init_addresses_), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("trace_1"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST));
    ASSERT_FALSE(client_wrapper_->finishWrite("", "trace_1", "", {}, {}));
    // Wait for the replacement pointer to be published under its production
    // lock. This retains the old ten-second bound without waiting forever for
    // the removed, transient rr_other_working_ flag.
    bool published = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::shared_lock<std::shared_mutex> lock(client_wrapper_->reinit_mutex_);
            published = client_wrapper_->meta_client_map_.at("").get() == raw_new_default_meta_client;
        }
        if (published) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(published);
    EXPECT_CALL(*raw_new_default_meta_client, FinishWrite(Eq("trace_2"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_2", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    ASSERT_EQ(raw_new_default_meta_client, client_wrapper_->meta_client_map_.at("").get());
}

}  // namespace rtp_llm::kvcm::compatibility
