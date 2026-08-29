#include "gtest/gtest.h"
#include "torch/all.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

namespace rtp_llm {
namespace {

void fillScoreTokenIdsWithMemcpy(torch::Tensor&                    token_ids,
                                 const std::vector<torch::Tensor>& complete_token_ids,
                                 const std::vector<int64_t>&       seq_lens,
                                 int64_t                           score_len) {
    int64_t    batch_idx  = 0;
    auto*      dst        = token_ids.data_ptr<int32_t>();
    const auto dst_stride = token_ids.size(1);
    for (size_t stream_idx = 0; stream_idx < complete_token_ids.size(); ++stream_idx) {
        auto* src     = complete_token_ids[stream_idx].data_ptr<int32_t>();
        auto  seq_len = seq_lens[stream_idx];
        for (int64_t i = 0; i < score_len; ++i) {
            std::memcpy(dst + batch_idx * dst_stride, src, seq_len * sizeof(int32_t));
            ++batch_idx;
        }
    }
}

void fillScoreTokenIdsWithTorchCopy(torch::Tensor&                    token_ids,
                                    const std::vector<torch::Tensor>& complete_token_ids,
                                    const std::vector<int64_t>&       seq_lens,
                                    int64_t                           score_len) {
    int64_t batch_idx = 0;
    for (size_t stream_idx = 0; stream_idx < complete_token_ids.size(); ++stream_idx) {
        auto seq_len = seq_lens[stream_idx];
        token_ids.narrow(0, batch_idx, score_len)
            .narrow(1, 0, seq_len)
            .copy_(complete_token_ids[stream_idx].narrow(1, 0, seq_len).expand({score_len, seq_len}));
        batch_idx += score_len;
    }
}

template<typename Func>
double benchmarkUs(Func&& func, int iterations) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

}  // namespace

TEST(MtpBatchStreamProcessorPerfTest, ScoreTokenIdsTorchCopyVsMemcpy) {
    constexpr int64_t stream_count = 64;
    constexpr int64_t score_len    = 4;
    constexpr int64_t max_seq_len  = 65536;
    constexpr int     iterations   = 20;

    auto src_storage = torch::empty({stream_count, max_seq_len}, torch::kInt32);
    src_storage.random_(0, 32000);

    std::vector<torch::Tensor> complete_token_ids;
    std::vector<int64_t>       seq_lens;
    complete_token_ids.reserve(stream_count);
    seq_lens.reserve(stream_count);
    for (int64_t i = 0; i < stream_count; ++i) {
        complete_token_ids.push_back(src_storage.narrow(0, i, 1));
        seq_lens.push_back(max_seq_len - (i % 8) * 128);
    }

    auto pinned_i32 = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    auto dst_memcpy = torch::empty({stream_count * score_len, max_seq_len + score_len}, pinned_i32);
    auto dst_torch  = torch::empty({stream_count * score_len, max_seq_len + score_len}, pinned_i32);

    dst_memcpy.fill_(-1);
    dst_torch.fill_(-1);
    fillScoreTokenIdsWithMemcpy(dst_memcpy, complete_token_ids, seq_lens, score_len);
    fillScoreTokenIdsWithTorchCopy(dst_torch, complete_token_ids, seq_lens, score_len);
    ASSERT_TRUE(torch::equal(dst_memcpy, dst_torch));

    auto memcpy_us = benchmarkUs(
        [&]() { fillScoreTokenIdsWithMemcpy(dst_memcpy, complete_token_ids, seq_lens, score_len); }, iterations);
    auto torch_us = benchmarkUs(
        [&]() { fillScoreTokenIdsWithTorchCopy(dst_torch, complete_token_ids, seq_lens, score_len); }, iterations);

    std::cout << "[mtp-score-token-ids-copy] streams=" << stream_count << " score_len=" << score_len
              << " max_seq_len=" << max_seq_len << " iterations=" << iterations << " memcpy_us=" << memcpy_us
              << " torch_copy_us=" << torch_us << " speedup=" << (memcpy_us / torch_us) << std::endl;
}

}  // namespace rtp_llm
