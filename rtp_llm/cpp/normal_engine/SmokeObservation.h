#pragma once
// ISOLATED DIAGNOSTIC ONLY. Not part of the candidate/development patch.
#include <ATen/core/Dict.h>
#include <ATen/core/jit_type.h>
#include <torch/serialize.h>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <mutex>
#include <unistd.h>
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm::smoke_observation {
// Match the existing ModelInputsLogger pickle format. No mutation of inputs,
// request flags, sampler RNG, outputs, or graph enablement is performed.
using Dict = c10::impl::GenericDict;
constexpr uint64_t kBudget = 128ULL * 1024 * 1024;
constexpr uint64_t kMaxSteps = 512;
constexpr uint64_t kMaxFile = 16ULL * 1024 * 1024;
inline bool enabled() {
    static const bool value = [] {
        const char* env = std::getenv("RTP_SMOKE_CAUSAL_OBSERVATION");
        return env && std::string(env) == "1";
    }();
    return value;
}
inline std::atomic<uint64_t> bytes_written{0};
inline std::atomic<uint64_t> sequence{0};
inline std::filesystem::path directory() {
    const char* enabled = std::getenv("RTP_SMOKE_CAUSAL_OBSERVATION");
    const char* root = std::getenv("RTP_SMOKE_CAUSAL_DIR");
    if (!enabled || std::string(enabled) != "1" || !root) return {};
    auto p = std::filesystem::path(root);
    if (!p.is_absolute() || !std::filesystem::is_directory(p) || std::filesystem::is_symlink(p)) return {};
    return p;
}
// Same owned-directory/exclusive-file contract as ModelInputsLogger. No exists/open race.
class OwnedFd {
public:
    explicit OwnedFd(int fd): fd_(fd) {}
    ~OwnedFd() { if (fd_ >= 0) ::close(fd_); }
    OwnedFd(const OwnedFd&) = delete;
    OwnedFd& operator=(const OwnedFd&) = delete;
    int get() const { return fd_; }
private:
    int fd_;
};
inline bool reserveBytes(uint64_t size, uint64_t limit) {
    auto used = bytes_written.load();
    while (used <= limit && size <= limit - used) {
        if (bytes_written.compare_exchange_weak(used, used + size)) return true;
    }
    return false;
}
inline void writeExclusive(const std::filesystem::path& dir, const std::string& name,
                           const char* bytes, size_t size, size_t cap) {
    // Reserve 8 KiB for bounded errors even after tensor records reach their limit.
    const auto limit = cap <= 1024 ? kBudget : kBudget - 8 * 1024;
    if (size > cap || !reserveBytes(size, limit)) throw std::runtime_error("diagnostic byte cap");
    OwnedFd directory_fd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    struct stat st{};
    if (directory_fd.get() < 0 || ::fstat(directory_fd.get(), &st) != 0 ||
        !S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() || ::fchmod(directory_fd.get(), 0700) != 0)
        throw std::runtime_error("diagnostic directory is not owned and safe");
    OwnedFd file(::openat(directory_fd.get(), name.c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (file.get() < 0) throw std::runtime_error("exclusive diagnostic file creation failed");
    size_t offset = 0;
    while (offset < size) {
        const auto written = ::write(file.get(), bytes + offset, size - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            ::unlinkat(directory_fd.get(), name.c_str(), 0);
            throw std::runtime_error("diagnostic file write failed");
        }
        offset += static_cast<size_t>(written);
    }
}
inline std::atomic<uint64_t> error_sequence{0};
inline void error(const char* stage, const char* reason) noexcept {
    try {
        // Eight bounded records per server, also charged against the total byte cap.
        const auto number = error_sequence.fetch_add(1);
        if (number >= 8) return;
        auto dir = directory();
        if (dir.empty()) return;
        std::string message = std::string(stage).substr(0, 32) + ": " + std::string(reason).substr(0, 960) + "\n";
        writeExclusive(dir, "sampler-error-" + std::to_string(getpid()) + "-" + std::to_string(number) + ".txt",
                       message.data(), message.size(), 1024);
    } catch (...) {}  // Diagnostics never replace the original sampler result/exception.
}
struct Snapshot {
    torch::Tensor raw_logits;
    Dict payload{c10::StringType::get(), c10::AnyType::get()};
    std::filesystem::path dir;
    uint64_t number = 0;
    bool active = false;
};
inline c10::List<torch::Tensor> rngStates(const SamplerInputs& in) {
    c10::List<torch::Tensor> states;
    for (auto gen : in.generator) {
        if (!gen.defined()) {
            // The exact default CUDA generator is not guessed or advanced.
            states.push_back(torch::empty({0}, torch::kUInt8));
        } else {
            std::lock_guard<std::mutex> guard(gen.mutex());
            states.push_back(gen.get_state().clone());
        }
    }
    return states;
}
inline Snapshot begin(const SamplerInputs& in, const StreamGroups& groups) {
    Snapshot out;
    try {
        out.dir = directory();
        if (out.dir.empty()) return out;
        out.number = sequence.fetch_add(1);
        if (out.number >= kMaxSteps) {
            if (out.number == kMaxSteps) error("begin", "512-step diagnostic cap reached; evidence incomplete");
            return out;
        }
        const auto raw_size = static_cast<uint64_t>(in.logits.numel()) * in.logits.element_size();
        if (raw_size + 65536 > kMaxFile || bytes_written.load() + raw_size + 65536 > kBudget) {
            error("begin", "diagnostic byte budget reached; evidence incomplete");
            return out;
        }
        if (in.batch_size != in.batch_size_out) {
            error("begin", "unexpected beam transition; diagnostic does not reinterpret it");
            return out;
        }
        out.payload.insert("schema", int64_t(1));
        out.payload.insert("pid", int64_t(getpid()));
        out.payload.insert("rank", int64_t(0));
        out.payload.insert("sequence", int64_t(out.number));
        out.payload.insert("sampler_step_column", int64_t(in.step));
        out.payload.insert("rng_before", rngStates(in));
        c10::List<torch::Tensor> prefixes;
        for (int64_t row = 0; row < int64_t(in.batch_size); ++row) {
            const auto n = in.sequence_lengths.data_ptr<int32_t>()[row];
            prefixes.push_back(in.token_ids[row].slice(0, 0, n).clone());
        }
        out.payload.insert("valid_prefix_token_ids", prefixes);  // Never save uninitialized padding.
#define OBS_CONTROL(name) if (in.name.defined()) out.payload.insert(#name, in.name.detach().cpu().clone());
        OBS_CONTROL(input_lengths) OBS_CONTROL(sequence_lengths) OBS_CONTROL(num_beams_in)
        OBS_CONTROL(num_beams_out) OBS_CONTROL(top_k) OBS_CONTROL(top_p) OBS_CONTROL(temperature)
        OBS_CONTROL(repetition_penalty) OBS_CONTROL(presence_penalty) OBS_CONTROL(frequency_penalty)
        OBS_CONTROL(no_repeat_ngram_size) OBS_CONTROL(do_sample) OBS_CONTROL(finished_mask)
#undef OBS_CONTROL
        c10::List<int64_t> ids, row_starts, row_counts, seeds, seeds_present;
        c10::List<std::string> configs;
        int64_t row = 0;
        for (auto& stream : groups.allStreams()) {
            const auto count = stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
            ids.push_back(stream->streamId()); row_starts.push_back(row); row_counts.push_back(count);
            const auto config = stream->generateConfig();
            configs.push_back(config->debugString());
            seeds_present.push_back(config->random_seed.has_value() ? 1 : 0);
            seeds.push_back(config->random_seed.value_or(0));
            row += count;
        }
        if (row != int64_t(in.batch_size)) throw std::runtime_error("stream/sampler row mismatch");
        out.payload.insert("stream_ids", ids); out.payload.insert("row_starts", row_starts);
        out.payload.insert("row_counts", row_counts); out.payload.insert("generate_config_debug", configs);
        out.payload.insert("random_seed_present", seeds_present); out.payload.insert("random_seed", seeds);
        // Snapshot before preprocessLogits or sampling can mutate this storage.
        // This GPU clone changes allocator/timing; diagnostic runs are not acceptance.
        out.raw_logits = in.logits.detach().clone();
        out.active = true;
    } catch (const std::exception& ex) { error("begin", ex.what()); }
      catch (...) { error("begin", "unknown exception"); }
    return out;
}
inline void finish(Snapshot& snapshot, const SamplerInputs& in, const SamplerOutput& output) noexcept {
    if (!snapshot.active) return;
    try {
        // D2H only after the original sampler call. No extra CUDA graph capture,
        // no sampler invocation, no replacement RNG, no mutation of the return value.
        snapshot.payload.insert("raw_logits_before_processors", snapshot.raw_logits.cpu());
        auto after = in.logits.detach().cpu();
        auto top = after.topk(std::min<int64_t>(32, after.size(1)), -1);
        snapshot.payload.insert("post_sampler_top_values", std::get<0>(top));
        snapshot.payload.insert("post_sampler_top_ids", std::get<1>(top));
        snapshot.payload.insert("selected_token_ids", output.token_ids.select(1, in.step).detach().cpu());
        if (output.success.defined()) snapshot.payload.insert("success", output.success.detach().cpu());
        snapshot.payload.insert("rng_after", rngStates(in));
        snapshot.payload.insert("post_sampler_values_semantics", "storage after original sampler; not assumed normalized probabilities");
        auto bytes = torch::pickle_save(snapshot.payload);
        if (bytes.size() > kMaxFile || bytes.size() > kBudget - bytes_written.load()) {
            error("finish", "serialized budget reached; evidence incomplete"); return;
        }
        const auto name = "sampler-" + std::to_string(getpid()) + "-" + std::to_string(snapshot.number) + ".pt";
        writeExclusive(snapshot.dir, name, bytes.data(), bytes.size(), kMaxFile);
    } catch (const std::exception& ex) { error("finish", ex.what()); }
      catch (...) { error("finish", "unknown exception"); }
}
}  // namespace rtp_llm::smoke_observation
