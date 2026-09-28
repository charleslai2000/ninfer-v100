#pragma once

#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>
#include <stdexcept>
#include <utility>

namespace ninfer {

struct ContinuationDescriptor {
    std::uint64_t kv_handle = 0;
    std::uint64_t kv_frontier = 0;
    std::uint64_t position = 0;
    std::uint64_t recurrent_handle = 0;
    std::uint64_t recurrent_version = 0;
    std::uint64_t generation = 0;
    friend bool operator==(const ContinuationDescriptor&, const ContinuationDescriptor&) = default;
};

struct ContinuationCandidateSpec {
    std::uint64_t kv_frontier = 0;
    std::uint64_t position = 0;
    std::uint64_t recurrent_version = 0;
};

struct ContinuationCandidateViews {
    targets::qwen3_6::detail::KVAddressSpaceHandle kv;
    targets::qwen3_6::detail::StateImageSelectors recurrent;
    targets::qwen3_6::detail::StateImageHandle recurrent_handle;
};

class ContinuationCandidate;

// One logical sequence owner; its committed descriptor is the sole reader-visible mapping to
// separately stored KV and recurrent images. Readers must capture snapshot() once per execution.
class ContinuationOwner {
public:
    using KVStore = targets::qwen3_6::detail::KVAddressSpaceStore;
    using KVHandle = targets::qwen3_6::detail::KVAddressSpaceHandle;
    using StateStore = targets::qwen3_6::detail::StateImageStore;
    using StateHandle = targets::qwen3_6::detail::StateImageHandle;

    ContinuationOwner(ContinuationDescriptor initial, KVHandle kv, StateHandle state,
                      KVStore& kv_store, StateStore& state_store, std::uint32_t kv_entitlement,
                      std::int32_t rollback_row, std::int32_t candidate_row,
                      cudaStream_t stream = nullptr);
    ContinuationOwner(const ContinuationOwner&) = delete;
    ContinuationOwner& operator=(const ContinuationOwner&) = delete;

    [[nodiscard]] ContinuationDescriptor snapshot() const;
    [[nodiscard]] bool transaction_pending() const;
    [[nodiscard]] KVHandle active_kv_handle() const { return committed_kv(); }
    [[nodiscard]] StateHandle active_state_handle() const { return committed_state(); }
    [[nodiscard]] ContinuationCandidate begin(ContinuationCandidateSpec spec);

    // Test seam uses the real owner API and stores; set only while no transaction is pending.
    void set_failure_injection_for_test(std::int32_t stage);
    void set_candidate_row_for_test(std::int32_t row);

private:
    friend class ContinuationCandidate;
    [[nodiscard]] std::uint64_t kv_identity(KVHandle handle) const noexcept;
    [[nodiscard]] std::uint64_t state_identity(StateHandle handle) const noexcept;
    [[nodiscard]] KVHandle committed_kv() const;
    [[nodiscard]] StateHandle committed_state() const;
    void publish(std::uint64_t base_generation, ContinuationDescriptor next,
                 KVHandle kv, StateHandle state) noexcept;
    void abort(std::uint64_t base_generation) noexcept;

    mutable std::mutex mutex_;
    ContinuationDescriptor committed_;
    KVHandle committed_kv_;
    StateHandle committed_state_;
    KVStore* kv_store_;
    StateStore* state_store_;
    std::uint32_t kv_entitlement_;
    std::int32_t rollback_row_;
    std::int32_t candidate_row_;
    cudaStream_t stream_;
    bool pending_ = false;
    std::int32_t failure_stage_for_test_ = 0;
};

class ContinuationCandidate {
public:
    ContinuationCandidate() noexcept = default;
    ~ContinuationCandidate();
    ContinuationCandidate(ContinuationCandidate&& other) noexcept;
    ContinuationCandidate& operator=(ContinuationCandidate&&) = delete;
    ContinuationCandidate(const ContinuationCandidate&) = delete;
    ContinuationCandidate& operator=(const ContinuationCandidate&) = delete;

    [[nodiscard]] ContinuationCandidateViews views() const;
    void execute(const std::function<void(ContinuationCandidateViews)>& work);
    void prepare();
    void commit();
    void rollback();

private:
    friend class ContinuationOwner;
    using KVStore = ContinuationOwner::KVStore;
    using KVHandle = ContinuationOwner::KVHandle;
    using StateStore = ContinuationOwner::StateStore;
    using StateHandle = ContinuationOwner::StateHandle;

    explicit ContinuationCandidate(ContinuationOwner& owner, ContinuationDescriptor base,
                                   ContinuationCandidateSpec spec);
    void rollback_noexcept() noexcept;

    ContinuationOwner* owner_ = nullptr;
    ContinuationDescriptor base_;
    ContinuationCandidateSpec spec_;
    KVHandle kv_source_;
    KVHandle kv_candidate_;
    StateHandle state_source_;
    StateHandle state_candidate_;
    std::optional<targets::qwen3_6::detail::StateImageSourcePin> source_pin_;
    std::optional<targets::qwen3_6::detail::KVRollbackActivationLease> rollback_lease_;
    std::optional<targets::qwen3_6::detail::KVPrefixForkReservation> kv_fork_;
    bool source_quiesced_ = false;
    bool state_fork_started_ = false;
    bool prepared_ = false;
    bool executed_ = false;
    std::uint64_t finalized_state_version_ = 0;
    bool published_ = false;
    std::int32_t fail_during_execute_for_test_ = 0;
};

} // namespace ninfer
