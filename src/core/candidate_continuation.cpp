#include "core/candidate_continuation.h"

#include <limits>

namespace ninfer {
namespace {
std::uint64_t kv_id(ContinuationOwner::KVHandle handle) noexcept {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&handle));
}
std::uint64_t state_id(ContinuationOwner::StateHandle handle) noexcept {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&handle));
}
} // namespace

ContinuationOwner::ContinuationOwner(ContinuationDescriptor initial, KVHandle kv, StateHandle state,
    KVStore& kv_store, StateStore& state_store, std::uint32_t kv_entitlement,
    std::int32_t rollback_row, std::int32_t candidate_row, cudaStream_t stream)
    : committed_(initial), committed_kv_(kv), committed_state_(state), kv_store_(&kv_store),
      state_store_(&state_store), kv_entitlement_(kv_entitlement), rollback_row_(rollback_row),
      candidate_row_(candidate_row), stream_(stream) {
    if (initial.generation == 0 || !kv_store.valid(kv) || !state_store.valid(state) ||
        !kv_store.active(kv) || kv_store.committed_frontier(kv) != initial.kv_frontier ||
        state_store.content_epoch(state) != initial.recurrent_version) {
        throw std::invalid_argument("initial descriptor does not match real stores");
    }
    committed_.kv_handle = kv_id(kv);
    committed_.recurrent_handle = state_id(state);
}

ContinuationDescriptor ContinuationOwner::snapshot() const {
    std::lock_guard lock(mutex_);
    return committed_;
}
bool ContinuationOwner::transaction_pending() const {
    std::lock_guard lock(mutex_);
    return pending_;
}
ContinuationOwner::KVHandle ContinuationOwner::committed_kv() const {
    std::lock_guard lock(mutex_); return committed_kv_;
}
ContinuationOwner::StateHandle ContinuationOwner::committed_state() const {
    std::lock_guard lock(mutex_); return committed_state_;
}

void ContinuationOwner::set_candidate_row_for_test(std::int32_t row) {
    std::lock_guard lock(mutex_);
    if (pending_) throw std::logic_error("cannot change candidate row during transaction");
    candidate_row_ = row;
}

void ContinuationOwner::set_failure_injection_for_test(std::int32_t stage) {
    std::lock_guard lock(mutex_);
    if (pending_) throw std::logic_error("cannot change failure injection during transaction");
    failure_stage_for_test_ = stage;
}

ContinuationCandidate ContinuationOwner::begin(ContinuationCandidateSpec spec) {
    ContinuationDescriptor base;
    std::int32_t fail_stage = 0;
    {
        std::lock_guard lock(mutex_);
        if (pending_) throw std::logic_error("continuation transaction already pending");
        if (spec.kv_frontier < committed_.kv_frontier || spec.position < committed_.position ||
            spec.recurrent_version <= committed_.recurrent_version ||
            committed_.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("candidate descriptor does not advance continuation");
        pending_ = true;
        base = committed_;
        fail_stage = std::exchange(failure_stage_for_test_, 0);
    }
    ContinuationCandidate tx(*this, base, spec);
    try {
        tx.state_source_ = committed_state();
        tx.kv_source_ = committed_kv();
        tx.source_pin_.emplace(state_store_->pin_candidate_source(tx.state_source_));
        if (fail_stage == 1) throw std::runtime_error("injected after recurrent source pin");
        tx.rollback_lease_.emplace(kv_store_->deactivate_with_rollback_lease(tx.kv_source_));
        tx.source_quiesced_ = true;
        if (fail_stage == 2) throw std::runtime_error("injected after KV rollback lease transfer");
        auto destination = kv_store_->create_inactive();
        if (!destination) throw std::runtime_error("candidate KV address allocation failed");
        tx.kv_candidate_ = *destination;
        if (base.kv_frontier != 0) {
            tx.kv_fork_.emplace(kv_store_->prepare_prefix_fork(
                tx.kv_source_, tx.kv_candidate_, static_cast<std::uint32_t>(base.kv_frontier),
                kv_entitlement_, candidate_row_));
        } else {
            auto activation=kv_store_->prepare_activation(tx.kv_candidate_,kv_entitlement_,candidate_row_);
            kv_store_->commit_activation(std::move(activation),stream_);
        }
        if (fail_stage == 3) throw std::runtime_error("injected after candidate KV fork");
        auto state_destination = state_store_->reserve_destination();
        if (!state_destination) throw std::runtime_error("candidate recurrent destination allocation failed");
        tx.state_candidate_ = *state_destination;
        if (fail_stage == 4) throw std::runtime_error("injected after recurrent destination reserve");
        (void)state_store_->begin_fork(tx.state_source_, tx.state_candidate_);
        tx.state_fork_started_ = true;
        tx.fail_during_execute_for_test_ = fail_stage;
        return tx;
    } catch (...) {
        tx.rollback_noexcept();
        throw;
    }
}

void ContinuationOwner::publish(std::uint64_t generation, ContinuationDescriptor next,
                                KVHandle kv, StateHandle state) noexcept {
    std::lock_guard lock(mutex_);
    if (!pending_ || committed_.generation != generation) std::terminate();
    committed_ = next;
    committed_kv_ = kv;
    committed_state_ = state;
    pending_ = false;
}
void ContinuationOwner::abort(std::uint64_t generation) noexcept {
    std::lock_guard lock(mutex_);
    if (!pending_ || committed_.generation != generation) std::terminate();
    pending_ = false;
}

ContinuationCandidate::ContinuationCandidate(ContinuationOwner& owner,
    ContinuationDescriptor base, ContinuationCandidateSpec spec)
    : owner_(&owner), base_(base), spec_(spec) {}
ContinuationCandidate::~ContinuationCandidate() { rollback_noexcept(); }
ContinuationCandidate::ContinuationCandidate(ContinuationCandidate&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), base_(other.base_), spec_(other.spec_),
      kv_source_(other.kv_source_), kv_candidate_(other.kv_candidate_),
      state_source_(other.state_source_), state_candidate_(other.state_candidate_),
      source_pin_(std::move(other.source_pin_)), rollback_lease_(std::move(other.rollback_lease_)),
      kv_fork_(std::move(other.kv_fork_)), source_quiesced_(other.source_quiesced_),
      state_fork_started_(other.state_fork_started_), prepared_(other.prepared_),
      executed_(other.executed_), finalized_state_version_(other.finalized_state_version_),
      published_(other.published_), fail_during_execute_for_test_(other.fail_during_execute_for_test_) {
    other.source_quiesced_ = false;
    other.state_fork_started_ = false;
    other.prepared_ = false;
    other.executed_ = false;
    other.published_ = true;
}

ContinuationCandidateViews ContinuationCandidate::views() const {
    if (!owner_) throw std::logic_error("candidate is not active");
    const auto selectors = owner_->state_store_->selectors(state_source_, state_candidate_);
    return {kv_candidate_, selectors, state_candidate_};
}
ContinuationOwner::KVStore& ContinuationCandidate::kv_store_for_execution() const {
    if (!owner_ || !prepared_ || executed_) throw std::logic_error("candidate KV view is outside execution lifetime");
    return *owner_->kv_store_;
}

void ContinuationCandidate::execute(const std::function<void(ContinuationCandidateViews)>& work) {
    if (!owner_ || executed_ || !work) throw std::logic_error("candidate cannot execute");
    try {
        auto& kv = *owner_->kv_store_;
        if (kv_fork_) {
            if (kv_fork_->needs_tail_copy()) {
                kv.physical_pool().copy_page(kv.prefix_fork_tail_source(*kv_fork_),
                                             kv.prefix_fork_tail_destination(*kv_fork_),
                                             owner_->stream_);
            }
            // Store-local KV finalization leaves the address private to the descriptor.
            kv.prepare_prefix_fork_candidate(std::move(*kv_fork_), owner_->stream_);
            kv_fork_.reset();
        }
        prepared_ = true;
        const auto candidate_views = ContinuationCandidateViews{
            kv_candidate_, owner_->state_store_->selectors(state_source_, state_candidate_),
            state_candidate_};
        if (fail_during_execute_for_test_ == 5)
            throw std::runtime_error("injected after candidate KV write");
        work(candidate_views);
        if (!owner_->kv_store_->valid(kv_candidate_)) throw std::logic_error("candidate KV handle changed during execution");
        if (fail_during_execute_for_test_ == 6)
            throw std::runtime_error("injected after candidate state write");
        finalized_state_version_ = owner_->state_store_->finalize_candidate_state(
            state_source_, state_candidate_);
        owner_->state_store_->prepare_fork_candidate(state_source_, state_candidate_);
        state_fork_started_ = false;
        if (fail_during_execute_for_test_ == 7 || fail_during_execute_for_test_ == 8)
            throw std::runtime_error(fail_during_execute_for_test_ == 7
                ? "injected after recurrent finalize" : "injected before descriptor publish");
        executed_ = true;
    } catch (...) {
        rollback_noexcept();
        throw;
    }
}

void ContinuationCandidate::prepare() {
    if (!owner_ || !executed_ || !prepared_ || finalized_state_version_ == 0)
        throw std::logic_error("candidate stores are not prepared");
}

void ContinuationCandidate::commit() {
    if (!owner_ || !prepared_ || !executed_) throw std::logic_error("candidate is not prepared for publication");
    ContinuationDescriptor next{
        .kv_handle = kv_id(kv_candidate_), .kv_frontier = spec_.kv_frontier,
        .position = spec_.position, .recurrent_handle = state_id(state_candidate_),
        .recurrent_version = finalized_state_version_,
        .generation = base_.generation + 1};
    if (next.kv_frontier != spec_.kv_frontier || next.position != spec_.position ||
        next.recurrent_version <= base_.recurrent_version)
        throw std::logic_error("candidate descriptor validation failed");

    // The descriptor assignment under the sequence-owner lock is the only committed publication.
    ContinuationOwner* owner = owner_;
    owner->publish(base_.generation, next, kv_candidate_, state_candidate_);
    published_ = true;
    // Old resources become reclaimable only after descriptor publication. Release the old address
    // first so its execution row is returned, then release recurrent source and lease.
    if (!owner->kv_store_->release_after_deactivate(kv_source_)) std::terminate();
    if (source_pin_) (void)source_pin_->release();
    rollback_lease_.reset();
    if (!owner->state_store_->release(state_source_)) std::terminate();
}

void ContinuationCandidate::rollback() { rollback_noexcept(); }
void ContinuationCandidate::rollback_noexcept() noexcept {
    if (!owner_ || published_) return;
    if (state_fork_started_ && owner_->state_store_->can_abort_fork(state_source_, state_candidate_)) {
        owner_->state_store_->abort_fork(state_source_, state_candidate_);
        state_fork_started_ = false;
    }
    if (state_candidate_.valid()) (void)owner_->state_store_->release(state_candidate_);
    kv_fork_.reset();
    if (kv_candidate_.valid()) {
        if (owner_->kv_store_->active(kv_candidate_)) owner_->kv_store_->deactivate(kv_candidate_);
        (void)owner_->kv_store_->release(kv_candidate_);
    }
    if (source_quiesced_) {
        owner_->kv_store_->restore_with_rollback_lease(std::move(*rollback_lease_), owner_->stream_);
        rollback_lease_.reset();
        source_quiesced_ = false;
    }
    if (source_pin_ && source_pin_->valid()) (void)source_pin_->release();
    owner_->abort(base_.generation);
    owner_ = nullptr;
}

} // namespace ninfer
