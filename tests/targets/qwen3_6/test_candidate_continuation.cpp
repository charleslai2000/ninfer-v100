#include "core/device.h"
#include "core/candidate_continuation.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <ninfer/targets/qwen3_6/state_image.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

namespace {
namespace q36 = ninfer::targets::qwen3_6;
namespace store = ninfer::targets::qwen3_6::detail;

void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void real_store_transaction(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 16,
        .geometry = {.page_tokens = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                     .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                     .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}}}};
    const auto page_layout = ninfer::plan_device_kv_page_pool(builder, page_spec);
    const auto table_layout = ninfer::plan_kv_execution_tables(builder,
        {.logical_page_capacity = 8, .table_rows = 5});
    const auto state_spec = q36::StateImageSpec{
        .linear = {.layers=1, .conv_channels=8, .conv_width=3, .value_heads=2,
                   .value_head_dim=4, .key_head_dim=4, .slot_count=4,
                   .conv_dtype=ninfer::DType::BF16}, .hidden=8};
    const auto state_layout = q36::plan_state_image_device_pool(builder, state_spec);
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool tables(backing, table_layout, physical_pages);
    store::LogicalKVPageStore pages(physical_pages, 24);
    store::KVAddressSpaceStore addresses(pages, tables, 8, 8);
    q36::StateImageDevicePool state_physical(backing, state_layout);
    store::StateImageStore states(state_physical, nullptr, 8);

    const auto source_kv = addresses.create_active(4, 0);
    check(source_kv.has_value(), "source KV allocation");
    addresses.ensure_mapped_to_tokens(*source_kv, 130, device.stream);
    addresses.commit_frontier(*source_kv, 130);
    const auto source_state = states.reserve_reset(device.stream);
    check(source_state.has_value(), "source state allocation");
    device.synchronize();
    const auto initial_epoch = states.content_epoch(*source_state);
    const auto source_page0 = addresses.logical_page(*source_kv, 0);
    const auto source_page1 = addresses.logical_page(*source_kv, 1);
    const auto source_page2 = addresses.logical_page(*source_kv, 2);
    (void)source_page0;
    (void)source_page1;
    (void)source_page2;
    const auto allocated_before = physical_pages.allocated_pages();
    const auto reserved_before = physical_pages.reserved_pages();

    using Pin = store::StateImageSourcePin;
    using RollbackLease = store::KVRollbackActivationLease;
    using Fork = store::KVPrefixForkReservation;
    static_assert(std::is_move_constructible_v<Pin> && !std::is_move_assignable_v<Pin> &&
                  !std::is_copy_constructible_v<Pin>);
    static_assert(std::is_move_constructible_v<RollbackLease> && !std::is_move_assignable_v<RollbackLease> &&
                  !std::is_copy_constructible_v<RollbackLease>);
    static_assert(std::is_move_constructible_v<Fork> && !std::is_move_assignable_v<Fork> &&
                  !std::is_copy_constructible_v<Fork>);

    ninfer::ContinuationDescriptor initial{0, 130, 130, 0, initial_epoch, 1};
    ninfer::ContinuationOwner owner(initial, *source_kv, *source_state, addresses, states,
                                    4, 3, 2, device.stream);
    const auto rollback_bytes = [&] {
        const auto view = state_physical.slot_view(states.physical_slot(*source_state));
        std::vector<std::uint8_t> bytes;
        const auto conv = state_physical.linear().all_layers_view();
        for (std::uint32_t layer = 0; layer < conv.spec.layers; ++layer) {
            auto tensor = state_physical.linear().conv_slot(layer, states.physical_slot(*source_state));
            const std::size_t offset = bytes.size();
            bytes.resize(offset + tensor.bytes());
            CUDA_CHECK(cudaMemcpy(bytes.data() + offset, tensor.data, tensor.bytes(),
                                  cudaMemcpyDeviceToHost));
        }
        for (std::uint32_t layer = 0; layer < conv.spec.layers; ++layer) {
            auto tensor = state_physical.linear().recurrent_slot(layer, states.physical_slot(*source_state));
            const std::size_t offset = bytes.size();
            bytes.resize(offset + tensor.bytes());
            CUDA_CHECK(cudaMemcpy(bytes.data() + offset, tensor.data, tensor.bytes(),
                                  cudaMemcpyDeviceToHost));
        }
        return bytes;
    };
    const auto committed = owner.snapshot();
    const auto source_bytes = rollback_bytes();
    const auto original_generation = owner.snapshot().generation;
    const auto rows_free_before = tables.free_row_count();
    const auto source_growth_before = addresses.growth_reservation(*source_kv);
    const auto states_occupied_before = states.occupied();
    for (std::int32_t stage = 1; stage <= 8; ++stage) {
        owner.set_failure_injection_for_test(stage);
        bool injected = false;
        try {
            auto candidate = owner.begin({.kv_frontier=130, .position=130,
                                          .recurrent_version=initial_epoch + 1});
            if (stage >= 5) {
                candidate.execute([&](ninfer::ContinuationCandidateViews views) {
                    auto conv = state_physical.linear().conv_slot(
                        0, states.physical_slot(views.recurrent_handle));
                    CUDA_CHECK(cudaMemset(conv.data, static_cast<int>(0x40 + stage), conv.bytes()));
                });
            }
        } catch (const std::runtime_error&) { injected = true; }
        check(injected, "owner failure injection reached requested boundary");
        if (!(owner.snapshot() == committed && !owner.transaction_pending() &&
                  addresses.active(*source_kv) && addresses.bound_row(*source_kv) == 0 &&
                  addresses.growth_reservation(*source_kv) == source_growth_before &&
                  addresses.committed_frontier(*source_kv) == 130 &&
                  states.content_epoch(*source_state) == initial_epoch &&
                  rollback_bytes() == source_bytes &&
                  physical_pages.allocated_pages() == allocated_before &&
                  physical_pages.reserved_pages() == reserved_before && pages.occupied() == 3 &&
                  addresses.occupied() == 1 && states.occupied() == states_occupied_before &&
                  tables.free_row_count() == rows_free_before)) {
            std::cerr << "injection stage " << stage << " accounting: pending="
                      << owner.transaction_pending() << " active=" << addresses.active(*source_kv)
                      << " row=" << addresses.bound_row(*source_kv)
                      << " pages=" << pages.occupied() << " addresses=" << addresses.occupied()
                      << " states=" << states.occupied()
                      << " rows_free=" << tables.free_row_count()
                      << " phys_alloc=" << physical_pages.allocated_pages()
                      << " phys_reserved=" << physical_pages.reserved_pages() << '\n';
            check(false, "injected rollback restores descriptor, source data/epoch/row and pool accounting");
        }
    }
    {
        auto candidate = owner.begin({.kv_frontier=130, .position=130,
                                      .recurrent_version=initial_epoch + 1});
        const auto views = candidate.views();
        check(views.kv != *source_kv, "candidate KV identity is private");
        const auto cand_state_slot = states.physical_slot(views.recurrent_handle);
        const auto old_state_slot = states.physical_slot(*source_state);
        check(cand_state_slot != old_state_slot, "candidate state destination is private");
        // Minimal host-controlled candidate write; no model/kernel execution.
        auto conv0 = state_physical.linear().conv_slot(0, cand_state_slot);
        CUDA_CHECK(cudaMemset(conv0.data, 0x5a, conv0.bytes()));
        candidate.execute([](ninfer::ContinuationCandidateViews) {});
        check(states.content_epoch(views.recurrent_handle) > initial_epoch &&
                  states.content_epoch(*source_state) == initial_epoch,
              "store-finalized candidate version advances without changing source epoch");
        candidate.rollback();
    }
    check(owner.snapshot() == committed && owner.snapshot().generation == original_generation &&
              owner.snapshot().kv_frontier == 130 && addresses.active(*source_kv) &&
              states.content_epoch(*source_state) == initial_epoch && rollback_bytes() == source_bytes &&
              physical_pages.allocated_pages() == allocated_before,
          "owner rollback after finalize preserves descriptor, source bytes and real store accounting");

    {
        auto candidate = owner.begin({.kv_frontier=130, .position=130,
                                      .recurrent_version=initial_epoch + 2});
        candidate.rollback();
    }
    check(owner.snapshot() == committed && addresses.active(*source_kv) &&
              addresses.committed_frontier(*source_kv) == 130 &&
              states.content_epoch(*source_state) == initial_epoch &&
              physical_pages.allocated_pages() == allocated_before,
          "repeated begin/rollback restores source and accounting baseline");

    const auto precommit_descriptor = owner.snapshot();
    const auto precommit_bytes = rollback_bytes();
    const auto old_state_slot = states.physical_slot(*source_state);
    store::KVAddressSpaceHandle published_kv_handle;
    store::StateImageHandle published_state_handle;
    {
        auto candidate = owner.begin({.kv_frontier=130, .position=130,
                                      .recurrent_version=initial_epoch + 2});
        const auto views = candidate.views();
        check(source_page2.valid() && source_page0.valid() && source_page1.valid(),
              "source exposes full-page and partial-tail logical identities before private fork");
        published_kv_handle = views.kv;
        published_state_handle = views.recurrent_handle;
        auto conv0 = state_physical.linear().conv_slot(0, states.physical_slot(views.recurrent_handle));
        CUDA_CHECK(cudaMemset(conv0.data, 0x33, conv0.bytes()));
        candidate.execute([](ninfer::ContinuationCandidateViews) {});
        check(owner.snapshot() == precommit_descriptor,
              "reader remains on old descriptor after both private store prepares");
        check(states.content_epoch(views.recurrent_handle) > precommit_descriptor.recurrent_version &&
                  states.content_epoch(*source_state) == precommit_descriptor.recurrent_version,
              "finalized recurrent candidate has store-issued version; source unchanged");
        candidate.commit();
    }
    const auto published = owner.snapshot();
    const auto read_before = precommit_descriptor;
    check(read_before == committed && read_before.generation == 1 &&
              read_before.kv_frontier == 130 && read_before.position == 130 &&
              read_before.recurrent_version == initial_epoch,
          "pre-publish reader sees complete old descriptor generation");
    check(published.generation == precommit_descriptor.generation + 1 &&
              published.kv_handle != precommit_descriptor.kv_handle &&
              published.recurrent_handle != precommit_descriptor.recurrent_handle &&
              published.recurrent_version > precommit_descriptor.recurrent_version &&
              published.kv_frontier == 130 && published.position == 130,
          "single descriptor publication exposes complete new generation");
    check(!addresses.valid(*source_kv) && !states.valid(*source_state),
          "old KV and recurrent source retire only after descriptor publication");
    const auto read_after = owner.snapshot();
    check(read_after == published && read_after.generation == 2 &&
              read_after.kv_frontier == 130 && read_after.position == 130 &&
              read_after.recurrent_version == states.content_epoch(published_state_handle),
          "post-publish reader sees complete new descriptor generation");
    check(addresses.active(published_kv_handle) &&
              addresses.committed_frontier(published_kv_handle) == published.kv_frontier &&
              states.content_epoch(published_state_handle) == published.recurrent_version,
          "published descriptor resolves the finalized candidate stores");
    check(states.valid(published_state_handle) &&
              states.content_epoch(published_state_handle) == published.recurrent_version &&
              states.physical_slot(published_state_handle) != old_state_slot,
          "published recurrent candidate handle/version is committed");
    // Stress the actual mutex-backed snapshot/publish contract. Rollbacks occur between commits;
    // reader-visible generations carry a deterministic position/frontier invariant.
    std::atomic<bool> stop_readers{false};
    std::atomic<bool> reader_failed{false};
    std::atomic<std::uint64_t> reads{0};
    std::mutex expected_mutex;
    std::vector<ninfer::ContinuationDescriptor> expected_descriptors{committed,published};
    std::vector<std::thread> readers;
    for (int reader = 0; reader < 4; ++reader) {
        readers.emplace_back([&] {
            while (!stop_readers.load(std::memory_order_acquire)) {
                const auto view = owner.snapshot();
                bool coherent=false;
                {
                    std::lock_guard lock(expected_mutex);
                    coherent=std::any_of(expected_descriptors.begin(),expected_descriptors.end(),
                        [&](const auto& expected) {
                            return expected.generation==view.generation &&
                                expected.kv_handle==view.kv_handle &&
                                expected.kv_frontier==view.kv_frontier &&
                                expected.position==view.position &&
                                expected.recurrent_handle==view.recurrent_handle &&
                                expected.recurrent_version==view.recurrent_version;
                        });
                }
                if (!coherent || view.kv_frontier != 130 || view.position < 130 ||
                    view.recurrent_version <= initial_epoch) {
                    reader_failed.store(true, std::memory_order_release);
                    break;
                }
                ++reads;
            }
        });
    }
    for (std::uint64_t generation = 3; generation <= 13; ++generation) {
        owner.set_candidate_row_for_test(static_cast<std::int32_t>(generation % 5));
        const auto before = owner.snapshot();
        try {
        {
            auto candidate = owner.begin({.kv_frontier=130, .position=before.position + 1,
                                          .recurrent_version=before.recurrent_version + 1});
            const auto next_views = candidate.views();
            candidate.execute([](ninfer::ContinuationCandidateViews) {});
            if ((generation % 2) == 0) candidate.rollback();
            else {
                const auto old=owner.snapshot();
                const auto new_version=states.content_epoch(next_views.recurrent_handle);
                {
                    std::lock_guard expected_lock(expected_mutex);
                    expected_descriptors.push_back({
                        .kv_handle=old.kv_handle, .kv_frontier=130,
                        .position=before.position+1,
                        .recurrent_handle=old.recurrent_handle,
                        .recurrent_version=new_version, .generation=old.generation+1});
                }
                candidate.commit();
                published_kv_handle = next_views.kv;
                published_state_handle = next_views.recurrent_handle;
                {
                    std::lock_guard expected_lock(expected_mutex);
                    expected_descriptors.back().kv_handle=owner.snapshot().kv_handle;
                    expected_descriptors.back().recurrent_handle=owner.snapshot().recurrent_handle;
                }
            }
        }
        {
            const auto current=owner.snapshot();
            std::lock_guard expected_lock(expected_mutex);
            if ((generation % 2) == 0)
                check(current==before,"rollback keeps reader-visible old generation");
        }
        } catch (...) {
            stop_readers.store(true, std::memory_order_release);
            for (auto& reader : readers) if (reader.joinable()) reader.join();
            throw;
        }
    }
    stop_readers.store(true, std::memory_order_release);
    for (auto& reader : readers) reader.join();
    check(!reader_failed.load(std::memory_order_acquire) && reads.load() > 0,
          "concurrent readers observe only coherent published generations");
    device.synchronize();
    const auto final_descriptor = owner.snapshot();
    check(final_descriptor.generation == 8 && final_descriptor.position == 136 &&
              addresses.active(published_kv_handle) &&
              states.content_epoch(published_state_handle) == final_descriptor.recurrent_version,
          "concurrent commit/rollback sequence converges to coherent final generation");
    addresses.deactivate(published_kv_handle);
    check(addresses.release(published_kv_handle) && states.release(published_state_handle),
          "published sequence reset/release lifecycle closes");
    check(pages.occupied() == 0 && physical_pages.allocated_pages() == 0 &&
              physical_pages.reserved_pages() == 0 && states.occupied() == 0,
          "real stores return to zero ownership");
}
} // namespace

int main() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || count == 0) return 77;
    try {
        ninfer::DeviceContext device(0);
        real_store_transaction(device);
        device.synchronize();
        std::cout << "real candidate store transaction checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "real candidate store transaction failed: " << error.what() << '\n';
        return 1;
    }
}
