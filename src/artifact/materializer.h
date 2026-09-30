#pragma once

#include "artifact/binder.h"
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <memory>
#include <string>
#include <span>
#include <vector>

namespace ninfer::artifact {

struct MaterializationStats {
    std::uint64_t file_bytes              = 0;
    std::uint64_t h2d_bytes               = 0;
    std::uint64_t device_capacity_bytes   = 0;
    std::uint64_t retained_resource_bytes = 0;
    std::uint64_t peak_staging_bytes      = 0;
    std::size_t tensor_count              = 0;
    std::size_t resource_count            = 0;
    double upload_seconds                 = 0.0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact();
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    void* device_data(ObjectHandle handle) const;
    std::span<const std::byte> resource_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_resource_bytes(ObjectHandle handle);

    const MaterializationStats& stats() const noexcept { return stats_; }

    DeviceArena& device_arena();
    void* device_allocate(std::size_t bytes, std::size_t alignment = 256);
    void reset_device_allocations();
    [[nodiscard]] std::uint64_t device_allocation_generation() const noexcept { return device_generation_; }
    [[nodiscard]] std::weak_ptr<int> lifetime_token() const noexcept { return lifetime_; }
    void register_device_cleanup(void* owner, std::function<void()> cleanup);
    void unregister_device_cleanup(void* owner) noexcept;

private:
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                            DeviceContext&, const StartupObserver*);

    struct ObjectStorage {
        void* device = nullptr;
        std::vector<std::byte> resource;
    };

    std::unique_ptr<DeviceArena> device_arena_;
    std::unique_ptr<DeviceArena> persistent_arena_;
    std::uint64_t device_generation_ = 1;
    std::vector<std::pair<void*,std::function<void()>>> device_cleanups_;
    std::shared_ptr<int> lifetime_{std::make_shared<int>(0)};
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device,
                                 const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::artifact
