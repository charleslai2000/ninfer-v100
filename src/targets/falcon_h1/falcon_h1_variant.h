#pragma once

#include "falcon_h1_binder.h"
#include "falcon_h1_program.h"

namespace ninfer::targets::falcon_h1 {
struct Variant {
    [[nodiscard]] static ModelGeometry resolve_geometry(const BoundModel& bound,
                                                         std::uint64_t context_limit) {
        return geometry_from_artifact(bound,context_limit);
    }
    [[nodiscard]] static Program make_program(const BoundModel& bound,
        std::uint64_t context_limit, std::uint32_t kv_entitlement, std::int32_t row_count,
        std::uint32_t sequence_capacity, Program::KVStore& kv, Program::StateStore& states,
        qwen3_6::StateImageDevicePool& physical) {
        return Program(geometry_from_artifact(bound,context_limit),kv_entitlement,row_count,
                       sequence_capacity,kv,states,physical);
    }
};
} // namespace ninfer::targets::falcon_h1
