#pragma once

#include "falcon_h1_program.h"
#include "falcon_h1_variant.h"
#include <memory>
#include <string_view>

namespace ninfer::targets::falcon_h1 {
struct Package {
    static constexpr std::string_view model_id="falcon-h1-7b";
    static constexpr std::string_view target_key="falcon_h1_7b";
    using ProgramType=Program;
    using VariantType=Variant;
    [[nodiscard]] static ProgramType make_program(const BoundModel& bound, std::uint64_t context_limit,
        std::uint32_t kv_entitlement, std::int32_t row_count, std::uint32_t sequence_capacity,
        Program::KVStore& kv, Program::StateStore& states,
        qwen3_6::StateImageDevicePool& physical) {
        return Variant::make_program(bound,context_limit,kv_entitlement,row_count,
                                     sequence_capacity,kv,states,physical);
    }
};
} // namespace ninfer::targets::falcon_h1
