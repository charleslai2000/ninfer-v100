#pragma once
#include "falcon_h1_program.h"
namespace ninfer::targets::falcon_h1 {
// Per-invocation metadata only; owns no request or physical KV/state resources.
struct ExecutionContext {
    std::uint64_t sequence_id=0;
    ContinuationDescriptor committed;
    std::uint64_t context_limit=0;
    [[nodiscard]] static ExecutionContext capture(const Program& program,std::uint64_t sequence_id) {
        const auto descriptor=program.sequence(sequence_id).snapshot();
        if(descriptor.position>program.geometry().context_limit)
            throw std::out_of_range("Falcon execution position exceeds configured context limit");
        return {sequence_id,descriptor,program.geometry().context_limit};
    }
};
} // namespace ninfer::targets::falcon_h1
