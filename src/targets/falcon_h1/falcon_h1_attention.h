#pragma once
#include "falcon_h1_binder.h"
#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/candidate_continuation.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace ninfer::targets::falcon_h1 {
struct Program;
struct FalconAttentionTaps { void* q=nullptr; void* k_pre_scale=nullptr; void* k=nullptr; void* v=nullptr; void* post_rope_q=nullptr; void* post_rope_k=nullptr; void* candidate_k=nullptr; void* candidate_v=nullptr; void* pre_o=nullptr; void* o_projection=nullptr; void* output=nullptr; };
[[nodiscard]] std::size_t attention_workspace_bytes(std::size_t tokens);
void execute_attention(Program&,ContinuationCandidate&,const ContinuationDescriptor&,std::uint64_t,std::uint32_t,const BoundModel&,const artifact::MaterializedArtifact&,const Tensor&,Tensor&,DeviceSpan,std::size_t,cudaStream_t,FalconAttentionTaps={});
void execute_attention_candidate(Program&,ContinuationCandidate&,const ContinuationCandidateViews&,const ContinuationDescriptor&,std::uint64_t,std::uint32_t,const BoundModel&,const artifact::MaterializedArtifact&,const Tensor&,Tensor&,DeviceSpan,std::size_t,cudaStream_t,FalconAttentionTaps={});
} // namespace ninfer::targets::falcon_h1
