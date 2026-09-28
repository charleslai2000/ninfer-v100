#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
namespace ninfer::ops::detail {
struct GptqG128View {
 const std::uint32_t* qweight=nullptr;
 const std::uint32_t* qzeros=nullptr;
 const std::uint16_t* scales=nullptr;
 const std::int32_t* input_perm=nullptr;
 std::int32_t n=0,k=0,groups=0;
 std::uint64_t qzeros_offset=0,scales_offset=0;
};
void gptq_g128_linear_f32(const float* x,const GptqG128View& w,float* y,std::int32_t m,cudaStream_t stream);
}
