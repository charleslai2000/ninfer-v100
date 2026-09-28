#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace ninfer::ops::detail::falcon_mamba_ssd {
inline constexpr int kChunkSize=256,kHeads=24,kHeadDim=128,kStateDim=256;
std::size_t workspace_bytes(int tokens);
void launch(const float* x,const float* b,const float* c,const float* dt,const float* a,const float* d,const float* state_in,float* raw,float* state_out,int tokens,void* workspace,std::size_t workspace_size,cudaStream_t stream);
void launch_coordinate_trace(const float* x,const float* b,const float* dt,const float* a,const float* state_in,float* trace,int tokens,cudaStream_t stream);
}
