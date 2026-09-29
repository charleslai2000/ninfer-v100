#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace flash_v100 {
constexpr int KV_CACHE_DTYPE_FP16=0, KV_CACHE_DTYPE_FP8_E4M3=1, KV_CACHE_DTYPE_FP8_E5M2=2;
template<int D> struct FlashV100Traits { static constexpr int BLOCK_M_16=16,BLOCK_N_16=512,BLOCK_M_32=32,BLOCK_N_32=256,BLOCK_M_64=64,BLOCK_N_64=128,BLOCK_M_128=32,BLOCK_N_128=176,BLOCK_M_256=32,BLOCK_N_256=64,WARPS_PER_BLOCK=16,THREADS_PER_WARP=32,THREADS_PER_BLOCK=512,THREADS_PER_ROW=512/BLOCK_M,PER_UINT4=8,d_stride_uint4=(D+7)/8; };
__device__ __forceinline__ float load_kv_cache_float(const void* p, int64_t i, float scale) { return __half2float(static_cast<const __half*>(p)[i])*scale; }
}
