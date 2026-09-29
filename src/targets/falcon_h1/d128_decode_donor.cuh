// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, D. Skryabin.
// Faithful isolated extraction/adaptation of 1CatAI/1Cat-vLLM
// flash-attention-v100/kernel/flash_decode_paged.cu at commit
// fcf59f8e9ae50c186333e98e5cf6aae705f320de. See donor LICENSE.
// Framework API, routing, workspace allocation and unrelated kernel code omitted.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace ninfer::targets::falcon_h1::donor_d128 {
constexpr int kWarpSize=32, kThreadsPerBlock=256, kWarpsPerBlock=8;
constexpr int WARP=kWarpSize, THREADS=kThreadsPerBlock, WARPS=kWarpsPerBlock, PART=256;
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for(int off=16;off>0;off>>=1) v+=__shfl_down_sync(0xffffffff,v,off);
  return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for(int off=16;off>0;off>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffff,v,off));
  return v;
}
template<int N> __device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float sh[N]; __shared__ float result;
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  val=warp_reduce_sum(val); if(lane==0) sh[warp]=val; __syncthreads();
  val=threadIdx.x<N?sh[lane]:0.f;
  if(warp==0){val=warp_reduce_sum(val);if(lane==0)result=val;}
  __syncthreads();return result;
}
template<int N> __device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float sh[N]; __shared__ float result;
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  val=warp_reduce_max(val); if(lane==0) sh[warp]=val; __syncthreads();
  val=threadIdx.x<N?sh[lane]:-1.0e20f;
  if(warp==0){val=warp_reduce_max(val);if(lane==0)result=val;}
  __syncthreads();return result;
}

// Donor dot_qk_half2 arithmetic/operation order, with only cache K's scalar
// representation adapted from __half to __nv_bfloat16. Q is already rounded
// from BF16 to donor FP16 by the wrapper; each K is converted at load.
template<int D>
__device__ __forceinline__ float dot_qk_bf16_cache(const __half* __restrict__ q,
 const __nv_bfloat16* __restrict__ k,const int64_t base,const int lane) {
  float acc=0.f;
#pragma unroll
  for(int i=lane;i<D/2;i+=kWarpSize){
    const float2 qv=__half22float2(reinterpret_cast<const __half2*>(q)[i]);
    const int64_t j=base+static_cast<int64_t>(i)*2;
    const __half k0h=__float2half_rn(__bfloat162float(k[j]));
    const __half k1h=__float2half_rn(__bfloat162float(k[j+1]));
    const float2 kv=__half22float2(__halves2half2(k0h,k1h));
    acc=fmaf(qv.x,kv.x,acc); acc=fmaf(qv.y,kv.y,acc);
  }
  return warp_reduce_sum(acc);
}

// NInfer physical planes are D-major per page/head; element address is
// D*Page*(kv_head+KVH*physical_page)+D*page_offset+d.
// NInfer tensors store K and V separately; these element-stride expressions
// describe direct persistent reads, with no temporary KV view.
__global__ void partition(
 const __nv_bfloat16* __restrict__ q_bf16,
 const __nv_bfloat16* __restrict__ k_pages,const __half* __restrict__ v_pages,
 __half* __restrict__ partial_out,float* __restrict__ max_logits,float* __restrict__ exp_sums,
 const int* __restrict__ block_table,int history,int physical_pages,
 int qh,int kvh,float scale) {
  const int head=blockIdx.x, part=blockIdx.y;
  if(head>=qh)return;
  const int part_size=256, start=part*part_size;
  const int donor_num_parts=(history+part_size-1)/part_size;
  if(start>=history)return;
  const int end=min(start+part_size,history), n=end-start;
  const int kv_head=head/(qh/kvh);
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  __shared__ __half q_shared[128];
  __shared__ float scores[256];
  __shared__ int pages[256],offs[256];
  for(int d=threadIdx.x;d<128;d+=blockDim.x)
    q_shared[d]=__float2half_rn(__bfloat162float(q_bf16[(int64_t)head*128+d]));
  for(int i=threadIdx.x;i<n;i+=blockDim.x){int tok=start+i,logical=tok>>6,off=tok&63;
    int pg=block_table[logical]; if(pg<0||pg>=physical_pages){pages[i]=-1;offs[i]=off;}else{pages[i]=pg;offs[i]=off;}}
  __syncthreads();
  float local_max=-1.0e20f;
  for(int i=warp;i<n;i+=kWarpsPerBlock){int pg=pages[i];if(pg<0)continue;
    int64_t base=(int64_t)128*64*(kv_head+kvh*pg)+(int64_t)128*offs[i];
    float s=dot_qk_bf16_cache<128>(q_shared,k_pages,base,lane);
    if(lane==0){s*=scale;scores[i]=s;local_max=fmaxf(local_max,s);}}
  float part_max=block_reduce_max<kWarpsPerBlock>(local_max);
  float local_sum=0.f;
  for(int i=threadIdx.x;i<n;i+=blockDim.x){float p=__expf(scores[i]-part_max);scores[i]=p;local_sum+=p;}
  float part_sum=block_reduce_sum<kWarpsPerBlock>(local_sum);
  float inv=part_sum>0.f?1.f/part_sum:0.f; __syncthreads();
  for(int d=threadIdx.x;d<128;d+=blockDim.x){float acc=0.f;
    for(int i=0;i<n;i++){int pg=pages[i];if(pg<0)continue;int64_t ix=(int64_t)128*64*(kv_head+kvh*pg)+(int64_t)128*offs[i]+d;
      acc=fmaf(scores[i],__half2float(v_pages[ix]),acc);}
    partial_out[((int64_t)head*gridDim.y+part)*128+d]=__float2half(acc*inv);}
  if(threadIdx.x==0){max_logits[(int64_t)head*donor_num_parts+part]=part_max;exp_sums[(int64_t)head*donor_num_parts+part]=part_sum;}
}

// Donor cross-partition reduction. Caller supplies P=ceil(history/256), so
// PARTITION_SIZE is the donor scalar reducer's fixed partition profile.
__global__ void reduce(const __half* partial,const float* maxima,
 const float* sums,__nv_bfloat16* out,int history,int partitions) {
  const int head=blockIdx.x;if(head>=12)return;extern __shared__ float sh[];
  float* ms=sh;float* ws=sh+partitions;float lm=-1.0e20f;
  for(int p=threadIdx.x;p<partitions;p+=blockDim.x){ms[p]=maxima[(int64_t)head*partitions+p];lm=fmaxf(lm,ms[p]);}
  float gm=block_reduce_max<kWarpsPerBlock>(lm);float ls=0.f;
  for(int p=threadIdx.x;p<partitions;p+=blockDim.x){ws[p]=sums[(int64_t)head*partitions+p]*__expf(ms[p]-gm);ls+=ws[p];}
  float gs=block_reduce_sum<kWarpsPerBlock>(ls);float inv=gs>0.f?1.f/gs:0.f;__syncthreads();
  for(int d=threadIdx.x;d<128;d+=blockDim.x){float a=0.f;for(int p=0;p<partitions;p++)a=fmaf(ws[p],__half2float(partial[((int64_t)head*partitions+p)*128+d]),a);const __half donor_out=__float2half(a*inv);out[(int64_t)head*128+d]=__float2bfloat16_rn(__half2float(donor_out));}
  (void)history;
}
} // namespace ninfer::targets::falcon_h1::donor_d128
