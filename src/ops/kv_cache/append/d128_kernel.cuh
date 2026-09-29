#pragma once
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/math.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
template<typename Metadata>
__global__ void kv_cache_append_d128_bf16(const __nv_bfloat16* __restrict__ k,const __nv_bfloat16* __restrict__ v,const int* __restrict__ positions,Metadata metadata,__nv_bfloat16* __restrict__ cache_k,__half* __restrict__ cache_v,int width){constexpr int D=128,KVH=2,VE=8;int token=(int)blockIdx.x;int unit=(int)threadIdx.x;if(token>=metadata.valid_tokens(width))return;int head=unit/(D/VE),vec=unit%(D/VE);if(head>=KVH)return;int pos=positions[token];int pg=paged_kv_physical_page(metadata.block_table(),pos);int d=vec*VE;int64_t src=(int64_t)token*KVH*D+(int64_t)head*D+d;int64_t dst=paged_kv_element_offset<D,KVH>(pg,head,pos&63,d);int4 kb=load_vec<int4>(k+src);int4 vb=bf16x8_bits_to_f16x8_bits(load_vec<int4>(v+src));store_vec(cache_k+dst,kb);store_vec(cache_v+dst,vb);}
}
