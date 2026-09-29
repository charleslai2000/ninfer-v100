// BSD-3-Clause donor-derived D128 paged-prefill core; see donor-source-map.md
// and LICENSE. Donor: 1CatAI/1Cat-vLLM fcf59f8e9ae50c186333e98e5cf6aae705f320de,
// flash-attention-v100/kernel/fused_mha_forward_paged.cu (D128 BM32/BN176 path).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
namespace ninfer::targets::falcon_h1::donor_prefill {
using namespace nvcuda::wmma;
constexpr int D=128,QH=12,KVH=2,BM=32,BN=176,WARPS=16,THREADS=512;
constexpr int QSTR=128,KVSTR=128,SSTR=176,OSTR=128;
struct alignas(128) SmemLayout {
  alignas(16) __half q[BM*QSTR];
  union { alignas(16) __half k[BN*KVSTR]; alignas(16) __half v[BN*KVSTR]; } reuse_kv;
  union { alignas(16) float s[BM*SSTR]; alignas(16) __half p[BM*SSTR]; } reuse_sp;
  alignas(32) float o[BM*OSTR];
  alignas(16) float row_max[BM],row_sum[BM];
};
static_assert(sizeof(SmemLayout)<98304,"D128 donor shared memory must fit Volta");
__device__ __forceinline__ int64_t offset(int page,int head,int in_page,int d){return int64_t(D)*64*(head+KVH*page)+int64_t(D)*in_page+d;}

__global__ __launch_bounds__(THREADS,2) void prefill_d128(
 const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
 const __half* __restrict__ V,__nv_bfloat16* __restrict__ Out,
 const int* __restrict__ table,int T,int past_length,float softmax_scale){
 extern __shared__ __align__(128) unsigned char raw[];auto& sh=*reinterpret_cast<SmemLayout*>(raw);
 const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,qh=blockIdx.y,qrow=blockIdx.x*BM,kvh=qh/6;
 const int valid_q=min(BM,T-qrow),visible_kv_length=past_length+T;
 for(int x=tid;x<valid_q*D;x+=THREADS){const int r=x/D,d=x%D;sh.q[r*QSTR+d]=__float2half_rn(__bfloat162float(Q[(qrow+r)*QH*D+qh*D+d]));}
 if(tid<BM){sh.row_max[tid]=-1.e30f;sh.row_sum[tid]=0.f;}
 for(int x=tid;x<BM*D;x+=THREADS)sh.o[(x/D)*OSTR+x%D]=0.f;
 __syncthreads();
 const int ntiles=(visible_kv_length+BN-1)/BN;
 for(int block_n=0;block_n<ntiles;++block_n){
  const int start=block_n*BN,valid_k=min(BN,visible_kv_length-start);
  for(int x=tid;x<valid_k*D;x+=THREADS){const int r=x/D,d=x%D,pos=start+r;sh.reuse_kv.k[r*KVSTR+d]=__float2half_rn(__bfloat162float(K[offset(table[pos>>6],kvh,pos&63,d)]));}
  __syncthreads();
  const int tiles_m=(valid_q+15)/16,tiles_n=(valid_k+15)/16,total=tiles_m*tiles_n;
  const int tiles_per_warp=(total+WARPS-1)/WARPS;
  const unsigned row_causal=(lane&1)+((lane>>2)&1)*8+((lane>>4)&1)*4;
  const unsigned col_causal=((lane>>1)&1)*2+((lane>>3)&1)*8;
  for(int ti=0;ti<tiles_per_warp;++ti){const int g=warp*tiles_per_warp+ti;if(g>=total)break;const int tm=(g/tiles_n)*16,tn=(g%tiles_n)*16;
   fragment<matrix_a,16,16,16,half,row_major>a;fragment<matrix_b,16,16,16,half,col_major>b;fragment<accumulator,16,16,16,float>acc;fill_fragment(acc,0.f);
#pragma unroll
   for(int kt=0;kt<D/16;++kt){const int ko=kt*16;load_matrix_sync(a,sh.q+tm*QSTR+ko,QSTR);load_matrix_sync(b,sh.reuse_kv.k+tn*KVSTR+ko,KVSTR);mma_sync(acc,a,b,acc);}
#pragma unroll
   for(int i=0;i<acc.num_elements;++i){const unsigned c=col_causal+(i&1)+((i>>2)&1)*4,r=row_causal+((i>>1)&1)*2;const int gm=qrow+tm+r,gn=start+tn+c;const bool valid=gm<qrow+valid_q&&gn<start+valid_k&&gn<=past_length+gm;acc.x[i]=valid?acc.x[i]*softmax_scale:-1.e30f;}
   store_matrix_sync(sh.reuse_sp.s+tm*SSTR+tn,acc,SSTR,mem_row_major);
  }
  __syncthreads();
  const int pv_m=(valid_q+15)/16,pv_d=D/16,total_pv=pv_m*pv_d,pv_per_warp=(total_pv+WARPS-1)/WARPS;
  for(int sub_start=0;sub_start<valid_k;sub_start+=BN){const int sub_valid=min(BN,valid_k-sub_start);const int row=tid/16,thread_row=tid%16;__half* p_row=sh.reuse_sp.p+row*SSTR;const int vec_cols=sub_valid>>2,vecs=(vec_cols+15)/16,tail=vec_cols<<2;float exp_diff=1.f;__half2 half_buffer[20];int tail_col=-1;__half tail_value=__float2half(0.f);
   if(tid<valid_q*16){const unsigned mask=valid_q==BM?0xffffffffu:__activemask();const int leader=__ffs(mask)-1;float* score=sh.reuse_sp.s+row*SSTR+sub_start;float thread_max=-1.e30f;float4*score4=reinterpret_cast<float4*>(score);
#pragma unroll 4
    for(int j=0;j<vecs;++j){const int vc=thread_row+j*16;if(vc<vec_cols){const float4 z=score4[vc];thread_max=fmaxf(thread_max,fmaxf(fmaxf(z.x,z.y),fmaxf(z.z,z.w)));}}
    for(int c=tail+thread_row;c<sub_valid;c+=16)thread_max=fmaxf(thread_max,score[c]);
#pragma unroll
    for(int off=8;off>0;off>>=1)thread_max=fmaxf(thread_max,__shfl_down_sync(mask,thread_max,off,16));
    const float row_max=__shfl_sync(mask,thread_max,leader,16),old_max=sh.row_max[row],new_max=fmaxf(old_max,row_max);exp_diff=__expf(old_max-new_max);float thread_sum=0.f;int vc_base=thread_row,hidx=0;
#pragma unroll 4
    for(int j=0;j<vecs;++j,vc_base+=16)if(vc_base<vec_cols){const float4 z=score4[vc_base];const float e0=__expf(fmaxf(z.x-new_max,-80.f)),e1=__expf(fmaxf(z.y-new_max,-80.f)),e2=__expf(fmaxf(z.z-new_max,-80.f)),e3=__expf(fmaxf(z.w-new_max,-80.f));thread_sum+=(e0+e1)+(e2+e3);half_buffer[hidx++]=__float22half2_rn(make_float2(e0,e1));half_buffer[hidx++]=__float22half2_rn(make_float2(e2,e3));}
    for(int c=tail+thread_row;c<sub_valid;c+=16){const float e=__expf(fmaxf(score[c]-new_max,-80.f));thread_sum+=e;tail_col=c;tail_value=__float2half_rn(e);}
#pragma unroll
    for(int off=8;off>0;off>>=1)thread_sum+=__shfl_down_sync(mask,thread_sum,off,16);
    const float row_sum=__shfl_sync(mask,thread_sum,leader,16);if(thread_row==0){sh.row_sum[row]=exp_diff*sh.row_sum[row]+row_sum;sh.row_max[row]=new_max;}
   }
   // P0011 donor race fix: score-read/reduce phase ends before aliased P stores.
   __syncthreads();
   if(tid<valid_q*16){int hidx=0,vc_base=thread_row;
#pragma unroll 4
    for(int j=0;j<vecs;++j,vc_base+=16)if(vc_base<vec_cols){const int base=vc_base*4;reinterpret_cast<__half2*>(p_row+base)[0]=half_buffer[hidx++];reinterpret_cast<__half2*>(p_row+base)[1]=half_buffer[hidx++];}
    if(tail_col>=0)p_row[tail_col]=tail_value;for(int c=tail+thread_row;c<BN;c+=16)if(c>=sub_valid)p_row[c]=__float2half(0.f);
    if(block_n>0||sub_start>0){float4* out4=reinterpret_cast<float4*>(sh.o+row*OSTR);for(int x=thread_row;x<D/4;x+=16){float4 z=out4[x];z.x*=exp_diff;z.y*=exp_diff;z.z*=exp_diff;z.w*=exp_diff;out4[x]=z;}}
   }
   __syncthreads();
   const int nk=(BN+15)/16;
   for(int ti=0;ti<pv_per_warp;++ti){const int g=warp*pv_per_warp+ti;if(g>=total_pv)break;const int tm=(g/pv_d)*16,td=(g%pv_d)*16;if(tm>=valid_q)continue;fragment<matrix_a,16,16,16,half,row_major>a;fragment<matrix_b,16,16,16,half,row_major>b;fragment<accumulator,16,16,16,float>acc;load_matrix_sync(acc,sh.o+tm*OSTR+td,OSTR,mem_row_major);
#pragma unroll
    for(int kt=0;kt<nk;++kt){const int ko=kt*16;if(ko>=sub_valid)break;const int pos=start+sub_start+ko,phys=table[pos>>6],off=pos&63;load_matrix_sync(a,sh.reuse_sp.p+tm*SSTR+ko,SSTR);const __half* vtile=V+int64_t(D)*64*(kvh+KVH*phys)+int64_t(D)*off+td;load_matrix_sync(b,vtile,D);mma_sync(acc,a,b,acc);}
    store_matrix_sync(sh.o+tm*OSTR+td,acc,OSTR,mem_row_major);
   }
   __syncthreads();
  }
 }
 for(int x=tid;x<valid_q*D;x+=THREADS){const int row=x/D,d=x%D;const float inv=1.f/fmaxf(sh.row_sum[row],1.e-24f);Out[(qrow+row)*QH*D+qh*D+d]=__float2bfloat16_rn(sh.o[row*OSTR+d]*inv);}
}
} // namespace ninfer::targets::falcon_h1::donor_prefill
