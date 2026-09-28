#include "ops/linear/gptq/gptq_g128.h"
#include "core/device.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
__global__ void gptq_scalar_kernel(const float* __restrict__ x,const std::uint32_t* __restrict__ qweight,
 const std::uint32_t* __restrict__ qzeros,const std::uint16_t* __restrict__ scales,
 const std::int32_t* __restrict__ perm,float* __restrict__ y,int n,int k,int groups) {
 const int col=blockIdx.x*blockDim.x+threadIdx.x;
 const int row=blockIdx.y;
 if(col>=n)return;
 float acc=0.0f;
 for(int kk=0;kk<k;++kk){
  const int qrow=kk>>3;const int nib=(kk&7)*4;
  const std::uint32_t packed=qweight[static_cast<std::int64_t>(qrow)*n+col];
  const int code=static_cast<int>((packed>>nib)&15u);
  const int group=kk>>7;
  const std::uint32_t zword=qzeros[static_cast<std::int64_t>(group)*(n/8)+(col>>3)];
  const int zero=static_cast<int>((zword>>((col&7)*4))&15u)+1;
  const float scale=__half2float(__ushort_as_half(scales[static_cast<std::int64_t>(group)*n+col]));
  const float weight=scale*static_cast<float>(code-zero);
  const float xv=x[static_cast<std::int64_t>(row)*k+perm[kk]];
  acc=fmaf(xv,weight,acc);
 }
 y[static_cast<std::int64_t>(row)*n+col]=acc;
}
}
void gptq_g128_linear_f32(const float*x,const GptqG128View&w,float*y,std::int32_t m,cudaStream_t stream){
 if(!x||!y||!w.qweight||!w.qzeros||!w.scales||!w.input_perm||m<=0||w.n<=0||w.k<=0||w.k%128||w.n%8)throw std::invalid_argument("gptq_g128_linear_f32: invalid view/shape");
 constexpr int threads=128;dim3 block(threads);dim3 grid((w.n+threads-1)/threads,m);
 gptq_scalar_kernel<<<grid,block,0,stream>>>(x,w.qweight,w.qzeros,w.scales,w.input_perm,y,w.n,w.k,w.groups);
 CUDA_CHECK(cudaGetLastError());
}
}
