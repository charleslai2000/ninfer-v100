#pragma once
#include "artifact/materializer.h"
#include "targets/falcon_h1/falcon_h1_binder.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace ninfer::targets::falcon_h1 {
struct MambaDecodeStateView {float* compact_conv=nullptr;float* ssm=nullptr;};
struct MambaDecodeTaps {float *in_proj=nullptr,*z=nullptr,*xbc=nullptr,*dt_split=nullptr,*conv=nullptr,*x=nullptr,*b=nullptr,*c=nullptr,*dt=nullptr,*da=nullptr,*raw_ssm=nullptr,*gated=nullptr,*gated_norm=nullptr,*out_proj=nullptr,*conv_state=nullptr,*ssm_state=nullptr;};
struct MambaDecodeWorkspace {float *projection=nullptr,*conv_output=nullptr,*x=nullptr,*b=nullptr,*c=nullptr,*dt=nullptr,*da=nullptr,*raw_ssm=nullptr,*gated=nullptr,*gated_norm=nullptr,*output=nullptr;std::size_t bytes=0;};
std::size_t mamba_decode_workspace_bytes();
MambaDecodeWorkspace mamba_decode_workspace_view(float*,std::size_t);
void mamba_decode(const BoundModel&,const artifact::MaterializedArtifact&,std::size_t,const float*,MambaDecodeStateView,MambaDecodeWorkspace,MambaDecodeTaps,cudaStream_t);
struct MambaPrefillTaps {float *in_proj=nullptr,*z=nullptr,*xbc=nullptr,*dt_split=nullptr,*conv=nullptr,*x=nullptr,*b=nullptr,*c=nullptr,*dt=nullptr,*A=nullptr,*dA=nullptr,*raw_ssm=nullptr,*gated=nullptr,*gated_norm=nullptr,*out_proj=nullptr,*conv_state=nullptr,*ssm_state=nullptr;};
struct MambaPrefillWorkspace {float *projection=nullptr,*conv_output=nullptr,*x=nullptr,*b=nullptr,*c=nullptr,*b_heads=nullptr,*c_heads=nullptr,*dt=nullptr,*a=nullptr,*decays=nullptr,*dA=nullptr,*raw_ssm=nullptr,*gated=nullptr,*gated_norm=nullptr,*output=nullptr,*d=nullptr,*ssm_out=nullptr;void *ssd=nullptr;std::size_t bytes=0,tokens=0;};
std::size_t mamba_prefill_workspace_bytes(std::size_t);
MambaPrefillWorkspace mamba_prefill_workspace_view(float*,std::size_t,std::size_t);
void mamba_prefill(const BoundModel&,const artifact::MaterializedArtifact&,std::size_t,const float*,std::size_t,MambaDecodeStateView,MambaPrefillWorkspace,MambaPrefillTaps,cudaStream_t);
}
