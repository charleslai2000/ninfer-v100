#pragma once
#include "falcon_h1_program.h"
#include "falcon_h1_gptq.h"
#include "falcon_h1_attention.h"
#include "falcon_h1_mamba.h"
#include <cstddef>
namespace ninfer::targets::falcon_h1 {
std::size_t layer_workspace_bytes(std::size_t tokens);
std::size_t layer_reference_workspace_bytes(std::size_t tokens);
void execute_layer_reference(Program&,std::uint64_t,std::uint32_t,const BoundModel&,const artifact::MaterializedArtifact&,GptqExecution&,const Tensor&,Tensor&,DeviceSpan,std::size_t,cudaStream_t);
void capture_layer_reference_taps(Program&,std::uint64_t,std::uint32_t,const BoundModel&,const artifact::MaterializedArtifact&,GptqExecution&,const Tensor&,Tensor&,DeviceSpan,std::size_t,cudaStream_t,const std::string& directory);
void execute_layer(Program&,std::uint64_t,std::uint32_t,const BoundModel&,const artifact::MaterializedArtifact&,GptqExecution&,const Tensor&,Tensor&,DeviceSpan,std::size_t,cudaStream_t,FalconAttentionTaps={});
}
