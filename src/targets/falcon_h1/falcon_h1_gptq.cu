#include "falcon_h1_gptq.h"
#include "artifact/v3_reader.h"
#include "core/device.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
namespace ninfer::targets::falcon_h1 {
GptqExecution::~GptqExecution() { reset(); }
void GptqExecution::stage(const BoundModel& model, artifact::MaterializedArtifact& mat) {
    if(owner_ && (lifetime_.expired() || owner_!=&mat)) reset();
    if(owner_==&mat && generation_!=mat.device_allocation_generation()) views_.clear();
    if(owner_!=&mat) { owner_=&mat; lifetime_=mat.lifetime_token(); mat.register_device_cleanup(this,[this]{views_.clear();owner_=nullptr;generation_=0;lifetime_.reset();}); }
    generation_=mat.device_allocation_generation(); views_.clear();
    if(model.gptq_permutations.size()!=model.gptq_geometry.size()||model.gptq_host_permutations.size()!=model.gptq_permutations.size()) throw std::logic_error("GPTQ binder geometry/permutation inventory mismatch");
    for (const auto& [name, stage] : model.gptq_permutations) {
        const std::string weight_name=name;
        if(weight_name.find("/ffn/")==std::string::npos) throw std::logic_error("invalid GPTQ permutation binding");
        const auto geo_it=model.gptq_geometry.find(weight_name);if(geo_it==model.gptq_geometry.end())throw std::logic_error("GPTQ geometry missing for "+weight_name);const auto geo=geo_it->second; const std::size_t expected=static_cast<std::size_t>(geo[1])*sizeof(std::int32_t);
        if(stage.bytes!=expected) throw std::logic_error("GPTQ canonical permutation byte extent mismatch");
        const auto& bound_perm=model.tensors.at(weight_name+"/input_perm");
        if(bound_perm.handle.index!=stage.canonical.index) throw std::logic_error("GPTQ canonical input_perm binding mismatch");
        auto* canonical=static_cast<const std::int32_t*>(mat.device_data(stage.canonical));
        auto* device_perm=static_cast<std::int32_t*>(mat.device_allocate(stage.bytes,256));
        CUDA_CHECK(cudaMemcpy(device_perm,canonical,stage.bytes,cudaMemcpyDeviceToDevice));
        const auto& host_perm=model.gptq_host_permutations.at(weight_name);
        std::vector<std::int32_t> verify(host_perm.size());
        CUDA_CHECK(cudaMemcpy(verify.data(),device_perm,stage.bytes,cudaMemcpyDeviceToHost));
        if(verify!=host_perm) throw std::logic_error("GPTQ staged device permutation differs from canonical I32 artifact data");
        const auto& w=model.tensors.at(weight_name); auto* base=static_cast<const std::byte*>(mat.device_data(w.handle));
        const auto g=artifact::v3::gptq_g128_geometry(geo[1],geo[0]);
        views_.emplace(weight_name,ops::detail::GptqG128View{reinterpret_cast<const std::uint32_t*>(base),reinterpret_cast<const std::uint32_t*>(base+g.qzeros_offset),reinterpret_cast<const std::uint16_t*>(base+g.scales_offset),device_perm,static_cast<std::int32_t>(geo[0]),static_cast<std::int32_t>(geo[1]),static_cast<std::int32_t>(geo[2]),g.qzeros_offset,g.scales_offset});
    }
}
void GptqExecution::reset() { auto* old=owner_; views_.clear(); owner_=nullptr; generation_=0; lifetime_.reset(); if(old) old->unregister_device_cleanup(this); }
void GptqExecution::linear(const std::string& name,const float* x,float* y,std::int32_t m,cudaStream_t stream) const {
    if(!owner_ || lifetime_.expired() || generation_!=owner_->device_allocation_generation()) throw std::logic_error("GPTQ staged pointer lifetime has expired");
    const auto it=views_.find(name); if(it==views_.end()) throw std::out_of_range("GPTQ projection was not staged");
    ops::detail::gptq_g128_linear_f32(x,it->second,y,m,stream);
}
}
