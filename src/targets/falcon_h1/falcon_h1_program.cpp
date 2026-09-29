#include "falcon_h1_program.h"
#include "falcon_h1_attention.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::falcon_h1 {
namespace qwen_store = ninfer::targets::qwen3_6::detail;
namespace {
std::uint64_t integer(const BoundModel& b, const char* key) {
    const auto it=b.config_integers.find(key);
    if(it==b.config_integers.end() || it->second==0)
        throw std::invalid_argument(std::string("Falcon Program missing/invalid bound config integer: ")+key);
    return it->second;
}
double scalar(const BoundModel& b, const char* key) {
    const auto it=b.config_scalars.find(key);
    if(it==b.config_scalars.end()) throw std::invalid_argument(std::string("Falcon Program missing bound scalar: ")+key);
    return it->second;
}
std::vector<double> array(const BoundModel& b,const char* key) {
    const auto it=b.config_arrays.find(key);
    if(it==b.config_arrays.end()) throw std::invalid_argument(std::string("Falcon Program missing bound array: ")+key);
    return it->second;
}
}

ModelGeometry geometry_from_artifact(const BoundModel& b, std::uint64_t context_limit) {
    ModelGeometry g;
    g.hidden_size=integer(b,"hidden_size"); g.intermediate_size=integer(b,"intermediate_size");
    g.vocab_size=integer(b,"vocab_size"); g.layer_count=integer(b,"num_hidden_layers");
    g.attention_heads=integer(b,"num_attention_heads"); g.kv_heads=integer(b,"num_key_value_heads");
    g.head_dim=integer(b,"head_dim"); g.conv_width=integer(b,"mamba_d_conv");
    g.ssm_heads=integer(b,"mamba_n_heads"); g.ssm_state=integer(b,"mamba_d_state");
    g.ssm_chunk=integer(b,"mamba_chunk_size"); g.ssm_groups=integer(b,"mamba_n_groups");
    g.ssm_head_dim=integer(b,"mamba_d_head");
    g.recurrent_conv_channels=2*integer(b,"mamba_d_ssm")+2*integer(b,"mamba_n_groups")*integer(b,"mamba_d_state");
    g.recurrent_conv_width=g.conv_width; g.recurrent_value_heads=g.ssm_heads;
    g.recurrent_value_head_dim=g.ssm_head_dim; g.recurrent_key_head_dim=g.ssm_state;
    const auto layer_types=b.layer_types;
    if(layer_types.size()!=g.layer_count)
        throw std::invalid_argument("Falcon artifact layer type inventory is missing/incomplete");
    g.context_limit=context_limit==0 ? integer(b,"max_position_embeddings") : context_limit;
    if(context_limit>integer(b,"max_position_embeddings"))
        throw std::invalid_argument("Falcon Program context limit exceeds artifact maximum");
    g.embedding_multiplier=scalar(b,"embedding_multiplier");
    g.attention_in_multiplier=scalar(b,"attention_in_multiplier");
    g.attention_out_multiplier=scalar(b,"attention_out_multiplier");
    g.key_multiplier=scalar(b,"key_multiplier"); g.ssm_in_multiplier=scalar(b,"ssm_in_multiplier");
    g.ssm_out_multiplier=scalar(b,"ssm_out_multiplier"); g.rms_norm_eps=scalar(b,"rms_norm_eps");
    g.rope_theta=scalar(b,"rope_theta"); g.ssm_multipliers=array(b,"ssm_multipliers");
    g.mlp_multipliers=array(b,"mlp_multipliers");
    g.layer_types=layer_types;
    if(g.layer_count!=b.layer_count || g.layer_count==0 || g.layer_count>std::numeric_limits<std::uint32_t>::max() ||
       g.attention_heads*g.head_dim==0 || g.kv_heads==0 || g.ssm_heads*g.ssm_head_dim!=g.hidden_size ||
       g.recurrent_conv_channels!=2*g.ssm_heads*g.ssm_head_dim+2*g.ssm_groups*g.ssm_state ||
       g.recurrent_value_heads!=g.ssm_heads || g.recurrent_value_head_dim!=g.ssm_head_dim ||
       g.recurrent_key_head_dim!=g.ssm_state || g.ssm_multipliers.size()!=5 || g.mlp_multipliers.size()!=2)
        throw std::invalid_argument("Falcon Program artifact geometry closure failed");
    return g;
}

struct Program::Sequence {
    std::uint64_t id=0;
    qwen_store::KVAddressSpaceHandle kv;
    qwen_store::StateImageHandle state;
    std::unique_ptr<ContinuationOwner> owner;
};

Program::Program(ModelGeometry geometry, std::uint32_t kv_entitlement, std::int32_t row_count,
                 std::uint32_t sequence_capacity, KVStore& kv, StateStore& states,
                 qwen3_6::StateImageDevicePool& state_physical)
    : geometry_(std::move(geometry)), kv_entitlement_(kv_entitlement), row_count_(row_count),
      sequence_capacity_(sequence_capacity), kv_(&kv), states_(&states),
      state_physical_(&state_physical) {
    if(geometry_.layer_count!=44 || geometry_.context_limit==0 || kv_entitlement_==0 ||
       row_count_<1 || sequence_capacity_==0 || sequence_capacity_>static_cast<std::uint32_t>(row_count_))
        throw std::invalid_argument("Falcon Program construction geometry/capacity is invalid");
    if(geometry_.layer_types.size()!=geometry_.layer_count)
        throw std::invalid_argument("Falcon Program requires artifact-bound layer inventory");
    layers_.reserve(static_cast<std::size_t>(geometry_.layer_count));
    for(std::uint32_t i=0;i<geometry_.layer_count;++i) {
        if(geometry_.layer_types[i]!="falcon_h1_hybrid")
            throw std::invalid_argument("Falcon Program artifact layer type is unsupported");
        layers_.push_back({i,{LayerStage::SharedPreNorm,LayerStage::AttentionBranch,
                             LayerStage::MambaBranch,LayerStage::ResidualMerge,
                             LayerStage::PreFfnNorm,LayerStage::GptqFfn}});
    }
    sequences_.reserve(sequence_capacity_);
}
Program::~Program() {
    while(!sequences_.empty()) {
        const auto id=sequences_.back()->id;
        try { release_sequence(id); }
        catch(...) { std::terminate(); }
    }
}

std::size_t Program::attention_workspace_bytes(std::size_t tokens) const {
    return ::ninfer::targets::falcon_h1::attention_workspace_bytes(tokens);
}

std::size_t Program::configure_attention_kv_storage(LayoutBuilder& layout,
                                                       std::uint32_t physical_page_count) const {
    if (physical_page_count == 0) throw std::invalid_argument("Falcon KV page count is zero");
    const auto schema = paged_kv_storage_layout(KvCacheStorage::BFloat16, static_cast<std::int32_t>(geometry_.head_dim));
    auto pool = plan_device_kv_page_pool(layout, DeviceKVPagePoolSpec{
        .page_group_count=physical_page_count,
        .geometry={.page_tokens=kPagedKVPageSize,
                   .device_plane_order=PagedKVPlaneOrder::PageMajor,
                   .planes={{.dtype=schema.key.data_dtype,.leading_extent=schema.key.data_leading_extent,
                             .head_extent=static_cast<std::int32_t>(geometry_.kv_heads)},
                            {.dtype=schema.value.data_dtype,.leading_extent=schema.value.data_leading_extent,
                             .head_extent=static_cast<std::int32_t>(geometry_.kv_heads)}}}});
    return pool.payload_bytes();
}

void Program::run_attention(std::uint64_t sequence_id, std::uint32_t layer,
                             const BoundModel& model,
                             const artifact::MaterializedArtifact& materialized,
                             const Tensor& input, Tensor& output, DeviceSpan workspace,
                             std::size_t workspace_bytes, cudaStream_t stream) {
    auto& owner = sequence(sequence_id);
    const ContinuationDescriptor base = owner.snapshot();
    const auto tokens = static_cast<std::uint64_t>(input.ne[1]);
    if (tokens == 0 || base.position != base.kv_frontier ||
        tokens > geometry_.context_limit - base.position ||
        base.recurrent_version == std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument("Falcon Program Attention continuation extent is invalid");
    }
    auto candidate = owner.begin({.kv_frontier=base.kv_frontier+tokens,
                                  .position=base.position+tokens,
                                  .recurrent_version=base.recurrent_version+1});
    execute_attention(*this, candidate, base, base.position, layer, model, materialized,
                      input, output, workspace, workspace_bytes, stream);
    candidate.prepare();
    candidate.commit();
}

std::uint64_t Program::create_sequence(std::int32_t row) {
    if(sequences_.size()>=sequence_capacity_ || row<0 || row>=row_count_)
        throw std::runtime_error("Falcon sequence capacity/row unavailable");
    if(next_id_==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Falcon sequence id exhausted");
    auto kv=kv_->create_inactive();
    if(!kv) throw std::runtime_error("Falcon sequence KV address allocation failed");
    try {
        auto activation=kv_->prepare_activation(*kv,kv_entitlement_,row);
        kv_->commit_activation(std::move(activation));
    } catch(...) { (void)kv_->release(*kv); throw; }
    auto state=states_->reserve_reset(nullptr);
    if(!state) { if(!kv_->release_after_deactivate(*kv)) std::terminate(); throw std::runtime_error("Falcon sequence recurrent state allocation failed"); }
    try {
        const auto epoch=states_->content_epoch(*state);
        const auto id=next_id_++;
        auto sequence=std::make_unique<Sequence>(); sequence->id=id; sequence->kv=*kv; sequence->state=*state;
        ContinuationDescriptor descriptor{0,0,0,0,epoch,1};
        sequence->owner=std::make_unique<ContinuationOwner>(descriptor,*kv,*state,*kv_,*states_,
                                                            kv_entitlement_,row,row_count_>1?row+1:row);
        sequences_.push_back(std::move(sequence));
        return id;
    } catch(...) {
        if(kv_->active(*kv)) kv_->deactivate(*kv);
        (void)kv_->release(*kv); (void)states_->release(*state); throw;
    }
}

ContinuationOwner& Program::sequence(std::uint64_t id) {
    const auto it=std::find_if(sequences_.begin(),sequences_.end(),[&](const auto& s){return s->id==id;});
    if(it==sequences_.end()) throw std::out_of_range("Falcon sequence id is stale");
    return *(*it)->owner;
}
const ContinuationOwner& Program::sequence(std::uint64_t id) const {
    const auto it=std::find_if(sequences_.begin(),sequences_.end(),[&](const auto& s){return s->id==id;});
    if(it==sequences_.end()) throw std::out_of_range("Falcon sequence id is stale");
    return *(*it)->owner;
}
void Program::set_failure_injection_for_test(std::uint64_t id, std::int32_t stage) {
    sequence(id).set_failure_injection_for_test(stage);
}
FalconContinuationSnapshot Program::fork_snapshot(std::uint64_t id) const {
    return {sequence(id).snapshot()};
}
void Program::reset_sequence(std::uint64_t id,std::int32_t row) {
    auto& s=sequence(id);
    if(s.transaction_pending()) throw std::logic_error("cannot reset Falcon sequence during candidate transaction");
    const auto old_id=id;
    const auto it=std::find_if(sequences_.begin(),sequences_.end(),[&](const auto& item){return item->id==old_id;});
    if(it==sequences_.end()) throw std::out_of_range("Falcon sequence id is stale");
    auto& slot=**it;
    slot.kv=slot.owner->active_kv_handle();
    slot.state=slot.owner->active_state_handle();
    if(row<0 || row>=row_count_)
        throw std::invalid_argument("Falcon reset requires a valid execution row");
    const auto next_state=states_->reserve_reset(nullptr);
    if(!next_state) throw std::runtime_error("Falcon reset recurrent state allocation failed");
    const auto old_kv=slot.kv;
    const auto old_state=slot.state;
    auto next_kv=kv_->create_inactive();
    if(!next_kv) { (void)states_->release(*next_state); throw std::runtime_error("Falcon reset KV address allocation failed"); }
    if(kv_->active(old_kv)) kv_->deactivate(old_kv);
    if(!kv_->release(old_kv) || !states_->release(old_state))
        throw std::runtime_error("Falcon reset could not retire source sequence resources");
    try {
        auto activation=kv_->prepare_activation(*next_kv,kv_entitlement_,row);
        kv_->commit_activation(std::move(activation));
    } catch(...) {
        (void)kv_->release(*next_kv);
        (void)states_->release(*next_state);
        throw;
    }
    slot.kv=*next_kv; slot.state=*next_state;
    const auto epoch=states_->content_epoch(slot.state);
    slot.owner=std::make_unique<ContinuationOwner>(
        ContinuationDescriptor{0,0,0,0,epoch,1},slot.kv,slot.state,*kv_,*states_,
        kv_entitlement_,row,row_count_>1?row+1:row);
}
void Program::release_sequence(std::uint64_t id) {
    const auto it=std::find_if(sequences_.begin(),sequences_.end(),[&](const auto& s){return s->id==id;});
    if(it==sequences_.end()) throw std::out_of_range("Falcon sequence id is stale");
    auto& s=**it;
    s.kv=s.owner->active_kv_handle(); s.state=s.owner->active_state_handle();
    if(s.owner->transaction_pending()) throw std::logic_error("cannot release Falcon sequence during candidate transaction");
    if(!kv_->release_after_deactivate(s.kv) || !states_->release(s.state)) std::terminate();
    sequences_.erase(it);
}

} // namespace ninfer::targets::falcon_h1
