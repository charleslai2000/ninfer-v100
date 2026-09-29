#include "core/arena.h"
#include "core/candidate_continuation.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "targets/falcon_h1/falcon_h1_program.h"
#include "targets/falcon_h1/falcon_h1_package.h"
#include "targets/falcon_h1/falcon_h1_execution_context.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"
#include <ninfer/targets/qwen3_6/state_image.h>
#include <iostream>
#include <stdexcept>
namespace { void check(bool x,const char*m){if(!x)throw std::runtime_error(m);} }
int main(){try{
 using namespace ninfer;namespace q=ninfer::targets::qwen3_6;namespace f=ninfer::targets::falcon_h1;namespace d=ninfer::targets::qwen3_6::detail;
 nlohmann::json config={{"hidden_size",3072},{"intermediate_size",12288},{"vocab_size",130049},{"num_hidden_layers",44},{"num_attention_heads",12},{"num_key_value_heads",2},{"head_dim",128},{"mamba_d_conv",4},{"mamba_n_heads",24},{"mamba_d_state",256},{"mamba_chunk_size",256},{"mamba_n_groups",1},{"mamba_d_head",128},{"mamba_d_ssm",3072},{"max_position_embeddings",8192},{"model_type","falcon_h1"},{"gptq_format","gptq_q4_g128_fp16"},{"gptq_layout","gptq_canonical_k128_v1"},{"input_perm","explicit_i32_auxiliary"},{"torch_dtype","bfloat16"},{"mamba_expand",2},{"mamba_norm_before_gate",false},{"mamba_rms_norm",true},{"mamba_proj_bias",false},{"mamba_conv_bias",true},{"rms_norm_eps",1e-5},{"rope_theta",1e11},{"attention_bias",false},{"mlp_bias",false},{"projectors_bias",false},{"tie_word_embeddings",false},{"mamba_use_mlp",true},{"embedding_multiplier",5.656854249492381},{"attention_in_multiplier",1.0},{"attention_out_multiplier",0.10416666666666667},{"key_multiplier",0.030690398488999456},{"ssm_in_multiplier",0.4166666666666667},{"ssm_out_multiplier",0.11785113019775793},{"lm_head_multiplier",0.013020833333333334},{"ssm_multipliers",{0.3535533905932738,0.25,0.1767766952966369,0.5,0.3535533905932738}},{"mlp_multipliers",{0.2946278254943948,0.032552083333333336}},{"quantization_config",{{"bits",4},{"group_size",128},{"desc_act",true},{"sym",true},{"quant_method","gptq"},{"checkpoint_format","gptq"},{"pack_dtype","int32"}}}};
 auto bound=f::bind_config_only(config);auto g=f::geometry_from_artifact(bound,0);check(g.layer_count==44&&g.context_limit==8192&&g.recurrent_conv_channels==6656,"artifact-driven geometry");
 auto malformed=config;malformed.erase("mamba_n_groups");bool rejected=false;try{(void)f::bind_config_only(malformed);}catch(const std::exception&){rejected=true;}check(rejected,"malformed config rejected deterministically");
 LayoutBuilder b;DeviceKVPagePoolSpec ps{.page_group_count=16,.geometry={.page_tokens=64,.device_plane_order=PagedKVPlaneOrder::PageMajor,.planes={{.dtype=DType::BF16,.leading_extent=128,.head_extent=2},{.dtype=DType::FP16,.leading_extent=128,.head_extent=2}}}};auto pl=plan_device_kv_page_pool(b,ps);auto tl=plan_kv_execution_tables(b,{.logical_page_capacity=8,.table_rows=6});q::StateImageSpec ss{.linear={.layers=44,.conv_channels=static_cast<int>(g.recurrent_conv_channels),.conv_width=static_cast<int>(g.recurrent_conv_width),.value_heads=static_cast<int>(g.recurrent_value_heads),.value_head_dim=static_cast<int>(g.recurrent_value_head_dim),.key_head_dim=static_cast<int>(g.recurrent_key_head_dim),.slot_count=6,.conv_dtype=DType::BF16},.hidden=static_cast<int>(g.hidden_size)};auto sl=q::plan_state_image_device_pool(b,ss);
 DeviceArena arena(b.finish(256));DeviceSpan backing{arena.base(),arena.capacity()};DeviceKVPagePool physical(backing,pl);KVExecutionTablePool tables(backing,tl,physical);d::LogicalKVPageStore pages(physical,32);d::KVAddressSpaceStore addresses(pages,tables,8,8);q::StateImageDevicePool state_physical(backing,sl);d::StateImageStore states(state_physical,nullptr,8);DeviceContext device(0);
 auto program=f::Package::make_program(bound,0,4,6,2,addresses,states,state_physical);check(program.layers().size()==44,"44-layer inventory");for(std::size_t i=0;i<44;++i)check(program.layers()[i].layer==i&&program.layers()[i].stages.size()==6,"formal dispatch slots");
 auto id=program.create_sequence(0);auto ctx=f::ExecutionContext::capture(program,id);check(ctx.committed.generation==1&&program.sequence_count()==1,"sequence create/context");
 {auto tx=program.sequence(id).begin({.kv_frontier=0,.position=1,.recurrent_version=ctx.committed.recurrent_version+1});tx.execute([](ContinuationCandidateViews){});check(program.fork_snapshot(id).descriptor.generation==1,"snapshot during candidate");tx.rollback();}check(program.sequence(id).snapshot()==ctx.committed,"rollback restores sequence");
 {auto tx=program.sequence(id).begin({.kv_frontier=0,.position=1,.recurrent_version=ctx.committed.recurrent_version+1});tx.execute([](ContinuationCandidateViews){});tx.rollback();}check(program.sequence(id).snapshot()==ctx.committed,"rollback retry");
 {auto tx=program.sequence(id).begin({.kv_frontier=0,.position=1,.recurrent_version=ctx.committed.recurrent_version+1});tx.execute([](ContinuationCandidateViews){});tx.commit();}
 auto decoded=program.sequence(id).snapshot();check(decoded.kv_frontier==0&&decoded.position==1&&decoded.generation==2&&decoded.recurrent_version>ctx.committed.recurrent_version,"prefill to decode descriptor continuity");
 auto snap=program.fork_snapshot(id);check(snap.descriptor==decoded,"committed snapshot");
 program.reset_sequence(id,4);check(program.sequence_count()==1&&program.sequence(id).snapshot().position==0&&program.sequence(id).snapshot().kv_frontier==0,"reset reinitializes sequence");
 {auto tx=program.sequence(id).begin({.kv_frontier=3,.position=3,.recurrent_version=program.sequence(id).snapshot().recurrent_version+1});tx.execute([&](ContinuationCandidateViews views){auto& store=tx.kv_store_for_execution();store.ensure_mapped_to_tokens(views.kv,3,device.stream);auto cache=store.execution_layer_view(views.kv,0,128,2,KvCacheStorage::BFloat16);(void)cache;store.commit_frontier(views.kv,3);});tx.commit();}check(program.sequence(id).snapshot().kv_frontier==3,"candidate KV frontier publish");
 program.release_sequence(id);check(program.sequence_count()==0&&pages.occupied()==0&&physical.allocated_pages()==0&&physical.reserved_pages()==0&&states.occupied()==0,"release closes accounting");
 std::cout<<"Falcon Program minimal lifecycle checks passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<"Falcon Program test failed: "<<e.what()<<'\n';return 1;}}
