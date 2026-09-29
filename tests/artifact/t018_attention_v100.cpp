#include "artifact/reader.h"
#include "artifact/materializer.h"
#include "targets/falcon_h1/falcon_h1_binder.h"
#include "targets/falcon_h1/falcon_h1_package.h"
#include "targets/falcon_h1/falcon_h1_attention.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"
#include <ninfer/targets/qwen3_6/state_image.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void write_raw(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("tap write failed: " + path.string());
}
}

int main(int argc, char** argv) {
 try {
  if (argc < 6 || argc > 8)
   throw std::runtime_error("usage ARTIFACT INPUT_F32 TAP_DIR LAYER POSITION [FAIL_STAGE] [HANDOFF_T]");
  using namespace ninfer;
  namespace f = targets::falcon_h1;
  namespace q = targets::qwen3_6;
  namespace d = targets::qwen3_6::detail;
  constexpr int H=3072,QH=12,KVH=2,D=128;
  const int layer=std::stoi(argv[4]), position=std::stoi(argv[5]);
  const int fail_stage=argc>=7?std::stoi(argv[6]):0;
  const int handoff=argc>=8?std::stoi(argv[7]):0;
  const std::filesystem::path input_path(argv[2]), outdir(argv[3]);
  const auto input_bytes=std::filesystem::file_size(input_path);
  if(input_bytes==0 || input_bytes%(H*sizeof(float))) throw std::runtime_error("input must be complete F32 hidden rows");
  const int T=static_cast<int>(input_bytes/(H*sizeof(float)));
  if(layer<0 || layer>=44 || position<0 || (fail_stage && (fail_stage<1 || fail_stage>8))) throw std::runtime_error("invalid layer/position/failure stage");
  if(handoff && (handoff<1 || handoff>T)) throw std::runtime_error("handoff suffix must fit input rows");
  std::filesystem::create_directories(outdir);

  artifact::Reader reader(argv[1]);
  auto bound=f::bind(reader);
  DeviceContext device(0);
  auto materialized=artifact::materialize(reader,bound.materialization,device);
  LayoutBuilder layout;
  constexpr int physical_page_count=64, logical_page_capacity=32, row_count=4;
  DeviceKVPagePoolSpec page_spec{.page_group_count=physical_page_count,.geometry={.page_tokens=64,.device_plane_order=PagedKVPlaneOrder::PageMajor,.planes={}}};
  page_spec.geometry.planes.reserve(88);
  for(int li=0;li<44;++li){page_spec.geometry.planes.push_back({.dtype=DType::BF16,.leading_extent=D,.head_extent=KVH});page_spec.geometry.planes.push_back({.dtype=DType::FP16,.leading_extent=D,.head_extent=KVH});}
  auto page_layout=plan_device_kv_page_pool(layout,page_spec);
  auto table_layout=plan_kv_execution_tables(layout,{.logical_page_capacity=logical_page_capacity,.table_rows=row_count});
  q::StateImageSpec state_spec{.linear={.layers=44,.conv_channels=6656,.conv_width=4,.value_heads=24,
   .value_head_dim=128,.key_head_dim=256,.slot_count=2,.conv_dtype=DType::BF16},.hidden=H};
  auto state_layout=q::plan_state_image_device_pool(layout,state_spec);
  DeviceArena backing(layout.finish(256));
  DeviceSpan backing_span{backing.base(),backing.capacity()};
  DeviceKVPagePool physical_pages(backing_span,page_layout);
  KVExecutionTablePool execution_tables(backing_span,table_layout,physical_pages);
  d::LogicalKVPageStore logical_pages(physical_pages,physical_page_count);
  d::KVAddressSpaceStore address_store(logical_pages,execution_tables,row_count,logical_page_capacity);
  q::StateImageDevicePool state_physical(backing_span,state_layout);
  d::StateImageStore state_store(state_physical,nullptr,2);
  auto program=f::Package::make_program(bound,8192,logical_page_capacity,row_count,2,address_store,state_store,state_physical);
  const auto sid=program.create_sequence(0);

  std::vector<float> input_f32(static_cast<std::size_t>(T)*H);
  std::ifstream fin(input_path,std::ios::binary);
  fin.read(reinterpret_cast<char*>(input_f32.data()),static_cast<std::streamsize>(input_bytes));
  if(!fin || fin.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("input fixture read/size mismatch");
  std::vector<__nv_bfloat16> host(input_f32.size());
  for(std::size_t i=0;i<host.size();++i) host[i]=__float2bfloat16_rn(input_f32[i]);
  auto base=program.sequence(sid).snapshot();
  if(position!=static_cast<int>(base.position)) throw std::runtime_error("position must equal committed descriptor position");
  const auto candidate_spec=ContinuationCandidateSpec{.kv_frontier=base.kv_frontier+T,.position=base.position+T,.recurrent_version=base.recurrent_version+1};
  if(fail_stage>=1&&fail_stage<=4){
   for(int injected=1;injected<=4;++injected){program.sequence(sid).set_failure_injection_for_test(injected);bool did_throw=false;try{auto abandoned=program.sequence(sid).begin(candidate_spec);}catch(const std::runtime_error&){did_throw=true;}if(!did_throw||program.sequence(sid).snapshot()!=base||program.sequence(sid).transaction_pending())throw std::runtime_error("begin rollback invariant failed");}
   program.release_sequence(sid);std::cout<<"T018_INJECTED_BEGIN_ROLLBACK_PASS stages=1-4\n";return 0;
  }
  if(fail_stage>=5) program.set_failure_injection_for_test(sid,fail_stage);
  auto candidate=program.sequence(sid).begin(candidate_spec);

  DeviceBuffer input_dev(host.size()*2),output_dev(host.size()*2);
  input_dev.copy_from_host(host.data(),host.size()*2);
  Tensor input(input_dev.p,DType::BF16,{H,T}),output(output_dev.p,DType::BF16,{H,T});
  const std::size_t ws_bytes=program.attention_workspace_bytes(T);
  DeviceBuffer ws(ws_bytes);
  std::vector<__nv_bfloat16> qv(static_cast<std::size_t>(T)*QH*D),kp(static_cast<std::size_t>(T)*KVH*D),kv(kp.size()),vv(kp.size());
  std::vector<__nv_bfloat16> qr(qv.size()),kr(kp.size()),pre(qv.size()),op(host.size()),out(host.size());
  DeviceBuffer qd(qv.size()*2),kpd(kp.size()*2),kd(kv.size()*2),vd(vv.size()*2),qrd(qr.size()*2),krd(kr.size()*2),pred(pre.size()*2),opd(op.size()*2),outd(out.size()*2),ckd(kp.size()*2),cvd(vv.size()*2);
  if(fail_stage==0) {
   f::execute_attention(program,candidate,base,position,layer,bound,materialized,input,output,
    {ws.p,ws.bytes},ws.bytes,device.stream,{.q=qd.p,.k_pre_scale=kpd.p,.k=kd.p,.v=vd.p,
      .post_rope_q=qrd.p,.post_rope_k=krd.p,.candidate_k=ckd.p,.candidate_v=cvd.p,.pre_o=pred.p,.o_projection=opd.p,.output=outd.p});
   device.synchronize();
   qd.copy_to_host(qv.data(),qv.size()*2);kpd.copy_to_host(kp.data(),kp.size()*2);kd.copy_to_host(kv.data(),kv.size()*2);vd.copy_to_host(vv.data(),vv.size()*2);
   qrd.copy_to_host(qr.data(),qr.size()*2);krd.copy_to_host(kr.data(),kr.size()*2);pred.copy_to_host(pre.data(),pre.size()*2);opd.copy_to_host(op.data(),op.size()*2);outd.copy_to_host(out.data(),out.size()*2);
   const std::pair<const char*,const std::vector<__nv_bfloat16>*> taps[]={{"q.bf16",&qv},{"k_pre_scale.bf16",&kp},{"k.bf16",&kv},{"v.bf16",&vv},{"q_rope.bf16",&qr},{"k_rope.bf16",&kr},{"attention_pre_o.bf16",&pre},{"o_projection.bf16",&op},{"output.bf16",&out}};
   for(const auto&[name,val]:taps) write_raw(outdir/name,val->data(),val->size()*2);
   std::vector<__nv_bfloat16> candidate_k(kp.size());std::vector<__half> candidate_v(vv.size());ckd.copy_to_host(candidate_k.data(),candidate_k.size()*2);cvd.copy_to_host(candidate_v.data(),candidate_v.size()*2);write_raw(outdir/"candidate_k.bf16",candidate_k.data(),candidate_k.size()*2);write_raw(outdir/"candidate_v.f16",candidate_v.data(),candidate_v.size()*2);
   std::ofstream meta(outdir/"run.json");meta<<"{\"layer\":"<<layer<<",\"tokens\":"<<T<<",\"position_start\":"<<position<<",\"kv_frontier_start\":"<<base.kv_frontier<<",\"artifact\":\""<<argv[1]<<"\",\"input_f32\":\""<<argv[2]<<"\"}\n";
   write_raw(outdir/"input.bf16",host.data(),host.size()*2);std::vector<std::int32_t> host_positions(T);for(int i=0;i<T;++i)host_positions[i]=position+i;write_raw(outdir/"positions.i32",host_positions.data(),host_positions.size()*4);
   if(!meta) throw std::runtime_error("failed writing run metadata");
   candidate.prepare();candidate.commit();
  } else {
   bool thrown=false;
   try {
    f::execute_attention(program,candidate,base,position,layer,bound,materialized,input,output,{ws.p,ws.bytes},ws.bytes,device.stream);
   } catch(const std::runtime_error&) { thrown=true; }
   if(!thrown) throw std::runtime_error("candidate failure injection did not fire");
   device.synchronize();
   if(program.sequence(sid).snapshot()!=base || program.sequence(sid).transaction_pending()) throw std::runtime_error("injected rollback changed committed descriptor/pending state");
   program.release_sequence(sid);
   std::cout<<"T018_INJECTED_ROLLBACK_PASS stage="<<fail_stage<<'\n';
   return 0;
  }

  const auto committed=program.sequence(sid).snapshot();
  if(committed.kv_frontier!=base.kv_frontier+T) throw std::runtime_error("committed KV frontier mismatch");
  if(handoff) {
   const auto suffix=static_cast<std::uint64_t>(handoff);
   std::vector<__nv_bfloat16> shost(host.begin(),host.begin()+suffix*H);
   DeviceBuffer si(shost.size()*2),so(shost.size()*2),sws(program.attention_workspace_bytes(handoff));
   si.copy_from_host(shost.data(),shost.size()*2);Tensor sint(si.p,DType::BF16,{H,handoff}),sout(so.p,DType::BF16,{H,handoff});
   const std::size_t qn=static_cast<std::size_t>(handoff)*QH*D,kn=static_cast<std::size_t>(handoff)*KVH*D;
   DeviceBuffer hq(qn*2),hkp(kn*2),hk(kn*2),hv(kn*2),hqr(qn*2),hkr(kn*2),hck(kn*2),hcv(kn*2),hpre(qn*2),hop(static_cast<std::size_t>(handoff)*H*2),hout(static_cast<std::size_t>(handoff)*H*2);
   auto tx=program.sequence(sid).begin({.kv_frontier=committed.kv_frontier+suffix,.position=committed.position+suffix,.recurrent_version=committed.recurrent_version+1});
   f::execute_attention(program,tx,committed,committed.position,layer,bound,materialized,sint,sout,{sws.p,sws.bytes},sws.bytes,device.stream,{.q=hq.p,.k_pre_scale=hkp.p,.k=hk.p,.v=hv.p,.post_rope_q=hqr.p,.post_rope_k=hkr.p,.candidate_k=hck.p,.candidate_v=hcv.p,.pre_o=hpre.p,.o_projection=hop.p,.output=hout.p});
   device.synchronize();std::vector<__nv_bfloat16> sout_host(shost.size());so.copy_to_host(sout_host.data(),sout_host.size()*2);
   for(auto z:sout_host) if(!std::isfinite(__bfloat162float(z))) throw std::runtime_error("handoff decode produced nonfinite output");
   device.synchronize();std::vector<__nv_bfloat16> hqv(qn),hkpre(kn),hkv(kn),hvv(kn),hqrv(qn),hkrv(kn),hprev(qn),hopv(static_cast<std::size_t>(handoff)*H),houtv(hopv.size()),hckv(kn);std::vector<__half> hcvv(kn);hq.copy_to_host(hqv.data(),qn*2);hkp.copy_to_host(hkpre.data(),kn*2);hk.copy_to_host(hkv.data(),kn*2);hv.copy_to_host(hvv.data(),kn*2);hqr.copy_to_host(hqrv.data(),qn*2);hkr.copy_to_host(hkrv.data(),kn*2);hpre.copy_to_host(hprev.data(),qn*2);hop.copy_to_host(hopv.data(),hopv.size()*2);hout.copy_to_host(houtv.data(),houtv.size()*2);hck.copy_to_host(hckv.data(),kn*2);hcv.copy_to_host(hcvv.data(),kn*2);const auto hd=outdir/"handoff-taps";std::filesystem::create_directories(hd);write_raw(hd/"q.bf16",hqv.data(),qn*2);write_raw(hd/"k_pre_scale.bf16",hkpre.data(),kn*2);write_raw(hd/"k.bf16",hkv.data(),kn*2);write_raw(hd/"v.bf16",hvv.data(),kn*2);write_raw(hd/"q_rope.bf16",hqrv.data(),qn*2);write_raw(hd/"k_rope.bf16",hkrv.data(),kn*2);write_raw(hd/"attention_pre_o.bf16",hprev.data(),qn*2);write_raw(hd/"o_projection.bf16",hopv.data(),hopv.size()*2);write_raw(hd/"output.bf16",houtv.data(),houtv.size()*2);write_raw(hd/"candidate_k.bf16",hckv.data(),kn*2);write_raw(hd/"candidate_v.f16",hcvv.data(),kn*2);
   tx.rollback();if(program.sequence(sid).snapshot()!=committed)throw std::runtime_error("handoff rollback changed committed descriptor");
  }
  program.release_sequence(sid);
  std::cout<<"T018_PROGRAM_PASS layer="<<layer<<" T="<<T<<" position="<<position<<" handoff="<<handoff<<'\n';
  return 0;
 } catch(const std::exception&e) { std::cerr<<"T018_FAIL "<<e.what()<<'\n';return 1; }
}
