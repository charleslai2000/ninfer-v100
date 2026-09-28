#include "falcon_h1_binder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <variant>

namespace ninfer::targets::falcon_h1 {
namespace {
using namespace artifact;
using Json = nlohmann::json;

[[noreturn]] void invalid(const std::string& message) { throw ArtifactError("Falcon-H1 binder: " + message); }
std::uint64_t cfg_u64(const Json& c, const char* key) {
    if (!c.contains(key) || !c.at(key).is_number_integer() || c.at(key).get<std::int64_t>() <= 0) invalid(std::string("invalid/missing config field ")+key);
    return c.at(key).get<std::uint64_t>();
}
double cfg_f64(const Json& c, const char* key) {
    if (!c.contains(key) || !c.at(key).is_number()) invalid(std::string("invalid/missing config field ")+key);
    const double v=c.at(key).get<double>(); if(!std::isfinite(v)) invalid(std::string("nonfinite config field ")+key); return v;
}
void require_array_size(const Json& c,const char* key,std::size_t n) {
    if(!c.contains(key)||!c.at(key).is_array()||c.at(key).size()!=n) invalid(std::string("invalid config array ")+key);
    for(const auto& x:c.at(key)) if(!x.is_number()||!std::isfinite(x.get<double>())) invalid(std::string("invalid config array ")+key);
}
struct Spec { std::string name; NumericFormat fmt; StorageLayout layout; std::vector<std::uint64_t> shape; };
void add(std::vector<Spec>& s,std::string n,NumericFormat f,StorageLayout l,std::initializer_list<std::uint64_t> sh){s.push_back({std::move(n),f,l,sh});}
std::string layer_name(std::uint64_t i,std::string_view tail){return "text/layers."+std::to_string(i)+"."+std::string(tail);}
std::string ffn_name(std::uint64_t i,std::string_view proj){return "text/layers/"+std::to_string(i)+"/ffn/"+std::string(proj);}
bool frontend_resource(std::string_view n){return n=="frontend/tokenizer.json"||n=="frontend/tokenizer_config.json"||n=="frontend/special_tokens_map.json"||n=="frontend/chat_template.jinja"||n=="frontend/generation_config.json";}

std::vector<Spec> expected(const Json& c) {
 const auto h=cfg_u64(c,"hidden_size"), ff=cfg_u64(c,"intermediate_size"), vocab=cfg_u64(c,"vocab_size"), layers=cfg_u64(c,"num_hidden_layers"), ah=cfg_u64(c,"num_attention_heads"), kh=cfg_u64(c,"num_key_value_heads"), hd=cfg_u64(c,"head_dim");
 const auto md=cfg_u64(c,"mamba_d_conv"), nh=cfg_u64(c,"mamba_n_heads"), ds=cfg_u64(c,"mamba_d_state"), chunk=cfg_u64(c,"mamba_chunk_size"), groups=cfg_u64(c,"mamba_n_groups"), dhead=cfg_u64(c,"mamba_d_head");
 if(layers!=44 || h!=3072 || ff!=12288 || vocab!=130049 || ah*hd!=1536 || kh*hd!=256 || ah%kh || md!=4 || nh!=24 || ds!=256 || chunk!=256 || groups!=1 || dhead!=128 || nh*dhead!=3072) invalid("artifact geometry incompatible with frozen Falcon-H1 contract");
 if(c.value("model_type",std::string{})!="falcon_h1" || c.value("gptq_format",std::string{})!="gptq_q4_g128_fp16" || c.value("gptq_layout",std::string{})!="gptq_canonical_k128_v1" || c.value("input_perm",std::string{})!="explicit_i32_auxiliary") invalid("model/GPTQ config contract mismatch");
 if(c.value("torch_dtype",std::string{})!="bfloat16" || c.value("mamba_d_head",0)!=128 || c.value("mamba_d_ssm",0)!=3072 || c.value("mamba_expand",0)!=2 || c.value("mamba_norm_before_gate",true) || c.value("mamba_rms_norm",false)!=true || c.value("mamba_proj_bias",true) || c.value("mamba_conv_bias",false)!=true) invalid("attention/Mamba semantic geometry mismatch");
 if(c.value("rms_norm_eps",0.0)!=1.0e-5 || c.value("rope_theta",0.0)!=1.0e11 || c.value("attention_bias",true) || c.value("mlp_bias",true) || c.value("projectors_bias",true)) invalid("normalization/position/bias contract mismatch");
 if(c.value("tie_word_embeddings",true) || !c.value("mamba_use_mlp",false) || c.value("mamba_norm_before_gate",true) || !c.value("mamba_rms_norm",false)) invalid("unsupported model semantic config");
 for(auto k: {"embedding_multiplier","attention_in_multiplier","attention_out_multiplier","key_multiplier","ssm_in_multiplier","ssm_out_multiplier","lm_head_multiplier","rms_norm_eps","rope_theta"}) (void)cfg_f64(c,k);
 require_array_size(c,"ssm_multipliers",5); require_array_size(c,"mlp_multipliers",2);
 const std::array<double,5> expected_ssm={0.3535533905932738,0.25,0.1767766952966369,0.5,0.3535533905932738};
 for(std::size_t i=0;i<expected_ssm.size();++i) if(c.at("ssm_multipliers")[i].get<double>()!=expected_ssm[i]) invalid("SSM multiplier contract mismatch");
 const std::array<double,2> expected_mlp={0.2946278254943948,0.032552083333333336};
 for(std::size_t i=0;i<expected_mlp.size();++i) if(c.at("mlp_multipliers")[i].get<double>()!=expected_mlp[i]) invalid("MLP multiplier contract mismatch");
 const std::array<std::pair<const char*,double>,9> expected_scalars={{{"embedding_multiplier",5.656854249492381},{"attention_in_multiplier",1.0},{"attention_out_multiplier",0.10416666666666667},{"key_multiplier",0.030690398488999456},{"ssm_in_multiplier",0.4166666666666667},{"ssm_out_multiplier",0.11785113019775793},{"lm_head_multiplier",0.013020833333333334},{"rms_norm_eps",1.0e-5},{"rope_theta",1.0e11}}};
 for(const auto& [key,value]:expected_scalars) if(cfg_f64(c,key)!=value) invalid(std::string("config scalar mismatch: ")+key);
 if(!c.contains("quantization_config") || c.at("quantization_config").value("bits",0)!=4 || c.at("quantization_config").value("group_size",0)!=128 || !c.at("quantization_config").value("desc_act",false) || !c.at("quantization_config").value("sym",false) || c.at("quantization_config").value("quant_method",std::string{})!="gptq" || c.at("quantization_config").value("checkpoint_format",std::string{})!="gptq" || c.at("quantization_config").value("pack_dtype",std::string{})!="int32") invalid("quantization config mismatch");
 std::vector<Spec> s; s.reserve(888);
 add(s,"text/token_embedding",NumericFormat::BF16,StorageLayout::ContiguousLeV1,{vocab,h});
 add(s,"text/final_layernorm",NumericFormat::BF16,StorageLayout::ContiguousLeV1,{h});
 add(s,"text/lm_head",NumericFormat::BF16,StorageLayout::ContiguousLeV1,{vocab,h});
 for(std::uint64_t i=0;i<layers;++i){
  add(s,layer_name(i,"input_layernorm.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{h});
  add(s,layer_name(i,"pre_ff_layernorm.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{h});
  add(s,layer_name(i,"self_attn.q_proj.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{ah*hd,h});
  add(s,layer_name(i,"self_attn.k_proj.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{kh*hd,h});
  add(s,layer_name(i,"self_attn.v_proj.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{kh*hd,h});
  add(s,layer_name(i,"self_attn.o_proj.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{h,ah*hd});
  add(s,layer_name(i,"mamba.in_proj.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{6680,h});
  add(s,layer_name(i,"mamba.out_proj.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{h,h});
  add(s,layer_name(i,"mamba.conv1d.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{3584,1,md});
  add(s,layer_name(i,"mamba.conv1d.bias"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{3584});
  add(s,layer_name(i,"mamba.A_log"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{nh});
  add(s,layer_name(i,"mamba.D"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{nh});
  add(s,layer_name(i,"mamba.dt_bias"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{nh});
  add(s,layer_name(i,"mamba.norm.weight"),NumericFormat::BF16,StorageLayout::ContiguousLeV1,{h});
  for(auto [role,n,k]: {std::tuple<std::string_view,std::uint64_t,std::uint64_t>{"gate_proj",ff,h},{"up_proj",ff,h},{"down_proj",h,ff}}){
   const auto name=ffn_name(i,role); add(s,name,NumericFormat::GPTQ_Q4_G128_FP16,StorageLayout::GPTQ_CANONICAL_K128_V1,{n,k}); add(s,name+"/input_perm",NumericFormat::I32,StorageLayout::ContiguousLeV1,{k});
  }
 }
 return s;
}
}

BoundModel bind_config_only(const Json& config) {
 if(!config.is_object()) invalid("text config must be an object");
 const auto specs=expected(config);
 BoundModel out; out.config=config; out.layer_count=cfg_u64(config,"num_hidden_layers");
 for(auto key:{"hidden_size","intermediate_size","vocab_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim","mamba_d_conv","mamba_n_heads","mamba_d_state","mamba_chunk_size","mamba_n_groups","mamba_d_head","mamba_d_ssm","max_position_embeddings"}) out.config_integers.emplace(key,cfg_u64(config,key));
 for(auto key:{"embedding_multiplier","attention_in_multiplier","attention_out_multiplier","key_multiplier","ssm_in_multiplier","ssm_out_multiplier","lm_head_multiplier","rms_norm_eps","rope_theta"}) out.config_scalars.emplace(key,cfg_f64(config,key));
 for(auto key:{"ssm_multipliers","mlp_multipliers"}) { auto& v=out.config_arrays[key]; for(const auto& x:config.at(key)) v.push_back(x.get<double>()); }
 for(std::uint64_t i=0;i<out.layer_count;++i) out.layer_types.emplace_back("falcon_h1_hybrid");
 return out;
}

void validate_directory(const Json& doc) {
 if(!doc.contains("components") || !doc.at("components").contains("text") || !doc.at("components").at("text").contains("config")) invalid("missing components.text.config");
 const Json config=doc.at("components").at("text").at("config");
 const auto specs=expected(config);
 std::set<std::string> expected_names; for(const auto& s:specs) if(!expected_names.insert(s.name).second) invalid("internal duplicate role spec: "+s.name);
 if(!doc.contains("objects") || !doc.at("objects").is_array()) invalid("missing object inventory");
 std::map<std::string,std::size_t> actual_count;
 for(const auto& o:doc.at("objects")){if(!o.is_object()||!o.contains("name")||!o.at("name").is_string()) invalid("malformed object descriptor");const auto n=o.at("name").get<std::string>();if(frontend_resource(n)) { if(o.value("kind",std::string{})!="resource"||o.value("encoding",std::string{})!="raw-bytes-v1")invalid("frontend resource representation mismatch: "+n);continue;} if(!expected_names.contains(n)) invalid("unexplained artifact object: "+n);++actual_count[n];}
 for(const auto& n:expected_names)if(actual_count[n]!=1)invalid("missing/duplicate role "+n);
 for(const auto& s:specs){const Json* found=nullptr;for(const auto& o:doc.at("objects"))if(o.at("name")==s.name){found=&o;break;}if(found==nullptr||found->value("kind",std::string{})!="tensor"||found->value("shape",std::vector<std::uint64_t>{})!=s.shape)invalid("wrong/missing tensor shape: "+s.name);const auto expected_format=std::string(format_name(s.fmt));std::string expected_layout=std::string(layout_name(s.layout));if(s.layout==StorageLayout::ContiguousLeV1)expected_layout="contiguous_le_v1";std::string artifact_format=expected_format;if(s.fmt==NumericFormat::BF16)artifact_format="bf16";else if(s.fmt==NumericFormat::FP32)artifact_format="fp32";else if(s.fmt==NumericFormat::I32)artifact_format="int32";if(found->value("format",std::string{})!=artifact_format||found->value("layout",std::string{})!=expected_layout)invalid("wrong tensor representation: "+s.name);}
 if(!doc.contains("bindings")||!doc.at("bindings").is_array()||!doc.contains("uses")||!doc.at("uses").is_array())invalid("binding/use inventory absent");
 std::set<std::string> binding_names;std::map<std::string,std::size_t> binding_count;
 for(const auto& b:doc.at("bindings")){if(!b.is_object()||!b.contains("name")||!b.at("name").is_string()||!b.contains("object")||!b.at("object").is_string()||!binding_names.insert(b.at("name").get<std::string>()).second)invalid("malformed/duplicate binding");++binding_count[b.at("object").get<std::string>()];}
 for(const auto& n:expected_names)if(binding_count[n]!=((n.find("/ffn/")!=std::string::npos&&n.find("/input_perm")==std::string::npos)?2U:1U))invalid("binding object closure mismatch: "+n);
 for(const auto& n:binding_names){auto slash=n.rfind("/parts");if(slash!=std::string::npos){if(!expected_names.contains(n.substr(0,slash)))invalid("unexplained binding: "+n);}else if(!expected_names.contains(n)&&!frontend_resource(n))invalid("unexplained binding: "+n);}
 for(const auto& s:specs)if(!binding_names.contains(s.name))invalid("missing binding: "+s.name);
 for(const auto& o:doc.at("objects")) if(o.is_object()&&o.contains("name")&&frontend_resource(o.at("name").get<std::string>())&&!binding_names.contains(o.at("name").get<std::string>())) invalid("missing frontend resource binding");
 std::set<std::string> expected_perm,found_perm;for(std::uint64_t i=0;i<44;++i)for(auto p:{"gate_proj","up_proj","down_proj"})expected_perm.insert(ffn_name(i,p));
 for(const auto& u:doc.at("uses")){if(!u.is_object()||!u.contains("parameter")||!u.contains("input")||!u.contains("binding"))invalid("malformed use record");if(u.at("input")=="input_perm"){auto p=u.at("parameter").get<std::string>();if(!expected_perm.contains(p)||!found_perm.insert(p).second||u.at("binding")!=p+"/input_perm")invalid("wrong/duplicate input_perm use");}}
 if(found_perm!=expected_perm)invalid("input_perm use closure mismatch");
}
void validate(const Reader& reader) {
 if(reader.v3_directory().is_null() || reader.v3_directory().empty()) invalid("v3 directory required");
 validate_directory(reader.v3_directory());
}

BoundModel bind(Reader& reader) {
 validate(reader);
 const auto& doc=reader.v3_directory();
 const Json config=doc.at("components").at("text").at("config");
 const auto specs=expected(config);
 Binder binder(reader); BoundModel out; out.config=config; out.layer_count=cfg_u64(config,"num_hidden_layers");
 for(const auto& o:doc.at("objects"))if(o.is_object()&&o.contains("name")&&frontend_resource(o.at("name").get<std::string>())){const auto h=binder.require_resource(o.at("name").get<std::string>(),ResourceEncoding::RawBytesV1);binder.retain_on_host(h);}
 out.resource_count=0;
 for(const auto& o:doc.at("objects"))if(o.is_object()&&o.contains("name")&&frontend_resource(o.at("name").get<std::string>()))++out.resource_count;
 for(auto key:{"hidden_size","intermediate_size","vocab_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim","mamba_d_conv","mamba_n_heads","mamba_d_state","mamba_chunk_size","mamba_n_groups","mamba_d_head","mamba_d_ssm","max_position_embeddings"}) out.config_integers.emplace(key,cfg_u64(config,key));
 for(std::uint64_t i=0;i<out.layer_count;++i) out.layer_types.emplace_back("falcon_h1_hybrid");
 for(auto key:{"embedding_multiplier","attention_in_multiplier","attention_out_multiplier","key_multiplier","ssm_in_multiplier","ssm_out_multiplier","lm_head_multiplier","rms_norm_eps","rope_theta"}) out.config_scalars.emplace(key,cfg_f64(config,key));
 for(auto key:{"ssm_multipliers","mlp_multipliers"}) { auto& v=out.config_arrays[key]; for(const auto& x:config.at(key)) v.push_back(x.get<double>()); }
 for(const auto& s:specs) {
  const auto handle=binder.require_tensor(s.name,s.fmt,s.layout,s.shape);
  const auto payload=binder.payload(handle);
  if(payload.data.size()!=tensor_encoded_size(s.layout,s.fmt,s.shape)) invalid("payload size mismatch: "+s.name);
  if(s.fmt==NumericFormat::I32){const auto k=s.shape.at(0);std::vector<bool> seen(k,false);for(std::uint64_t j=0;j<k;++j){std::int32_t value=0;std::memcpy(&value,payload.data.data()+j*sizeof(value),sizeof(value));if(value<0||static_cast<std::uint64_t>(value)>=k||seen[static_cast<std::size_t>(value)])invalid("input_perm is not a permutation: "+s.name);seen[static_cast<std::size_t>(value)]=true;}}
  binder.materialize_on_device(handle);
  out.tensors.emplace(s.name,BoundTensor{handle,s.name});
  out.device_bytes_by_format[std::string(format_name(s.fmt))]+=payload.data.size();
 }
 out.materialization=binder.finish();
 if(out.tensors.size()!=specs.size() || out.layer_count!=44 || out.materialization.device_objects.size()!=specs.size() || out.materialization.host_objects.size()!=out.resource_count) invalid("binding closure mismatch");
 return out;
}

} // namespace ninfer::targets::falcon_h1
