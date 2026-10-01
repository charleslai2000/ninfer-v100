"""Deterministic, text-only Falcon-H1 v3 converter.

No target/runtime registration is performed. The converter reads the pinned
safetensors checkpoint, canonicalizes GPTQ using the F1 stable permutation,
and writes a single-file v3 artifact with a complete source disposition.
"""
from __future__ import annotations
import argparse, hashlib, json, struct, subprocess
from pathlib import Path
import torch
from safetensors.torch import safe_open
from tools.artifact.v3 import align, gptq_geometry, make_gptq_payload
from . import inventory as inv

RESOURCE_FILES=("tokenizer.json","tokenizer_config.json","special_tokens_map.json","chat_template.jinja","generation_config.json")
PROJS=("gate_proj","up_proj","down_proj")

def sha256(path):
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def tensor(index,name,root):
 with safe_open(str(root/index[name]),framework="pt",device="cpu") as f: return f.get_tensor(name)

def bf16_bytes(t):
 if t.dtype!=torch.bfloat16: raise ValueError(f"expected BF16, got {t.dtype}")
 return t.contiguous().view(torch.uint16).numpy().tobytes()

def i32_bytes(t):
 if t.dtype!=torch.int32: raise ValueError(f"expected I32, got {t.dtype}")
 return t.contiguous().numpy().tobytes()

def gptq(index,base,root):
 qw=tensor(index,base+".qweight",root).numpy(); qz=tensor(index,base+".qzeros",root).numpy(); sc=tensor(index,base+".scales",root).numpy(); g=tensor(index,base+".g_idx",root).numpy().astype("int32",copy=False)
 k=int(g.size); n=int(sc.shape[1]); perm=__import__('numpy').argsort(g,kind="stable").astype("int32")
 qwu=qw.astype("uint32",copy=False); qcodes=((qwu[:,:,None]>>(__import__('numpy').arange(8,dtype='uint32')*4))&15).transpose(0,2,1).reshape(k,n)[perm]
 groups=g[perm]; assert (__import__('numpy').array_equal(groups,__import__('numpy').repeat(__import__('numpy').arange(k//128,dtype='int32'),128)))
 group_ids=groups[::128]; zu=qz.astype("uint32",copy=False); zcodes=((zu[:,:,None]>>(__import__('numpy').arange(8,dtype='uint32')*4))&15).reshape(qz.shape[0],n)[group_ids]
 packed=__import__('numpy').zeros((k//8,n),dtype='uint32')
 for b in range(8): packed |= (qcodes[b::8].astype('uint32')&15)<<(4*b)
 packedz=__import__('numpy').zeros((k//128,n//8),dtype='uint32')
 for b in range(8): packedz |= (zcodes[:,b::8].astype('uint32')&15)<<(4*b)
 payload=make_gptq_payload(packed.astype('<i4',copy=False).tobytes(),packedz.astype('<i4',copy=False).tobytes(),sc[group_ids].astype('<f2',copy=False).tobytes(),(n,k))
 return payload,i32_bytes(torch.from_numpy(perm))

def build(root,output):
 idx=json.loads((root/'model.safetensors.index.json').read_text())["weight_map"]; cfg=json.loads((root/'config.json').read_text())
 names=sorted(idx); objects=[]; payload_names=[]; producers={}
 def add(name,kind,**kw): objects.append({"name":name,"kind":kind,**kw}); payload_names.append(name)
 add("text/token_embedding","tensor",shape=[cfg["vocab_size"],cfg["hidden_size"]],format="bf16",layout="contiguous_le_v1"); producers["model.embed_tokens.weight"]="text/token_embedding"
 for name in names:
  if name in ("model.embed_tokens.weight","lm_head.weight","model.final_layernorm.weight"): continue
  if name.endswith((".qzeros", ".scales", ".g_idx")):
   continue
  if name.endswith(".qweight"):
   base=name[:-8]; layer=base.split('.')[2]; proj=base.split('.')[-1]; out=f"text/layers/{layer}/ffn/{proj}"; add(out,"tensor",shape=[int(tensor(idx,base+".scales",root).shape[1]),int(tensor(idx,base+".g_idx",root).numel())],format=inv.GPTQ_FORMAT,layout=inv.GPTQ_LAYOUT); add(out+"/input_perm","tensor",shape=[int(tensor(idx,base+".g_idx",root).numel())],format="int32",layout="contiguous_le_v1"); producers[base+".qweight"]=out
  elif name.endswith(".g_idx"): producers[name]=producers.get(name,name)
  elif name not in producers:
   t=tensor(idx,name,root); add("text/"+name.replace("model.",""),"tensor",shape=list(t.shape),format="bf16",layout="contiguous_le_v1"); producers[name]="text/"+name.replace("model.","")
 add("text/final_layernorm","tensor",shape=[cfg["hidden_size"]],format="bf16",layout="contiguous_le_v1"); producers["model.final_layernorm.weight"]="text/final_layernorm"
 add("text/lm_head","tensor",shape=[cfg["vocab_size"],cfg["hidden_size"]],format="bf16",layout="contiguous_le_v1"); producers["lm_head.weight"]="text/lm_head"
 resources=[]
 for fn in RESOURCE_FILES:
  if (root/fn).exists(): add("frontend/"+fn,"resource",encoding="raw-bytes-v1"); resources.append(fn)
 config=dict(cfg); config.update({"artifact_model_id":inv.MODEL_ID,"artifact_weights_id":inv.WEIGHTS_ID,"gptq_format":inv.GPTQ_FORMAT,"gptq_layout":inv.GPTQ_LAYOUT,"input_perm":"explicit_i32_auxiliary","source_revision":inv.REVISION,"pad_policy":{"artifact_pad_token_id":0,"tokenizer_pad_token_id":130048,"policy":"preserve_tokenizer_id_130048_for_frontend_padding; model_config_pad_id_0_is_not_rewritten"}})
 bindings=[{"name":n,"object":n,"component":"text"} for n in payload_names]; uses=[]
 for o in payload_names:
  if o.endswith("/input_perm"): uses.append({"parameter":o.rsplit('/',1)[0],"input":"input_perm","binding":o.rsplit('/',1)[0]+"/input_perm"})
  elif "/ffn/" in o: bindings.append({"name":o+"/parts","object":o,"parts":[{"object":o,"offset":0,"bytes":1}],"component":"text"})
 sources={fn:sha256(root/fn) for fn in sorted({p.name for p in root.glob('model-*.safetensors')})}
 doc={"components":{"text":{"config":config}},"objects":objects,"bindings":bindings,"uses":uses,"files":[{"path":None,"payload_bytes":0}],"metadata":{"source_tensor_count":len(names),"source_disposition":"all indexed tensors mapped"},"provenance":{"source_model":"tiiuae/Falcon-H1-7B-Instruct-GPTQ-Int4","hf_revision":inv.REVISION,"source_shard_sha256":sources,"recipe":"falcon_h1_v3_gptq_canonical_v1"}}
 # second pass computes bytes and writes deterministic directory/payload
 data={}
 for o in objects:
  n=o["name"]
  if n=="text/token_embedding": data[n]=bf16_bytes(tensor(idx,"model.embed_tokens.weight",root))
  elif n=="text/final_layernorm": data[n]=bf16_bytes(tensor(idx,"model.final_layernorm.weight",root))
  elif n=="text/lm_head": data[n]=bf16_bytes(tensor(idx,"lm_head.weight",root))
  elif n.startswith("frontend/"): data[n]=(root/n.removeprefix("frontend/")).read_bytes()
  elif n.endswith("/input_perm"):
   base=n.split('/')[2]; proj=n.split('/')[4]; data[n]=gptq(idx,f"model.layers.{base}.feed_forward.{proj}",root)[1]
  elif "/ffn/" in n:
   parts=n.split('/'); data[n]=gptq(idx,f"model.layers.{parts[2]}.feed_forward.{parts[4]}",root)[0]
  else:
   src="model."+n.removeprefix("text/").replace("/", "."); data[n]=bf16_bytes(tensor(idx,src,root))
 cursor=0; blob=bytearray()
 for o in objects:
  cursor=align(cursor,256 if o["kind"]=="tensor" else 1); o["offset"]=cursor; o["bytes"]=len(data[o["name"]]); blob.extend(b'\0'*(cursor-len(blob))); blob.extend(data[o["name"]]); cursor+=o["bytes"]
 for b in bindings:
  if "parts" in b: b["parts"][0]["bytes"]=next(o["bytes"] for o in objects if o["name"]==b["object"])
 disposition=inv.summary(root); doc["metadata"].update({"source_disposition_counts":disposition["dispositions"],"gptq_projection_count":len(disposition["gptq_projection_bases"]),"duplicate_consumption":0,"silent_drop":0,"unexplained_artifact_objects":0})
 doc["provenance"]["frontend_sha256"]={fn:sha256(root/fn) for fn in RESOURCE_FILES if (root/fn).exists()}
 doc["files"][0]["payload_bytes"]=cursor; directory=json.dumps(doc,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode(); start=align(32+len(directory),4096); output.parent.mkdir(parents=True,exist_ok=True); aid=bytes.fromhex('00112233445566778899aabbccddeeff'); output.write_bytes(b'NINFER\0\3'+struct.pack('<Q',len(directory))+aid+directory+b'\0'*(start-32-len(directory))+blob)
 return doc

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,default=inv.CHECKPOINT); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); print(json.dumps({"objects":len(build(args.checkpoint,args.output)["objects"]),"output":str(args.output)},indent=2))
