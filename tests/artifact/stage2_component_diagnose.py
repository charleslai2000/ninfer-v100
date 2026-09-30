#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0,'/home/ubuntu/workspaces/3rdparty/ninfer-v100-falcon-h1-7b')
from tools.artifact.v3 import V3Inspector
from tools.reference import falcon_h1_attention as ar
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--layer',type=int,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--native-attn',type=Path,required=True);p.add_argument('--native-output',type=Path,required=True);p.add_argument('--native-layer',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.checkpoint/'config.json').read_text());idx=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
def w(n):
 with safe_open(str(a.checkpoint/idx[n]),framework='pt',device='cpu') as f:return f.get_tensor(n).float().numpy()
vi=V3Inspector(a.artifact); l=a.layer;base=f'model.layers.{l}.'
def artbf(name,shape):return torch.from_numpy(np.frombuffer(vi.payload(name),dtype='<u2').copy()).view(torch.bfloat16).float().numpy().reshape(shape)
def metric(u,v):
 d=np.asarray(u,dtype=np.float32)-np.asarray(v,dtype=np.float32);return {'max_abs':float(abs(d).max()),'rel_l2':float(np.linalg.norm(d)/max(np.linalg.norm(v),1e-30))}
x=torch.from_numpy(np.fromfile(a.input,dtype='<u2').copy()).view(torch.bfloat16).float().reshape(-1,3072).numpy();an=artbf(f'text/layers.{l}.input_layernorm.weight',(3072,));norm=ar.rmsnorm(x,w(base+'input_layernorm.weight'),cfg['rms_norm_eps']);pre=ar.rmsnorm(x,an,cfg['rms_norm_eps'])

for r,heads,mul in [('q',12,1),('k',2,cfg['key_multiplier']),('v',2,1)]:
 src=(pre*cfg['attention_in_multiplier'])@w(base+f'self_attn.{r}_proj.weight').T*mul
 got=(ar.rmsnorm(x,an,cfg['rms_norm_eps'])*cfg['attention_in_multiplier'])@artbf(f'text/layers.{l}.self_attn.{r}_proj.weight',(heads*128,3072)).T*mul
 print('projection',r,metric(got,src))
vi=V3Inspector(a.artifact); ff=f'text/layers/{l}/ffn/'; perm=lambda n:np.frombuffer(vi.payload(ff+n+'/input_perm'),dtype='<i4').copy()
gate=gr.matmul(vi.payload(ff+'gate_proj'),(12288,3072),perm('gate_proj'),pre)
up=gr.matmul(vi.payload(ff+'up_proj'),(12288,3072),perm('up_proj'),pre)
prod=torch.nn.functional.silu(torch.from_numpy(gate))*torch.from_numpy(up)
down=gr.matmul(vi.payload(ff+'down_proj'),(3072,12288),perm('down_proj'),prod.numpy())
down=torch.from_numpy(down).to(torch.bfloat16).float().numpy()*cfg['mlp_multipliers'][1]
merged=torch.from_numpy(x+((ar.attention(*(lambda q,k,v:(q.reshape(-1,12,128),k.reshape(-1,2,128),v.reshape(-1,2,128)))((pre*cfg['attention_in_multiplier'])@w(base+'self_attn.q_proj.weight').T, (pre*cfg['attention_in_multiplier'])@w(base+'self_attn.k_proj.weight').T*cfg['key_multiplier'], (pre*cfg['attention_in_multiplier'])@w(base+'self_attn.v_proj.weight').T)),np.arange(len(x)),cfg['rope_theta'],128**-.5)[0].reshape(-1,1536)@w(base+'self_attn.o_proj.weight').T)*cfg['attention_out_multiplier'] + torch.zeros_like(x).numpy()).to(torch.bfloat16).float().numpy()
expect=merged+down
native=torch.from_numpy(np.fromfile(a.native_layer,dtype='<u2').copy()).view(torch.bfloat16).float().numpy().reshape(expect.shape)
print('final_vs_gptq_only_diagnostic',metric(native,expect))
