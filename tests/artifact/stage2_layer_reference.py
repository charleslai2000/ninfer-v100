#!/usr/bin/env python3
"""Same-artifact Falcon-H1 one-layer oracle; non-qualification comparison utility."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, "/home/ubuntu/workspaces/3rdparty/ninfer-v100-falcon-h1-7b")
from tools.artifact.v3 import V3Inspector
from tools.reference import falcon_h1_attention as attref
from tools.reference import falcon_h1_gptq as gptqref
from tools.reference import falcon_h1_mamba as mambaref

def load_tensor(index, root, name):
    with safe_open(str(root/index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).float().numpy()
def bf16_input(path, tokens):
    return torch.from_numpy(np.fromfile(path, dtype="<u2").copy()).view(torch.bfloat16).float().numpy().reshape(tokens,3072)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--artifact',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True); ap.add_argument('--native-output',type=Path,required=True)
    ap.add_argument('--layer',type=int,choices=[0,22,43],required=True); ap.add_argument('--tokens',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    cfg=json.loads((a.checkpoint/'config.json').read_text())
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    w=lambda name: load_tensor(index,a.checkpoint,f'model.layers.{a.layer}.{name}')
    x=bf16_input(a.input,a.tokens); norm=attref.rmsnorm(x,w('input_layernorm.weight'),cfg['rms_norm_eps'])
    scaled=norm*cfg['attention_in_multiplier']; q=scaled@w('self_attn.q_proj.weight').T
    k=(scaled@w('self_attn.k_proj.weight').T)*cfg['key_multiplier']; v=scaled@w('self_attn.v_proj.weight').T
    q=q.reshape(a.tokens,12,128); k=k.reshape(a.tokens,2,128); v=v.reshape(a.tokens,2,128)
    at,*_=attref.attention(q,k,v,np.arange(a.tokens),cfg['rope_theta'],1/np.sqrt(128))
    ah=(at.reshape(a.tokens,1536)@w('self_attn.o_proj.weight').T)*cfg['attention_out_multiplier']
    keys=['in_proj.weight','conv1d.weight','conv1d.bias','A_log','D','dt_bias','norm.weight','out_proj.weight']
    mw={key:torch.from_numpy(w('mamba.'+key)) for key in keys}
    mt,_=mambaref.f3_recurrence(mw,torch.from_numpy(norm),cfg,initial_conv=torch.zeros(3584,3),initial_ssm=torch.zeros(24,128,256))
    mh=mt['out'].numpy()*cfg['ssm_out_multiplier']; merged=x+ah+mh
    merged=torch.from_numpy(merged).to(torch.bfloat16).float().numpy()
    pre=attref.rmsnorm(merged,w('pre_ff_layernorm.weight'),cfg['rms_norm_eps'])
    vi=V3Inspector(a.artifact); ff=f'text/layers/{a.layer}/ffn/'
    payload=lambda n:vi.payload(n)
    perm=lambda proj:np.frombuffer(payload(ff+proj+'/input_perm'),dtype='<i4').copy()
    gate=gptqref.matmul(payload(ff+'gate_proj'),(12288,3072),perm('gate_proj'),np.asarray(pre,dtype=np.float32))
    up=gptqref.matmul(payload(ff+'up_proj'),(12288,3072),perm('up_proj'),np.asarray(pre,dtype=np.float32))
    prod=torch.nn.functional.silu(torch.from_numpy(gate))*torch.from_numpy(up)
    down=gptqref.matmul(payload(ff+'down_proj'),(3072,12288),perm('down_proj'),prod.numpy())
    down=torch.from_numpy(down).to(torch.bfloat16).float().numpy()*cfg['mlp_multipliers'][1]
    expect=(merged+down).astype(np.float32)
    got=torch.from_numpy(np.fromfile(a.native_output,dtype='<u2').copy()).view(torch.bfloat16).float().numpy().reshape(expect.shape)
    d=got-expect; report={'layer':a.layer,'tokens':a.tokens,'same_artifact':True,'max_abs':float(np.max(np.abs(d))),'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(expect),1e-30)),'nonfinite':int((~np.isfinite(got)).sum()),'qualification':'NON_V100'}
    a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
