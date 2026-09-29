#!/usr/bin/env python3
"""Same-artifact Falcon layer Attention reference for T018 qualification diagnostics."""
from __future__ import annotations
import argparse
from pathlib import Path
import struct
import numpy as np
def rope(x, positions, theta):
    x=np.asarray(x,dtype=np.float32).copy();d=x.shape[-1];half=d//2
    inv=theta**(-np.arange(0,d,2,dtype=np.float32)/d)
    angle=np.asarray(positions,dtype=np.float32)[...,None]*inv;c=np.cos(angle);s=np.sin(angle)
    while c.ndim<x.ndim:c=c[:,None,:];s=s[:,None,:]
    a=x[...,:half].copy();b=x[...,half:].copy()
    return np.concatenate((a*c-b*s,b*c+a*s),axis=-1)


def attention(q,k,v,positions,theta,scale,initial_k=None,initial_v=None):
    q=rope(q,positions,theta);k=rope(k,positions,theta)
    if initial_k is not None:kall=np.concatenate((initial_k,k),axis=0);vall=np.concatenate((initial_v,v),axis=0);past=len(initial_k)
    else:kall=k;vall=v;past=0
    t,h,d=q.shape;reps=h//k.shape[1];outs=[];probs=[]
    for ti in range(t):
        headout=[];headprob=[];upto=past+ti+1
        for hi in range(h):
            kvh=hi//reps;scores=kall[:upto,kvh]@q[ti,hi]*scale;scores-=np.max(scores);p=np.exp(scores);p/=p.sum();headout.append(p@vall[:upto,kvh]);headprob.append(p)
        outs.append(np.stack(headout));probs.append(headprob)
    return np.stack(outs),probs,kall,vall,q,k


def read_artifact(path):
    data = Path(path).read_bytes()
    if data[:8] != b"NINFER\0\3":
        raise ValueError("not a NInfer v3 artifact")
    n = struct.unpack_from("<Q", data, 8)[0]
    directory = __import__("json").loads(data[32:32 + n])
    payload_offset = (32 + n + 4095) // 4096 * 4096
    return data, directory, payload_offset


class ArtifactView:
    def __init__(self, path):
        import hashlib
        self.data, self.directory, self.payload_offset = read_artifact(path)
        self.sha256 = hashlib.sha256(self.data).hexdigest()

    def payload(self, name):
        obj = next(o for o in self.directory["objects"] if o["name"] == name)
        start = self.payload_offset + obj["offset"]
        return self.data[start:start + obj["bytes"]]


def tensor(artifact, name):
    obj = next(o for o in artifact.directory["objects"] if o["name"] == name)
    if obj["format"] != "bf16":
        raise ValueError(f"unexpected weight format for {name}: {obj['format']}")
    u = np.frombuffer(artifact.payload(name), dtype="<u2").copy().astype(np.uint32)
    return (u << 16).view(np.float32).reshape(obj["shape"])


def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("hidden_f32", type=Path, help="[T,3072] float32 input; rounded at Program BF16 boundary")
    ap.add_argument("output_npz", type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--position", type=int, required=True)
    ap.add_argument("--history", type=Path, help="optional same-artifact historical K/V NPZ")
    args = ap.parse_args()
    art = ArtifactView(args.artifact)
    cfg = art.directory["components"]["text"]["config"]
    raw = np.fromfile(args.hidden_f32, dtype="<f4")
    H = int(cfg["hidden_size"])
    if raw.size % H:
        raise ValueError("hidden fixture does not contain complete hidden rows")
    T = raw.size // H
    x = bf16(raw.reshape(T, H))
    positions = np.arange(args.position, args.position + T, dtype=np.int64)
    prefix = f"text/layers.{args.layer}.self_attn."
    xn = bf16(x * np.float32(cfg["attention_in_multiplier"]))
    q = bf16(xn @ tensor(art, prefix + "q_proj.weight").T)
    k0 = bf16(xn @ tensor(art, prefix + "k_proj.weight").T)
    v0 = bf16(xn @ tensor(art, prefix + "v_proj.weight").T)
    qh = q.reshape(T, 12, 128)
    kh_pre_scale = k0.reshape(T, 2, 128)
    kh = bf16(kh_pre_scale * np.float32(cfg["key_multiplier"]))
    vh = v0.reshape(T, 2, 128)
    initial_k = initial_v = None
    if args.history:
        with np.load(args.history) as hist:
            initial_k = np.asarray(hist["k"], dtype=np.float32)
            initial_v = np.asarray(hist["v"], dtype=np.float32)
        if initial_k.ndim != 3 or initial_k.shape[1:] != (2, 128) or initial_v.shape != initial_k.shape:
            raise ValueError("history NPZ must contain k/v [history,2,128]")
    out, probs, _, _, qr, kr = attention(qh, kh, vh, positions, float(cfg["rope_theta"]),
                                           128 ** -0.5, initial_k, initial_v)
    probs_array = np.zeros((T, 12, len(initial_k) + T if initial_k is not None else T), dtype=np.float32)
    for ti, row in enumerate(probs):
        for hi, values in enumerate(row):
            probs_array[ti, hi, :len(values)] = values
    pre_o = bf16(out)
    o_linear = bf16(pre_o.reshape(T, 1536) @ tensor(art, prefix + "o_proj.weight").T)
    output = bf16(o_linear * np.float32(cfg["attention_out_multiplier"]))
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, hidden=x, q=q, k_pre_scale=kh_pre_scale, k=kh, v=v0,
             q_rope=bf16(qr), k_rope=bf16(kr), attention_probabilities=probs_array,
             pre_o=pre_o, o_projection=o_linear, output=output, positions=positions)
    print(f"REFERENCE_PASS layer={args.layer} T={T} position={args.position} history={0 if initial_k is None else len(initial_k)} artifact_sha256={art.sha256} output={args.output_npz}")


if __name__ == "__main__":
    main()
