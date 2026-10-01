"""Version-3 inspection substrate and Falcon GPTQ canonical codec.

This module is deliberately target-agnostic: it validates framing, typed
objects, references, and component config without importing a target package.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, mmap, struct
from pathlib import Path
from typing import Any, Iterable

MAGIC = b"NINFER\x00\x03"
PREFIX = struct.Struct("<8sQ16s")
ALIGN = 4096
ARTIFACT_ID_BYTES = 16
PLANE_ALIGN = 256
GPTQ_FORMAT = "gptq_q4_g128_fp16"
GPTQ_LAYOUT = "gptq_canonical_k128_v1"
RAW = "raw-bytes-v1"

def align(v:int, a:int)->int: return (v+a-1)//a*a

def gptq_geometry(shape):
    n,k = map(int, shape)
    if n<=0 or k<=0 or k%128 or n%8: raise ValueError("GPTQ G128 requires positive [N,K], K divisible by 128 and N divisible by 8")
    groups=k//128
    qbytes=(k//8)*n*4
    zbytes=groups*(n//8)*4
    sbytes=groups*n*2
    zo=align(qbytes,PLANE_ALIGN); so=align(zo+zbytes,PLANE_ALIGN)
    return {"k":k,"n":n,"groups":groups,"qweight_bytes":qbytes,"qzeros_bytes":zbytes,"scales_bytes":sbytes,"qzeros_offset":zo,"scales_offset":so,"bytes":so+sbytes}

def _unpack_qweight(q, k, n):
    q=q.cast("I"); out=[[0]*n for _ in range(k)]
    for i in range(k//8):
        for j in range(n):
            w=q[(i*n)+j]
            for b in range(8): out[i*8+b][j]=(w>>(4*b))&15
    return out

def decode_gptq(payload:bytes, shape, input_perm=None):
    g=gptq_geometry(shape); n,k=g["n"],g["k"]
    q=memoryview(payload)[:g["qweight_bytes"]]; z=memoryview(payload)[g["qzeros_offset"]:g["qzeros_offset"]+g["qzeros_bytes"]]; s=memoryview(payload)[g["scales_offset"]:g["scales_offset"]+g["scales_bytes"]]
    out=[]
    for i in range(k):
        row=[]; group=i//128
        for j in range(n):
            qw=struct.unpack_from("<I",q,((i//8)*n+j)*4)[0]; code=(qw>>((i&7)*4))&15
            zw=struct.unpack_from("<I",z,(group*(n//8)+j//8)*4)[0]; zero=(zw>>((j&7)*4))&15
            scale=struct.unpack_from("<e",s,(group*n+j)*2)[0]
            row.append(float(scale)*(code-(zero+1)))
        out.append(row)
    # Generic codec returns canonical [N,K] only; input_perm is a separate
    # consumer concern and is deliberately not applied here.
    return [[out[i][j] for i in range(k)] for j in range(n)]

def restore_source_order(canonical, input_perm):
    """Return W_src from W_can where W_can[:,j] = W_src[:,input_perm[j]]."""
    k=len(input_perm)
    if any(len(row)!=k for row in canonical) or sorted(input_perm)!=list(range(k)):
        raise V3Error("input_perm must be a permutation matching canonical K")
    source=[[0]*k for _ in canonical]
    for j,source_k in enumerate(input_perm):
        for n,row in enumerate(canonical): source[n][source_k]=row[j]
    return source

def make_gptq_payload(qweight:bytes,qzeros:bytes,scales:bytes,shape):
    g=gptq_geometry(shape)
    if len(qweight)!=g["qweight_bytes"] or len(qzeros)!=g["qzeros_bytes"] or len(scales)!=g["scales_bytes"]: raise ValueError("GPTQ plane byte count mismatch")
    return qweight + bytes(g["qzeros_offset"]-len(qweight)) + qzeros + bytes(g["scales_offset"]-(g["qzeros_offset"]+len(qzeros))) + scales

@dataclass(frozen=True)
class V3Object:
    name:str; kind:str; offset:int; bytes:int; shape:tuple[int,...]=(); format:str|None=None; layout:str|None=None; encoding:str|None=None
    def json(self):
        d={"name":self.name,"kind":self.kind,"offset":self.offset,"bytes":self.bytes}
        if self.kind=="tensor": d.update(shape=list(self.shape),format=self.format,layout=self.layout)
        else: d.update(encoding=self.encoding)
        return d

class V3Error(ValueError): pass

def validate_directory(doc:dict[str,Any]):
    required={"components","objects","bindings","uses","files"}
    optional={"metadata","provenance"}
    if not required.issubset(doc) or set(doc)-required-optional: raise V3Error("v3 root members must contain components, objects, bindings, uses, files")
    if not isinstance(doc["files"], list) or not doc["files"]: raise V3Error("files must be a nonempty array")
    if not isinstance(doc["objects"], list) or not doc["objects"]: raise V3Error("objects must be a nonempty array")
    if not isinstance(doc["bindings"], list) or not isinstance(doc["uses"], list): raise V3Error("bindings and uses must be arrays")
    if not isinstance(doc["components"],dict) or not isinstance(doc["components"].get("text"),dict): raise V3Error("components.text is required")
    if not isinstance(doc["components"]["text"].get("config"),dict): raise V3Error("components.text.config is required")
    if any(x in doc["components"] for x in ("vision", "mtp", "proposal", "media")):
        raise V3Error("unsupported component must not be required")
    names=set()
    for o in doc["objects"]:
        if not isinstance(o,dict) or set(o)-{"name","kind","offset","bytes","shape","format","layout","encoding"}: raise V3Error("malformed object")
        if "offset" in o and (not isinstance(o["offset"],int) or o["offset"]<0): raise V3Error("invalid object offset")
        if "bytes" in o and (not isinstance(o["bytes"],int) or o["bytes"]<=0): raise V3Error("invalid object bytes")
        if not isinstance(o.get("name"),str) or o["name"] in names: raise V3Error("duplicate/invalid object name")
        names.add(o["name"])
        if o.get("kind") not in ("tensor","resource"): raise V3Error("invalid object kind")
        if o["kind"]=="tensor" and (not isinstance(o.get("shape"),list) or not o["shape"] or any(not isinstance(x,int) or x<=0 for x in o["shape"]) or not isinstance(o.get("format"),str) or not isinstance(o.get("layout"),str)): raise V3Error("malformed tensor")
        if o["kind"]=="resource" and not isinstance(o.get("encoding"),str): raise V3Error("malformed resource")
    binding_names=set()
    for b in doc["bindings"]:
        if not isinstance(b,dict) or not isinstance(b.get("name"),str) or not isinstance(b.get("object"),str) or b["object"] not in names: raise V3Error("binding references missing object")
        if b.get("name") in binding_names: raise V3Error("duplicate binding")
        binding_names.add(b.get("name"))
        for part in b.get("parts",[]):
            if not isinstance(part,dict) or part.get("object") not in names or int(part.get("offset",-1))<0 or int(part.get("bytes",0))<=0: raise V3Error("invalid binding part")
    pairs=set()
    for u in doc["uses"]:
        if not isinstance(u,dict) or not isinstance(u.get("binding"),str) or u["binding"] not in binding_names: raise V3Error("use references missing binding")
        pair=(u.get("parameter"),u.get("input"));
        if pair in pairs: raise V3Error("duplicate parameter/input use")
        pairs.add(pair)
        for a in u.get("auxiliaries",[]):
            if a not in binding_names: raise V3Error("invalid auxiliary binding")
    return doc

class V3Writer:
    def __init__(self,path,doc):
        self.path=Path(path); self.doc=dict(doc); raw=self.doc.pop("artifact_id", "00000000000000000000000000000000"); self.doc=validate_directory(self.doc); self.payload=bytearray()
        self.artifact_id=bytes.fromhex(raw)
        if len(self.artifact_id)!=ARTIFACT_ID_BYTES: raise V3Error("artifact_id must be 16 bytes")
    def add(self,name,data):
        if any(o["name"]==name for o in self.doc["objects"]): raise V3Error("duplicate object")
        o=next(x for x in self.doc["objects"] if x["name"]==name) if False else None
        raise V3Error("use planned payload writer; offsets must be precomputed")
    def write(self,payloads:dict[str,bytes]):
        objects=self.doc["objects"]; cursor=0; blob=bytearray();
        for o in objects:
            if o["name"] not in payloads: raise V3Error("missing payload: "+o["name"])
            cursor=align(cursor,1 if o["kind"]=="resource" else PLANE_ALIGN); o["offset"]=cursor; o["bytes"]=len(payloads[o["name"]]); blob.extend(b"\0"*(cursor-len(blob))); blob.extend(payloads[o["name"]]); cursor+=o["bytes"]
        if isinstance(self.doc.get("files"), list) and self.doc["files"]:
            self.doc["files"][0]["payload_bytes"] = cursor
            self.doc["files"][0].setdefault("path", None)
        directory=json.dumps(self.doc,ensure_ascii=False,separators=(",",":"),sort_keys=True).encode()
        start=align(PREFIX.size+len(directory),ALIGN); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("wb") as f: f.write(PREFIX.pack(MAGIC,len(directory),self.artifact_id)); f.write(directory); f.write(b"\0"*(start-f.tell())); f.write(blob)

class V3Inspector:
    def __init__(self,path):
        self.path=Path(path); data=self.path.read_bytes()
        if len(data)<PREFIX.size or data[:8]!=MAGIC: raise V3Error("not NInfer v3")
        n=PREFIX.unpack_from(data)[1]; self.artifact_id=data[16:32]; self.directory=json.loads(data[32:32+n]); validate_directory(self.directory); self.payload_offset=align(32+n,ALIGN); self.data=data
        if self.payload_offset + self.directory["files"][0].get("payload_bytes", 0) > len(data): raise V3Error("payload out of bounds")
    def payload(self,name):
        o=next(o for o in self.directory["objects"] if o["name"]==name); return self.data[self.payload_offset+o["offset"]:self.payload_offset+o["offset"]+o["bytes"]]
    def sha256(self): return hashlib.sha256(self.data).hexdigest()
