"""Pinned Falcon-H1 source inventory and v3 artifact role mapping.

This module contains only source-to-artifact inventory declarations. It does
not register a runtime target or construct an executable model.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MODEL_ID = "falcon-h1-7b"
WEIGHTS_ID = "gptq-int4"
TARGET_KEY = "falcon_h1_7b"
CHECKPOINT = Path("/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4")
REVISION = "52036ae497a2c0c253e17f98cc8652f2f7082ae3"
GPTQ_FORMAT = "gptq_q4_g128_fp16"
GPTQ_LAYOUT = "gptq_canonical_k128_v1"
BF16_FORMAT = "bf16"
DIRECT_LAYOUT = "contiguous_le_v1"
I32_FORMAT = "int32"
RAW_ENCODING = "raw-bytes-v1"

@dataclass(frozen=True)
class SourceTensor:
    name: str
    disposition: str
    artifact_name: str
    role: str


def source_names(checkpoint: Path = CHECKPOINT) -> tuple[str, ...]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return tuple(sorted(index["weight_map"]))


def classify(name: str) -> str:
    if ".feed_forward." in name and name.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
        return "gptq_auxiliary_source"
    if ".feed_forward." in name and any(name.endswith(f".{x}") for x in ("gate_proj", "up_proj", "down_proj")):
        return "gptq_projection_source"
    return "bf16_tensor_source"


def inventory(checkpoint: Path = CHECKPOINT) -> tuple[SourceTensor, ...]:
    out=[]
    for name in source_names(checkpoint):
        if name.endswith((".qweight", ".qzeros", ".scales")):
            base=name.rsplit(".",1)[0]
            out.append(SourceTensor(name,"gptq_plane_source",base,"gptq_plane"))
        elif name.endswith(".g_idx"):
            base=name.rsplit(".",1)[0]
            out.append(SourceTensor(name,"gptq_auxiliary_source",base+".input_perm","input_perm_source"))
        else:
            out.append(SourceTensor(name,"bf16_tensor_source",name,"bf16_tensor"))
    return tuple(out)


def summary(checkpoint: Path = CHECKPOINT) -> dict[str, object]:
    rows=inventory(checkpoint)
    return {"model_id":MODEL_ID,"revision":REVISION,"source_tensor_count":len(rows),"dispositions":{k:sum(r.disposition==k for r in rows) for k in sorted({r.disposition for r in rows})},"gptq_projection_bases":sorted({r.artifact_name for r in rows if r.role=="gptq_plane"})}

if __name__ == "__main__":
    print(json.dumps(summary(),indent=2))
