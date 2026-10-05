# S4 artifact audit — existing traces and decode JSONL

Date: 2026-10-05. Offline analysis only; no GPU work was performed.

## Existing Nsight Systems artifacts

Re-ran installed `nsys stats` on three prior S3 reports:

| Report | CUDA API / GPU kernel evidence | Finding |
|---|---|---|
| `s3-v100-nsys-decode.nsys-rep` | CUDA kernel and CUDA API reports both skipped: no CUDA trace/kernel data | Client-side capture does not include server activity. |
| `s3-v100-prewarmed.nsys-rep` | CUDA kernel and CUDA API reports skipped: no CUDA trace/kernel data | Prewarmed variant still contains no server CUDA work. |
| `s3-v100-traced-server.nsys-rep` | API summary exists; **0 CUDA kernels**. 1,048 `cuGetProcAddress_v2`, 2 `cudaGetDeviceProperties_v2`, 2 `cuInit`; NVTX shows `NCCL:ncclCommInitRankConfig` 340.14ms and 2 `CCCL:cub::DeviceScan::InclusiveScan` ranges. | Trace terminated during scheduler/NCCL/CUB warmup; no Falcon forward timeline. Does not answer active/idle/replay or owner share. |

These findings reaffirm no valid historical GPU owner trace; do not use S3 time report as S4 hardware evidence. A single distinct capture—warmed stable window, graph-level, bounded NVTX, CUDA-only—is still the authorized next attempt; if it fails, inspect crash/tool/runtime interaction and do not spawn a second harness.

## Recoverable existing S4 throughput rows

Original S4 writers emitted multiple JSON documents separated by literal bytes `\\n`, so strict `json.loads()` on the full file fails. The separators are unambiguous; each individual object was independently parsed, correctness flags inspected, and complete compact newline-delimited normalized files generated under `/hy-tmp/sglang-tmp/`. This repairs the delimiter for analysis, not the source capture. The canonical damaged inputs were left untouched.

Strictly correct three-repeat subsets:

| Input | n | Aggregate tok/s mean ± sample SD | Exact scheduler batch histogram |
|---|---:|---:|---|
| `s4-v100-c96-sparse.jsonl` | 3 | 391.20 ± 0.98 | 256 decode passes at 96 (and 0 at others) |
| `s4-v100-c104.jsonl` | 3 | 400.74 ± 0.80 | 256 at 104 |
| `s4-v100-c112-itl-correct.jsonl` | 3 | 411.20 ± 1.20 | 256 at 112 |
| `s4-v100-c112-unstreamed.jsonl` | 3 | 316.47 ± 0.68 | 512 at 53 + 256 at 6; this was token-pool/admission-split, not full batch |
| `s4-v100-c120.jsonl` | 3 | 311.33 ± 0.72 | 256 at 96 + 256 at 24; split |
| `s4-v100-c128-unstreamed.jsonl` | 3 | 322.73 ± 0.26 | 512 at 53 + 256 at 22; split |

`c96` and `c112` sparse graph-vs-throughput values are not new repeats and still lack continuously sampled peak power/utilization; they are observations only, not saturation/ceiling proof. The c96 sparse hash (newline-normalized) is `6b021e7954419b74617f9e45a9c5f4237acb98d113e3c509a1fd93e71d1180c5`; c104 `0ef1e72e36602b3e483025f2e17f9d54b1155dcd7ccc168a5930de18f8f3db89`; c112 `a64c9f4724aac89558284214060303b44fd02bfd4b25ced3d637956bfa2d551e`. Hashes are for normalized files, not original damaged byte streams.

## S4 visibility / limitation

At audit time V100 measured 1MiB, 0% GPU, ~43W idle. Nsight Systems existing reports have no usable kernel activity; NCU cannot enumerate counters with current `RmProfilingAdminOnly=1`; no DCGM tool/service exists. No S4 NVIDIA-backed semantic owner ledger or ceiling can currently be asserted.
