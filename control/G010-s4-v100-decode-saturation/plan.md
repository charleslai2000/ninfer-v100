# Plan: S4 V100 single-card decode and prefill ceiling

## Current evidence

- Frozen S1 Falcon serving baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e` remains the only runtime; no model math/kernel changes. RTX 3090 historical ~495 tok/s is a reference only and has not been remeasured.
- Existing 32GB V100 decode observations (1K prompt/256 output), recovered and revalidated from literal-`\\n`-separated source rows: c96 sparse mean 391.20±0.98 tok/s (n=3), c104 400.74±0.80 (n=3), c112 411.20±1.20 (n=3). All outputs were complete/nonzero, graph histogram showed exactly 256 decode forwards at intended batch, and row-wise normalized JSONL is hash-checked under `/hy-tmp/sglang-logs/*validated.jsonl`. c112 leaves ~1.34GiB. These demonstrate neither hardware ceiling nor fully bounded software residual.
- c120 was KV-admission split 96+24, invalid as a full-batch saturation point. Previous graph memory result: c96 sparse ~0.30GB vs complete set ~0.51GB; c112 sparse `[1,112]` ~0.31GB vs complete ~0.55GB, without observed throughput loss. Revalidate from actual histogram.
- Existing S3 report audit: `s3-v100-nsys-decode` and `s3-v100-prewarmed` have no CUDA/kernel data; `s3-v100-traced-server` contains ~340ms NCCL init, 2 CUB inclusive-scan NVTX events, and no CUDA kernels (warmup ended before a Falcon forward). This is not an owner trace. Do one distinct minimally intrusive, warmed graph-level CUDA/NVTX capture with no CPU sampling/backtrace; if it crashes, inspect interaction and stop adding harnesses.
- Nsight Compute 2025.1.1 and Nsight Systems binaries exist on gpushare-v100. `ncu --query-metrics --devices 0` gives `ERR_NVGPUCTRPERM`; `/proc/driver/nvidia/params` says `RmProfilingAdminOnly:1`. `perf_event_paranoid=4`; Nsight reports CPU perf sampling unavailable. DCGM tools/service are absent.
- SSH session uid is root, but no authorized reversible module-admin procedure has been established; no driver module/sysctl state has been changed. Need legitimate profiling permission before counter claims.
- A c96 S4 server was stopped before sending requests after GPU showed 29,670MiB foreign occupancy. Subsequent device check showed 1MiB/0% and no S4 process. No foreign process was terminated.
- Curves/corrupt harness artifacts and correctness caveats are recorded in `experiments/S004-v100-decode-saturation/README.md` and `experiment.md`. Newer final protocol is in `NVIDIA-PROFILING-BLOCKER.md`.

## Decisive frontier

First: use authorized NVIDIA/DCGM permission route or establish DCGM availability; capture stable decode using one bounded warm NVTX range with graph-level trace and CUDA-only activity. Produce ordered semantic owner ledger and targeted NCU hardware metrics on the top few owners. Do not start prefill until decode saturation and owner analysis close.

## Remaining sequence

1. When shared V100 idle: repeat validated covered c96/c104/c108/c112 full-batch points (and justified intermediate points), at least three correct repeats; compact one-row-per-wave JSONL, histogram, graph/eager, VRAM, power/clocks/utilization. Optimize sparse graph set by actual scheduler distribution. Mark `MAX_THROUGHPUT`; do not enforce 2GiB safety reserve as user now prioritizes maximum stable throughput; prevent OOM/instability.
2. Install/use NVIDIA DCGM only with approved host process; enumerate actual supported metric groups. Capture repeated stable decode/prefill DCGM samples and A/B telemetry overhead.
3. One warm steady-state Nsight Systems server capture, graph-level, no CPU sampling/backtrace, bounded with NVTX; verify GPU timeline, GPU-active/idle fraction, replay gaps, API launches/sync, kernel families and counts. If crash, preserve logs and analyze tool/runtime interaction; no second harness.
4. Join kernels to Falcon callsites/semantic roles. Rank by total device time/count. Target NCU to leading few exact kernels with `--kernel-name`/range and documented metrics. If permissions cannot be obtained, use DCGM + CUDA-event real-shape bench but do not say NCU proved efficiency.
5. For each leading residual inspect mature frozen `sglang-V100`/upstream/1Cat V100 donor routes; quantify realistic recoverable software budget. No kernel work before new explicit authorization.
6. Once decode is closed, separately benchmark 1K/4K/~8K prefill with warmed minimal capture and its own owner ledger; assess prefill graphs only with service A/B.
7. Complete optimization/service A/B, residual before/after, and evidence-bounded software vs hardware remainder. SKU economics/8-card estimates remain paused.

## Current decisive state

ACTIVE. No NVIDIA-tool-backed Falcon owner ledger, DCGM metric capture, valid Nsight GPU timeline, targeted NCU result, prefill owner ledger, or proven saturation ceiling exists. Exact blocker is counter profiling restricted by driver parameter plus absent DCGM tooling; next decisive condition is authorized profiling-enablement procedure and stable isolated V100 window.
