# Plan: S4 V100 single-card decode and prefill ceiling

## Current evidence

- Frozen S1 Falcon serving baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e` remains the only runtime; no model math/kernel changes. RTX 3090 historical ~495 tok/s is a reference only and has not been remeasured.
- Existing 32GB V100 decode observations (1K prompt/256 output), recovered and revalidated from literal-`\\n`-separated source rows: c96 sparse mean 391.20±0.98 tok/s (n=3), c104 400.74±0.80 (n=3), c112 411.20±1.20 (n=3). All outputs were complete/nonzero, graph histogram showed exactly 256 decode forwards at intended batch, and row-wise normalized JSONL is hash-checked under `/hy-tmp/sglang-logs/*validated.jsonl`. c112 leaves ~1.34GiB. These demonstrate neither hardware ceiling nor fully bounded software residual.
- c120 was KV-admission split 96+24, invalid as a full-batch saturation point. Previous graph memory result: c96 sparse ~0.30GB vs complete set ~0.51GB; c112 sparse `[1,112]` ~0.31GB vs complete ~0.55GB, without observed throughput loss. Revalidate from actual histogram.
- Existing S3 report audit: `s3-v100-nsys-decode` and `s3-v100-prewarmed` have no CUDA/kernel data; `s3-v100-traced-server` contains ~340ms NCCL init, 2 CUB inclusive-scan NVTX events, and no CUDA kernels. This is not an owner trace. The latest S4 wrapped launch loaded and warmed without crash; one env-gated NVTX hook was installed around graph96 decode worker forwards, but no Nsight report was emitted. This is a capture trigger/visibility failure, not confirmed crash. Full A/B and rollback details are in `experiments/S004-v100-decode-saturation/NSYS-CAPTURE-ATTEMPT-2026-10-05.md`.
- Nsight Compute 2025.1.1 `ERR_NVGPUCTRPERM` (even sudo) is an accepted, confirmed permission limitation. Current policy prohibits retries, driver/global changes and DCGM installation; targeted NCU is deferred unless already-elevated execution becomes available without host changes. Nsight Systems 2024.6.2 and no-CPU sampling configuration are present.
- Passwordless sudo is available, but does not bypass driver counter policy; no global driver state was changed. This does not block Nsight Systems or in-process event timing.
- A c96 S4 server was stopped before sending requests after GPU showed 29,670MiB foreign occupancy. Subsequent device check showed 1MiB/0% and no S4 process. No foreign process was terminated.
- Curves/correctness caveats are recorded in `experiments/S004-v100-decode-saturation/README.md` and `experiment.md`. JSON normalization audit is in `PROFILE-ARTIFACT-AUDIT.md`. Capture launch, missing report, A/B and rollback are documented in `NSYS-CAPTURE-ATTEMPT-2026-10-05.md`. The NVTX patch has been removed; no profile report/GPU timeline exists. Do not add speculative profiler harnesses. Diagnose existing range delivery or use in-process CUDA-event/NVTX fallback.

## Decisive frontier

First: establish a verifiable way for existing NVTX range from the model worker to reach the supported process-wrapped Nsight capture; two launches did not produce reports, though neither crashed. Do not add another profiler harness without concrete diagnosis. If no low-risk range delivery path exists, use already authorized in-process CUDA-event/NVTX instrumentation as fallback and clearly mark missing Nsight evidence. Defer NCU/DCGM and do not start prefill until decode owner coverage closes.

## Remaining sequence

1. Close any remaining decode curve point only if scheduler/hardware evidence justifies it; no new broad sweep. Keep graph96 A/B data as current accepted operating observation.
2. Establish existing Nsight range visibility from the model-worker process without writing a new speculative harness. If no direct range verification path exists after the two no-report wrapped launches, use already authorized in-process CUDA event/NVTX owner timing or explicitly leave timeline unresolved.
3. Under policy, defer NCU/DCGM. From any successful Systems capture (or CUDA-event fallback), join kernels to Falcon callsites/semantic roles, quantify GPU-active/idle/gap and rank total device time/count. No optimization before a validated ledger.
4. Inspect mature frozen/upstream/1Cat donor routes for top evidence-backed residuals; no kernel work before separate explicit authorization.
5. For each leading residual inspect mature frozen `sglang-V100`/upstream/1Cat V100 donor routes; quantify realistic recoverable software budget. No kernel work before new explicit authorization.
6. Once decode is closed, separately benchmark 1K/4K/~8K prefill with warmed minimal capture and its own owner ledger; assess prefill graphs only with service A/B.
7. Complete optimization/service A/B, residual before/after, and evidence-bounded software vs hardware remainder. SKU economics/8-card estimates remain paused.

## Current decisive state

ACTIVE. No valid Nsight GPU timeline, decode owner ledger, prefill owner ledger, or proven ceiling exists. Two supported profile-wrapped launches completed warmup and c96 A/B without crashing, but did not yield a `.nsys-rep`; the second included env-gated worker NVTX around graph96 and still lacked confirmed capture. Instrumentation was rolled back to prior disposable-source state, and the GPU is idle. NCU/DCGM are deferred per current user policy. Next decisive issue is verifiable NVTX process-scope range delivery or an already-authorized in-process CUDA event/NVTX fallback.
