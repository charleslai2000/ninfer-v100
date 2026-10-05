# Goal: S4 V100 decode saturation and CUDA-Graph coverage optimization

## Authorized outcome

Starting from frozen Falcon serving correctness baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e`, establish the single-card V100-SXM2-32GB Falcon-H1 decode saturation point and separately establish prefill throughput/owners. Prioritize NVIDIA DCGM/Nsight hardware telemetry, targeted NCU on dominant kernels, and an ordered semantic owner ledger; graph coverage/sets are to be optimized against real scheduler batches. Pause purchase economics, 8-card estimates, and serving feature work. No new kernels or model-math changes without later explicit authorization. Preserve the S3 graph8 result and corrected Mamba accounting without reopening them.

## Scope and constraints

- Run only on the shared V100 using the frozen `sglang-V100` source/environment; RTX 3090 results are fixed historical references and must not be accessed/requalified.
- Separate prefill/decode; fixed ~1K prompt and >=256 output tokens; radix disabled initially; exact graph capture sizes must cover tested regimes.
- At least three stable correct repeats per concurrency/config; record throughput per request/aggregate/ITL, graph/eager counters, scheduler batch sizes, peak VRAM, power/utilization.
- Use supported Nsight Systems launch/capture-range for one low-distortion steady decode timeline. CUDA+NVTX, graph granularity, no CPU sampling/backtrace. If a first attempt fails, inspect tool/runtime interaction; do not add another profiler harness after an equivalent failure.
- Current NCU permission failure is accepted/confirmed by user; do not retry NCU under this state. No driver reload/global settings or DCGM host-engine installation. Existing elevated NCU may be considered later only if already available without changing host state. Use authorized in-process CUDA events/NVTX if NSight timeline cannot be established.
- Decode first: fixed 1K prompt/256+ output, near-100% graph coverage, stable full intended batches, graph memory/power/VRAM and aggregate/per-request TPS; maximize stable throughput, avoid OOM/instability. Then measure isolated 1K/4K/8K prefill separately. Never mix phase ledgers.
- Optimization order: profile → owner ledger → largest row → mature donor → implementation only after authorization → operator/service A/B → re-profile. No kernel implementation is authorized in this goal revision.
- Do not begin prefill until decode saturation and the decode owner ledger are closed. Defer SKU economics and multi-card estimates.

## Completion condition

A valid low-distortion Nsight Systems steady-decode timeline plus ordered semantic owner ledger (or a reproducible Nsight incompatibility with crash evidence and validated in-process CUDA-event/NVTX ledger) explains >=90% GPU-active forward time; gaps, replay cadence, graph coverage and top measured residual owners are ranked. Only then close decode saturation and proceed to separately bounded prefill owners. No V100 ceiling from throughput alone. Stop for a new kernel/architecture only after dominant residual and software-recoverable budget are evidenced.

## Result

ACTIVE. Prior S3 accepted anchor: c8/1K/256, graph4 75.72 vs graph8 154.00 tok/s (+103.4%), graph8 replay ~100%; marginal Mamba state 0.1320 GiB/request plus one fixed dummy slot. Full S4 results pending.
