# Goal: S4 V100 decode saturation and CUDA-Graph coverage optimization

## Authorized outcome

Starting from frozen Falcon serving correctness baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e`, establish the single-card V100-SXM2-32GB Falcon-H1 decode saturation point and separately establish prefill throughput/owners. Prioritize NVIDIA DCGM/Nsight hardware telemetry, targeted NCU on dominant kernels, and an ordered semantic owner ledger; graph coverage/sets are to be optimized against real scheduler batches. Pause purchase economics, 8-card estimates, and serving feature work. No new kernels or model-math changes without later explicit authorization. Preserve the S3 graph8 result and corrected Mamba accounting without reopening them.

## Scope and constraints

- Run only on the shared V100 using the frozen `sglang-V100` source/environment; RTX 3090 results are fixed historical references and must not be accessed/requalified.
- Separate prefill/decode; fixed ~1K prompt and >=256 output tokens; radix disabled initially; exact graph capture sizes must cover tested regimes.
- At least three stable correct repeats per concurrency/config; record throughput per request/aggregate/ITL, graph/eager counters, scheduler batch sizes, peak VRAM, power/utilization.
- Establish DCGM metric support/permissions and Nsight Systems CUDA-only warm-window capture. No CPU sampling/backtrace; avoid repeating the failed full-server setup. Resolve performance-counter access only through an authorized, reversible NVIDIA-supported admin procedure.
- NCU only on a few dominant, ledger-identified kernels; if counters remain unavailable after authorized remedy, use explicitly bounded DCGM plus CUDA-event real-tensor microbenchmarks and report missing evidence.
- Decode first: fixed 1K prompt/256+ output, near-100% graph coverage, stable full intended batches, graph memory/power/VRAM and aggregate/per-request TPS; maximize stable throughput, avoid OOM/instability. Then measure isolated 1K/4K/8K prefill separately. Never mix phase ledgers.
- Optimization order: profile → owner ledger → largest row → mature donor → implementation only after authorization → operator/service A/B → re-profile. No kernel implementation is authorized in this goal revision.
- Do not begin prefill until decode saturation and the decode owner ledger are closed. Defer SKU economics and multi-card estimates.

## Completion condition

NVIDIA-tool-backed decode and isolated prefill owner ledgers explain most GPU-active forward time; targeted NCU or explicitly documented DCGM/CUDA-event substitutes quantify dominant kernel efficiency; decode saturation and phase-separated prefill operating points are repeatable; graph set is optimized from scheduler histogram; optimization A/B results and unrecovered residual are ranked; evidence distinguishes software-recoverable from hardware-irreducible gap. No ceiling claim from throughput alone. Stop for a new kernel/architecture only after a dominant residual is demonstrated and donor/software recovery is bounded.

## Result

ACTIVE. Prior S3 accepted anchor: c8/1K/256, graph4 75.72 vs graph8 154.00 tok/s (+103.4%), graph8 replay ~100%; marginal Mamba state 0.1320 GiB/request plus one fixed dummy slot. Full S4 results pending.
