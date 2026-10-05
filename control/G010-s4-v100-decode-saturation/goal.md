# Goal: S4 V100 decode saturation and CUDA-Graph coverage optimization

## Authorized outcome

Starting from frozen Falcon serving correctness baseline `07343bd165855e9cd35ae63acea6cee5819c8c4e`, determine the steady-state decode throughput saturation curve on the 32GB V100 by increasing concurrency and maximizing scheduler batch formation/CUDA-Graph coverage. No new kernels and no model math changes. Preserve the S3 graph8 result and corrected Mamba accounting without reopening them.

## Scope and constraints

- Run only on the shared V100 using the frozen `sglang-V100` source/environment; RTX 3090 results are fixed historical references and must not be accessed/requalified.
- Separate prefill/decode; fixed ~1K prompt and >=256 output tokens; radix disabled initially; exact graph capture sizes must cover tested regimes.
- At least three stable correct repeats per concurrency/config; record throughput per request/aggregate/ITL, graph/eager counters, scheduler batch sizes, peak VRAM, power/utilization.
- Use existing framework telemetry and CUDA events/NVTX with low overhead; validate overhead against uninstrumented throughput. Do not retry the known-failing Nsight server capture or privileged NCU counters.
- Quantify graph set cost/coverage/padding and choose smallest useful set. Update 16GB feasibility with weights 8.09GB, reserve, graph memory, KV and Mamba `(S+1)*0.1320 GiB`, retaining >=2 GiB margin; estimate only, no physical qualification.
- Do not start prefill graph work in S4.

## Completion condition

A correct repeated decode concurrency curve reaches plateau/regression under high graph coverage; graph-set tradeoff is quantified; scheduler/eager losses are bounded with validated low-overhead telemetry; V100 optimized TPS and 3090 ratio are reported without hardware-only attribution unless software loss is bounded; 16GB c8/c12/c16 estimates are explicit; production config is recommended. Stop only if further decode gains demonstrably require a new dominant kernel/architecture change.

## Result

ACTIVE. Prior S3 accepted anchor: c8/1K/256, graph4 75.72 vs graph8 154.00 tok/s (+103.4%), graph8 replay ~100%; marginal Mamba state 0.1320 GiB/request plus one fixed dummy slot. Full S4 results pending.
