# S004 — V100 decode saturation and CUDA-Graph coverage optimization

## Frozen identity and scope

- Falcon correctness baseline: `07343bd165855e9cd35ae63acea6cee5819c8c4e`.
- Execute only on `gpushare-v100` Tesla V100-SXM2-32GB using frozen source/runtime extraction `/hy-tmp/sglang-V100-s2-07343bd165` and pinned checkpoint revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`.
- RTX 3090 is fixed reference only; no access/build/run/profile/requalification.
- No new kernels or model math. Prefill graph optimization deferred.
- The frozen `sglang-V100` repository is not mounted in the current local runtime. Current durable S4 experiment and control records are under `ninfer-v100-falcon-h1-7b/experiments/S004-v100-decode-saturation/` and its `control/G010-s4-v100-decode-saturation/`; remote raw logs are under `/hy-tmp/sglang-logs/`. Earlier local attempts under `.pi/experiments` in a different repository are not authoritative.

## Objective

Determine V100 steady-state decode throughput saturation by increasing concurrency and maximizing graph replay coverage. Record aggregate/per-request tok/s, ITL, graph/eager pass deltas, actual scheduler batch histogram, VRAM, power and utilization. Compare graph sets from observed distribution; use smallest set covering useful operating region. Add/validate only low-overhead framework/CUDA-event telemetry; no repeated Nsight injection or privileged NCU counter attempts.

## Workload and controls

~1K prompt, 256+ generated tokens, radix disabled initially, identical correctness harness; c=1,2,4,8,12,16,24 as memory/scheduler permits, 3+ stable correct repeats. Every graph capture set must cover target batch sizes; if 24 is impossible, record concrete failure and proceed within resource envelope. Separate config effects from batch-cap effects. Check output correctness each run.

## Historical accepted anchor and references

S3 graph-set A/B on 32GB V100, c8/1K/256: capture `[1,2,4]` at max-running=8 = 75.72 tok/s, decode graph/eager passes 0/256 per repeat; capture `[1,2,4,8]` = 154.00 tok/s, 255/0, 256/0, 256/0. Improvement +103.4%. Outputs correct; prompt remained eager. Mamba marginal state = 0.1320 GiB/logical request plus one fixed 0.1320 GiB dummy slot. Full result and raw hashes in sibling `../E021-s3-v100-optimization/results.md`.

Frozen 3090 historic references: c8 371.2 tok/s, c12 485.7, c24 peak 495.1, steady B16 459.53. They are not fresh comparisons.

## Observed interim results (final milestone still active)

Every listed concurrency wave uses 1K prompt, 256 output, disabled radix cache, frozen V100 runtime and a graph set containing the target batch. Correctness gate is a complete non-streamed response with 256 nonzero token IDs per request; different requests can differ in a few token IDs under concurrent greedy reduction, so cross-request ID equality is not required. SGLang's native graph pass counter plus the disposable-source histogram give replay and actual batch sizes. Peak memory/utilization/power snapshots still need complete per-wave durable captures.

| Concurrency | mean aggregate tok/s (3 repeats) | mean per-request tok/s | mean ITL ms | dominant decode batch graph coverage | relevant resources |
|---:|---:|---:|---:|---|---|
| 1 | 31.86 | 31.86 | ~30.2 | graph1, ~100% | — |
| 2 | 57.23 | 28.62 | ~33.0 | graph2, ~100% | — |
| 4 | 99.65 | 24.91 | ~36.6 | graph4, ~100% | — |
| 8 | 153.00 | 19.13 | ~46.3 | graph8, ~100% | S3 result independently accepted |
| 12 | 172.74 | 14.40 | ~61.2 | graph12, ~100% | — |
| 16 | 203.92 | 12.74 | ~67.9 | graph16, ~100% | — |
| 24 | 245.49 | 10.23 | ~82.8 | graph24, ~100% | graph set `[1,2,4,8,12,16,24]` |
| 32 | 282.24 | 8.82 | ~94.3 | graph32, ~100% | earlier max_bs24 config was KV/request limited and gave invalid lower point (~213.6); correct point used max_total_tokens 49,152, graph set to 32 |
| 40 | 306.66 | 7.67 | ~107.1 | graph40, ~100% | — |
| 48 | 314.57 | 6.55 | ~124.8 | graph48, ~100% | — |
| 56 | 339.25 | 6.06 | ~133.1 | graph56, ~100% | — |
| 64 | 359.17 | 5.61 | ~138.5 | graph64, ~100% | 8.09GB weights, KV 5.50GB (token capacity 131,072), graph 0.51GB in comparison server |
| 80 | 375.07 | 4.69 | ~168.2 | graph80, ~100% | same c96 server/pools |
| 96 | 391.33 | 4.08 | ~192.5 | graph96, ~100% | 8.09GB weights, KV 7.04GB (167,664 tokens), graph 0.51GB; startup free headroom 2.54GB |
| 104 | 400.74 | 3.85 | not captured validly | graph104, ~100% | KV 6.00GB, graph 0.31GB, but startup free headroom <2GB |
| 112 | **410.64** | 3.66 | **210.8** | graph112, ~100% | KV 6.21GB (148,080 tokens), graph 0.31GB; Mamba cache size 112 => state increment 14.79GiB plus padding; startup free headroom only 1.34GB |
| 120 | 311.33 (not plateau evidence) | 2.59 | — | observed batch96 + batch24, both graphs | KV pool max 123,344 tokens; 120×(1024+256) workload triggers admission/batch splitting |
| 128 | 322.73 (not plateau evidence) | 2.52 | — | observed batch53 + batch22 | 68,688-token effective cap, KV-limited; full batch128 did not form |

The sustained c112 point is the largest measured full batch before the minimum 2GiB margin is violated; it is a **resource-bound best point, not a demonstrated unconstrained GPU throughput plateau**. Aggregate throughput still rose from c96→104→112. At c120, `max_total_tokens=260000` did not ensure enough finalized pool: actual 123,344 tokens, so the run split 96+24 and regressed. At c128 with the normal memory fraction, effective pool 68,688 tokens caused 53+22 batches. Lowering `mem_fraction_static` to 0.80 left only 8,880 tokens and caused pathological 6/2 microbatches; that run was stopped. Do not interpret these resource-limited curves as decoder saturation. Additional concurrency beyond 112 needs a changed memory-capacity plan, not just more graph entries.

### Graph set tradeoff

For c96, fixed `mem_fraction_static=.92`, exact same model/runtime/KV pool and concurrency:

| capture set | graph memory | graph pass/batch observations | c96 mean TPS | result |
|---|---:|---|---:|---|
| `[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96]` | 0.51GB | 255–256 graph96 forwards per 256-token request wave; histogram exactly 96×256 | ~391.39 | correct |
| `[1,96]` | 0.30GB | 255–256 graph96 forwards; histogram exactly 96×256 | ~391.14 | correct |

Sparse graph set saves ~0.21GB with no material throughput change at steady c96. For c112, `[1,112]` used 0.31GB; the broader `[1,2,4,...,112]` set used 0.55GB and both gave ~410 tok/s. S1 graph padding uses the smallest captured size >= batch; tested sustained requests hit exact sizes, so padding waste was zero for decode waves. Initial prefill was eager and contributes its separate prefill_none count; exclude it from decode coverage.

### Telemetry and measurement caveats

- The disposable-source S4 histogram adds one integer increment per decode iteration behind `SGLANG_S4_BATCH_HISTOGRAM=1`; `/server_info` exposes cumulative counts. The graph/eager pass counter is framework-maintained per completed decode forward. A request wave of output length 256 therefore expects ~256 decode passes per output token; measurements showed graph passes 255/256, and histogram showed exact batch 96/112 counts. This is low overhead by source inspection, but an explicit same-config telemetry-on vs off benchmark pair is not yet recorded; report the overhead validation as remaining.
- Early streaming runner incorrectly treated returned cumulative `output_ids` as SSE deltas and allowed incomplete/nonzero token outputs. Those early cases are excluded from accepted correctness data. Valid reruns use non-stream response completeness checks; corrected SSE parsing does cumulative-length differencing and measured c112 mean ITL ~210.8ms.
- Several early raw JSONL files contain literal `\\n` separators and/or oversized request payload rows; they are not valid manifests. The final compact-runner file was intended to correct this but still did not persist correctly in the current visible execution. Terminal results and server logs are observed; no claim of clean final JSONL artifact is made yet.
- GPU readings taken after requests were generally idle snapshots (e.g. ~59–73W, 0% utilization), not peak-over-wave telemetry. A 250ms `nvidia-smi` sampler script exists, but was not successfully wrapped around the accepted measurements. Peak power/utilization per workload remains unverified.

### Capture set memory details

- graph set `[1,2,4,8,12,16,24]`: 0.21GB for graph4 benchmark; `[1,2,4,8]`: 0.22GB.
- c32 `[1,2,4,8,12,16,24,32]`: 0.32GB.
- c48 `[1,2,4,8,12,16,24,32,40,48]`: 0.37GB.
- c64 `[1,2,4,8,12,16,24,32,40,48,56,63,64]`: 0.43GB. Exact 64 batch required token pool >=~82K; with 90,112 configured tokens the runtime admitted 64 and sustained c64 was ~360 tok/s. The earlier c64 pool configured at 81,920 admitted only batch63 due token availability and gave misleading ~308 tok/s.
- c96 sparse capture reduces graph buffers from 0.51GB to 0.30GB. This extra memory headroom is measurable (~0.21GB), but on 32GB it does not increase TPS.

### Fixed RTX 3090 reference comparison

c8 V100 153.0 / historical 3090 371.2 = **0.412×**; c12 V100 172.7 / 3090 485.7 = **0.356×**; c24 V100 245.5 / historical 3090 c24 495.1 = **0.496×**. For optimized c112 no like-batch 3090 point exists; compare only to 3090 peak gives 410.6/495.1 = **0.829×**, explicitly unlike-batch and not a hardware ceiling ratio. Scheduler/eager loss is bounded at tested graph-covered batches; owner kernel and GPU busy/idle attribution remain unmeasured, so no residual difference is attributed exclusively to hardware.

### 16GB conditional short-context estimates (no physical qualification)

S2 baseline uses 8.09GB weights, residual runtime/static allowance rounded up to 2.5GiB, graphs budget 0.25GiB (S4 sparse sets require 0.30GB at c96/c112), and >=2GiB safety margin. FP16 KV is ~45,056 bytes/token (44KiB). Mamba per logical request is 0.1320GiB plus one 0.1320GiB dummy slot. At 1K prompt+256 output per request, 4K tokens cover up to 4 requests conservatively; 24,576 token pool recommended by S2 covers 8 such requests. The S2 conservative operational default remains c4 and c8 only for short contexts after actual-device validation. S2 already recommended 16GiB 24,576 total tokens, default max-running 4, graphs 1/2/4, no piecewise, >=2GiB remaining. This remains conditional, not physical qualification.

An S4 high-concurrency extrapolation from the actual 32GB configuration is infeasible to assert on 16GB: Mamba alone uses (S+1)×0.1320GiB, so c8/c12/c16 = 1.19/1.72/2.24GiB; KV for 24,576 tokens ~1.03GiB and graph ~0.30GB. Combined with weights 8.09GB + runtime reserve 2.5GiB + >=2GiB safety, c8 is a plausible *budget candidate* (~15.1GiB before conventions/allocator); c12 is marginal/no safety (~15.6GiB); c16 exceeds (~16.1GiB). S2 keeps the operational recommendation c4 by default and c8 as the conditional short-context cap; do not promote c12/c16 without a physical 16GB run and exact allocator/cap evidence.

## Remaining S4 work

S4 acceptance is not yet met. Required remaining tasks: (1) produce clean, parseable raw per-wave JSONL for c1-112 after final harness correction; (2) get c96/c112 telemetry on/off overhead paired; (3) capture per-wave peak power, utilization and VRAM with SMI sampler; (4) quantify eager pass rate/padding from exact histogram at operating points and graph-set same-workload A/B; (5) establish >=2GiB-headroom highest useful c point and rank/root bottleneck; (6) finish 16GB budget with all conventions explicit; (7) update report/control and push commit. No profiler retries or prefill-graph tests in S4.
