# S4 progress record — 2026-10-05

This is an ACTIVE interim evidence log, not the S4 completion report. The latest priority supersedes interim plans to examine SKU economics or 8-card estimates. First establish 32GB V100 decode NVIDIA owner ledger/saturation; prefill separately only after decode closes. The canonical runtime source is frozen S1 (`07343bd165855e9cd35ae63acea6cee5819c8c4e`) in remote extraction `/hy-tmp/sglang-V100-s2-07343bd165`. RTX 3090 was not accessed. The local `sglang-V100` repository is not mounted in this execution runtime; this record and control state therefore live in the currently mounted `ninfer-v100-falcon-h1-7b` repository. Earlier `.pi` records copied to a different SGLang checkout were removed from consideration.

## Valid completed evidence

- Workload: 1,024-token prompt, 256 decode tokens, no radix, exact graph at each tested operating batch; 32GB V100-SXM2-16GB; only frozen S1 runtime/model math.
- Valid correctness gate: non-stream `/generate` full response contained 256 nonzero output IDs per request, no errors. Simultaneous request outputs can differ by a few token IDs under concurrent greedy floating-point reductions; exact across-request equality is not required.
- Throughput curve (all entries have 3 terminal-printed correct repeats unless marked): c1=31.86, c2=57.23, c4=99.65, c8=153.00, c12=172.74, c16=203.92, c24=245.49, c32=282.24, c40=306.66, c48=314.57, c56=339.25, c64=359.17, c80=375.07, c96=391.33, c104=400.74, c112=410.64 aggregate tok/s.
- The final compact S4 histogram records at c112 show graph batch112 histogram 255–256 per 256-token output wave, confirming exact-batch graph replay. Full graph set costs 0.55GB; sparse `[1,112]` costs 0.31GB. At c96, sparse `[1,96]` costs 0.30GB vs complete interval/sparse set 0.51GB; throughput is ~391 tok/s either way. Selection is based on exact operating point; do not use `[1,96]` as a general multi-concurrency set.
- c112 max-running112 on mem_fraction_static .92: startup available after pool/capture 1.34GiB (<2GiB safety threshold), weight 8.09GB, KV 6.21GB/148,080 token capacity, graph 0.31GB, Mamba pool max 112. This is the measured full-batch c112 peak, but not the valid production choice under stated safety rule.
- c96 on mem_fraction_static .92: startup free 2.54GiB, weights 8.09GB, KV 7.04GB/167,664 tokens, graph 0.51GB; sparse graph reduces graph memory to 0.30GB and free headroom to ~2.75GB, with unchanged throughput. Use c96 sparse as the current highest c candidate meeting the 2GiB margin, subject to final resource-sampler validation.
- c120 had only a finalized 123,344-token KV pool and formed batches96+24; ~311 tok/s, not a hardware saturation point. c128 at normal mem fraction finalized only 68,688 tokens and formed batches53+22; ~323 tok/s. Both are KV-limited. A trial with mem_fraction_static=.80 finalized 8,880 tokens and generated pathological microbatches; stopped, rejected.
- Corrected Mamba formula is frozen at 0.1320GiB/logical request + one fixed dummy state slot. State sizes at c96/112 are ~12.8/14.9GiB including dummy, respectively.
- Fixed historical 3090 ratios: V100 c8 153.0 / 3090 c8 371.2 = 0.412; c12 172.7 / 485.7 = 0.356; c24 245.5 / 495.1 = 0.496. V100 c112 410.6 / 3090 peak c24 495.1 = 0.829 is unlike-batch only; not a hardware-ceiling comparison.

## Invalid/excluded data and instrumentation caveats

- Early SSE harness counted the current cumulative `output_ids` array as a delta and accepted partially observed results; exclude its correctness results. Later non-stream full-response gate fixes output completeness. The corrected SSE parser differences cumulative lengths and estimates per-token arrival time from packets, but timings are coarse, and two earlier attempted files remain damaged.
- Several early remote JSONL files have literal `\\n` separators and/or oversized per-request arrays (Linux PIPE_BUF atomic-write issue), so they are not parseable raw artifacts. Terminal result lines and server logs provide the observations above; no hash is claimed for valid c1–112 JSONL. A final attempt to produce a valid c112 JSONL repeated the same record-format failure. This remains an artifact defect to correct before final S4 acceptance.
- A disposable-source opt-in `SGLANG_S4_BATCH_HISTOGRAM` remains in the remote extraction; it adds one integer increment per completed decode forward and serves a cumulative histogram. For the latest capture attempt, an additional env-gated NVTX `range_start/end` was installed around graph96 worker forwards; two profile-wrapped launches produced no report and it has now been removed. The remote `scheduler.py` matches frozen S1 blob plus the pre-existing histogram-only patch; no profile NVTX code or S4 server process remains. NCU/CUPTI permissions are deferred as directed.
- Power/SMI samples were occasional idle snapshots (typically 59–73W, 0% GPU utilization), not peak-over-wave measures. `snapshot_smi.py` sampler exists but was not successfully wrapped around accepted benchmark waves. Per-wave peak GPU utilization, power and VRAM remain unmeasured.
- No S4 profiler run; this follows user's instruction to avoid known-failing server Nsight capture/privileged counters. S3 Nsight/NCU limitations remain. Prefill graphs remain explicitly deferred.

## Conditional 16GB estimate

Use S2 stated budget: 8.09GB weights, 2.5GiB runtime/static allowance, ~0.25GiB graph allowance, ~2GiB mandatory headroom, FP16 KV 45,056B/token (~44KiB), and Mamba `(S+1)*0.1320GiB`.

- 24,576-token KV pool costs ~1.03GiB. c8 Mamba state costs ~1.19GiB incl dummy; c12 ~1.72GiB; c16 ~2.24GiB. Combined footprint estimates: c8 roughly 15.1GiB, c12 ~15.6GiB (no robust 2GiB margin), c16 ~16.1GiB (over nominal 16GiB). These accounting estimates use mixed decimal/GiB conventions and are conservative screening, not physical qualification.
- Keep S2 c4 default/24,576 total-token candidate. c8 is only plausibly deployable for short contexts if actual 16GB startup shows >=2GiB free; c12 is marginal and c16 does not fit in the estimate. No physical 16GB claim.

## Still required for S4 completion

- Repeat c96 sparse/full paired samples with validated compact JSONL recording and same memory configuration; record graph memory tradeoff and pass/batch histograms.
- Use `snapshot_smi.py` correctly for peak VRAM, utilization, power/temperature during each wave; validate opt-in histogram overhead against disabled telemetry.
- Confirm highest useful operating point with >=2GiB startup reserve (candidate c96 sparse); classify c112 as throughput-only, below requested reserve. Determine if safe max is c96 or another point between 96 and 112 under a modestly adjusted reserve/pool setting.
- Finalize exact production CLI, graph sets, historical 3090 comparisons and bounded top remaining decode owner uncertainty. Do not infer hardware bottleneck without actual GPU busy/idle/owner attribution.
- Update this report/control, verify all output files, stop S4 server and confirm V100 1MiB/0%, then commit/push the relevant stage in the correct repository when its mounted path is available.
