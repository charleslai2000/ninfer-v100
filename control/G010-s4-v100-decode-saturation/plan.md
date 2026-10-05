# Plan: S4 V100 decode saturation and graph coverage

## Current evidence

- Frozen S1 V100 base is correct; no model/kernel changes. V100 access remains only on `gpushare-v100` extraction `/hy-tmp/sglang-V100-s2-07343bd165`; final active benchmark server has been stopped, GPU 1 MiB/0%.
- Correctness test bug found and fixed for measurements: original harness accepted partial SSE deltas and did not require all 256 token IDs nonzero. Use complete non-stream responses as gate; do not infer correctness from SSE array length. Different requests may produce slightly different greedy token sequences, so exact cross-request sequence equality is not a correctness condition.
- Source-derived SGLang decode-batch histogram instrumentation is opt-in and does one integer increment per completed decode step. It was applied only to disposable remote extraction, not repo source.
- Curves show full-batch throughput continues rising through c112. c120 was KV-limited (actual batches 96+24, ~311 tok/s). c128 at normal static reserve hit max_total token-pool cap and split into smaller batches. Lowering `mem_fraction_static` to .80 left only 8,880 KV tokens and caused unusably tiny microbatches; rejected.
- At c112/max-running112, `mem_fraction_static=.92`, graph set `[1,112]`: weights 8.09GB, KV 6.21GB/148,080 tokens, graph 0.31GB, pool Mamba allocated 112 slots (temporal 14.57GB logged, plus conv), available after pool/capture only 1.34GB. This does not meet required 2GB safety margin.
- c112 with full capture `[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112]` consumed 0.55GB graph vs 0.31GB `[1,112]`; both measured ~410 tok/s, so the sparse exact operating-point set saves ~0.24GB without measurable throughput loss for sustained c112. Need choose stable production baseline respecting headroom.
- c96 same-policy complete requests about 391 tok/s; c104 about 400.7; c112 ~410.4 (three repeats, all outputs complete/nonzero). Historical 3090 c24 peak 495.1 is not a comparable batch, do not overclaim ratio.
- Some remote JSONL artifacts from early harness versions are corrupt (literal `\\n` delimiters and oversized request detail). Trust printed observations and correct separate logs; final compact JSONL runner still needs validation/final clean recording. Keep that issue explicit.

## Decisive frontier

Establish best deployable concurrency at >=2GB GPU reserve; verify c96 sparse/full set memory and throughput tradeoff with valid compact JSONL. Quantify scheduler graph/eager pass totals plus true batch histogram. Validate telemetry overhead once against uninstrumented config. Finish short-context 16GB candidate budget, exact configs, rank resource samples and clean service shutdown; then commit/push final experiment/control results.

## Established

- Frozen baseline and S3 accepted graph8/Mamba facts as listed above.
- S4 has measured covered concurrency to c112. Best full-batch point is ~410.7 tok/s at c112, graph replay 255–256/256, exact batch112 histogram; however available VRAM reserve is only 1.34 GiB, short of required >=2GiB. Thus c112 is measured but not yet the deployable recommendation.
- c96 clean pool has 8.09GB weights, 7.04GB KV for 167,664 tokens, 0.51GB full graph set, 2.54GB startup headroom. Sparse `[1,96]` changes graph memory to 0.30GB and leaves ~2.75GB; throughput is ~391 tok/s either way.
- c120+ at the existing memory plan does not form one full batch after prefill (KV-bound); c120 splits to 96+24 and regresses. c128 normal pool also splits to 53+22. `mem_fraction_static=.80` leaves only 8,880 tokens, pathological microbatches, and is rejected.
- Source histogram instrumentation is opt-in, in-memory one-counter-per-step, applied only to disposable remote extraction. No permanent runtime/kernel math changes.
- Current S4 report/work materials are under `experiments/S004-v100-decode-saturation/` in the visible `ninfer-v100-falcon-h1-7b` workspace; remote logs remain at `/hy-tmp/sglang-logs/`. The local `sglang-V100` checkout is not mounted in this current runtime. No other-repo path is authoritative.

## Decisive frontier

Finish highest operating point with >=2GiB headroom and graph-set selection; validate telemetry overhead, retain exact scheduler batch/passes and power/utilization sampling. Correctly parse streaming cumulative output to obtain ITL and produce parseable compact raw manifests. Complete conditional 16GB estimates and commit/push final records only after acceptance.

## Next action

Run matched c96 full vs sparse graph set with final compact JSONL; sample GPU resources over the entire wave; use idle teardown after measurement. Then derive 16GB budget and finalize task/plan/frontier.
