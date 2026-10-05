# Nsight Systems warmed launch attempt — 2026-10-05

## Host/launch conditions

- Frozen S1 server launched using installed Nsight Systems **2024.6.2.225**, as a normal child application; not `--attach-pid`.
- Command (full args recorded in remote logs `/hy-tmp/sglang-logs/s4-v100-nsys-launch.log` for the initial miss and `s4-v100-nsys-launch2.log` for the env-gated-marker attempt):
  `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --backtrace=none --cuda-graph-trace=graph --capture-range=nvtx --nvtx-capture='S4_DECODE_RANGE@default' --capture-range-end=stop --kill=none --force-overwrite=false --output=/hy-tmp/sglang-logs/s4-v100-nsys-c96 python -m sglang.launch_server ... --cuda-graph-bs 1 96 --max-running-requests 96 --disable-radix-cache --disable-piecewise-cuda-graph --enable-layerwise-nvtx-marker`.
- Server load, graph capture, tokenizer/model warmup all finished. Graph memory .30 GB, KV 8.27GB/196,976 tokens; process stable, HTTP responsive after load. No profiler range was triggered during server startup.
- Initial launch had no scheduler `S4_DECODE_RANGE` hook, so that attempt did not trigger collection. A temporary one-forward hook was later added to the disposable S1 extraction around `forward_batch_generation()` for decode batch96, gated by `SGLANG_S4_CAPTURE_RANGE=1`; launcher environment was confirmed by `/proc`. Runtime batch histogram and SGLang counters advanced to 256×96, so hook conditions and API calls were exercised. However Nsight did not produce a `.nsys-rep` or stats file when the wrapped service was terminated. It remains unproven whether `torch.cuda.nvtx.range_start/end` generated an NVTX range visible to Nsight in this SGLang worker/process, or if the range selected within the worker was filtered by CLI/domain/pipe handling. The frozen source scheduler imports `torch.cuda.nvtx`; API smoke test returned range id 0. Launch log has no crash/SIGSEGV; server ran stable. A profile-range miss is more likely than runtime crash, but direct marker/timeline confirmation is absent. No valid GPU timeline exists.
- The temporary scheduler range patch was removed with `restore_nsys_patch.py`. Byte comparison shows the remote scheduler now exactly matches frozen S1 source except for the pre-existing S4 batch histogram instrumentation. Server and all Nsight processes were gracefully exited; GPU is 1MiB/0%.

## Workload A/B around the attempted capture

Correct c96 sparse `[1,96]`, same server config and graph, 1K prompt/256 output, full 256 outputs and 112 nonzero IDs per request, 256/256 decode graph96 forward histogram:

| phase | aggregate tok/s | mean ITL | row hash |
|---|---:|---:|---|
| before attempted trigger, A | 391.4457 | 192.30 ms | `de935b47af5dfcfe7b5ad7770610515219f580936618852b8782ed1b142891e4` |
| after trigger setup, B | 391.3466 | 192.30 ms | `7bb2302566ab6310c921aaa08ab7eab7b916ef5aa48eee98f11428b5325b04b9` |

Difference −0.0253% (well within repeated-run noise). This is **not** a profiler overhead A/B: Nsight collection was never triggered. A prior first attempt had an invalid criterion requiring all generated token IDs nonzero; actual valid streams contain 112 nonzero and 144 zero IDs. That rejected row is preserved as `/hy-tmp/sglang-logs/s4-v100-nsys-c96-ab-A.jsonl` (no profile run was active). Do not cite it as correctness-valid.

## Determination and follow-up

This has not demonstrated a profiler/runtime crash. It is a capture-start failure/miss with no report and no SIGSEGV. Do not call Nsight unusable. The second wrapped process's c96 A/B waves each yielded full 256 output IDs, 112 nonzero IDs, exact 256×graph96 histogram and rates 391.6934 vs 391.4321 tok/s (−0.0669%). Their difference is not profiler overhead because collection start was never confirmed. A subsequent correctness-valid 256-token c96 trigger wave returned 391.1729 tok/s, 112 nonzero IDs and graph96 256/256, also without a valid timeline.

No further profiler harness should be added. Before any additional execution, only use the existing environment-gated range patch if the SGLang NVTX marker can be verified at the Nsight process scope without a speculative second capture. Otherwise classify Nsight range launch as not yet operational for this runtime and use the user-authorized in-process CUDA-event/NVTX fallback, clearly distinguishing measured coverage from missing timeline. No valid GPU timeline, owner ledger, idle-gap attribution or profiler-overhead measurement exists; no kernel changes were made.
