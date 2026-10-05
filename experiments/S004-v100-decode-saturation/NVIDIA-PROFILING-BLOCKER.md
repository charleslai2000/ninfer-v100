# S4 NVIDIA-first profiling continuation — 2026-10-05

## Scope correction

The latest user instruction supersedes the prior S4 ordering. Single-card Falcon-H1-7B V100-SXM2-32GB decode and prefill ceilings are the only objective. Pause SKU economics, 8-card estimates and new serving features. Prior measured decode results remain observations, not ceilings. Do not begin prefill until decode saturation and decode owner analysis are closed.

## Shared-device state

- Only target remains `gpushare-v100`, V100-SXM2-32GB.
- I started an S4 sparse c96 candidate on port 30003, inspected it before benchmark requests, and immediately stopped it after detecting a 29,670MiB foreign allocation. No S4 collection ran in that attempted window. Follow-up checks show 1MiB/0%, no Falcon/S4 PID and no CUDA process. No unrelated PID was killed.
- Actual frozen serving extraction is still `/hy-tmp/sglang-V100-s2-07343bd165`; no source/runtime change made.

## Recovered S4 curve rows

Audited all `s4-v100-*.jsonl` files. Source output uses literal `\\n` separators rather than linefeeds; individual compact JSON objects are independently valid. `scripts/normalize_s4_jsonl.py` now deterministically parses every object, emits strict line-delimited JSON, reparses every output row, verifies a final newline and emits source/output hashes. Strictly correct rows were normalized/copied to:

- `/hy-tmp/sglang-logs/s4-v100-c96-sparse.validated.jsonl` SHA256 `6b021e7954419b74617f9e45a9c5f4237acb98d113e3c509a1fd93e71d1180c5`, 3 rows, mean `391.20±0.98 tok/s`, exact graph histogram 256×batch96.
- `/hy-tmp/sglang-logs/s4-v100-c104.validated.jsonl` SHA256 `0ef1e72e36602b3e483025f2e17f9d54b1155dcd7ccc168a5930de18f8f3db89`, 3 rows, mean `400.74±0.80 tok/s`, exact 256×batch104.
- `/hy-tmp/sglang-logs/s4-v100-c112-itl-correct.validated.jsonl` SHA256 `a64c9f4724aac89558284214060303b44fd02bfd4b25ced3d637956bfa2d551e`, 3 rows, mean `411.20±1.20 tok/s`, exact 256×batch112.

Per-row correctness and all-full-nonzero gates are true. Source hashes are stored in `PROFILE-ARTIFACT-AUDIT.md`; normalized byte hashes above do not purport to equal the flawed raw inputs.

## NVIDIA-tool/permission findings

- CUDA 12.8 includes Nsight Compute 2025.1.1 and Nsight Systems; binaries are `/usr/local/cuda-12.8/bin/ncu` (wrapper to `/opt/nvidia/nsight-compute/2025.1.1`) and `/usr/local/cuda-12.8/bin/nsys`.
- `ncu --query-metrics --devices 0` reports `ERR_NVGPUCTRPERM`.
- `cat /proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`.
- SSH session has uid 0, but that alone does not authorize changing shared-host driver settings or rebooting/reloading a module. No `NVreg_RestrictProfilingToAdminUsers` modprobe configuration was found in `/etc/modprobe.d` or `/lib/modprobe.d`; safe mode activation may require driver reload/reboot and is not undertaken here without explicit operational authorization.
- Nsight Systems reports timestamp counter support, but CPU perf sampling/process-tree support is denied because kernel `perf_event_open` fails and `perf_event_paranoid=4`. User's requested capture does not require CPU sampling; use `--sample=none` and minimal CUDA-only collection if authorized. The previous S3 server capture crash makes warm-window collection a single bounded attempt, not a reason to add another harness after failure.
- No DCGM binaries/service/socket are present (`dcgmi`, `dcgmproftester` missing; no DCGM paths under `/usr`/`/opt`).

## Minimal-impact next measurement protocol

Only run after confirming V100 is otherwise idle:
1. Secure an admin-approved temporary performance-counter enablement procedure. Record before/after module parameters and exact rollback route; don't change module state casually.
2. Start frozen S1 server and reach steady decode c96 `[1,96]` or c112 `[1,112]` with full-batch correct requests. Use existing full-output correctness client and compact one-record-per-wave file; include a final line-feed and parse the file immediately.
3. A/B no-profiler vs DCGM / Nsight windows, same config/repeats. If no DCGM installation is approved, use existing `nvidia-smi dmon` for continuous utilization, power, clocks, memory (not tensor/occupancy) and Nsight Systems only for a warmed bounded window.
4. First Nsight attempt: graph-level trace (do not expand graph nodes), CUDA/NVTX only, `--sample=none`, no backtrace, `--capture-range=nvtx` and `--capture-range-end=stop`; mark one short steady decode range. Do not spawn profiler on HTTP client alone: attach/capture the server process. Inspect trace for GPU kernels before interpreting. If it crashes again, save logs, inspect compatibility/source path and stop further server-capture experiments per user instruction.
5. Extract major kernel name families and total device durations by CUDA kernel from the stable capture. Join semantic categories from runtime call path/source; do not claim semantic attribution solely from kernel names.
6. Select only largest attributable kernel family for targeted NCU. Restrict with `--kernel-name` plus launch skip/count or NVTX range; if metrics still denied, do not repeat attempts without the approved permission change.
7. Prefill 1K/4K/8K can be staged only after decode completion; isolate single request, capture steady prefill range, summarize separately from decode.

## Residual gap

S4 is ACTIVE. No DCGM, NSight server timeline, NCU metrics, measured prefill owner ledger, hardware-backed saturation point, or final residual budget has yet been established. User-facing work can proceed after legitimate profiling access and a stable isolated workload window; until then no claim of maximum TPS or ceiling is warranted.
