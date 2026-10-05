# Current frontier

## G010-s4-v100-decode-saturation
- T001 V100 decode saturation and graph coverage — ACTIVE
  - frontier: establish authorized DCGM/NCU counter access, then a bounded warm-window CUDA-only Nsight Systems trace and decode semantic owner/efficiency ledger; pause prefill/economics until decode closes.
  - task: `G010-s4-v100-decode-saturation/tasks/T001-decode-saturation-and-graph-coverage.md`

G009-S3-FALCON-V100-RTX3090-FORENSICS
- T001-ab-forensics-and-measured-optimization — ACTIVE
  - frontier: resolve valid V100 Systems CUDA trace (client capture missed server; fresh server under nsys segfaulted in kvcache-store JIT warmup); NCU currently ERR_NVGPUCTRPERM. Prewarm JIT outside capture/try safe attach, then targeted owner profiling and separate decode/prefill optimization. RTX 3090 fixed reference only.
  - task: control/G009-s3-falcon-v100-rtx3090-forensics/tasks/T001-ab-forensics-and-measured-optimization.md

G007-PRODUCTION-HYBRID-LAYER
- T001-production-hybrid-layer-composition — ACTIVE
  - frontier: fix diagnostic candidate restore using H7 generation-2/frontier-7 state+input; require byte-exact StateImage view readback before Mamba taps (prior fresh-zero taps are INVALID / state-unpaired)
  - task: control/G007-production-hybrid-layer/tasks/T001-production-hybrid-layer-composition.md
  - unblock/reopen: artifact-v3 recovery PASS at exact expected SHA and environment smoke passed; remaining frontier is numerical/atomicity Stage 2 qualification
