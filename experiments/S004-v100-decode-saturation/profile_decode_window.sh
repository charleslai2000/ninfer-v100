#!/usr/bin/env bash
# Single bounded warm decode capture; CUDA graph launches are kept at graph granularity.
set -euo pipefail
: "${SERVER_PID:?Set SERVER_PID to the warmed SGLang scheduler/model-worker PID}"
: "${OUT:?Set OUT to a new /hy-tmp/sglang-logs/s4-v100-*.nsys-rep path}"
[[ "$OUT" == /hy-tmp/sglang-logs/s4-v100-*.nsys-rep ]]
[[ ! -e "$OUT" ]]
kill -0 "$SERVER_PID"
if [[ "${LAYERWISE_NVTX_ENABLED:-0}" != 1 ]]; then
  echo "Start frozen S1 with --enable-layerwise-nvtx-marker; set LAYERWISE_NVTX_ENABLED=1" >&2
  exit 2
fi
# SGLang hooks emit stack-balanced, per-module NVTX dictionary messages in the
# default domain. Capture the first model-module forward range only, after a
# warmup request; do not expand graph nodes, collect CPU samples, or collect stacks.
/usr/local/cuda-12.8/bin/nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=graph \
  --capture-range=nvtx \
  --nvtx-capture='.*@default' \
  --capture-range-end=stop \
  --duration=45 \
  --force-overwrite=false \
  --output="$OUT" \
  --attach-pid="$SERVER_PID"
test -s "$OUT"
/usr/local/cuda-12.8/bin/nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum "$OUT" > "${OUT%.nsys-rep}.stats.txt"
if ! grep -q "CUDA GPU Kernel Summary" "${OUT%.nsys-rep}.stats.txt"; then
  echo "No CUDA kernel activity in bounded capture; preserve the report and stop here." >&2
  exit 3
fi
