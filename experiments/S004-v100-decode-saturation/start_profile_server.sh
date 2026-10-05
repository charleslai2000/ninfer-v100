#!/usr/bin/env bash
# Start only on an otherwise-idle assigned V100. Frozen S1 runtime; S4 profiling overlay.
set -euo pipefail
MODEL=/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4
EXTRACT=/hy-tmp/sglang-V100-s2-07343bd165
LOG=/hy-tmp/sglang-logs/s4-v100-nsys-server.log
PYTHONPATH="$EXTRACT/python" HF_HOME=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub \
TRITON_CACHE_DIR=/hy-tmp/triton-cache TORCHINDUCTOR_CACHE_DIR=/hy-tmp/torchinductor-cache \
TMPDIR=/hy-tmp/sglang-tmp CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
python -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 \
  --mem-fraction-static 0.92 --max-running-requests 96 --max-total-tokens 200000 \
  --cuda-graph-max-bs 96 --cuda-graph-bs 1 96 \
  --disable-radix-cache --disable-piecewise-cuda-graph --trust-remote-code \
  --enable-metrics --show-time-cost --enable-layerwise-nvtx-marker \
  >"$LOG" 2>&1 &
echo $!
