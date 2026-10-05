#!/usr/bin/env bash
# Reversible single-use NSys capture launcher. Run only when assigned V100 is idle.
set -euo pipefail
MODEL=/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4
EXTRACT=/hy-tmp/sglang-V100-s2-07343bd165
OUT=/hy-tmp/sglang-logs/s4-v100-nsys-c96
[[ ! -e "${OUT}.nsys-rep" && ! -e "${OUT}.qdstrm" ]]
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-v100
export PYTHONPATH="$EXTRACT/python" HF_HOME=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub
export TRITON_CACHE_DIR=/hy-tmp/triton-cache TORCHINDUCTOR_CACHE_DIR=/hy-tmp/torchinductor-cache
export TMPDIR=/hy-tmp/sglang-tmp CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export SGLANG_S4_BATCH_HISTOGRAM=1 SGLANG_S4_CAPTURE_RANGE=1 SGLANG_S4_CAPTURE_BS=96
exec /usr/local/cuda-12.8/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --backtrace=none \
  --cuda-graph-trace=graph --capture-range=nvtx \
  --nvtx-capture='S4_DECODE_RANGE@default' --capture-range-end=stop \
  --kill=none --force-overwrite=false --output="$OUT" \
  "$CONDA_PREFIX/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name Falcon-H1-7B-Instruct-GPTQ-Int4 \
  --host 127.0.0.1 --port 30003 --tp-size 1 --dtype float16 \
  --mem-fraction-static 0.92 --max-running-requests 96 --max-total-tokens 200000 \
  --cuda-graph-max-bs 96 --cuda-graph-bs 1 96 \
  --disable-radix-cache --disable-piecewise-cuda-graph --trust-remote-code \
  --enable-metrics --show-time-cost
