#!/usr/bin/env bash
set -euo pipefail
ARTIFACT=${ARTIFACT:-/hy-tmp/f6c2c-prefill/artifact.ninfer}
BIN=${BIN:-/hy-tmp/ninfer-t018-qual-build/tests/ninfer_t018_attention_v100_test}
SRC=${SRC:-/hy-tmp/ninfer-t018-authoritative}
IN=${IN:-/hy-tmp/f6c2c-prefill/matrix}
OUT=${OUT:-/hy-tmp/t018-final-matrix}
PY=${PY:-/hy-tmp/ninfer-tools/bin/python}
mkdir -p "$OUT"
run_case(){
 local l=$1 t=$2 pos=$3 input=$4 name=$5 fail=${6:-0} handoff=${7:-0}
 local dir="$OUT/$name"; mkdir -p "$dir/taps"
 if [[ $fail != 0 ]]; then "$BIN" "$ARTIFACT" "$input" "$dir/taps" "$l" "$pos" "$fail" | tee "$dir/program.log"; else "$BIN" "$ARTIFACT" "$input" "$dir/taps" "$l" "$pos" 0 "$handoff" | tee "$dir/program.log"; fi
 if [[ $fail == 0 ]]; then
  "$PY" "$SRC/tests/artifact/t018_attention_reference.py" "$ARTIFACT" "$input" "$dir/reference.npz" --layer "$l" --position "$pos"
  "$PY" "$SRC/tests/artifact/t018_attention_compare.py" "$dir/reference.npz" "$dir/taps" | tee "$dir/taps.jsonl"
 fi
}
sha256sum "$ARTIFACT" > "$OUT/artifact.sha256"
# L0 T7 retains the frozen, already validated F6C2C input identity.
# Larger T cases use exact-size deterministic hidden fixtures generated from the
# same seeded F6C2C input recipe; each file is recorded before execution.
for l in 0 22 43; do
 for t in 7 65 257; do
  fixture="$OUT/inputs/L${l}-T${t}.f32"; mkdir -p "$(dirname "$fixture")"
  if [[ $t == 7 ]]; then cp "$IN/valid-case-01-replay-L0-T7-zero-zero-input.f32" "$fixture"; fi
  if [[ ! -f $fixture ]]; then "$PY" "$SRC/tests/artifact/t018_make_hidden_fixture.py" "$fixture" "$l" "$t"; fi
  handoff=0; [[ $l == 0 && $t == 7 ]] && handoff=4
  run_case "$l" "$t" 0 "$fixture" "L${l}-T${t}" 0 "$handoff"
 done
done
# Candidate rollback injection after CUDA writes at the frozen L0/T7 case.
run_case 0 7 0 "$IN/valid-case-01-replay-L0-T7-zero-zero-input.f32" rollback-begin-stages 1
