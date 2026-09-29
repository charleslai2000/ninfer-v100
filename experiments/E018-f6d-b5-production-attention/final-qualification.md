# T018 final bounded qualification record

Status: **PASS (within the assigned production Attention scope)**

Clean checkpoint commit: `f61aff7ad2809d59698462a7bd88d6eee144fb49` (`f61aff7a`, `Integrate Falcon H1 D128 production attention`); authoritative worktree `/tmp/ninfer-t018-authoritative` clean after commit.

## Fixed production length semantics
For committed history H and query count T, past length is H, candidate extent `[H,H+T)`, positions `H+i`, prefill query i reads keys through H+i, T=1 donor decode receives total visible length H+1, and commit frontier is H+T. No T019/T020 transaction ownership or rollback contract or persistent KV layout changed.

Pinned donor decode consumer uses `seq_len` as total visible KV: `query_pos=seq_len-1`, partitioning and reads are bounded by `seq_len`. The handoff tap-export segfault was in test plumbing: host code dereferenced the device-resident block table to resolve V page IDs. Replaced only the export with a device gather from candidate K/V pages/table; production attention semantics were not changed to work around it.

## Production same-artifact tap matrix
V100 sm_70, CUDA 12.8.61, artifact SHA256 `41586ac915088947742765d6980134914161d10f6e053a94024ba7cc7df1fb21`. Current relevant source hashes: prefill `bce9fe7055e95d525adca93133e409508472521fd047ad7aaf3631e6d563fbac`; attention adapter `dbbd010532422a9c516714559ddabe12ad14eac8e0cd1cadbe1720987f39934f`; capture harness `1f1e2239e27562b3a4672354380800c59dd72995ac4acbd625dcbec315a54164`; final V100 qualification executable `ca10f042934593767e891d4c534320c07db6577298be8c45986953f7268ef1a3`.

All 9 cases L0/L22/L43 × T7/T65/T257 ran on production path after length-semantic fix. Every Q, K pre-scale, K, V, RoPE-Q, RoPE-K, pre-O, O projection and output tap was finite and the same-artifact comparator returned `T018_SAME_ARTIFACT_TAPS_PASS`. Comparator reports per-tap max_abs/relative-L2 in `/hy-tmp/t018-final-postfix/L{0,22,43}-T{7,65,257}/metrics.jsonl`. Per-case peak diagnostic max_abs (over ordered taps): L0 T7 0.00390625; L0 T65 0.00390625; L0 T257 0.0078125; L22 T7 0.001953125; L22 T65 0.001953125; L22 T257 0.001953125; L43 T7 0.00390625; L43 T65 0.00390625; L43 T257 0.00390625. Comparator threshold is a diagnostic 0.02, not a numerical production acceptance specification.

## Nonzero-history handoff taps
For each H=7/65/257, production first executed/committed a prefix using the actual artifact and hidden fixture, exported the cache-resident BF16 post-scale K and FP16 V, then ran 4 continuation query rows at positions H..H+3 through production; candidate continuation was rolled back and committed descriptor remained unchanged. Same-artifact reference used that cache-export history NPZ and exact suffix input. All ordered taps finite and each comparator returned `T018_SAME_ARTIFACT_TAPS_PASS`.

| H+4 | max_abs per ordered Q/K-pre/K/V/Q-RoPE/K-RoPE/pre-O/O-proj/output | rel-L2 envelope | nonfinite |
|---|---|---|---|
| 7+4 | 0.0009765625, 0.0000305176, 0.0000009537, 0, 0.0009765625, 0.0000009537, 0.0009765625, 0.015625, 0.001953125 | max 0.00175746 | 0 |
| 65+4 | 0.0009765625, 0, 0, 0, 0.0009765625, 0, 0.0004882813, 0.0004882813, 0.0000610352 | max 0.00125969 | 0 |
| 257+4 | 0.001953125, 0, 0, 0, 0.001953125, 0, 0.0001220703, 0.0002441406, 0.0000305176 | max 0.00091751 | 0 |

Full metrics and captures: `/hy-tmp/t018-history-handoff/H{7,65,257}/handoff-metrics.jsonl`; reference histories are built from `/prefix-taps/candidate_{k,v}` using persistent cache dtypes. This validates the actual history read and candidate append values. Existing `7+4/65+4/257+4` smoke rollback checks also passed.

## Transaction/visibility, regressions
- T7 commit → T1 history decode → rollback: pass; committed descriptor unchanged.
- T1 with committed history at layers 0/22/43: pass, finite output.
- Handoffs 7+4/65+4/257+4 rollback: pass; committed descriptor unchanged.
- Injected rollback stages 1–4 (candidate begin) and 5–8 (candidate execution/finalization): pass; no committed frontier/descriptor change and no pending transaction.
- T016 pinned D128 decoder regression: histories 1/63/64/65/257 all completed, finite; donor-post-BF16 difference zero for all five. F5A post-BF16 max_abs/relative-L2: H1 0/0; H63 4.88281e-4/0.00126095; H64 4.88281e-4/0.00127032; H65 4.88281e-4/0.00141531; H257 2.44141e-4/0.00141956. Zero nonfinite. Executable `/hy-tmp/f6d-b4-adapter-build/f6d_b4a_decode`.
- T017 contract reconciliation: E017 is explicitly an adapter-only exploratory contract (D-major/page64 address, BF16 Q/K edge, FP16 direct V, QH12/KVH2, BF16 output bridge), not the production history-aware Program contract or a production-kernel oracle. Reran its address probes and frozen T7/T65/T257 numerical diagnostics: all probes pass; all cases finite, V100-vs-F5A-host-BF16 max_abs 0.00195312, relative-L2 0.000591712/0.000621812/0.000672070. The previously reported nonfinite came from an incomplete earlier draft binary (`build/e017_prefill`) and is preserved as historical failed exploratory evidence, not the final adapter regression. No frozen valid E017 contract failure remains.
- D256 existing Attention regression `/hy-tmp/f6d-b4-prefill/d256-build/tests/ninfer_softmax_attention_test`: causal public contract, packed, context and aggregate Attention all PASS.

## Bounded donor fidelity
Pinned source SHA256 `848d8f64228bfd206d8485a9e7d9614d13a3d873a3f56c7e319559ad885b8971` at `1CatAI/1Cat-vLLM@fcf59f8e9ae50c186333e98e5cf6aae705f320de`. Production T>1 kernel exposes BM32/BN176/WARPS16/512-thread profile; WMMA 16x16x16 QK/PV; GQA `qh/6`; causal masking; FP32 online max/sum/output recurrence; score/probability aliased storage with donor P0011 barrier; final normalization. Adapter differences are NInfer page-table/page64 D-major cache staging/access, BF16 Q/K to FP16 WMMA operand boundary, native FP16 V direct page access, BF16 output, and launch/interface/workspace. This is the requested bounded source/profile closure, not a byte-identical or line-by-line donor claim. No staging/repack/second KV cache was added.

## Scope boundary
T018 production Attention integration and the prescribed bounded qualification are complete. No 44-layer hybrid/model orchestration, generation, CUDA Graph or benchmark work was performed. Raw runs stay on V100 under `/hy-tmp/t018-final-postfix` and `/hy-tmp/t018-history-handoff`.
