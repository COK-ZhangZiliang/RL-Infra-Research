# Low-Precision Batch-Invariant TOP Projection Analysis

- Benchmark records: `10`
- Shape: `seq=512, hidden=1024, vocab=8192`
- Contract: same-mode single-sample vs mixed-batch sampled-logprob bitwise equality.

## Main Result

| dtype | torch already batch-invariant | strict fallback overhead | batch_inv_persistent overhead | certified_top_projection overhead | certified bitwise | speedup vs strict |
|---|---:|---:|---:|---:|---:|---:|
| `bfloat16` | 5/5 | 1.370 | 1.285 | 1.010 | 5/5 | 1.455x |
| `float16` | 5/5 | 1.373 | 1.345 | 1.003 | 5/5 | 1.452x |

## Batch 8 Detail

| dtype | strict fallback overhead | batch_inv_persistent overhead | certified_top_projection overhead | certified bitwise | certified speedup vs strict |
|---|---:|---:|---:|---:|---:|
| `bfloat16` | 1.563 | 1.365 | 0.984 | `True` | 1.589x |
| `float16` | 1.568 | 1.381 | 1.000 | `True` | 1.568x |

## FP8 / Lower-Than-FP16 Probe

| dtype | input creation | torch matmul | certified_top_projection | note |
|---|---|---|---|---|
| `float8_e4m3fn` | `ok` | `error` | `error` | NotImplementedError('"addmm_cuda" not implemented for \'Float8_e4m3fn\''); certified=NotImplementedError('"addmm_cuda" not implemented for \'Float8_e4m3fn\'') |
| `float8_e5m2` | `ok` | `error` | `error` | NotImplementedError('"addmm_cuda" not implemented for \'Float8_e5m2\''); certified=NotImplementedError('"addmm_cuda" not implemented for \'Float8_e5m2\'') |
| `float8_e4m3fnuz` | `ok` | `error` | `error` | NotImplementedError('"addmm_cuda" not implemented for \'Float8_e4m3fnuz\''); certified=NotImplementedError('"addmm_cuda" not implemented for \'Float8_e4m3fnuz\'') |
| `float8_e5m2fnuz` | `ok` | `error` | `error` | NotImplementedError('"addmm_cuda" not implemented for \'Float8_e5m2fnuz\''); certified=NotImplementedError('"addmm_cuda" not implemented for \'Float8_e5m2fnuz\'') |

## Interpretation

- In BF16/FP16, regular `torch_batched_3d` is already batch-invariant in the measured contract.
- The strict canonical fallback still costs about 1.37x on average in this focused rerun.
- `batch_inv_persistent` is batch-invariant, but it is slower than the certified torch fast path for BF16/FP16.
- `certified_top_projection` preserves batch invariance and avoids the strict fallback overhead by routing BF16/FP16 to the certified torch path.
