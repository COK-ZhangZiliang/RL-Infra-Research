# Certified Batch-Invariant Operator Plan

- Fast-path overhead gate: `<=1.05x`
- Certificate scope: only the measured dtype, math mode, shape, backend, GPU, and batch set.

| dtype | TF32 | shape | selected mode | kind | mean overhead | bitwise runs | reason |
|---|---|---|---|---|---:|---:|---|
| `bfloat16` | `False` | `512x1024x8192` | `certified_top_projection` | `certified_fast_path` | 1.009 | 5/5 | 0-diff across all measured batches and mean overhead <= 1.05x |
| `float16` | `False` | `512x1024x8192` | `certified_top_projection` | `certified_fast_path` | 1.000 | 5/5 | 0-diff across all measured batches and mean overhead <= 1.05x |
| `float32` | `False` | `512x1024x8192` | `certified_top_projection` | `certified_fast_path` | 0.868 | 5/5 | 0-diff across all measured batches and mean overhead <= 1.05x |
| `float32` | `True` | `512x1024x8192` | `certified_top_projection` | `certified_fast_path` | 0.992 | 5/5 | 0-diff across all measured batches and mean overhead <= 1.05x |

## Interpretation

- `certified_fast_path` means TOP can use the regular backend for this measured contract with near-zero operator overhead.
- `strict_fallback` means correctness is available, but the near-zero overhead goal is not yet met.
- The remaining research target is every `strict_fallback` or `unresolved` contract.
