# Certified TOP Projection Validation

- Overall passed: `True`
- Overhead gate: `<=1.05x`

## Operator Contracts

| dtype | TF32 | shape | runs | bitwise | mean overhead | max invariant diff | max drift vs torch single | passed |
|---|---|---|---:|---:|---:|---:|---:|---|
| `bfloat16` | `False` | `512x1024x8192` | 5 | 5/5 | 1.009 | 0.000000 | 0.000000 | `True` |
| `float16` | `False` | `512x1024x8192` | 5 | 5/5 | 1.000 | 0.000000 | 0.000000 | `True` |
| `float32` | `False` | `512x1024x8192` | 5 | 5/5 | 0.868 | 0.000000 | 0.003169 | `True` |
| `float32` | `True` | `512x1024x8192` | 5 | 5/5 | 0.992 | 0.000000 | 0.000000 | `True` |

## Planner Contracts

| dtype | TF32 | selected mode | selected kind | passed |
|---|---|---|---|---|
| `bfloat16` | `False` | `certified_top_projection` | `certified_fast_path` | `True` |
| `float16` | `False` | `certified_top_projection` | `certified_fast_path` | `True` |
| `float32` | `False` | `certified_top_projection` | `certified_fast_path` | `True` |
| `float32` | `True` | `certified_top_projection` | `certified_fast_path` | `True` |
