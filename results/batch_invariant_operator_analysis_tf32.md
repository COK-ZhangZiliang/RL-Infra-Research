# Batch-Invariant Operator Overhead Analysis

- Status: `new_benchmark_analyzed`
- Benchmark config records: `5`
- Benchmark mode rows: `40`

## Mode Summary

| mode | runs | ok | batch-invariant logprob runs | mean matmul ms | mean overhead vs torch | max invariant diff | max drift vs torch single |
|---|---:|---:|---:|---:|---:|---:|---:|
| `batch_inv_persistent` | 5 | 5 | 5 | 1.654 | 1.069 | 0.000000 | 0.003355 |
| `canonical_per_sample_loop` | 5 | 5 | 5 | 2.132 | 1.339 | 0.000000 | 0.000000 |
| `certified_top_projection` | 5 | 5 | 5 | 1.504 | 0.992 | 0.000000 | 0.000000 |
| `certified_top_projection_0drift` | 5 | 5 | 5 | 2.144 | 1.344 | 0.000000 | 0.000000 |
| `torch_batched_3d` | 5 | 5 | 5 | 1.522 | 1.000 | 0.000000 | 0.000000 |
| `torch_flattened_2d` | 5 | 5 | 5 | 1.517 | 1.002 | 0.000000 | 0.000000 |
| `tp_inv` | 5 | 5 | 5 | 3.565 | 2.193 | 0.000000 | 0.003344 |
| `tp_inv_optim` | 5 | 5 | 5 | 3.443 | 2.091 | 0.000000 | 0.003344 |

## Mode x Dtype Summary

| mode | dtype | runs | ok | batch-invariant logprob runs | mean overhead vs torch | max invariant diff | max drift vs torch single |
|---|---|---:|---:|---:|---:|---:|---:|
| `batch_inv_persistent` | `float32` | 5 | 5 | 5 | 1.069 | 0.000000 | 0.003355 |
| `canonical_per_sample_loop` | `float32` | 5 | 5 | 5 | 1.339 | 0.000000 | 0.000000 |
| `certified_top_projection` | `float32` | 5 | 5 | 5 | 0.992 | 0.000000 | 0.000000 |
| `certified_top_projection_0drift` | `float32` | 5 | 5 | 5 | 1.344 | 0.000000 | 0.000000 |
| `torch_batched_3d` | `float32` | 5 | 5 | 5 | 1.000 | 0.000000 | 0.000000 |
| `torch_flattened_2d` | `float32` | 5 | 5 | 5 | 1.002 | 0.000000 | 0.000000 |
| `tp_inv` | `float32` | 5 | 5 | 5 | 2.193 | 0.000000 | 0.003344 |
| `tp_inv_optim` | `float32` | 5 | 5 | 5 | 2.091 | 0.000000 | 0.003344 |

## Batch 8 Detail

| dtype | mode | status | matmul ms | overhead vs torch | logprob bitwise | max invariant diff | max drift vs torch single | error |
|---|---|---|---:|---:|---|---:|---:|---|
| `float32` | `batch_inv_persistent` | `ok` | 2.067 | 1.071 | `True` | 0.000000 | 0.002899 | `` |
| `float32` | `canonical_per_sample_loop` | `ok` | 2.717 | 1.407 | `True` | 0.000000 | 0.000000 | `` |
| `float32` | `certified_top_projection` | `ok` | 1.902 | 0.985 | `True` | 0.000000 | 0.000000 | `` |
| `float32` | `certified_top_projection_0drift` | `ok` | 2.751 | 1.425 | `True` | 0.000000 | 0.000000 | `` |
| `float32` | `torch_batched_3d` | `ok` | 1.931 | 1.000 | `True` | 0.000000 | 0.000000 | `` |
| `float32` | `torch_flattened_2d` | `ok` | 1.931 | 1.000 | `True` | 0.000000 | 0.000000 | `` |
| `float32` | `tp_inv` | `ok` | 4.525 | 2.344 | `True` | 0.000000 | 0.002895 | `` |
| `float32` | `tp_inv_optim` | `ok` | 4.403 | 2.281 | `True` | 0.000000 | 0.002895 | `` |

## Current Evidence

- New CUDA benchmark records are present and analyzed in this report.
- `torch_batched_3d` is the no-TOP performance baseline.
- Batch-invariance is measured as each mode's mixed-batch sample 0 versus the same mode's single-sample execution.
- `canonical_per_sample_loop` is the strict TOP fallback baseline because it preserves this contract across all measured dtypes and batch sizes.
- Existing persistent batch-invariant kernels satisfy the measured batch-invariance contract, but they are not uniformly near-zero overhead for every dtype.

## Design Implication

- The poor baseline is not TOP itself; it is the canonical per-sample fallback that sacrifices batched GEMM efficiency.
- The target operator should keep deterministic/canonical accumulation order inside each output row while scheduling many rows/tiles together persistently.
- Near-zero overhead means the valid batch-invariant mode should be within about 5% of `torch_batched_3d` for the same shape.
- Generic RL optimizations such as sampled-logprob-only computation should be reported separately because they do not by themselves optimize the TOP-induced operator overhead.
