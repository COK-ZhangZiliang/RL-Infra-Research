# Batch-Invariant Operator Overhead Analysis

- Status: `new_benchmark_analyzed`
- Benchmark config records: `15`
- Benchmark mode rows: `120`

## Mode Summary

| mode | runs | ok | batch-invariant logprob runs | mean matmul ms | mean overhead vs torch | max invariant diff | max drift vs torch single |
|---|---:|---:|---:|---:|---:|---:|---:|
| `batch_inv_persistent` | 15 | 15 | 15 | 1.269 | 1.202 | 0.000000 | 0.003169 |
| `canonical_per_sample_loop` | 15 | 15 | 15 | 1.764 | 1.387 | 0.000000 | 0.000000 |
| `certified_top_projection` | 15 | 15 | 15 | 1.096 | 0.959 | 0.000000 | 0.003169 |
| `certified_top_projection_0drift` | 15 | 15 | 15 | 1.747 | 1.366 | 0.000000 | 0.000000 |
| `torch_batched_3d` | 15 | 15 | 11 | 1.203 | 1.000 | 0.000004 | 0.000004 |
| `torch_flattened_2d` | 15 | 15 | 11 | 1.241 | 1.042 | 0.000004 | 0.000004 |
| `tp_inv` | 15 | 15 | 15 | 1.955 | 1.553 | 0.000000 | 0.015941 |
| `tp_inv_optim` | 15 | 5 | 5 | 3.406 | 1.485 | 0.000000 | 0.003158 |

## Mode x Dtype Summary

| mode | dtype | runs | ok | batch-invariant logprob runs | mean overhead vs torch | max invariant diff | max drift vs torch single |
|---|---|---:|---:|---:|---:|---:|---:|
| `batch_inv_persistent` | `bfloat16` | 5 | 5 | 5 | 1.341 | 0.000000 | 0.000000 |
| `batch_inv_persistent` | `float16` | 5 | 5 | 5 | 1.358 | 0.000000 | 0.000000 |
| `batch_inv_persistent` | `float32` | 5 | 5 | 5 | 0.908 | 0.000000 | 0.003169 |
| `canonical_per_sample_loop` | `bfloat16` | 5 | 5 | 5 | 1.398 | 0.000000 | 0.000000 |
| `canonical_per_sample_loop` | `float16` | 5 | 5 | 5 | 1.391 | 0.000000 | 0.000000 |
| `canonical_per_sample_loop` | `float32` | 5 | 5 | 5 | 1.371 | 0.000000 | 0.000000 |
| `certified_top_projection` | `bfloat16` | 5 | 5 | 5 | 1.009 | 0.000000 | 0.000000 |
| `certified_top_projection` | `float16` | 5 | 5 | 5 | 1.000 | 0.000000 | 0.000000 |
| `certified_top_projection` | `float32` | 5 | 5 | 5 | 0.868 | 0.000000 | 0.003169 |
| `certified_top_projection_0drift` | `bfloat16` | 5 | 5 | 5 | 1.403 | 0.000000 | 0.000000 |
| `certified_top_projection_0drift` | `float16` | 5 | 5 | 5 | 1.388 | 0.000000 | 0.000000 |
| `certified_top_projection_0drift` | `float32` | 5 | 5 | 5 | 1.308 | 0.000000 | 0.000000 |
| `torch_batched_3d` | `bfloat16` | 5 | 5 | 5 | 1.000 | 0.000000 | 0.000000 |
| `torch_batched_3d` | `float16` | 5 | 5 | 5 | 1.000 | 0.000000 | 0.000000 |
| `torch_batched_3d` | `float32` | 5 | 5 | 1 | 1.000 | 0.000004 | 0.000004 |
| `torch_flattened_2d` | `bfloat16` | 5 | 5 | 5 | 1.025 | 0.000000 | 0.000000 |
| `torch_flattened_2d` | `float16` | 5 | 5 | 5 | 1.011 | 0.000000 | 0.000000 |
| `torch_flattened_2d` | `float32` | 5 | 5 | 1 | 1.091 | 0.000004 | 0.000004 |
| `tp_inv` | `bfloat16` | 5 | 5 | 5 | 1.556 | 0.000000 | 0.015941 |
| `tp_inv` | `float16` | 5 | 5 | 5 | 1.540 | 0.000000 | 0.001981 |
| `tp_inv` | `float32` | 5 | 5 | 5 | 1.563 | 0.000000 | 0.003158 |
| `tp_inv_optim` | `bfloat16` | 5 | 0 | 0 | - | - | - |
| `tp_inv_optim` | `float16` | 5 | 0 | 0 | - | - | - |
| `tp_inv_optim` | `float32` | 5 | 5 | 5 | 1.485 | 0.000000 | 0.003158 |

## Batch 8 Detail

| dtype | mode | status | matmul ms | overhead vs torch | logprob bitwise | max invariant diff | max drift vs torch single | error |
|---|---|---|---:|---:|---|---:|---:|---|
| `bfloat16` | `batch_inv_persistent` | `ok` | 1.218 | 1.395 | `True` | 0.000000 | 0.000000 | `` |
| `bfloat16` | `canonical_per_sample_loop` | `ok` | 1.397 | 1.601 | `True` | 0.000000 | 0.000000 | `` |
| `bfloat16` | `certified_top_projection` | `ok` | 0.879 | 1.007 | `True` | 0.000000 | 0.000000 | `` |
| `bfloat16` | `certified_top_projection_0drift` | `ok` | 1.416 | 1.622 | `True` | 0.000000 | 0.000000 | `` |
| `bfloat16` | `torch_batched_3d` | `ok` | 0.873 | 1.000 | `True` | 0.000000 | 0.000000 | `` |
| `bfloat16` | `torch_flattened_2d` | `ok` | 0.930 | 1.066 | `True` | 0.000000 | 0.000000 | `` |
| `bfloat16` | `tp_inv` | `ok` | 1.485 | 1.701 | `True` | 0.000000 | 0.015941 | `` |
| `bfloat16` | `tp_inv_optim` | `error` | - | - | `None` | - | - | `OutOfResources(114688, 101376, 'shared memory')` |
| `float16` | `batch_inv_persistent` | `ok` | 1.244 | 1.401 | `True` | 0.000000 | 0.000000 | `` |
| `float16` | `canonical_per_sample_loop` | `ok` | 1.420 | 1.599 | `True` | 0.000000 | 0.000000 | `` |
| `float16` | `certified_top_projection` | `ok` | 0.880 | 0.990 | `True` | 0.000000 | 0.000000 | `` |
| `float16` | `certified_top_projection_0drift` | `ok` | 1.423 | 1.603 | `True` | 0.000000 | 0.000000 | `` |
| `float16` | `torch_batched_3d` | `ok` | 0.888 | 1.000 | `True` | 0.000000 | 0.000000 | `` |
| `float16` | `torch_flattened_2d` | `ok` | 0.917 | 1.032 | `True` | 0.000000 | 0.000000 | `` |
| `float16` | `tp_inv` | `ok` | 1.484 | 1.671 | `True` | 0.000000 | 0.001981 | `` |
| `float16` | `tp_inv_optim` | `error` | - | - | `None` | - | - | `OutOfResources(114688, 101376, 'shared memory')` |
| `float32` | `batch_inv_persistent` | `ok` | 2.361 | 0.848 | `True` | 0.000000 | 0.002636 | `` |
| `float32` | `canonical_per_sample_loop` | `ok` | 3.969 | 1.425 | `True` | 0.000000 | 0.000000 | `` |
| `float32` | `certified_top_projection` | `ok` | 2.355 | 0.846 | `True` | 0.000000 | 0.002636 | `` |
| `float32` | `certified_top_projection_0drift` | `ok` | 4.011 | 1.440 | `True` | 0.000000 | 0.000000 | `` |
| `float32` | `torch_batched_3d` | `ok` | 2.785 | 1.000 | `False` | 0.000003 | 0.000003 | `` |
| `float32` | `torch_flattened_2d` | `ok` | 2.778 | 0.998 | `False` | 0.000003 | 0.000003 | `` |
| `float32` | `tp_inv` | `ok` | 4.504 | 1.618 | `True` | 0.000000 | 0.002626 | `` |
| `float32` | `tp_inv_optim` | `ok` | 4.372 | 1.570 | `True` | 0.000000 | 0.002626 | `` |

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
