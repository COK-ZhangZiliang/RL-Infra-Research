# Batch-Invariant Operator Coverage Audit

## CUDA Registered Ops

| operator | benchmark case | covered |
|---|---|---|
| `aten::_log_softmax` | `log_softmax` | `True` |
| `aten::addmm` | `addmm` | `True` |
| `aten::bmm` | `bmm` | `True` |
| `aten::mean.dim` | `mean_last_dim` | `True` |
| `aten::mm` | `mm` | `True` |
| `aten::mm.dtype` | `mm` | `True` |
| `aten::rms_norm` | `rms_norm` | `True` |

## TP Exported Ops

| operator | benchmark case | covered |
|---|---|---|
| `matmul_tp_inv` | `tp_row_linear` | `True` |
| `matmul_tp_persistent` | `tp_row_linear` | `True` |
| `moe_sum_tree_reduce` | `moe_sum_tree_reduce` | `True` |
| `tree_all_reduce_sum` | `tp_invariant_all_reduce` | `True` |

## Row-Wise Optimization Config

- Env: `SGLANG_BATCH_INVARIANT_OPS_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE`
- Default max vectorized columns: `8192`

## Uncovered

- CUDA registered ops: `[]`
- TP exported ops: `[]`
