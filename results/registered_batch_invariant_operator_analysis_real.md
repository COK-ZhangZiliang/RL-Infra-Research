# Registered Batch-Invariant Operator Analysis

- Benchmark payloads: `2`
- Mode rows: `1620`

## Operator Summary

| op | TF32 | deterministic | dtype | mode | runs | ok | bitwise | mean ms | mean overhead | max invariant diff |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `addmm` | `False` | `False` | `bfloat16` | `batch_inv_addmm` | 5 | 5 | 5/5 | 0.973 | 1.179 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `torch_addmm` | 5 | 5 | 5/5 | 0.815 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch_inv_addmm` | 5 | 5 | 5/5 | 0.976 | 1.162 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `torch_addmm` | 5 | 5 | 5/5 | 0.821 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch_inv_addmm` | 5 | 5 | 5/5 | 1.785 | 0.819 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `torch_addmm` | 5 | 5 | 2/5 | 2.154 | 1.000 | 0.000008 |
| `addmm` | `True` | `False` | `bfloat16` | `batch_inv_addmm` | 5 | 5 | 5/5 | 0.956 | 1.184 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `torch_addmm` | 5 | 5 | 5/5 | 0.802 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch_inv_addmm` | 5 | 5 | 5/5 | 0.980 | 1.171 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `torch_addmm` | 5 | 5 | 5/5 | 0.817 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch_inv_addmm` | 5 | 5 | 5/5 | 1.624 | 1.036 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `torch_addmm` | 5 | 5 | 5/5 | 1.554 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch_inv_bmm` | 5 | 5 | 5/5 | 0.091 | 5.184 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `torch_bmm` | 5 | 5 | 5/5 | 0.017 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch_inv_bmm` | 5 | 5 | 5/5 | 0.099 | 5.282 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `torch_bmm` | 5 | 5 | 5/5 | 0.018 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch_inv_bmm` | 5 | 5 | 5/5 | 0.084 | 3.501 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `torch_bmm` | 5 | 5 | 1/5 | 0.024 | 1.000 | 0.000027 |
| `bmm` | `True` | `False` | `bfloat16` | `batch_inv_bmm` | 5 | 5 | 5/5 | 0.090 | 5.072 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `torch_bmm` | 5 | 5 | 5/5 | 0.018 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch_inv_bmm` | 5 | 5 | 5/5 | 0.093 | 5.212 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `torch_bmm` | 5 | 5 | 5/5 | 0.018 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch_inv_bmm` | 5 | 5 | 5/5 | 0.100 | 5.498 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `torch_bmm` | 5 | 5 | 5/5 | 0.018 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch_inv_log_softmax` | 20 | 20 | 20/20 | 0.197 | 1.906 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `legacy_chunked_log_softmax` | 20 | 20 | 20/20 | 0.230 | 1.886 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `torch_log_softmax` | 20 | 20 | 20/20 | 0.139 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch_inv_log_softmax` | 20 | 20 | 20/20 | 0.197 | 1.841 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `legacy_chunked_log_softmax` | 20 | 20 | 20/20 | 0.230 | 1.810 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `torch_log_softmax` | 20 | 20 | 20/20 | 0.136 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch_inv_log_softmax` | 20 | 20 | 20/20 | 0.354 | 1.437 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `legacy_chunked_log_softmax` | 20 | 20 | 20/20 | 0.444 | 1.878 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `torch_log_softmax` | 20 | 20 | 20/20 | 0.253 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch_inv_log_softmax` | 20 | 20 | 20/20 | 0.192 | 1.815 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `legacy_chunked_log_softmax` | 20 | 20 | 20/20 | 0.229 | 1.841 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `torch_log_softmax` | 20 | 20 | 20/20 | 0.139 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch_inv_log_softmax` | 20 | 20 | 20/20 | 0.197 | 1.743 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `legacy_chunked_log_softmax` | 20 | 20 | 20/20 | 0.233 | 1.871 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `torch_log_softmax` | 20 | 20 | 20/20 | 0.138 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch_inv_log_softmax` | 20 | 20 | 20/20 | 0.358 | 1.643 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `legacy_chunked_log_softmax` | 20 | 20 | 20/20 | 0.450 | 2.166 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `torch_log_softmax` | 20 | 20 | 20/20 | 0.254 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch_inv_mean` | 20 | 20 | 20/20 | 0.082 | 2.102 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `legacy_chunked_mean` | 20 | 20 | 20/20 | 0.082 | 2.174 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `torch_mean` | 20 | 20 | 20/20 | 0.061 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch_inv_mean` | 20 | 20 | 20/20 | 0.085 | 2.224 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `legacy_chunked_mean` | 20 | 20 | 20/20 | 0.085 | 2.268 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `torch_mean` | 20 | 20 | 20/20 | 0.061 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch_inv_mean` | 20 | 20 | 20/20 | 0.129 | 1.781 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `legacy_chunked_mean` | 20 | 20 | 20/20 | 0.127 | 1.699 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `torch_mean` | 20 | 20 | 20/20 | 0.113 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch_inv_mean` | 20 | 20 | 20/20 | 0.086 | 2.373 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `legacy_chunked_mean` | 20 | 20 | 20/20 | 0.081 | 2.069 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `torch_mean` | 20 | 20 | 20/20 | 0.061 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch_inv_mean` | 20 | 20 | 20/20 | 0.086 | 2.183 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `legacy_chunked_mean` | 20 | 20 | 20/20 | 0.085 | 2.097 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `torch_mean` | 20 | 20 | 20/20 | 0.061 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch_inv_mean` | 20 | 20 | 20/20 | 0.127 | 1.591 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `legacy_chunked_mean` | 20 | 20 | 20/20 | 0.125 | 1.482 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `torch_mean` | 20 | 20 | 20/20 | 0.114 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch_inv_persistent` | 5 | 5 | 5/5 | 0.984 | 1.268 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `torch_mm` | 5 | 5 | 5/5 | 0.758 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch_inv_persistent` | 5 | 5 | 5/5 | 0.999 | 1.336 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `torch_mm` | 5 | 5 | 5/5 | 0.732 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch_inv_persistent` | 5 | 5 | 5/5 | 1.726 | 0.807 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `torch_mm` | 5 | 5 | 1/5 | 2.096 | 1.000 | 0.000008 |
| `mm` | `True` | `False` | `bfloat16` | `batch_inv_persistent` | 5 | 5 | 5/5 | 0.982 | 1.287 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `torch_mm` | 5 | 5 | 5/5 | 0.742 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch_inv_persistent` | 5 | 5 | 5/5 | 1.006 | 1.330 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `torch_mm` | 5 | 5 | 5/5 | 0.746 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch_inv_persistent` | 5 | 5 | 5/5 | 1.561 | 1.043 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `torch_mm` | 5 | 5 | 5/5 | 1.495 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `moe_sum_tree_reduce` | 10 | 10 | 10/10 | 0.562 | 4.955 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `moe_sum_tree_reduce_v0` | 5 | 5 | 5/5 | 0.373 | 3.694 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `moe_sum_tree_reduce_v1` | 10 | 10 | 10/10 | 0.579 | 5.004 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `moe_sum_tree_reduce_v2` | 10 | 10 | 6/10 | 0.153 | 1.588 | 16.375000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `torch_sum` | 10 | 10 | 10/10 | 0.105 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `moe_sum_tree_reduce` | 10 | 10 | 10/10 | 0.520 | 4.617 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `moe_sum_tree_reduce_v0` | 5 | 5 | 5/5 | 0.370 | 3.706 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `moe_sum_tree_reduce_v1` | 10 | 10 | 10/10 | 0.532 | 4.595 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `moe_sum_tree_reduce_v2` | 10 | 10 | 6/10 | 0.127 | 1.431 | 17.140625 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `torch_sum` | 10 | 10 | 10/10 | 0.104 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `moe_sum_tree_reduce` | 10 | 10 | 10/10 | 1.865 | 9.240 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `moe_sum_tree_reduce_v0` | 5 | 5 | 5/5 | 0.196 | 1.313 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `moe_sum_tree_reduce_v1` | 10 | 10 | 10/10 | 1.859 | 9.183 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `moe_sum_tree_reduce_v2` | 10 | 10 | 8/10 | 0.160 | 1.084 | 15.920801 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `torch_sum` | 10 | 10 | 10/10 | 0.188 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `moe_sum_tree_reduce` | 10 | 10 | 10/10 | 0.559 | 5.247 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `moe_sum_tree_reduce_v0` | 5 | 5 | 5/5 | 0.378 | 3.804 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `moe_sum_tree_reduce_v1` | 10 | 10 | 10/10 | 0.575 | 5.295 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `moe_sum_tree_reduce_v2` | 10 | 10 | 6/10 | 0.148 | 1.592 | 16.375000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `torch_sum` | 10 | 10 | 10/10 | 0.100 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `moe_sum_tree_reduce` | 10 | 10 | 10/10 | 0.524 | 4.669 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `moe_sum_tree_reduce_v0` | 5 | 5 | 5/5 | 0.370 | 3.890 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `moe_sum_tree_reduce_v1` | 10 | 10 | 10/10 | 0.532 | 4.677 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `moe_sum_tree_reduce_v2` | 10 | 10 | 6/10 | 0.128 | 1.526 | 17.140625 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `torch_sum` | 10 | 10 | 10/10 | 0.104 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `moe_sum_tree_reduce` | 10 | 10 | 10/10 | 1.877 | 9.433 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `moe_sum_tree_reduce_v0` | 5 | 5 | 5/5 | 0.212 | 1.684 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `moe_sum_tree_reduce_v1` | 10 | 10 | 10/10 | 1.868 | 9.299 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `moe_sum_tree_reduce_v2` | 10 | 10 | 8/10 | 0.159 | 1.046 | 15.920801 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `torch_sum` | 10 | 10 | 10/10 | 0.188 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch_inv_rms_norm` | 20 | 20 | 20/20 | 0.163 | 1.355 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `legacy_chunked_rms_norm` | 20 | 20 | 20/20 | 0.184 | 1.469 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `torch_rms_norm` | 20 | 20 | 20/20 | 0.175 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch_inv_rms_norm` | 20 | 20 | 20/20 | 0.168 | 1.407 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `legacy_chunked_rms_norm` | 20 | 20 | 20/20 | 0.182 | 1.320 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `torch_rms_norm` | 20 | 20 | 20/20 | 0.176 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch_inv_rms_norm` | 20 | 20 | 20/20 | 0.298 | 1.066 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `legacy_chunked_rms_norm` | 20 | 20 | 20/20 | 0.341 | 1.165 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `torch_rms_norm` | 20 | 20 | 20/20 | 0.338 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch_inv_rms_norm` | 20 | 20 | 20/20 | 0.164 | 1.378 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `legacy_chunked_rms_norm` | 20 | 20 | 20/20 | 0.183 | 1.415 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `torch_rms_norm` | 20 | 20 | 20/20 | 0.175 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch_inv_rms_norm` | 20 | 20 | 20/20 | 0.166 | 1.326 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `legacy_chunked_rms_norm` | 20 | 20 | 20/20 | 0.182 | 1.302 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `torch_rms_norm` | 20 | 20 | 20/20 | 0.176 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch_inv_rms_norm` | 20 | 20 | 20/20 | 0.299 | 1.140 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `legacy_chunked_rms_norm` | 20 | 20 | 20/20 | 0.341 | 1.192 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `torch_rms_norm` | 20 | 20 | 20/20 | 0.338 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `torch_mm` | 5 | 5 | 5/5 | 0.745 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `tp_inv` | 5 | 5 | 5/5 | 1.180 | 1.496 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `tp_inv_optim` | 5 | 0 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float16` | `torch_mm` | 5 | 5 | 5/5 | 0.745 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `tp_inv` | 5 | 5 | 5/5 | 1.181 | 1.504 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `tp_inv_optim` | 5 | 0 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float32` | `torch_mm` | 5 | 5 | 1/5 | 2.130 | 1.000 | 0.000008 |
| `tp_row_linear` | `False` | `False` | `float32` | `tp_inv` | 5 | 5 | 5/5 | 3.576 | 1.591 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `tp_inv_optim` | 5 | 5 | 5/5 | 3.379 | 1.529 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `torch_mm` | 5 | 5 | 5/5 | 0.741 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `tp_inv` | 5 | 5 | 5/5 | 1.179 | 1.509 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `tp_inv_optim` | 5 | 0 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float16` | `torch_mm` | 5 | 5 | 5/5 | 0.752 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `tp_inv` | 5 | 5 | 5/5 | 1.176 | 1.476 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `tp_inv_optim` | 5 | 0 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float32` | `torch_mm` | 5 | 5 | 5/5 | 1.518 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `tp_inv` | 5 | 5 | 5/5 | 3.431 | 2.160 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `tp_inv_optim` | 5 | 5 | 5/5 | 3.347 | 2.089 | 0.000000 |

## Shape Detail

| op | TF32 | deterministic | dtype | shape | mode | ok | bitwise | mean ms | mean overhead | max invariant diff |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| `addmm` | `False` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.189 | 1.251 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.151 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 2.330 | 1.228 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.898 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.371 | 1.084 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.342 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.680 | 1.146 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.594 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 1.297 | 1.187 | 0.000000 |
| `addmm` | `False` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.092 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.224 | 0.986 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.227 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 2.296 | 1.204 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.907 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.422 | 1.245 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.339 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.715 | 1.179 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.606 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 1.223 | 1.195 | 0.000000 |
| `addmm` | `False` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.024 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.326 | 0.816 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.400 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 4.489 | 0.833 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 0/1 | 5.392 | 1.000 | 0.000008 |
| `addmm` | `False` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.645 | 0.782 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.825 | 1.000 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 1.159 | 0.823 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 0/1 | 1.409 | 1.000 | 0.000008 |
| `addmm` | `False` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 2.305 | 0.841 | 0.000000 |
| `addmm` | `False` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 0/1 | 2.742 | 1.000 | 0.000008 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.194 | 1.264 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.154 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 2.344 | 1.219 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.924 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.380 | 1.102 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.344 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.681 | 1.147 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.594 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 1.182 | 1.188 | 0.000000 |
| `addmm` | `True` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.995 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.224 | 0.981 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.228 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 2.313 | 1.205 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.920 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.427 | 1.260 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.339 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.717 | 1.182 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.607 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 1.218 | 1.229 | 0.000000 |
| `addmm` | `True` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.991 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.330 | 1.123 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.294 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 3.992 | 1.076 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 3.709 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 0.636 | 0.949 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 0.670 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 1.153 | 1.001 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.153 | 1.000 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_addmm` | 1/1 | 1/1 | 2.007 | 1.033 | 0.000000 |
| `addmm` | `True` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_addmm` | 1/1 | 1/1 | 1.942 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=1, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.145 | 6.870 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=1, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=16, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.089 | 5.466 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=16, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=2, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.099 | 6.331 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=2, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=4, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.061 | 3.972 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=4, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=8, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.062 | 3.279 | 0.000000 |
| `bmm` | `False` | `False` | `bfloat16` | `batch=8, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=1, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.151 | 7.114 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=1, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=16, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.088 | 5.644 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=16, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=2, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.135 | 6.419 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=2, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=4, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.061 | 3.267 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=4, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=8, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.061 | 3.966 | 0.000000 |
| `bmm` | `False` | `False` | `float16` | `batch=8, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch=1, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.064 | 2.504 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch=1, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch=16, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.143 | 5.640 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch=16, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 0/1 | 0.025 | 1.000 | 0.000021 |
| `bmm` | `False` | `False` | `float32` | `batch=2, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.089 | 3.687 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch=2, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 0/1 | 0.024 | 1.000 | 0.000023 |
| `bmm` | `False` | `False` | `float32` | `batch=4, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.063 | 2.875 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch=4, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 0/1 | 0.022 | 1.000 | 0.000019 |
| `bmm` | `False` | `False` | `float32` | `batch=8, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.060 | 2.799 | 0.000000 |
| `bmm` | `False` | `False` | `float32` | `batch=8, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 0/1 | 0.022 | 1.000 | 0.000027 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=1, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.089 | 4.116 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=1, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=16, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.151 | 10.003 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=16, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=2, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.086 | 3.995 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=2, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=4, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.060 | 3.988 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=4, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=8, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.062 | 3.256 | 0.000000 |
| `bmm` | `True` | `False` | `bfloat16` | `batch=8, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=1, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.073 | 3.057 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=1, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=16, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.138 | 8.915 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=16, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=2, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.134 | 6.168 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=2, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=4, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.060 | 3.986 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=4, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=8, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.061 | 3.936 | 0.000000 |
| `bmm` | `True` | `False` | `float16` | `batch=8, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=1, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.088 | 4.195 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=1, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=16, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.147 | 7.251 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=16, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.020 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=2, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.137 | 8.967 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=2, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=4, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.065 | 3.714 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=4, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=8, k=128, m=512, n=128` | `batch_inv_bmm` | 1/1 | 1/1 | 0.061 | 3.363 | 0.000000 |
| `bmm` | `True` | `False` | `float32` | `batch=8, k=128, m=512, n=128` | `torch_bmm` | 1/1 | 1/1 | 0.018 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.100 | 5.176 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.039 | 2.008 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.086 | 1.738 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.086 | 1.732 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.087 | 3.873 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.085 | 3.764 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.088 | 2.473 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.054 | 1.508 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.036 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.046 | 0.992 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.048 | 1.054 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.046 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 1.256 | 1.948 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.254 | 1.944 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.645 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.165 | 0.591 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.296 | 1.062 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.279 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.324 | 0.770 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.622 | 1.479 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.421 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.051 | 4.123 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.056 | 4.535 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.187 | 2.121 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.178 | 2.018 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.088 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.058 | 1.451 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.058 | 1.448 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.040 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.065 | 1.064 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.090 | 1.474 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.061 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.042 | 3.495 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.039 | 3.229 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.330 | 1.970 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.328 | 1.960 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.167 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.047 | 0.652 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.071 | 0.995 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.072 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.746 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.166 | 1.451 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.114 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.038 | 1.604 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.036 | 1.545 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.633 | 1.939 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.632 | 1.936 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.327 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.632 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.146 | 1.093 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.134 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.164 | 0.771 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.317 | 1.488 | 0.000000 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.213 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.095 | 4.294 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.040 | 1.794 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.086 | 1.756 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.086 | 1.755 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.049 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.055 | 2.523 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.039 | 1.798 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.097 | 2.801 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.054 | 1.565 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.045 | 1.016 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.049 | 1.101 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 1.254 | 1.943 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.254 | 1.942 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.646 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.164 | 0.627 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.302 | 1.153 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.262 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.324 | 0.791 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.622 | 1.518 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.410 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.080 | 4.241 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.078 | 4.126 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.180 | 2.031 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.178 | 2.014 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.089 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.083 | 2.119 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.078 | 1.992 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.039 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.055 | 0.890 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.091 | 1.484 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.061 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.038 | 3.298 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.036 | 3.130 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.327 | 1.959 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.325 | 1.951 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.167 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.045 | 0.684 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.072 | 1.097 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.065 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.802 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.164 | 1.550 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.106 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.038 | 1.622 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.036 | 1.570 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.023 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.634 | 1.942 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.633 | 1.940 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.326 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.677 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.146 | 1.163 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.126 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.165 | 0.806 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.317 | 1.551 | 0.000000 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.204 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.040 | 2.058 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.039 | 2.023 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.164 | 1.517 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.164 | 1.514 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.108 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.040 | 1.441 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.053 | 1.893 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.028 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.045 | 0.954 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.090 | 1.892 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.047 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 1.028 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.116 | 1.402 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.083 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 2.500 | 1.649 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 2.500 | 1.649 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 1.517 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.324 | 0.999 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.622 | 1.918 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.324 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.642 | 0.994 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.250 | 1.934 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.646 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.052 | 3.978 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.050 | 3.840 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.013 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.325 | 1.633 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.327 | 1.644 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.199 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.055 | 1.206 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.087 | 1.924 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.982 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.164 | 1.894 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.087 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.043 | 1.942 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.038 | 1.721 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.640 | 1.666 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.637 | 1.660 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.384 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.084 | 0.996 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.160 | 1.892 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.085 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.165 | 0.992 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.320 | 1.929 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.166 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.045 | 1.054 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.054 | 1.265 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.043 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 1.260 | 1.659 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.259 | 1.657 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.760 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.164 | 0.997 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.326 | 1.978 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.165 | 1.000 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.323 | 0.993 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.630 | 1.935 | 0.000000 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.326 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.089 | 6.656 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.052 | 3.875 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.013 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.094 | 1.887 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.094 | 1.895 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.056 | 2.483 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.052 | 2.305 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.041 | 1.194 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.055 | 1.598 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.045 | 0.999 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.049 | 1.085 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.046 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 1.257 | 1.939 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.255 | 1.936 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.648 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.164 | 0.593 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.297 | 1.071 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.277 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.324 | 0.763 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.622 | 1.463 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.425 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.062 | 4.149 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.050 | 3.307 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.179 | 2.028 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.178 | 2.018 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.088 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.051 | 1.298 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.051 | 1.293 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.040 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.059 | 0.962 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.091 | 1.474 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.061 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.037 | 3.066 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.036 | 2.967 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.328 | 1.957 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.327 | 1.950 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.167 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.047 | 0.647 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.071 | 0.983 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.072 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.739 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.166 | 1.446 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.115 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.038 | 1.611 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.039 | 1.669 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.636 | 1.947 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.633 | 1.938 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.327 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.623 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.146 | 1.074 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.136 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.165 | 0.762 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.316 | 1.462 | 0.000000 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.216 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.107 | 3.186 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.084 | 2.494 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.091 | 1.829 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.089 | 1.787 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.056 | 2.548 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.052 | 2.386 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.060 | 1.715 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.053 | 1.523 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.035 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.046 | 1.017 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.049 | 1.087 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 1.253 | 1.940 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.253 | 1.939 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.646 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.164 | 0.628 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.296 | 1.131 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.261 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.324 | 0.785 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.622 | 1.505 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.413 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.083 | 4.301 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.078 | 4.050 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.178 | 2.023 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.178 | 2.024 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.088 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.080 | 2.032 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.085 | 2.166 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.039 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.071 | 1.166 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.090 | 1.477 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.061 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.038 | 3.259 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.036 | 3.082 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.327 | 1.957 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.327 | 1.953 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.167 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.045 | 0.675 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.071 | 1.072 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.066 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 0.795 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.164 | 1.533 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.107 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.038 | 1.612 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.039 | 1.647 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.023 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.633 | 1.940 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.632 | 1.936 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.326 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.084 | 0.662 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.146 | 1.145 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.128 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.168 | 0.786 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.316 | 1.476 | 0.000000 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.214 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.067 | 5.236 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.052 | 4.017 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.013 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.164 | 1.366 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.164 | 1.362 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.120 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.053 | 2.069 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.150 | 5.812 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.054 | 1.134 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.086 | 1.791 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.048 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.085 | 1.029 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.115 | 1.399 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.082 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 2.500 | 1.644 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 2.500 | 1.644 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 1.521 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.324 | 0.998 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.627 | 1.933 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.324 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.642 | 0.996 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.260 | 1.954 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.645 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.059 | 4.429 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.053 | 4.009 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.013 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.326 | 1.611 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.329 | 1.622 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.203 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.054 | 1.167 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.086 | 1.864 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.046 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.086 | 0.991 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.166 | 1.902 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.087 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.038 | 1.706 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.039 | 1.775 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.638 | 1.659 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.637 | 1.657 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.384 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.086 | 1.005 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.160 | 1.878 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.085 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.165 | 0.991 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.320 | 1.929 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.166 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.045 | 1.062 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.054 | 1.281 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.042 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 1.262 | 1.656 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 1.260 | 1.654 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.762 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.183 | 1.112 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.314 | 1.906 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.165 | 1.000 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `batch_inv_log_softmax` | 1/1 | 1/1 | 0.323 | 0.993 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_log_softmax` | 1/1 | 1/1 | 0.630 | 1.937 | 0.000000 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `torch_log_softmax` | 1/1 | 1/1 | 0.325 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.045 | 3.740 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.058 | 4.784 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.068 | 2.632 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.057 | 2.197 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.065 | 2.487 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.066 | 2.529 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.064 | 3.863 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.056 | 3.366 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.044 | 1.747 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.043 | 1.702 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.309 | 1.009 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.309 | 1.006 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.307 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 0.991 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.082 | 0.988 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.083 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 1.009 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.158 | 1.012 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.060 | 3.757 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.076 | 4.754 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.052 | 1.153 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.052 | 1.143 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.060 | 3.548 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.056 | 3.310 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.065 | 2.507 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.055 | 2.109 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 3.753 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.058 | 4.723 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.088 | 1.086 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.088 | 1.083 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.044 | 1.796 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.043 | 1.761 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.051 | 1.171 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.047 | 1.077 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.043 | 2.777 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.045 | 2.915 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.161 | 1.031 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.161 | 1.030 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 0.975 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.045 | 0.962 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.047 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 1.016 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.084 | 1.031 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.045 | 3.498 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.102 | 7.892 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.013 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.068 | 2.666 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.056 | 2.200 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 3.802 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.041 | 3.438 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.062 | 3.812 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.056 | 3.434 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.044 | 1.667 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.042 | 1.601 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.309 | 1.007 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.309 | 1.007 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.307 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 0.985 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.082 | 0.984 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.084 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 1.013 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.158 | 1.014 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.096 | 3.978 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.086 | 3.532 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.066 | 1.470 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.058 | 1.299 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.092 | 3.790 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.086 | 3.528 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.063 | 2.493 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.057 | 2.250 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.043 | 3.656 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.040 | 3.348 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.088 | 1.084 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.087 | 1.077 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.044 | 1.833 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.041 | 1.694 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 1.075 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.047 | 1.084 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.043 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.055 | 3.563 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.042 | 2.764 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.161 | 1.032 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.160 | 1.028 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.049 | 1.042 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.046 | 0.980 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.047 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 1.021 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.098 | 1.214 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.045 | 3.859 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.042 | 3.524 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.063 | 1.411 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.064 | 1.434 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.101 | 5.358 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.099 | 5.226 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.019 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 1.537 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.043 | 1.433 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.030 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.047 | 1.063 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.046 | 1.049 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.609 | 1.004 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.609 | 1.003 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.607 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 0.986 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.157 | 0.980 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.161 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.308 | 1.005 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.308 | 1.006 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.306 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.060 | 3.793 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.054 | 3.413 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.088 | 1.088 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.088 | 1.080 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.063 | 2.359 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.057 | 2.120 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.027 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.048 | 1.109 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.050 | 1.155 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.043 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.044 | 3.038 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.039 | 2.714 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.014 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.163 | 1.043 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.162 | 1.040 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.049 | 1.097 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.045 | 1.023 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 1.029 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.084 | 1.036 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.048 | 1.875 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.046 | 1.767 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.311 | 1.015 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.310 | 1.014 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.306 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 0.944 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.083 | 0.945 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.087 | 1.000 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 1.016 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.158 | 1.017 | 0.000000 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.097 | 5.863 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.048 | 2.903 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.102 | 3.804 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.096 | 3.580 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.027 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.085 | 5.338 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.056 | 3.501 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.048 | 2.972 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.050 | 3.065 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.043 | 1.731 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.041 | 1.626 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.312 | 1.017 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.310 | 1.012 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.306 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 0.980 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.082 | 0.977 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.084 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 1.010 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.158 | 1.014 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.064 | 3.994 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.056 | 3.522 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.052 | 1.191 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.052 | 1.178 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.059 | 3.603 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.055 | 3.304 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.063 | 2.482 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.055 | 2.175 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.043 | 3.606 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.039 | 3.308 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.088 | 1.084 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.089 | 1.087 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.082 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.043 | 1.738 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.043 | 1.750 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 1.053 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.047 | 1.080 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.045 | 2.946 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.051 | 3.279 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.161 | 1.031 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.160 | 1.028 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 0.987 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.045 | 0.966 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.047 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 1.020 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.084 | 1.031 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.064 | 2.562 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.093 | 3.711 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.097 | 3.398 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.089 | 3.137 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.028 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.067 | 3.964 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.061 | 3.606 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.063 | 3.850 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.060 | 3.633 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 1.842 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.041 | 1.652 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.309 | 1.009 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.308 | 1.006 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.307 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 0.989 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.094 | 1.127 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.084 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 1.009 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.158 | 1.012 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.096 | 3.971 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.085 | 3.528 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.065 | 1.480 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.061 | 1.381 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.059 | 3.597 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.054 | 3.288 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.060 | 2.393 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.057 | 2.255 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.045 | 3.659 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.039 | 3.151 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.012 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.089 | 1.095 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.087 | 1.078 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.043 | 1.791 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.041 | 1.687 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.024 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 1.055 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.047 | 1.076 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.045 | 2.897 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.040 | 2.573 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.161 | 1.033 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.161 | 1.028 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.049 | 1.046 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.046 | 0.985 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.047 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 1.024 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.083 | 1.032 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.063 | 4.008 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.040 | 2.595 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.063 | 1.395 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.068 | 1.515 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.047 | 1.192 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.042 | 1.079 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.039 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.045 | 1.784 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.041 | 1.622 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 1.043 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.045 | 1.007 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.611 | 1.002 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.611 | 1.003 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.609 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 0.984 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.157 | 0.981 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.160 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.308 | 1.006 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.310 | 1.013 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.306 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.059 | 3.709 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.059 | 3.701 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.089 | 1.089 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.088 | 1.085 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.060 | 2.231 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.055 | 2.034 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.027 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.061 | 1.380 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.056 | 1.280 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.046 | 3.167 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.040 | 2.762 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.015 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.163 | 1.043 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.163 | 1.042 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.047 | 1.052 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.046 | 1.030 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.044 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.083 | 1.025 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.084 | 1.038 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.081 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.044 | 1.718 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.048 | 1.879 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.311 | 1.014 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.310 | 1.013 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.306 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.084 | 0.960 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.082 | 0.943 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.087 | 1.000 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `batch_inv_mean` | 1/1 | 1/1 | 0.158 | 1.011 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_mean` | 1/1 | 1/1 | 0.158 | 1.016 | 0.000000 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `torch_mean` | 1/1 | 1/1 | 0.156 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.192 | 1.299 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.148 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 2.377 | 1.332 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.784 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.374 | 1.102 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.339 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.680 | 1.303 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.522 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 1.297 | 1.302 | 0.000000 |
| `mm` | `False` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.996 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.224 | 1.297 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.173 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 2.438 | 1.389 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.755 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.421 | 1.254 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.336 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.718 | 1.371 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.524 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 1.197 | 1.370 | 0.000000 |
| `mm` | `False` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.873 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.309 | 0.769 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.402 | 1.000 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 4.330 | 0.805 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 5.377 | 1.000 | 0.000007 |
| `mm` | `False` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.626 | 0.772 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 0.810 | 1.000 | 0.000007 |
| `mm` | `False` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 1.046 | 0.774 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 1.351 | 1.000 | 0.000008 |
| `mm` | `False` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 2.319 | 0.913 | 0.000000 |
| `mm` | `False` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 2.539 | 1.000 | 0.000007 |
| `mm` | `True` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.198 | 1.328 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.149 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 2.452 | 1.377 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.780 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.380 | 1.115 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.341 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.681 | 1.303 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.522 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 1.200 | 1.311 | 0.000000 |
| `mm` | `True` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.916 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.226 | 1.308 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.173 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 2.433 | 1.371 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.774 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.434 | 1.266 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.343 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.718 | 1.375 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.522 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 1.219 | 1.332 | 0.000000 |
| `mm` | `True` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.916 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.310 | 1.050 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.295 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 3.853 | 1.057 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 3.644 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 0.612 | 1.050 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.583 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 1.109 | 1.042 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.064 | 1.000 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `batch_inv_persistent` | 1/1 | 1/1 | 1.920 | 1.016 | 0.000000 |
| `mm` | `True` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.890 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.164 | 3.275 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.159 | 3.171 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.030 | 0.605 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.124 | 2.162 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.102 | 1.775 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.112 | 1.960 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.106 | 1.848 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.057 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 1.832 | 7.030 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 2.007 | 7.701 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.051 | 15.875000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.261 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.889 | 4.003 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.890 | 4.006 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.890 | 4.006 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.680 | 3.059 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.222 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.278 | 5.927 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.277 | 5.903 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.021 | 0.438 | 15.875000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.047 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.145 | 4.306 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.143 | 4.264 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.144 | 4.274 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.112 | 3.321 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.518 | 7.399 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.531 | 7.576 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.015 | 0.213 | 14.375000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.265 | 4.359 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.264 | 4.343 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.264 | 4.344 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.199 | 3.270 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.061 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.929 | 6.945 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.941 | 7.039 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.100 | 16.375000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.134 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.472 | 4.141 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.466 | 4.082 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.464 | 4.070 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.339 | 2.974 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.114 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.147 | 2.942 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.146 | 2.919 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.031 | 0.614 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.129 | 3.959 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.105 | 3.238 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.103 | 3.183 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.104 | 3.208 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.032 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 1.670 | 6.404 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 1.814 | 6.956 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.051 | 14.671875 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.261 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.856 | 3.854 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.890 | 4.007 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.855 | 3.850 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.530 | 2.385 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.222 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.258 | 5.258 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.257 | 5.223 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.030 | 0.613 | 14.882812 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.049 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.145 | 2.940 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.149 | 3.026 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.144 | 2.912 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.103 | 2.089 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.049 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.432 | 6.165 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.433 | 6.174 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.190 | 17.140625 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.256 | 4.308 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.255 | 4.298 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.255 | 4.301 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.161 | 2.707 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.059 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.850 | 6.361 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.862 | 6.452 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.100 | 15.640625 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.134 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.455 | 3.981 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.453 | 3.962 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.455 | 3.977 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.269 | 2.352 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.114 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.607 | 16.021 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.607 | 16.027 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.030 | 0.783 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.038 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.082 | 2.545 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.063 | 1.946 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.073 | 2.264 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.076 | 2.348 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.032 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 9.033 | 17.601 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 9.024 | 17.583 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.026 | 15.920801 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.513 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.458 | 1.050 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.457 | 1.049 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.456 | 1.047 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.725 | 1.664 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.436 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 1.162 | 16.817 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 1.163 | 16.842 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.027 | 0.398 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.069 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.082 | 1.386 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.075 | 1.274 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.075 | 1.278 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.114 | 1.940 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.059 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 2.292 | 17.163 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 2.278 | 17.060 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.013 | 0.100 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.134 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.137 | 1.195 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.132 | 1.159 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.133 | 1.162 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.205 | 1.790 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.114 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 4.543 | 17.487 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 4.529 | 17.436 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.051 | 14.614189 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.260 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.252 | 1.138 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.251 | 1.135 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.251 | 1.136 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.386 | 1.746 | 0.000000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.221 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.156 | 4.805 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.156 | 4.785 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.020 | 0.617 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.032 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.087 | 2.549 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.083 | 2.424 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.083 | 2.432 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.072 | 2.102 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 1.841 | 7.066 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 1.999 | 7.673 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.051 | 15.875000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.261 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.894 | 4.014 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.928 | 4.164 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.903 | 4.053 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.676 | 3.033 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.223 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.281 | 7.155 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.282 | 7.192 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.019 | 0.479 | 15.875000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.039 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.147 | 4.006 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.144 | 3.923 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.144 | 3.939 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.113 | 3.085 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.037 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.520 | 7.414 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.518 | 7.386 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.190 | 14.375000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.265 | 4.409 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.265 | 4.413 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.265 | 4.408 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.199 | 3.312 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.060 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.930 | 6.954 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.934 | 6.982 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.014 | 0.102 | 16.375000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.134 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.469 | 4.099 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.469 | 4.098 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.469 | 4.096 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.337 | 2.945 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.114 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.147 | 2.604 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.148 | 2.622 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.033 | 0.578 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.056 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.121 | 4.914 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.117 | 4.750 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.115 | 4.676 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.110 | 4.467 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 1.696 | 6.498 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 1.785 | 6.837 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.051 | 14.671875 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.261 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.865 | 3.891 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.891 | 4.012 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.866 | 3.898 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.534 | 2.403 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.222 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.258 | 5.205 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.256 | 5.167 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.030 | 0.596 | 14.882812 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.145 | 2.698 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.143 | 2.661 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.144 | 2.679 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.104 | 1.927 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.054 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.443 | 6.402 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.442 | 6.390 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.194 | 17.140625 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.069 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.236 | 3.993 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.235 | 3.972 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.235 | 3.976 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.148 | 2.504 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.059 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.866 | 6.450 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.869 | 6.471 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.099 | 15.640625 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.134 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.461 | 4.037 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.463 | 4.057 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.463 | 4.053 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.279 | 2.444 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.114 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.600 | 15.910 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.598 | 15.859 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.019 | 0.502 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.038 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.128 | 3.696 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.110 | 3.177 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.114 | 3.278 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.076 | 2.202 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=1, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.035 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 9.043 | 17.642 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 9.028 | 17.615 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.026 | 15.920801 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.513 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.455 | 1.048 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.452 | 1.041 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.458 | 1.054 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.723 | 1.663 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=16, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.435 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 1.175 | 16.665 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 1.173 | 16.641 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.019 | 0.266 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.071 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.140 | 2.316 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.113 | 1.865 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.101 | 1.674 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.126 | 2.075 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=2, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.061 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 2.310 | 17.216 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 2.292 | 17.086 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.014 | 0.101 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.134 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.141 | 1.231 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.140 | 1.216 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.141 | 1.224 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.215 | 1.871 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=4, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.115 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 4.534 | 17.478 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 4.524 | 17.437 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `moe_sum_tree_reduce_v2` | 1/1 | 0/1 | 0.013 | 0.052 | 14.614189 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=10` | `torch_sum` | 1/1 | 1/1 | 0.259 | 1.000 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce` | 1/1 | 1/1 | 0.248 | 1.125 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v0` | 1/1 | 1/1 | 0.247 | 1.120 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v1` | 1/1 | 1/1 | 0.248 | 1.125 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `moe_sum_tree_reduce_v2` | 1/1 | 1/1 | 0.374 | 1.698 | 0.000000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `batch=8, experts=64, hidden=1024, seq_len=512, topk=8` | `torch_sum` | 1/1 | 1/1 | 0.220 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.059 | 2.730 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.040 | 1.866 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.066 | 1.007 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.066 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.066 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.066 | 3.019 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.088 | 4.054 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.060 | 1.631 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.092 | 2.509 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.037 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.908 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.046 | 0.910 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.955 | 0.977 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.954 | 0.977 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.977 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.693 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.237 | 0.993 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.238 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.325 | 0.674 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.479 | 0.993 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.482 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.066 | 2.930 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.063 | 2.809 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.131 | 0.969 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.133 | 0.982 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.135 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.063 | 1.857 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.054 | 1.589 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.062 | 0.876 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.069 | 0.986 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.041 | 2.437 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.038 | 2.303 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.245 | 0.977 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.245 | 0.978 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.250 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.756 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.060 | 0.990 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.060 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.088 | 0.687 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.125 | 0.980 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.128 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.041 | 1.604 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.038 | 1.498 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.482 | 0.978 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.482 | 0.977 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.493 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 0.720 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.118 | 0.989 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.119 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.166 | 0.677 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.243 | 0.993 | 0.000000 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.245 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.103 | 2.988 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.041 | 1.194 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.068 | 1.027 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.067 | 1.021 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.066 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.060 | 3.556 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.057 | 3.338 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.059 | 1.699 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.056 | 1.631 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.911 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.046 | 0.917 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.050 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.952 | 0.974 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.953 | 0.976 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.977 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.169 | 0.709 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.240 | 1.010 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.238 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.325 | 0.674 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.478 | 0.992 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.482 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.085 | 1.976 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.092 | 2.121 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.043 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.132 | 0.968 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.132 | 0.967 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.137 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.104 | 2.321 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.056 | 1.262 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.045 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.059 | 0.843 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.070 | 0.997 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.048 | 2.990 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.039 | 2.401 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.245 | 0.981 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.245 | 0.982 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.249 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.045 | 0.749 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.060 | 0.996 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.060 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.085 | 0.672 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.127 | 0.996 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.127 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.043 | 1.710 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.041 | 1.624 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.484 | 0.983 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.483 | 0.981 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.492 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.087 | 0.735 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.118 | 0.996 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.119 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.673 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.243 | 0.992 | 0.000000 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.246 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.042 | 2.645 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.041 | 2.532 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.129 | 1.031 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.128 | 1.026 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.125 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.058 | 1.254 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.042 | 0.892 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.047 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.061 | 0.945 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.068 | 1.053 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.064 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.085 | 0.795 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.085 | 0.790 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.107 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 1.911 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 1.910 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 1.911 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.324 | 0.679 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.490 | 1.027 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.477 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.642 | 0.672 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.955 | 0.999 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.956 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.056 | 2.696 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.053 | 2.537 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.247 | 1.005 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.247 | 1.007 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.246 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.045 | 0.713 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.065 | 1.024 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.064 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 0.693 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.126 | 1.016 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.124 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.041 | 1.547 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.039 | 1.482 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.027 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.484 | 1.003 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.483 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.483 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.085 | 0.698 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.124 | 1.011 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.122 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.681 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.244 | 1.009 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.242 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.899 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.045 | 0.886 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.051 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.958 | 1.004 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.956 | 1.002 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.954 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.685 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.243 | 1.011 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.241 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.325 | 0.679 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.479 | 1.000 | 0.000000 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.479 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.059 | 3.322 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.039 | 2.228 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.018 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.098 | 1.397 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.111 | 1.579 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.059 | 2.777 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.041 | 1.930 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.059 | 1.714 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.095 | 2.747 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.900 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.046 | 0.905 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.051 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.956 | 0.974 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.952 | 0.971 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.981 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.692 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.238 | 0.999 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.239 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.326 | 0.675 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.479 | 0.992 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.483 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.055 | 2.579 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.057 | 2.667 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.131 | 0.957 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.132 | 0.962 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.137 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.055 | 1.653 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.053 | 1.590 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.033 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.062 | 0.881 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.070 | 0.996 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.042 | 2.535 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.037 | 2.254 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.017 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.261 | 1.044 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.246 | 0.982 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.250 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.754 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.060 | 0.998 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.061 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 0.673 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.125 | 0.978 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.128 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.042 | 1.656 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.039 | 1.554 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.482 | 0.978 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.483 | 0.980 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.493 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 0.717 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.118 | 0.991 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.119 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.676 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.243 | 0.993 | 0.000000 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.244 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.091 | 2.469 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.040 | 1.071 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.037 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.097 | 1.450 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.088 | 1.310 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.067 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.058 | 2.632 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.055 | 2.474 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.022 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.060 | 1.763 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.044 | 1.274 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.890 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.046 | 0.890 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.052 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.952 | 0.975 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.953 | 0.976 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.977 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.692 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.236 | 0.990 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.238 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.325 | 0.674 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.485 | 1.007 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.482 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 2.619 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.084 | 2.544 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.033 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.133 | 0.964 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.131 | 0.950 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.138 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.055 | 1.648 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.056 | 1.659 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.034 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.057 | 0.813 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.069 | 0.990 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.070 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.041 | 2.518 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.038 | 2.324 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.016 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.245 | 0.980 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.245 | 0.981 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.250 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.046 | 0.760 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.060 | 1.001 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.060 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 0.675 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.139 | 1.091 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.127 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.041 | 1.625 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.038 | 1.526 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.025 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.482 | 0.984 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.481 | 0.982 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.490 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.085 | 0.719 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.118 | 0.994 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.118 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.166 | 0.679 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.243 | 0.997 | 0.000000 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.244 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.060 | 3.370 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.041 | 2.289 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.018 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.128 | 1.031 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.127 | 1.026 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.124 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.058 | 1.902 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.055 | 1.803 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.030 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.061 | 0.959 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.068 | 1.061 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.064 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 0.798 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.085 | 0.792 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.107 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 1.913 | 1.001 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 1.911 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 1.912 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.324 | 0.679 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.479 | 1.003 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.477 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.643 | 0.672 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.958 | 1.002 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.956 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.056 | 2.640 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.052 | 2.448 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.021 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.248 | 0.997 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.248 | 0.998 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.248 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.057 | 0.893 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.067 | 1.035 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.064 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.086 | 0.688 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.126 | 1.005 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.125 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.041 | 1.570 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.039 | 1.479 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.026 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.483 | 0.989 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.482 | 0.987 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.489 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.085 | 0.695 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.124 | 1.010 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.123 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.682 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.244 | 1.005 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.242 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.045 | 0.880 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.045 | 0.876 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.052 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.957 | 1.002 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.958 | 1.002 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.956 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.165 | 0.684 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.243 | 1.009 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.241 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `batch_inv_rms_norm` | 1/1 | 1/1 | 0.325 | 0.677 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `legacy_chunked_rms_norm` | 1/1 | 1/1 | 0.480 | 1.000 | 0.000000 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `torch_rms_norm` | 1/1 | 1/1 | 0.480 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.148 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.213 | 1.437 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.766 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 2.989 | 1.692 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.339 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.427 | 1.259 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.522 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.812 | 1.554 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.948 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 1.459 | 1.538 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.174 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.246 | 1.418 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.783 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 2.956 | 1.658 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.336 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.432 | 1.284 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.522 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.809 | 1.549 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.909 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 1.464 | 1.611 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.404 | 1.000 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.528 | 1.306 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 0.572 | 1.414 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 5.423 | 1.000 | 0.000007 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 9.098 | 1.678 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 8.644 | 1.594 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 0.811 | 1.000 | 0.000007 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 1.200 | 1.480 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 1.134 | 1.398 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 1.387 | 1.000 | 0.000008 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 2.371 | 1.710 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 2.191 | 1.580 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 0/1 | 2.625 | 1.000 | 0.000007 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 4.680 | 1.783 | 0.000000 |
| `tp_row_linear` | `False` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 4.355 | 1.659 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.150 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.215 | 1.433 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.792 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 2.965 | 1.655 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.342 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.433 | 1.267 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.523 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.809 | 1.549 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.896 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 1.470 | 1.640 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.174 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.247 | 1.422 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.808 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 2.979 | 1.648 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.339 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.432 | 1.272 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.522 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.746 | 1.429 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.916 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 1.475 | 1.611 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float16` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 0/1 | 0/0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.295 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 0.603 | 2.044 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=1, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 0.572 | 1.942 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 3.710 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 8.800 | 2.372 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=16, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 8.593 | 2.316 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 0.583 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 1.177 | 2.019 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=2, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 1.114 | 1.911 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.067 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 2.300 | 2.156 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=4, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 2.235 | 2.095 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `torch_mm` | 1/1 | 1/1 | 1.935 | 1.000 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv` | 1/1 | 1/1 | 4.277 | 2.211 | 0.000000 |
| `tp_row_linear` | `True` | `False` | `float32` | `batch=8, hidden=1024, seq_len=512, vocab=8192` | `tp_inv_optim` | 1/1 | 1/1 | 4.219 | 2.181 | 0.000000 |

## Row-Wise Optimization Delta

| op | TF32 | deterministic | dtype | shape | path | runs | optimized ms | legacy ms | speedup | optimized overhead | legacy overhead | optimized bitwise |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.100 | 0.039 | 0.388 | 5.176 | 2.008 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.086 | 0.086 | 0.996 | 1.738 | 1.732 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.087 | 0.085 | 0.972 | 3.873 | 3.764 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.088 | 0.054 | 0.610 | 2.473 | 1.508 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.048 | 1.063 | 0.992 | 1.054 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 1.256 | 1.254 | 0.998 | 1.948 | 1.944 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.165 | 0.296 | 1.796 | 0.591 | 1.062 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.324 | 0.622 | 1.920 | 0.770 | 1.479 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.051 | 0.056 | 1.100 | 4.123 | 4.535 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.187 | 0.178 | 0.951 | 2.121 | 2.018 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.058 | 0.058 | 0.998 | 1.451 | 1.448 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.065 | 0.090 | 1.386 | 1.064 | 1.474 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.042 | 0.039 | 0.924 | 3.495 | 3.229 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.330 | 0.328 | 0.995 | 1.970 | 1.960 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.047 | 0.071 | 1.528 | 0.652 | 0.995 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.085 | 0.166 | 1.945 | 0.746 | 1.451 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.038 | 0.036 | 0.964 | 1.604 | 1.545 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.633 | 0.632 | 0.998 | 1.939 | 1.936 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.085 | 0.146 | 1.729 | 0.632 | 1.093 | 1/1 |
| `log_softmax` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.164 | 0.317 | 1.931 | 0.771 | 1.488 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.095 | 0.040 | 0.418 | 4.294 | 1.794 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.086 | 0.086 | 0.999 | 1.756 | 1.755 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.055 | 0.039 | 0.713 | 2.523 | 1.798 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.097 | 0.054 | 0.559 | 2.801 | 1.565 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.049 | 1.085 | 1.016 | 1.101 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 1.254 | 1.254 | 1.000 | 1.943 | 1.942 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.164 | 0.302 | 1.838 | 0.627 | 1.153 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.324 | 0.622 | 1.921 | 0.791 | 1.518 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.080 | 0.078 | 0.973 | 4.241 | 4.126 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.180 | 0.178 | 0.992 | 2.031 | 2.014 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.083 | 0.078 | 0.940 | 2.119 | 1.992 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.055 | 0.091 | 1.667 | 0.890 | 1.484 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.038 | 0.036 | 0.949 | 3.298 | 3.130 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.327 | 0.325 | 0.996 | 1.959 | 1.951 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.045 | 0.072 | 1.604 | 0.684 | 1.097 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.085 | 0.164 | 1.932 | 0.802 | 1.550 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.038 | 0.036 | 0.968 | 1.622 | 1.570 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.634 | 0.633 | 0.999 | 1.942 | 1.940 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.085 | 0.146 | 1.718 | 0.677 | 1.163 | 1/1 |
| `log_softmax` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.317 | 1.924 | 0.806 | 1.551 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.040 | 0.039 | 0.983 | 2.058 | 2.023 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.164 | 0.164 | 0.998 | 1.517 | 1.514 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.040 | 0.053 | 1.314 | 1.441 | 1.893 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.045 | 0.090 | 1.984 | 0.954 | 1.892 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.085 | 0.116 | 1.363 | 1.028 | 1.402 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 2.500 | 2.500 | 1.000 | 1.649 | 1.649 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.324 | 0.622 | 1.920 | 0.999 | 1.918 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.642 | 1.250 | 1.946 | 0.994 | 1.934 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.052 | 0.050 | 0.965 | 3.978 | 3.840 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.325 | 0.327 | 1.006 | 1.633 | 1.644 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.055 | 0.087 | 1.595 | 1.206 | 1.924 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.085 | 0.164 | 1.929 | 0.982 | 1.894 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.043 | 0.038 | 0.886 | 1.942 | 1.721 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.640 | 0.637 | 0.996 | 1.666 | 1.660 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.084 | 0.160 | 1.900 | 0.996 | 1.892 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.320 | 1.945 | 0.992 | 1.929 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.054 | 1.200 | 1.054 | 1.265 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 1.260 | 1.259 | 0.999 | 1.659 | 1.657 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.164 | 0.326 | 1.985 | 0.997 | 1.978 | 1/1 |
| `log_softmax` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.323 | 0.630 | 1.948 | 0.993 | 1.935 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.089 | 0.052 | 0.582 | 6.656 | 3.875 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.094 | 0.094 | 1.005 | 1.887 | 1.895 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.056 | 0.052 | 0.929 | 2.483 | 2.305 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.041 | 0.055 | 1.339 | 1.194 | 1.598 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.049 | 1.086 | 0.999 | 1.085 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 1.257 | 1.255 | 0.998 | 1.939 | 1.936 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.164 | 0.297 | 1.807 | 0.593 | 1.071 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.324 | 0.622 | 1.918 | 0.763 | 1.463 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.062 | 0.050 | 0.797 | 4.149 | 3.307 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.179 | 0.178 | 0.995 | 2.028 | 2.018 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.051 | 0.051 | 0.996 | 1.298 | 1.293 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.059 | 0.091 | 1.533 | 0.962 | 1.474 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.037 | 0.036 | 0.968 | 3.066 | 2.967 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.328 | 0.327 | 0.996 | 1.957 | 1.950 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.047 | 0.071 | 1.520 | 0.647 | 0.983 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.085 | 0.166 | 1.956 | 0.739 | 1.446 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.038 | 0.039 | 1.036 | 1.611 | 1.669 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.636 | 0.633 | 0.996 | 1.947 | 1.938 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.085 | 0.146 | 1.723 | 0.623 | 1.074 | 1/1 |
| `log_softmax` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.316 | 1.919 | 0.762 | 1.462 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.107 | 0.084 | 0.783 | 3.186 | 2.494 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.091 | 0.089 | 0.977 | 1.829 | 1.787 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.056 | 0.052 | 0.936 | 2.548 | 2.386 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.060 | 0.053 | 0.888 | 1.715 | 1.523 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.049 | 1.069 | 1.017 | 1.087 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 1.253 | 1.253 | 1.000 | 1.940 | 1.939 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.164 | 0.296 | 1.802 | 0.628 | 1.131 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.324 | 0.622 | 1.917 | 0.785 | 1.505 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.083 | 0.078 | 0.942 | 4.301 | 4.050 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.178 | 0.178 | 1.000 | 2.023 | 2.024 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.080 | 0.085 | 1.066 | 2.032 | 2.166 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.071 | 0.090 | 1.266 | 1.166 | 1.477 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.038 | 0.036 | 0.946 | 3.259 | 3.082 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.327 | 0.327 | 0.998 | 1.957 | 1.953 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.045 | 0.071 | 1.588 | 0.675 | 1.072 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.085 | 0.164 | 1.929 | 0.795 | 1.533 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.038 | 0.039 | 1.021 | 1.612 | 1.647 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.633 | 0.632 | 0.998 | 1.940 | 1.936 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.084 | 0.146 | 1.730 | 0.662 | 1.145 | 1/1 |
| `log_softmax` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.168 | 0.316 | 1.878 | 0.786 | 1.476 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.067 | 0.052 | 0.767 | 5.236 | 4.017 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.164 | 0.164 | 0.998 | 1.366 | 1.362 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.053 | 0.150 | 2.809 | 2.069 | 5.812 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.054 | 0.086 | 1.580 | 1.134 | 1.791 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.085 | 0.115 | 1.360 | 1.029 | 1.399 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 2.500 | 2.500 | 1.000 | 1.644 | 1.644 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.324 | 0.627 | 1.936 | 0.998 | 1.933 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.642 | 1.260 | 1.962 | 0.996 | 1.954 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.059 | 0.053 | 0.905 | 4.429 | 4.009 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.326 | 0.329 | 1.007 | 1.611 | 1.622 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.054 | 0.086 | 1.597 | 1.167 | 1.864 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.086 | 0.166 | 1.918 | 0.991 | 1.902 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.038 | 0.039 | 1.041 | 1.706 | 1.775 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.638 | 0.637 | 0.999 | 1.659 | 1.657 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.086 | 0.160 | 1.868 | 1.005 | 1.878 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.320 | 1.946 | 0.991 | 1.929 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.054 | 1.207 | 1.062 | 1.281 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 1.262 | 1.260 | 0.999 | 1.656 | 1.654 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.183 | 0.314 | 1.714 | 1.112 | 1.906 | 1/1 |
| `log_softmax` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.323 | 0.630 | 1.951 | 0.993 | 1.937 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.058 | 1.279 | 3.740 | 4.784 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.068 | 0.057 | 0.835 | 2.632 | 2.197 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.065 | 0.066 | 1.017 | 2.487 | 2.529 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.064 | 0.056 | 0.871 | 3.863 | 3.366 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.044 | 0.043 | 0.974 | 1.747 | 1.702 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.309 | 0.309 | 0.998 | 1.009 | 1.006 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.083 | 0.082 | 0.997 | 0.991 | 0.988 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.158 | 0.158 | 1.003 | 1.009 | 1.012 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.060 | 0.076 | 1.265 | 3.757 | 4.754 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.052 | 0.052 | 0.992 | 1.153 | 1.143 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.060 | 0.056 | 0.933 | 3.548 | 3.310 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.065 | 0.055 | 0.841 | 2.507 | 2.109 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.058 | 1.258 | 3.753 | 4.723 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.088 | 0.088 | 0.997 | 1.086 | 1.083 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.044 | 0.043 | 0.980 | 1.796 | 1.761 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.051 | 0.047 | 0.919 | 1.171 | 1.077 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.043 | 0.045 | 1.050 | 2.777 | 2.915 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.161 | 0.161 | 0.999 | 1.031 | 1.030 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.046 | 0.045 | 0.986 | 0.975 | 0.962 | 1/1 |
| `mean_last_dim` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.083 | 0.084 | 1.014 | 1.016 | 1.031 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.102 | 2.256 | 3.498 | 7.892 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.068 | 0.056 | 0.825 | 2.666 | 2.200 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.046 | 0.041 | 0.904 | 3.802 | 3.438 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.062 | 0.056 | 0.901 | 3.812 | 3.434 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.044 | 0.042 | 0.960 | 1.667 | 1.601 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.309 | 0.309 | 1.001 | 1.007 | 1.007 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.083 | 0.082 | 0.999 | 0.985 | 0.984 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.158 | 0.158 | 1.001 | 1.013 | 1.014 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.096 | 0.086 | 0.888 | 3.978 | 3.532 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.066 | 0.058 | 0.884 | 1.470 | 1.299 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.092 | 0.086 | 0.931 | 3.790 | 3.528 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.063 | 0.057 | 0.903 | 2.493 | 2.250 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.043 | 0.040 | 0.916 | 3.656 | 3.348 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.088 | 0.087 | 0.993 | 1.084 | 1.077 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.044 | 0.041 | 0.924 | 1.833 | 1.694 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.046 | 0.047 | 1.008 | 1.075 | 1.084 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.055 | 0.042 | 0.776 | 3.563 | 2.764 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.161 | 0.160 | 0.996 | 1.032 | 1.028 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.049 | 0.046 | 0.940 | 1.042 | 0.980 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.083 | 0.098 | 1.190 | 1.021 | 1.214 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.042 | 0.913 | 3.859 | 3.524 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.063 | 0.064 | 1.016 | 1.411 | 1.434 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.101 | 0.099 | 0.975 | 5.358 | 5.226 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.046 | 0.043 | 0.932 | 1.537 | 1.433 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.047 | 0.046 | 0.987 | 1.063 | 1.049 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.609 | 0.609 | 0.999 | 1.004 | 1.003 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.158 | 0.157 | 0.994 | 0.986 | 0.980 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.308 | 0.308 | 1.002 | 1.005 | 1.006 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.060 | 0.054 | 0.900 | 3.793 | 3.413 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.088 | 0.088 | 0.993 | 1.088 | 1.080 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.063 | 0.057 | 0.899 | 2.359 | 2.120 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.048 | 0.050 | 1.042 | 1.109 | 1.155 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.044 | 0.039 | 0.894 | 3.038 | 2.714 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.163 | 0.162 | 0.997 | 1.043 | 1.040 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.049 | 0.045 | 0.932 | 1.097 | 1.023 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.083 | 0.084 | 1.006 | 1.029 | 1.036 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.048 | 0.046 | 0.942 | 1.875 | 1.767 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.311 | 0.310 | 0.999 | 1.015 | 1.014 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.083 | 0.083 | 1.000 | 0.944 | 0.945 | 1/1 |
| `mean_last_dim` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.158 | 0.158 | 1.000 | 1.016 | 1.017 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.097 | 0.048 | 0.495 | 5.863 | 2.903 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.102 | 0.096 | 0.941 | 3.804 | 3.580 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.085 | 0.056 | 0.656 | 5.338 | 3.501 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.048 | 0.050 | 1.031 | 2.972 | 3.065 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.043 | 0.041 | 0.939 | 1.731 | 1.626 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.312 | 0.310 | 0.994 | 1.017 | 1.012 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.083 | 0.082 | 0.997 | 0.980 | 0.977 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.158 | 0.158 | 1.004 | 1.010 | 1.014 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.064 | 0.056 | 0.882 | 3.994 | 3.522 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.052 | 0.052 | 0.989 | 1.191 | 1.178 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.059 | 0.055 | 0.917 | 3.603 | 3.304 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.063 | 0.055 | 0.877 | 2.482 | 2.175 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.043 | 0.039 | 0.917 | 3.606 | 3.308 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.088 | 0.089 | 1.003 | 1.084 | 1.087 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.043 | 0.043 | 1.007 | 1.738 | 1.750 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.046 | 0.047 | 1.026 | 1.053 | 1.080 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.051 | 1.113 | 2.946 | 3.279 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.161 | 0.160 | 0.997 | 1.031 | 1.028 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.046 | 0.045 | 0.978 | 0.987 | 0.966 | 1/1 |
| `mean_last_dim` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.083 | 0.084 | 1.011 | 1.020 | 1.031 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.064 | 0.093 | 1.449 | 2.562 | 3.711 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.097 | 0.089 | 0.923 | 3.398 | 3.137 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.067 | 0.061 | 0.910 | 3.964 | 3.606 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.063 | 0.060 | 0.944 | 3.850 | 3.633 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.041 | 0.897 | 1.842 | 1.652 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.309 | 0.308 | 0.997 | 1.009 | 1.006 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.083 | 0.094 | 1.139 | 0.989 | 1.127 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.158 | 0.158 | 1.004 | 1.009 | 1.012 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.096 | 0.085 | 0.888 | 3.971 | 3.528 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.065 | 0.061 | 0.934 | 1.480 | 1.381 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.059 | 0.054 | 0.914 | 3.597 | 3.288 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.060 | 0.057 | 0.942 | 2.393 | 2.255 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.039 | 0.861 | 3.659 | 3.151 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.089 | 0.087 | 0.984 | 1.095 | 1.078 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.043 | 0.041 | 0.942 | 1.791 | 1.687 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.046 | 0.047 | 1.021 | 1.055 | 1.076 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.040 | 0.888 | 2.897 | 2.573 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.161 | 0.161 | 0.996 | 1.033 | 1.028 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.049 | 0.046 | 0.941 | 1.046 | 0.985 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.083 | 0.083 | 1.008 | 1.024 | 1.032 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.063 | 0.040 | 0.647 | 4.008 | 2.595 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.063 | 0.068 | 1.086 | 1.395 | 1.515 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.047 | 0.042 | 0.906 | 1.192 | 1.079 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.045 | 0.041 | 0.910 | 1.784 | 1.622 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.045 | 0.965 | 1.043 | 1.007 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.611 | 0.611 | 1.001 | 1.002 | 1.003 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.158 | 0.157 | 0.997 | 0.984 | 0.981 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.308 | 0.310 | 1.006 | 1.006 | 1.013 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.059 | 0.059 | 0.998 | 3.709 | 3.701 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.089 | 0.088 | 0.996 | 1.089 | 1.085 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.060 | 0.055 | 0.912 | 2.231 | 2.034 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.061 | 0.056 | 0.928 | 1.380 | 1.280 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.040 | 0.872 | 3.167 | 2.762 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.163 | 0.163 | 0.999 | 1.043 | 1.042 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.047 | 0.046 | 0.979 | 1.052 | 1.030 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.083 | 0.084 | 1.013 | 1.025 | 1.038 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.044 | 0.048 | 1.094 | 1.718 | 1.879 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.311 | 0.310 | 0.999 | 1.014 | 1.013 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.084 | 0.082 | 0.982 | 0.960 | 0.943 | 1/1 |
| `mean_last_dim` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.158 | 0.158 | 1.005 | 1.011 | 1.016 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.059 | 0.040 | 0.684 | 2.730 | 1.866 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.066 | 0.066 | 0.992 | 1.007 | 1.000 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.066 | 0.088 | 1.343 | 3.019 | 4.054 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.060 | 0.092 | 1.538 | 1.631 | 2.509 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.046 | 1.003 | 0.908 | 0.910 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.955 | 0.954 | 1.000 | 0.977 | 0.977 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.165 | 0.237 | 1.433 | 0.693 | 0.993 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.325 | 0.479 | 1.473 | 0.674 | 0.993 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.066 | 0.063 | 0.959 | 2.930 | 2.809 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.131 | 0.133 | 1.014 | 0.969 | 0.982 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.063 | 0.054 | 0.856 | 1.857 | 1.589 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.062 | 0.069 | 1.126 | 0.876 | 0.986 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.041 | 0.038 | 0.945 | 2.437 | 2.303 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.245 | 0.245 | 1.001 | 0.977 | 0.978 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.046 | 0.060 | 1.310 | 0.756 | 0.990 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.088 | 0.125 | 1.426 | 0.687 | 0.980 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.041 | 0.038 | 0.934 | 1.604 | 1.498 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.482 | 0.482 | 0.999 | 0.978 | 0.977 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.086 | 0.118 | 1.374 | 0.720 | 0.989 | 1/1 |
| `rms_norm` | `False` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.166 | 0.243 | 1.467 | 0.677 | 0.993 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.103 | 0.041 | 0.399 | 2.988 | 1.194 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.068 | 0.067 | 0.994 | 1.027 | 1.021 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.060 | 0.057 | 0.939 | 3.556 | 3.338 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.059 | 0.056 | 0.960 | 1.699 | 1.631 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.046 | 1.007 | 0.911 | 0.917 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.952 | 0.953 | 1.002 | 0.974 | 0.976 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.169 | 0.240 | 1.424 | 0.709 | 1.010 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.325 | 0.478 | 1.472 | 0.674 | 0.992 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.085 | 0.092 | 1.074 | 1.976 | 2.121 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.132 | 0.132 | 0.999 | 0.968 | 0.967 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.104 | 0.056 | 0.544 | 2.321 | 1.262 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.059 | 0.070 | 1.184 | 0.843 | 0.997 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.048 | 0.039 | 0.803 | 2.990 | 2.401 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.245 | 0.245 | 1.001 | 0.981 | 0.982 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.045 | 0.060 | 1.329 | 0.749 | 0.996 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.085 | 0.127 | 1.482 | 0.672 | 0.996 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.043 | 0.041 | 0.950 | 1.710 | 1.624 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.484 | 0.483 | 0.998 | 0.983 | 0.981 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.087 | 0.118 | 1.355 | 0.735 | 0.996 | 1/1 |
| `rms_norm` | `False` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.243 | 1.473 | 0.673 | 0.992 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.042 | 0.041 | 0.957 | 2.645 | 2.532 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.129 | 0.128 | 0.995 | 1.031 | 1.026 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.058 | 0.042 | 0.712 | 1.254 | 0.892 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.061 | 0.068 | 1.115 | 0.945 | 1.053 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.085 | 0.085 | 0.994 | 0.795 | 0.790 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 1.911 | 1.910 | 1.000 | 1.000 | 1.000 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.324 | 0.490 | 1.513 | 0.679 | 1.027 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.642 | 0.955 | 1.486 | 0.672 | 0.999 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.056 | 0.053 | 0.941 | 2.696 | 2.537 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.247 | 0.247 | 1.002 | 1.005 | 1.007 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.045 | 0.065 | 1.436 | 0.713 | 1.024 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.086 | 0.126 | 1.466 | 0.693 | 1.016 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.041 | 0.039 | 0.958 | 1.547 | 1.482 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.484 | 0.483 | 0.997 | 1.003 | 1.000 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.085 | 0.124 | 1.448 | 0.698 | 1.011 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.244 | 1.481 | 0.681 | 1.009 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.045 | 0.986 | 0.899 | 0.886 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.958 | 0.956 | 0.998 | 1.004 | 1.002 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.165 | 0.243 | 1.477 | 0.685 | 1.011 | 1/1 |
| `rms_norm` | `False` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.325 | 0.479 | 1.473 | 0.679 | 1.000 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.059 | 0.039 | 0.671 | 3.322 | 2.228 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.098 | 0.111 | 1.130 | 1.397 | 1.579 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.059 | 0.041 | 0.695 | 2.777 | 1.930 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.059 | 0.095 | 1.602 | 1.714 | 2.747 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.046 | 1.006 | 0.900 | 0.905 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.956 | 0.952 | 0.996 | 0.974 | 0.971 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.165 | 0.238 | 1.443 | 0.692 | 0.999 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.326 | 0.479 | 1.469 | 0.675 | 0.992 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.055 | 0.057 | 1.034 | 2.579 | 2.667 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.131 | 0.132 | 1.005 | 0.957 | 0.962 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.055 | 0.053 | 0.962 | 1.653 | 1.590 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.062 | 0.070 | 1.130 | 0.881 | 0.996 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.042 | 0.037 | 0.889 | 2.535 | 2.254 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.261 | 0.246 | 0.941 | 1.044 | 0.982 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.046 | 0.060 | 1.323 | 0.754 | 0.998 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.086 | 0.125 | 1.454 | 0.673 | 0.978 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.042 | 0.039 | 0.939 | 1.656 | 1.554 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.482 | 0.483 | 1.002 | 0.978 | 0.980 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.086 | 0.118 | 1.382 | 0.717 | 0.991 | 1/1 |
| `rms_norm` | `True` | `False` | `bfloat16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.243 | 1.469 | 0.676 | 0.993 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.091 | 0.040 | 0.434 | 2.469 | 1.071 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.097 | 0.088 | 0.903 | 1.450 | 1.310 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.058 | 0.055 | 0.940 | 2.632 | 2.474 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.060 | 0.044 | 0.723 | 1.763 | 1.274 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.046 | 0.046 | 1.000 | 0.890 | 0.890 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 0.952 | 0.953 | 1.001 | 0.975 | 0.976 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.165 | 0.236 | 1.431 | 0.692 | 0.990 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.325 | 0.485 | 1.493 | 0.674 | 1.007 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.086 | 0.084 | 0.971 | 2.619 | 2.544 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.133 | 0.131 | 0.985 | 0.964 | 0.950 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.055 | 0.056 | 1.006 | 1.648 | 1.659 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.057 | 0.069 | 1.218 | 0.813 | 0.990 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.041 | 0.038 | 0.923 | 2.518 | 2.324 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.245 | 0.245 | 1.001 | 0.980 | 0.981 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.046 | 0.060 | 1.317 | 0.760 | 1.001 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.086 | 0.139 | 1.617 | 0.675 | 1.091 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.041 | 0.038 | 0.939 | 1.625 | 1.526 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.482 | 0.481 | 0.998 | 0.984 | 0.982 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.085 | 0.118 | 1.384 | 0.719 | 0.994 | 1/1 |
| `rms_norm` | `True` | `False` | `float16` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.166 | 0.243 | 1.468 | 0.679 | 0.997 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=1024, seq_len=512` | `vectorized` | 1 | 0.060 | 0.041 | 0.679 | 3.370 | 2.289 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=16384, seq_len=512` | `fallback` | 1 | 0.128 | 0.127 | 0.995 | 1.031 | 1.026 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=4096, seq_len=512` | `vectorized` | 1 | 0.058 | 0.055 | 0.948 | 1.902 | 1.803 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=1, cols=8192, seq_len=512` | `vectorized` | 1 | 0.061 | 0.068 | 1.106 | 0.959 | 1.061 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=1024, seq_len=512` | `vectorized` | 1 | 0.086 | 0.085 | 0.992 | 0.798 | 0.792 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=16384, seq_len=512` | `fallback` | 1 | 1.913 | 1.911 | 0.999 | 1.001 | 1.000 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=4096, seq_len=512` | `vectorized` | 1 | 0.324 | 0.479 | 1.477 | 0.679 | 1.003 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=16, cols=8192, seq_len=512` | `vectorized` | 1 | 0.643 | 0.958 | 1.490 | 0.672 | 1.002 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=1024, seq_len=512` | `vectorized` | 1 | 0.056 | 0.052 | 0.927 | 2.640 | 2.448 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=16384, seq_len=512` | `fallback` | 1 | 0.248 | 0.248 | 1.001 | 0.997 | 0.998 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=4096, seq_len=512` | `vectorized` | 1 | 0.057 | 0.067 | 1.160 | 0.893 | 1.035 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=2, cols=8192, seq_len=512` | `vectorized` | 1 | 0.086 | 0.126 | 1.461 | 0.688 | 1.005 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=1024, seq_len=512` | `vectorized` | 1 | 0.041 | 0.039 | 0.942 | 1.570 | 1.479 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=16384, seq_len=512` | `fallback` | 1 | 0.483 | 0.482 | 0.998 | 0.989 | 0.987 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=4096, seq_len=512` | `vectorized` | 1 | 0.085 | 0.124 | 1.454 | 0.695 | 1.010 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=4, cols=8192, seq_len=512` | `vectorized` | 1 | 0.165 | 0.244 | 1.474 | 0.682 | 1.005 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=1024, seq_len=512` | `vectorized` | 1 | 0.045 | 0.045 | 0.995 | 0.880 | 0.876 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=16384, seq_len=512` | `fallback` | 1 | 0.957 | 0.958 | 1.000 | 1.002 | 1.002 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=4096, seq_len=512` | `vectorized` | 1 | 0.165 | 0.243 | 1.476 | 0.684 | 1.009 | 1/1 |
| `rms_norm` | `True` | `False` | `float32` | `batch=8, cols=8192, seq_len=512` | `vectorized` | 1 | 0.325 | 0.480 | 1.478 | 0.677 | 1.000 | 1/1 |

## Validity Findings

| op | TF32 | deterministic | dtype | mode | runs | failed | non-bitwise | mean ms | mean overhead | max invariant diff |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `addmm` | `False` | `False` | `float32` | `torch_addmm` | 5 | 0 | 3 | 2.154 | 1.000 | 0.000008 |
| `bmm` | `False` | `False` | `float32` | `torch_bmm` | 5 | 0 | 4 | 0.024 | 1.000 | 0.000027 |
| `mm` | `False` | `False` | `float32` | `torch_mm` | 5 | 0 | 4 | 2.096 | 1.000 | 0.000008 |
| `moe_sum_tree_reduce` | `False` | `False` | `bfloat16` | `moe_sum_tree_reduce_v2` | 10 | 0 | 4 | 0.153 | 1.588 | 16.375000 |
| `moe_sum_tree_reduce` | `False` | `False` | `float16` | `moe_sum_tree_reduce_v2` | 10 | 0 | 4 | 0.127 | 1.431 | 17.140625 |
| `moe_sum_tree_reduce` | `False` | `False` | `float32` | `moe_sum_tree_reduce_v2` | 10 | 0 | 2 | 0.160 | 1.084 | 15.920801 |
| `moe_sum_tree_reduce` | `True` | `False` | `bfloat16` | `moe_sum_tree_reduce_v2` | 10 | 0 | 4 | 0.148 | 1.592 | 16.375000 |
| `moe_sum_tree_reduce` | `True` | `False` | `float16` | `moe_sum_tree_reduce_v2` | 10 | 0 | 4 | 0.128 | 1.526 | 17.140625 |
| `moe_sum_tree_reduce` | `True` | `False` | `float32` | `moe_sum_tree_reduce_v2` | 10 | 0 | 2 | 0.159 | 1.046 | 15.920801 |
| `tp_row_linear` | `False` | `False` | `bfloat16` | `tp_inv_optim` | 5 | 5 | 0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float16` | `tp_inv_optim` | 5 | 5 | 0 | - | - | - |
| `tp_row_linear` | `False` | `False` | `float32` | `torch_mm` | 5 | 0 | 4 | 2.130 | 1.000 | 0.000008 |
| `tp_row_linear` | `True` | `False` | `bfloat16` | `tp_inv_optim` | 5 | 5 | 0 | - | - | - |
| `tp_row_linear` | `True` | `False` | `float16` | `tp_inv_optim` | 5 | 5 | 0 | - | - | - |

## Interpretation

- This report covers the CUDA operators actually registered by `enable_batch_invariant_mode`: `mm`, `addmm`, `_log_softmax`, `mean.dim`, `rms_norm`, optional `bmm`, plus SGLang TP row-linear kernels.
- Batch invariance is measured by comparing sample 0 in a mixed batch with the same mode run on that sample alone.
- The row-wise optimization delta compares the new vectorized row kernels with the previous fixed-1024 chunked kernels in the same process.
