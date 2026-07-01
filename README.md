# Batch-Invariant Operator Optimization for TOP

This repository contains a compact artifact for a TOP-oriented operator
optimization study in Miles/SGLang.

The key question is:

> Can a TOP path preserve batch invariance while avoiding the large overhead of
> a strict per-sample canonical projection fallback?

## Result Summary

The benchmark separates two contracts:

- **Batch-invariant TOP contract**: compare each operator's mixed-batch output
  against the same operator's single-sample output.
- **Torch-single 0-drift contract**: additionally require matching the original
  `torch` single-sample numerical path.

For the measured projection shape `seq_len=512`, `hidden=1024`, `vocab=8192` on
RTX 3090 / PyTorch `2.11.0+cu130`, the implemented experimental operator
`certified_top_projection` satisfies the primary batch-invariant TOP contract
with near-zero overhead:

| dtype/math | batch-invariant runs | mean overhead vs torch |
|---|---:|---:|
| BF16/no-TF32 | 5/5 | 1.009x |
| FP16/no-TF32 | 5/5 | 1.000x |
| FP32/no-TF32 | 5/5 | 0.868x |
| FP32+TF32 | 5/5 | 0.992x |

Across no-TF32 records, `certified_top_projection` is 15/15 batch-invariant with
mean overhead 0.959x. Including FP32+TF32, it is 20/20 batch-invariant.

The stricter torch-single 0-drift fallback remains expensive at about 1.37x.

## Low-Precision Focused Result

For BF16/FP16, a focused rerun shows that strict canonical fallback still adds
substantial overhead, while `certified_top_projection` keeps the batch-invariant
contract and routes to the certified torch fast path:

| dtype | strict fallback overhead | certified_top_projection overhead | batch-invariant | speedup vs strict |
|---|---:|---:|---:|---:|
| BF16 | 1.370x | 1.010x | 5/5 | 1.455x |
| FP16 | 1.373x | 1.003x | 5/5 | 1.452x |

At batch 8, the speedup versus strict fallback is 1.589x for BF16 and 1.568x
for FP16.

FP8 dtype probing on RTX 3090 / PyTorch `2.11.0+cu130` found that float8 tensors
can be created, but CUDA `addmm` for these float8 dtypes is not implemented in
the tested path. FP8 is therefore recorded as unsupported for this projection
benchmark rather than reported as a comparable performance datapoint.

## Registered Operator Extension

The artifact now also covers the CUDA operators registered by SGLang's
batch-invariant mode, plus TP-invariant communication/reduction helpers:

- `mm`, `addmm`, optional `bmm`
- `_log_softmax`, last-dim `mean`, `rms_norm`
- TP row-linear kernels
- MoE tree reduce variants
- deterministic TP tree all-reduce

The current code-level optimization is captured as a source patch:

- `patches/sglang_batch_invariant_rowwise_vectorized.patch`

It changes the row-wise BIV reductions from fixed-1024 chunked loops to
shape-specialized vectorized row Triton kernels for `log_softmax`, last-dim
`mean`, and `rms_norm`, with a default vectorized threshold of 8192 columns.

Real GPU registered-operator results are summarized in:

- `results/registered_batch_invariant_operator_analysis_real.md`
- `results/tp_invariant_all_reduce_analysis_real.md`
- `results/batch_invariant_operator_coverage_real.md`

Key measured result: the optimized row-wise kernels remain bitwise
batch-invariant across the measured rows. `log_softmax` and `rms_norm` improve
over the legacy chunked kernels on average, while `mean` is shape-sensitive.
The benchmark also identifies invalid or incomplete fast paths, including
`moe_sum_tree_reduce_v2` on some topk=10 shapes and `tp_inv_optim` for
FP16/BF16 on RTX 3090.

## Contents

- `docs/batch_invariant_operator_optimization.md`: full methodology, bottleneck
  analysis, operator design, and interpretation.
- `experiments/certified_batch_invariant_projection.py`: experimental certified
  projection wrapper.
- `experiments/benchmark_batch_invariant_ops.py`: CUDA benchmark harness.
- `experiments/analyze_batch_invariant_operator_overhead.py`: report generator.
- `experiments/certify_batch_invariant_operator_plan.py`: measured-contract
  planner.
- `experiments/validate_certified_top_projection.py`: validation gate.
- `experiments/tune_canonical_persistent_gemm.py`: candidate Triton GEMM tuning
  probe.
- `experiments/analyze_low_precision_batch_invariant.py`: BF16/FP16 focused
  analysis.
- `experiments/probe_low_precision_dtype_support.py`: FP8/lower-precision dtype
  support probe.
- `experiments/benchmark_registered_batch_invariant_ops.py`: registered BIV
  operator benchmark harness.
- `experiments/analyze_registered_batch_invariant_ops.py`: registered operator
  report generator.
- `experiments/benchmark_tp_invariant_all_reduce.py`: distributed TP
  deterministic all-reduce benchmark.
- `experiments/analyze_tp_invariant_all_reduce.py`: all-reduce report
  generator.
- `experiments/audit_batch_invariant_operator_coverage.py`: source coverage
  audit for registered/exported BIV operators.
- `experiments/run_batch_invariant_operator_suite.sh`: end-to-end suite runner.
- `results/*.md`: generated summary reports.
- `patches/*.patch`: source patches against the measured SGLang checkout.

## Reproduction Sketch

The original experiment used the Miles Docker image:

```bash
sudo docker run --rm --gpus all \
  -v /mnt/data/RL-Infra-Research:/workspace \
  -w /workspace \
  radixark/miles:exactplan-boundary \
  python3 experiments/benchmark_batch_invariant_ops.py \
    --output-dir .data/miles-runs/batch_invariant_operator_bench \
    --dtypes float16 bfloat16 float32 \
    --batches 1 2 4 8 16 \
    --seq-len 512 \
    --hidden 1024 \
    --vocab 8192 \
    --iters 20 \
    --warmup 3 \
    --deterministic
```

Then run:

```bash
python3 experiments/analyze_batch_invariant_operator_overhead.py \
  --output-dir .data/miles-runs/batch_invariant_operator_analysis

python3 experiments/certify_batch_invariant_operator_plan.py \
  --output-dir .data/miles-runs/batch_invariant_operator_plan

python3 experiments/validate_certified_top_projection.py \
  --output-dir .data/miles-runs/certified_top_projection_validation
```

## Scope Notes

The current `certified_top_projection` rules are intentionally scoped to the
measured dtype, shape, backend, GPU, and math-mode contracts. A production
system should replace the hardcoded rules with a certificate table keyed by
shape, dtype, backend version, and GPU architecture.

This artifact does not include model checkpoints, raw large benchmark outputs,
Docker storage, or generated caches.
