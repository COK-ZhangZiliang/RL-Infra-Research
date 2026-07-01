#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/data/RL-Infra-Research}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/.data/miles-runs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.data/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${PROJECT_ROOT}/.data/cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/.data/cache/huggingface/transformers}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.data/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.data/cache/pip}"
export RAY_TMPDIR="${RAY_TMPDIR:-${PROJECT_ROOT}/.data/ray}"
export SGLANG_BATCH_INVARIANT_OPS_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE="${SGLANG_BATCH_INVARIANT_OPS_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE:-8192}"

REGISTERED_BENCH_DIR="${RUN_ROOT}/registered_batch_invariant_operator_bench"
REGISTERED_ANALYSIS_DIR="${RUN_ROOT}/registered_batch_invariant_operator_analysis"
ALL_REDUCE_BENCH_DIR="${RUN_ROOT}/tp_invariant_all_reduce_bench"
ALL_REDUCE_ANALYSIS_DIR="${RUN_ROOT}/tp_invariant_all_reduce_analysis"
COVERAGE_DIR="${RUN_ROOT}/batch_invariant_operator_coverage"

COMMON_REGISTERED_ARGS=(
  --output-dir "${REGISTERED_BENCH_DIR}"
  --dtypes float16 bfloat16 float32
  --batches 1 2 4 8 16
  --seq-len "${SEQ_LEN:-512}"
  --hidden "${HIDDEN:-1024}"
  --vocab "${VOCAB:-8192}"
  --rowwise-cols 1024 4096 8192 16384
  --moe-hidden "${MOE_HIDDEN:-1024}"
  --moe-topks 8 10
  --moe-experts "${MOE_EXPERTS:-64}"
  --iters "${ITERS:-30}"
  --warmup "${WARMUP:-5}"
)

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/audit_batch_invariant_operator_coverage.py" \
  --output-dir "${COVERAGE_DIR}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/benchmark_registered_batch_invariant_ops.py" \
  "${COMMON_REGISTERED_ARGS[@]}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/benchmark_registered_batch_invariant_ops.py" \
  "${COMMON_REGISTERED_ARGS[@]}" \
  --allow-tf32

"${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/analyze_registered_batch_invariant_ops.py" \
  --bench-dir "${REGISTERED_BENCH_DIR}" \
  --output-dir "${REGISTERED_ANALYSIS_DIR}"

if [[ "${RUN_TP_ALL_REDUCE:-1}" == "1" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  torchrun --nproc-per-node "${NPROC_PER_NODE}" \
    "${PROJECT_ROOT}/experiments/benchmark_tp_invariant_all_reduce.py" \
    --output-dir "${ALL_REDUCE_BENCH_DIR}" \
    --backend nccl \
    --dtypes float16 bfloat16 float32 \
    --shapes 512,1024 4096,1024 8192,4096 \
    --iters "${ITERS:-30}" \
    --warmup "${WARMUP:-5}"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/experiments/analyze_tp_invariant_all_reduce.py" \
    --bench-dir "${ALL_REDUCE_BENCH_DIR}" \
    --output-dir "${ALL_REDUCE_ANALYSIS_DIR}"
fi

echo "Coverage: ${COVERAGE_DIR}/report.md"
echo "Registered operator analysis: ${REGISTERED_ANALYSIS_DIR}/report.md"
if [[ "${RUN_TP_ALL_REDUCE:-1}" == "1" ]]; then
  echo "TP all-reduce analysis: ${ALL_REDUCE_ANALYSIS_DIR}/report.md"
fi
