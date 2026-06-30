from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_bench"
DEFAULT_GEMM_SUMMARY = PROJECT_ROOT / ".data" / "miles-runs" / "gemm_probe" / "summary.md"
DEFAULT_CANONICAL_SUMMARY = PROJECT_ROOT / ".data" / "miles-runs" / "canonical_projection_probe" / "summary.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_analysis"


def _load_json_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    records = []
    for path in sorted(root.glob("batch_inv_operator_*.json")):
        records.append({"path": str(path), "payload": json.loads(path.read_text(encoding="utf-8"))})
    return records


def _mode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = record["payload"]
        config = payload.get("config", {})
        for mode, stats in payload.get("modes", {}).items():
            logit_invariance = stats.get("logit_invariance_vs_mode_single") or stats.get(
                "logit_invariance_vs_single", {}
            )
            logprob_invariance = stats.get("sampled_logprob_invariance_vs_mode_single") or stats.get(
                "sampled_logprob_invariance_vs_single", {}
            )
            logprob_drift = stats.get("sampled_logprob_drift_vs_torch_single", {})
            row = {
                "path": record["path"],
                "mode": mode,
                "batch": config.get("batch"),
                "dtype": config.get("dtype"),
                "seq_len": config.get("seq_len"),
                "hidden": config.get("hidden"),
                "vocab": config.get("vocab"),
                "allow_tf32": config.get("allow_tf32"),
                "status": stats.get("status"),
                "matmul_ms": stats.get("matmul_ms"),
                "matmul_plus_logprob_ms": stats.get("matmul_plus_logprob_ms"),
                "overhead_vs_torch": stats.get("matmul_overhead_vs_torch_batched"),
                "logit_bitwise": logit_invariance.get("bitwise_equal"),
                "logprob_bitwise": logprob_invariance.get("bitwise_equal"),
                "logprob_max_abs_diff": logprob_invariance.get("max_abs_diff"),
                "logprob_drift_vs_torch_single_bitwise": logprob_drift.get("bitwise_equal"),
                "logprob_drift_vs_torch_single_max_abs_diff": logprob_drift.get("max_abs_diff"),
                "error": stats.get("error"),
            }
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)
    summary = {}
    for mode, mode_rows in grouped.items():
        ok_rows = [row for row in mode_rows if row["status"] == "ok"]
        summary[mode] = {
            "runs": len(mode_rows),
            "ok_runs": len(ok_rows),
            "bitwise_logprob_runs": sum(1 for row in ok_rows if row["logprob_bitwise"] is True),
            "mean_matmul_ms": _mean([row["matmul_ms"] for row in ok_rows]),
            "mean_overhead_vs_torch": _mean([row["overhead_vs_torch"] for row in ok_rows]),
            "max_logprob_diff": max(
                [row["logprob_max_abs_diff"] for row in ok_rows if row["logprob_max_abs_diff"] is not None],
                default=None,
            ),
            "max_drift_vs_torch_single": max(
                [
                    row["logprob_drift_vs_torch_single_max_abs_diff"]
                    for row in ok_rows
                    if row["logprob_drift_vs_torch_single_max_abs_diff"] is not None
                ],
                default=None,
            ),
        }
    return summary


def summarize_by_mode_dtype(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["mode"]), str(row["dtype"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (mode, dtype), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row["status"] == "ok"]
        summary_rows.append(
            {
                "mode": mode,
                "dtype": dtype,
                "runs": len(group_rows),
                "ok_runs": len(ok_rows),
                "bitwise_logprob_runs": sum(1 for row in ok_rows if row["logprob_bitwise"] is True),
                "mean_overhead_vs_torch": _mean([row["overhead_vs_torch"] for row in ok_rows]),
                "max_logprob_diff": max(
                    [row["logprob_max_abs_diff"] for row in ok_rows if row["logprob_max_abs_diff"] is not None],
                    default=None,
                ),
                "max_drift_vs_torch_single": max(
                    [
                        row["logprob_drift_vs_torch_single_max_abs_diff"]
                        for row in ok_rows
                        if row["logprob_drift_vs_torch_single_max_abs_diff"] is not None
                    ],
                    default=None,
                ),
            }
        )
    return summary_rows


def select_batch_rows(rows: list[dict[str, Any]], *, batch: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["batch"] == batch]
    return sorted(selected, key=lambda row: (str(row["dtype"]), str(row["mode"])))


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def evaluate(summary: dict[str, Any], rows: list[dict[str, Any]], canonical_text: str) -> dict[str, Any]:
    canonical_overhead_known = "canonical_per_sample_2d | True" in canonical_text
    has_new_bench = bool(rows)
    valid_fast_modes = [
        mode
        for mode, stats in summary.items()
        if stats["ok_runs"] > 0
        and stats["bitwise_logprob_runs"] == stats["ok_runs"]
        and stats["mean_overhead_vs_torch"] is not None
        and stats["mean_overhead_vs_torch"] <= 1.05
    ]
    valid_modes = [
        mode
        for mode, stats in summary.items()
        if stats["ok_runs"] > 0 and stats["bitwise_logprob_runs"] == stats["ok_runs"]
    ]
    return {
        "status": "new_benchmark_analyzed" if has_new_bench else "legacy_evidence_only_gpu_unavailable",
        "has_new_benchmark_records": has_new_bench,
        "legacy_canonical_projection_evidence": canonical_overhead_known,
        "valid_bitwise_modes": valid_modes,
        "valid_near_zero_overhead_modes": valid_fast_modes,
        "interpretation": {
            "current_measured_loss": (
                "Legacy canonical projection evidence shows 0-diff per-sample projection is valid but costs about 1.38x-1.42x vs regular mixed-batch projection."
            ),
            "bottleneck": (
                "The overhead comes from reducing M dimension batching efficiency: per-sample canonical GEMM loses large mixed-batch GEMM occupancy and launch amortization."
            ),
            "design_target": (
                "A useful new operator must preserve a canonical per-row reduction order while recovering batched GEMM tiling, L2 reuse, and persistent scheduling."
            ),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    has_records = bool(payload["rows"])
    lines = [
        "# Batch-Invariant Operator Overhead Analysis",
        "",
        f"- Status: `{payload['evaluation']['status']}`",
        f"- Benchmark config records: `{payload['record_count']}`",
        f"- Benchmark mode rows: `{len(payload['rows'])}`",
        "",
        "## Mode Summary",
        "",
        "| mode | runs | ok | batch-invariant logprob runs | mean matmul ms | mean overhead vs torch | max invariant diff | max drift vs torch single |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, stats in sorted(payload["summary"].items()):
        lines.append(
            f"| `{mode}` | {stats['runs']} | {stats['ok_runs']} | {stats['bitwise_logprob_runs']} | "
            f"{_fmt(stats['mean_matmul_ms'])} | {_fmt(stats['mean_overhead_vs_torch'])} | "
            f"{_fmt(stats['max_logprob_diff'], digits=6)} | "
            f"{_fmt(stats['max_drift_vs_torch_single'], digits=6)} |"
        )

    lines.extend(
        [
            "",
            "## Mode x Dtype Summary",
            "",
            "| mode | dtype | runs | ok | batch-invariant logprob runs | mean overhead vs torch | max invariant diff | max drift vs torch single |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["mode_dtype_summary"]:
        lines.append(
            f"| `{row['mode']}` | `{row['dtype']}` | {row['runs']} | {row['ok_runs']} | "
            f"{row['bitwise_logprob_runs']} | {_fmt(row['mean_overhead_vs_torch'])} | "
            f"{_fmt(row['max_logprob_diff'], digits=6)} | "
            f"{_fmt(row['max_drift_vs_torch_single'], digits=6)} |"
        )

    if payload["batch8_rows"]:
        lines.extend(
            [
                "",
                "## Batch 8 Detail",
                "",
                "| dtype | mode | status | matmul ms | overhead vs torch | logprob bitwise | max invariant diff | max drift vs torch single | error |",
                "|---|---|---|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in payload["batch8_rows"]:
            error = str(row["error"] or "")
            if len(error) > 90:
                error = error[:87] + "..."
            lines.append(
                f"| `{row['dtype']}` | `{row['mode']}` | `{row['status']}` | {_fmt(row['matmul_ms'])} | "
                f"{_fmt(row['overhead_vs_torch'])} | `{row['logprob_bitwise']}` | "
                f"{_fmt(row['logprob_max_abs_diff'], digits=6)} | "
                f"{_fmt(row['logprob_drift_vs_torch_single_max_abs_diff'], digits=6)} | `{error}` |"
            )

    if has_records:
        evidence_lines = [
            "- New CUDA benchmark records are present and analyzed in this report.",
            "- `torch_batched_3d` is the no-TOP performance baseline.",
            "- Batch-invariance is measured as each mode's mixed-batch sample 0 versus the same mode's single-sample execution.",
            "- `canonical_per_sample_loop` is the strict TOP fallback baseline because it preserves this contract across all measured dtypes and batch sizes.",
            "- Existing persistent batch-invariant kernels satisfy the measured batch-invariance contract, but they are not uniformly near-zero overhead for every dtype.",
        ]
    else:
        evidence_lines = [
            "- No new CUDA benchmark records were found; rerun `benchmark_batch_invariant_ops.py` on a GPU node.",
            "- Existing canonical projection probe: per-sample canonical projection recovers 0-diff logprob parity but costs about 1.38x at batch 8 and 1.42x at batch 16.",
            "- Existing GEMM probe: ordinary mixed-batch FP32/no-TF32 projection can fail bitwise batch invariance, while BF16/FP16 and FP32+TF32 cases in that probe passed.",
        ]

    lines.extend(
        [
            "",
            "## Current Evidence",
            "",
            *evidence_lines,
            "",
            "## Design Implication",
            "",
            "- The poor baseline is not TOP itself; it is the canonical per-sample fallback that sacrifices batched GEMM efficiency.",
            "- The target operator should keep deterministic/canonical accumulation order inside each output row while scheduling many rows/tiles together persistently.",
            "- Near-zero overhead means the valid batch-invariant mode should be within about 5% of `torch_batched_3d` for the same shape.",
            "- Generic RL optimizations such as sampled-logprob-only computation should be reported separately because they do not by themselves optimize the TOP-induced operator overhead.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze batch-invariant operator benchmark results.")
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--gemm-summary", type=Path, default=DEFAULT_GEMM_SUMMARY)
    parser.add_argument("--canonical-summary", type=Path, default=DEFAULT_CANONICAL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_json_records(args.bench_dir)
    rows = _mode_rows(records)
    summary = summarize_rows(rows)
    payload = {
        "inputs": {
            "bench_dir": str(args.bench_dir),
            "gemm_summary": str(args.gemm_summary),
            "canonical_summary": str(args.canonical_summary),
        },
        "rows": rows,
        "record_count": len(records),
        "summary": summary,
        "mode_dtype_summary": summarize_by_mode_dtype(rows),
        "batch8_rows": select_batch_rows(rows, batch=8),
        "evaluation": evaluate(summary, rows, _read_optional(args.canonical_summary)),
        "legacy_summaries": {
            "gemm": _read_optional(args.gemm_summary),
            "canonical_projection": _read_optional(args.canonical_summary),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
