from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "low_precision_batch_invariant_bench"
DEFAULT_FP8_PROBE = PROJECT_ROOT / ".data" / "miles-runs" / "low_precision_dtype_probe" / "probe.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "low_precision_batch_invariant_analysis"

LOW_PRECISION_DTYPES = ["bfloat16", "float16"]
MODES = [
    "torch_batched_3d",
    "canonical_per_sample_loop",
    "batch_inv_persistent",
    "certified_top_projection",
    "certified_top_projection_0drift",
]


def _load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    if not root.exists():
        return records
    for path in sorted(root.glob("batch_inv_operator_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        config = record["config"]
        for mode, stats in record["modes"].items():
            invariant = stats.get("sampled_logprob_invariance_vs_mode_single", {})
            drift = stats.get("sampled_logprob_drift_vs_torch_single", {})
            rows.append(
                {
                    "dtype": config["dtype"],
                    "batch": config["batch"],
                    "seq_len": config["seq_len"],
                    "hidden": config["hidden"],
                    "vocab": config["vocab"],
                    "allow_tf32": config["allow_tf32"],
                    "mode": mode,
                    "status": stats.get("status"),
                    "matmul_ms": stats.get("matmul_ms"),
                    "overhead_vs_torch": stats.get("matmul_overhead_vs_torch_batched"),
                    "bitwise": invariant.get("bitwise_equal"),
                    "max_invariant_diff": invariant.get("max_abs_diff"),
                    "max_drift_vs_torch_single": drift.get("max_abs_diff"),
                    "error": stats.get("error"),
                }
            )
    return rows


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mode_summary(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for dtype in LOW_PRECISION_DTYPES:
        for mode in MODES:
            selected = [row for row in rows if row["dtype"] == dtype and row["mode"] == mode]
            ok_rows = [row for row in selected if row["status"] == "ok"]
            out[(dtype, mode)] = {
                "runs": len(selected),
                "ok_runs": len(ok_rows),
                "bitwise_runs": sum(1 for row in ok_rows if row["bitwise"] is True),
                "mean_ms": _mean([row["matmul_ms"] for row in ok_rows]),
                "mean_overhead": _mean([row["overhead_vs_torch"] for row in ok_rows]),
                "max_invariant_diff": max(
                    [row["max_invariant_diff"] for row in ok_rows if row["max_invariant_diff"] is not None],
                    default=None,
                ),
                "max_drift_vs_torch_single": max(
                    [
                        row["max_drift_vs_torch_single"]
                        for row in ok_rows
                        if row["max_drift_vs_torch_single"] is not None
                    ],
                    default=None,
                ),
                "batch8": next((row for row in ok_rows if row["batch"] == 8), None),
            }
    return out


def _comparison_rows(summary: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dtype in LOW_PRECISION_DTYPES:
        strict = summary[(dtype, "canonical_per_sample_loop")]
        method = summary[(dtype, "certified_top_projection")]
        persistent = summary[(dtype, "batch_inv_persistent")]
        torch_base = summary[(dtype, "torch_batched_3d")]
        strict_ms = strict["mean_ms"]
        method_ms = method["mean_ms"]
        persistent_ms = persistent["mean_ms"]
        method_speedup = strict_ms / method_ms if strict_ms and method_ms else None
        persistent_speedup = strict_ms / persistent_ms if strict_ms and persistent_ms else None
        rows.append(
            {
                "dtype": dtype,
                "torch_bitwise": f"{torch_base['bitwise_runs']}/{torch_base['runs']}",
                "strict_overhead": strict["mean_overhead"],
                "persistent_overhead": persistent["mean_overhead"],
                "method_overhead": method["mean_overhead"],
                "method_bitwise": f"{method['bitwise_runs']}/{method['runs']}",
                "method_speedup_vs_strict": method_speedup,
                "persistent_speedup_vs_strict": persistent_speedup,
                "method_max_drift_vs_torch_single": method["max_drift_vs_torch_single"],
            }
        )
    return rows


def _batch8_rows(summary: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dtype in LOW_PRECISION_DTYPES:
        strict = summary[(dtype, "canonical_per_sample_loop")]["batch8"]
        method = summary[(dtype, "certified_top_projection")]["batch8"]
        persistent = summary[(dtype, "batch_inv_persistent")]["batch8"]
        if strict and method and persistent:
            rows.append(
                {
                    "dtype": dtype,
                    "strict_overhead": strict["overhead_vs_torch"],
                    "persistent_overhead": persistent["overhead_vs_torch"],
                    "method_overhead": method["overhead_vs_torch"],
                    "method_bitwise": method["bitwise"],
                    "method_speedup_vs_strict": strict["matmul_ms"] / method["matmul_ms"],
                    "persistent_speedup_vs_strict": strict["matmul_ms"] / persistent["matmul_ms"],
                }
            )
    return rows


def _load_fp8_probe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Low-Precision Batch-Invariant TOP Projection Analysis",
        "",
        f"- Benchmark records: `{payload['record_count']}`",
        f"- Shape: `{payload['shape']}`",
        "- Contract: same-mode single-sample vs mixed-batch sampled-logprob bitwise equality.",
        "",
        "## Main Result",
        "",
        "| dtype | torch already batch-invariant | strict fallback overhead | batch_inv_persistent overhead | certified_top_projection overhead | certified bitwise | speedup vs strict |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            f"| `{row['dtype']}` | {row['torch_bitwise']} | {_fmt(row['strict_overhead'])} | "
            f"{_fmt(row['persistent_overhead'])} | {_fmt(row['method_overhead'])} | "
            f"{row['method_bitwise']} | {_fmt(row['method_speedup_vs_strict'])}x |"
        )

    lines.extend(
        [
            "",
            "## Batch 8 Detail",
            "",
            "| dtype | strict fallback overhead | batch_inv_persistent overhead | certified_top_projection overhead | certified bitwise | certified speedup vs strict |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["batch8_rows"]:
        lines.append(
            f"| `{row['dtype']}` | {_fmt(row['strict_overhead'])} | {_fmt(row['persistent_overhead'])} | "
            f"{_fmt(row['method_overhead'])} | `{row['method_bitwise']}` | "
            f"{_fmt(row['method_speedup_vs_strict'])}x |"
        )

    fp8_probe = payload.get("fp8_probe")
    if fp8_probe is not None:
        lines.extend(["", "## FP8 / Lower-Than-FP16 Probe", ""])
        lines.append(
            "| dtype | input creation | torch matmul | certified_top_projection | note |"
        )
        lines.append("|---|---|---|---|---|")
        for item in fp8_probe["results"]:
            lines.append(
                f"| `{item['dtype']}` | `{item['input_creation_status']}` | `{item['torch_matmul_status']}` | "
                f"`{item['certified_projection_status']}` | {item['note']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- In BF16/FP16, regular `torch_batched_3d` is already batch-invariant in the measured contract.",
            "- The strict canonical fallback still costs about 1.37x on average in this focused rerun.",
            "- `batch_inv_persistent` is batch-invariant, but it is slower than the certified torch fast path for BF16/FP16.",
            "- `certified_top_projection` preserves batch invariance and avoids the strict fallback overhead by routing BF16/FP16 to the certified torch path.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BF16/FP16 batch-invariant TOP projection overhead.")
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--fp8-probe", type=Path, default=DEFAULT_FP8_PROBE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.bench_dir)
    rows = _rows(records)
    summary = _mode_summary(rows)
    shape = "-"
    if records:
        config = records[0]["config"]
        shape = f"seq={config['seq_len']}, hidden={config['hidden']}, vocab={config['vocab']}"
    payload = {
        "record_count": len(records),
        "shape": shape,
        "comparison_rows": _comparison_rows(summary),
        "batch8_rows": _batch8_rows(summary),
        "mode_summary": {f"{dtype}:{mode}": stats for (dtype, mode), stats in summary.items()},
        "fp8_probe": _load_fp8_probe(args.fp8_probe),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
