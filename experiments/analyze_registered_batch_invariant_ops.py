from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "registered_batch_invariant_operator_bench"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "registered_batch_invariant_operator_analysis"


def _load_payloads(bench_dir: Path) -> list[dict[str, Any]]:
    if not bench_dir.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(bench_dir.glob("registered_batch_inv_ops_*.json"))]


def _rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        config = payload.get("config", {})
        max_vectorized_cols = config.get("batch_inv_max_vectorized_rowwise_block_size")
        allow_tf32 = bool(config.get("allow_tf32"))
        deterministic = bool(config.get("deterministic"))
        seed = config.get("seed")
        for record in payload.get("records", []):
            baseline_mode = record.get("baseline_mode")
            shape = record.get("shape", {})
            for mode, stats in record.get("modes", {}).items():
                invariant = stats.get("invariance_vs_single", {})
                rows.append(
                    {
                        "op": record.get("op"),
                        "dtype": record.get("dtype"),
                        "shape": shape,
                        "allow_tf32": allow_tf32,
                        "deterministic": deterministic,
                        "seed": seed,
                        "max_vectorized_rowwise_cols": max_vectorized_cols,
                        "mode": mode,
                        "is_baseline": mode == baseline_mode,
                        "status": stats.get("status"),
                        "latency_ms": stats.get("latency_ms"),
                        "overhead_vs_baseline": stats.get("overhead_vs_baseline"),
                        "bitwise": invariant.get("bitwise_equal"),
                        "max_abs_diff": invariant.get("max_abs_diff"),
                        "mean_abs_diff": invariant.get("mean_abs_diff"),
                        "error": stats.get("error"),
                    }
                )
    return rows


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _rowwise_cols(shape: dict[str, Any]) -> int | None:
    value = shape.get("cols")
    return int(value) if value is not None else None


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool, bool, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["op"]),
                bool(row["allow_tf32"]),
                bool(row["deterministic"]),
                str(row["dtype"]),
                str(row["mode"]),
            ),
            [],
        ).append(row)

    summary = []
    for (op, allow_tf32, deterministic, dtype, mode), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row["status"] == "ok"]
        summary.append(
            {
                "op": op,
                "allow_tf32": allow_tf32,
                "deterministic": deterministic,
                "dtype": dtype,
                "mode": mode,
                "runs": len(group_rows),
                "ok_runs": len(ok_rows),
                "bitwise_runs": sum(1 for row in ok_rows if row["bitwise"] is True),
                "mean_latency_ms": _mean([row["latency_ms"] for row in ok_rows]),
                "mean_overhead_vs_baseline": _mean([row["overhead_vs_baseline"] for row in ok_rows]),
                "max_abs_diff": max(
                    [row["max_abs_diff"] for row in ok_rows if row["max_abs_diff"] is not None],
                    default=None,
                ),
            }
        )
    return summary


def _shape_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool, bool, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        shape_key = json.dumps(row["shape"], sort_keys=True)
        grouped.setdefault(
            (
                str(row["op"]),
                bool(row["allow_tf32"]),
                bool(row["deterministic"]),
                str(row["dtype"]),
                shape_key,
                str(row["mode"]),
            ),
            [],
        ).append(row)

    summary = []
    for (op, allow_tf32, deterministic, dtype, shape_key, mode), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row["status"] == "ok"]
        summary.append(
            {
                "op": op,
                "allow_tf32": allow_tf32,
                "deterministic": deterministic,
                "dtype": dtype,
                "shape": json.loads(shape_key),
                "mode": mode,
                "runs": len(group_rows),
                "ok_runs": len(ok_rows),
                "bitwise_runs": sum(1 for row in ok_rows if row["bitwise"] is True),
                "mean_latency_ms": _mean([row["latency_ms"] for row in ok_rows]),
                "mean_overhead_vs_baseline": _mean([row["overhead_vs_baseline"] for row in ok_rows]),
                "max_abs_diff": max(
                    [row["max_abs_diff"] for row in ok_rows if row["max_abs_diff"] is not None],
                    default=None,
                ),
            }
        )
    return summary


def _legacy_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, bool, bool, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        shape_key = json.dumps(row["shape"], sort_keys=True)
        grouped.setdefault(
            (
                str(row["op"]),
                bool(row["allow_tf32"]),
                bool(row["deterministic"]),
                str(row["dtype"]),
                shape_key,
            ),
            [],
        ).append(row)

    optimized_names = {
        "log_softmax": "batch_inv_log_softmax",
        "mean_last_dim": "batch_inv_mean",
        "rms_norm": "batch_inv_rms_norm",
    }
    legacy_names = {
        "log_softmax": "legacy_chunked_log_softmax",
        "mean_last_dim": "legacy_chunked_mean",
        "rms_norm": "legacy_chunked_rms_norm",
    }
    for (op, allow_tf32, deterministic, dtype, shape_key), group_rows in sorted(grouped.items()):
        if op not in optimized_names:
            continue
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for row in group_rows:
            if row["status"] == "ok":
                by_mode.setdefault(str(row["mode"]), []).append(row)
        optimized_rows = by_mode.get(optimized_names[op], [])
        legacy_rows = by_mode.get(legacy_names[op], [])
        if not optimized_rows or not legacy_rows:
            continue
        optimized_latency = _mean([row["latency_ms"] for row in optimized_rows])
        legacy_latency = _mean([row["latency_ms"] for row in legacy_rows])
        if not optimized_latency or not legacy_latency:
            continue
        optimized_overhead = _mean([row["overhead_vs_baseline"] for row in optimized_rows])
        legacy_overhead = _mean([row["overhead_vs_baseline"] for row in legacy_rows])
        shape = json.loads(shape_key)
        cols = _rowwise_cols(shape)
        max_vectorized_cols = optimized_rows[0].get("max_vectorized_rowwise_cols")
        out.append(
            {
                "op": op,
                "allow_tf32": allow_tf32,
                "deterministic": deterministic,
                "dtype": dtype,
                "shape": shape,
                "runs": min(len(optimized_rows), len(legacy_rows)),
                "optimized_latency_ms": optimized_latency,
                "legacy_latency_ms": legacy_latency,
                "speedup_vs_legacy": legacy_latency / optimized_latency,
                "optimized_bitwise_runs": sum(1 for row in optimized_rows if row["bitwise"] is True),
                "legacy_bitwise_runs": sum(1 for row in legacy_rows if row["bitwise"] is True),
                "optimized_overhead_vs_baseline": optimized_overhead,
                "legacy_overhead_vs_baseline": legacy_overhead,
                "optimized_path": (
                    "vectorized"
                    if max_vectorized_cols is not None
                    and cols is not None
                    and cols <= max_vectorized_cols
                    else "fallback"
                ),
            }
        )
    return out


def _validity_findings(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for item in summary:
        failed_runs = int(item["runs"]) - int(item["ok_runs"])
        non_bitwise_runs = int(item["ok_runs"]) - int(item["bitwise_runs"])
        if failed_runs == 0 and non_bitwise_runs == 0:
            continue
        findings.append(
            {
                **item,
                "failed_runs": failed_runs,
                "non_bitwise_runs": non_bitwise_runs,
            }
        )
    return findings


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Registered Batch-Invariant Operator Analysis",
        "",
        f"- Benchmark payloads: `{payload['payload_count']}`",
        f"- Mode rows: `{len(payload['rows'])}`",
        "",
        "## Operator Summary",
        "",
        "| op | TF32 | deterministic | dtype | mode | runs | ok | bitwise | mean ms | mean overhead | max invariant diff |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["summary"]:
        lines.append(
            f"| `{item['op']}` | `{item['allow_tf32']}` | `{item['deterministic']}` | "
            f"`{item['dtype']}` | `{item['mode']}` | "
            f"{item['runs']} | {item['ok_runs']} | {item['bitwise_runs']}/{item['ok_runs']} | "
            f"{_fmt(item['mean_latency_ms'])} | {_fmt(item['mean_overhead_vs_baseline'])} | "
            f"{_fmt(item['max_abs_diff'], digits=6)} |"
        )

    lines.extend(
        [
            "",
            "## Shape Detail",
            "",
            "| op | TF32 | deterministic | dtype | shape | mode | ok | bitwise | mean ms | mean overhead | max invariant diff |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["shape_summary"]:
        shape = ", ".join(f"{key}={value}" for key, value in item["shape"].items())
        lines.append(
            f"| `{item['op']}` | `{item['allow_tf32']}` | `{item['deterministic']}` | "
            f"`{item['dtype']}` | `{shape}` | `{item['mode']}` | "
            f"{item['ok_runs']}/{item['runs']} | {item['bitwise_runs']}/{item['ok_runs']} | "
            f"{_fmt(item['mean_latency_ms'])} | {_fmt(item['mean_overhead_vs_baseline'])} | "
            f"{_fmt(item['max_abs_diff'], digits=6)} |"
        )

    lines.extend(
        [
            "",
            "## Row-Wise Optimization Delta",
            "",
            "| op | TF32 | deterministic | dtype | shape | path | runs | optimized ms | legacy ms | speedup | optimized overhead | legacy overhead | optimized bitwise |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["legacy_speedups"]:
        shape = ", ".join(f"{key}={value}" for key, value in item["shape"].items())
        bitwise = f"{item['optimized_bitwise_runs']}/{item['runs']}"
        lines.append(
            f"| `{item['op']}` | `{item['allow_tf32']}` | `{item['deterministic']}` | "
            f"`{item['dtype']}` | `{shape}` | `{item['optimized_path']}` | {item['runs']} | "
            f"{_fmt(item['optimized_latency_ms'])} | {_fmt(item['legacy_latency_ms'])} | "
            f"{_fmt(item['speedup_vs_legacy'])} | {_fmt(item['optimized_overhead_vs_baseline'])} | "
            f"{_fmt(item['legacy_overhead_vs_baseline'])} | {bitwise} |"
        )

    lines.extend(
        [
            "",
            "## Validity Findings",
            "",
            "| op | TF32 | deterministic | dtype | mode | runs | failed | non-bitwise | mean ms | mean overhead | max invariant diff |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["validity_findings"]:
        lines.append(
            f"| `{item['op']}` | `{item['allow_tf32']}` | `{item['deterministic']}` | "
            f"`{item['dtype']}` | `{item['mode']}` | {item['runs']} | "
            f"{item['failed_runs']} | {item['non_bitwise_runs']} | "
            f"{_fmt(item['mean_latency_ms'])} | {_fmt(item['mean_overhead_vs_baseline'])} | "
            f"{_fmt(item['max_abs_diff'], digits=6)} |"
        )
    if not payload["validity_findings"]:
        lines.append("| - | - | - | - | - | 0 | 0 | 0 | - | - | - |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This report covers the CUDA operators actually registered by `enable_batch_invariant_mode`: `mm`, `addmm`, `_log_softmax`, `mean.dim`, `rms_norm`, optional `bmm`, plus SGLang TP row-linear kernels.",
            "- Batch invariance is measured by comparing sample 0 in a mixed batch with the same mode run on that sample alone.",
            "- The row-wise optimization delta compares the new vectorized row kernels with the previous fixed-1024 chunked kernels in the same process.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze registered batch-invariant operator benchmark results.")
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = _load_payloads(args.bench_dir)
    rows = _rows(payloads)
    payload = {
        "inputs": {"bench_dir": str(args.bench_dir)},
        "payload_count": len(payloads),
        "rows": rows,
        "summary": _group_summary(rows),
        "shape_summary": _shape_summary(rows),
        "legacy_speedups": _legacy_speedups(rows),
    }
    payload["validity_findings"] = _validity_findings(payload["summary"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
