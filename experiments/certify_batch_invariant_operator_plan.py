from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_DIRS = [
    PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_bench",
    PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_bench_tf32",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_plan"

FAST_PATH_PRIORITY = [
    "certified_top_projection",
    "torch_batched_3d",
    "torch_flattened_2d",
    "batch_inv_persistent",
]
FALLBACK_PRIORITY = [
    "canonical_per_sample_loop",
]


def _load_records(bench_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bench_dir in bench_dirs:
        if not bench_dir.exists():
            continue
        for path in sorted(bench_dir.glob("batch_inv_operator_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_path"] = str(path)
            records.append(payload)
    return records


def _rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        config = record["config"]
        contract = {
            "dtype": config["dtype"],
            "allow_tf32": bool(config["allow_tf32"]),
            "seq_len": config["seq_len"],
            "hidden": config["hidden"],
            "vocab": config["vocab"],
        }
        for mode, stats in record["modes"].items():
            sampled = stats.get("sampled_logprob_invariance_vs_mode_single") or stats.get(
                "sampled_logprob_invariance_vs_single", {}
            )
            drift = stats.get("sampled_logprob_drift_vs_torch_single", {})
            rows.append(
                {
                    **contract,
                    "batch": config["batch"],
                    "mode": mode,
                    "status": stats.get("status"),
                    "matmul_ms": stats.get("matmul_ms"),
                    "overhead_vs_torch": stats.get("matmul_overhead_vs_torch_batched"),
                    "logprob_bitwise": sampled.get("bitwise_equal"),
                    "logprob_max_abs_diff": sampled.get("max_abs_diff"),
                    "logprob_drift_vs_torch_single_bitwise": drift.get("bitwise_equal"),
                    "logprob_drift_vs_torch_single_max_abs_diff": drift.get("max_abs_diff"),
                    "error": stats.get("error"),
                }
            )
    return rows


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _contract_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["dtype"], row["allow_tf32"], row["seq_len"], row["hidden"], row["vocab"])


def _summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    return {
        "runs": len(rows),
        "ok_runs": len(ok_rows),
        "bitwise_runs": sum(1 for row in ok_rows if row["logprob_bitwise"] is True),
        "mean_overhead_vs_torch": _mean([row["overhead_vs_torch"] for row in ok_rows]),
        "max_logprob_diff": max(
            [row["logprob_max_abs_diff"] for row in ok_rows if row["logprob_max_abs_diff"] is not None],
            default=None,
        ),
    }


def _eligible(stats: dict[str, Any], *, max_overhead: float | None = None) -> bool:
    if stats["runs"] == 0 or stats["ok_runs"] != stats["runs"] or stats["bitwise_runs"] != stats["runs"]:
        return False
    if max_overhead is not None:
        overhead = stats["mean_overhead_vs_torch"]
        if overhead is None or overhead > max_overhead:
            return False
    return True


def build_plan(rows: list[dict[str, Any]], *, max_fast_overhead: float) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_contract_key(row), []).append(row)

    contracts = []
    for key, contract_rows in sorted(grouped.items()):
        dtype, allow_tf32, seq_len, hidden, vocab = key
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for row in contract_rows:
            by_mode.setdefault(row["mode"], []).append(row)

        mode_summary = {mode: _summarize_mode(mode_rows) for mode, mode_rows in by_mode.items()}
        selected_mode = None
        selected_reason = ""
        selected_kind = "unresolved"

        for mode in FAST_PATH_PRIORITY:
            stats = mode_summary.get(mode)
            if stats and _eligible(stats, max_overhead=max_fast_overhead):
                selected_mode = mode
                selected_kind = "certified_fast_path"
                selected_reason = (
                    f"0-diff across all measured batches and mean overhead <= {max_fast_overhead:.2f}x"
                )
                break

        if selected_mode is None:
            for mode in FALLBACK_PRIORITY:
                stats = mode_summary.get(mode)
                if stats and _eligible(stats):
                    selected_mode = mode
                    selected_kind = "strict_fallback"
                    selected_reason = "0-diff across all measured batches, but not near-zero overhead"
                    break

        contracts.append(
            {
                "contract": {
                    "dtype": dtype,
                    "allow_tf32": allow_tf32,
                    "seq_len": seq_len,
                    "hidden": hidden,
                    "vocab": vocab,
                    "batches": sorted({row["batch"] for row in contract_rows}),
                },
                "selected_mode": selected_mode,
                "selected_kind": selected_kind,
                "selected_reason": selected_reason,
                "selected_stats": mode_summary.get(selected_mode) if selected_mode else None,
                "mode_summary": mode_summary,
            }
        )

    return {
        "max_fast_overhead": max_fast_overhead,
        "contracts": contracts,
    }


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Certified Batch-Invariant Operator Plan",
        "",
        f"- Fast-path overhead gate: `<={plan['max_fast_overhead']:.2f}x`",
        "- Certificate scope: only the measured dtype, math mode, shape, backend, GPU, and batch set.",
        "",
        "| dtype | TF32 | shape | selected mode | kind | mean overhead | bitwise runs | reason |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for item in plan["contracts"]:
        contract = item["contract"]
        stats = item.get("selected_stats") or {}
        shape = f"{contract['seq_len']}x{contract['hidden']}x{contract['vocab']}"
        bitwise = f"{stats.get('bitwise_runs', '-')}/{stats.get('runs', '-')}"
        lines.append(
            f"| `{contract['dtype']}` | `{contract['allow_tf32']}` | `{shape}` | "
            f"`{item['selected_mode']}` | `{item['selected_kind']}` | "
            f"{_fmt(stats.get('mean_overhead_vs_torch'))} | {bitwise} | {item['selected_reason']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `certified_fast_path` means TOP can use the regular backend for this measured contract with near-zero operator overhead.",
            "- `strict_fallback` means correctness is available, but the near-zero overhead goal is not yet met.",
            "- The remaining research target is every `strict_fallback` or `unresolved` contract.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select certified batch-invariant operators from benchmark records.")
    parser.add_argument("--bench-dirs", nargs="+", type=Path, default=DEFAULT_BENCH_DIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-fast-overhead", type=float, default=1.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.bench_dirs)
    rows = _rows(records)
    plan = build_plan(rows, max_fast_overhead=args.max_fast_overhead)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "plan.md").write_text(render_markdown(plan), encoding="utf-8")
    print(args.output_dir / "plan.md")


if __name__ == "__main__":
    main()
