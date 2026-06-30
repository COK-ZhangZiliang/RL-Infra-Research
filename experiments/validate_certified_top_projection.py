from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_analysis" / "report.json"
DEFAULT_ANALYSIS_TF32 = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_analysis_tf32" / "report.json"
DEFAULT_PLAN = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_plan" / "plan.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "certified_top_projection_validation"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contract_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["dtype"], bool(row["allow_tf32"]), row["seq_len"], row["hidden"], row["vocab"])


def _validate_analysis(payload: dict[str, Any], *, max_overhead: float) -> list[dict[str, Any]]:
    rows = [row for row in payload["rows"] if row["mode"] == "certified_top_projection"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_contract_key(row), []).append(row)

    results = []
    for key, group_rows in sorted(grouped.items()):
        dtype, allow_tf32, seq_len, hidden, vocab = key
        ok_rows = [row for row in group_rows if row["status"] == "ok"]
        bitwise_rows = [row for row in ok_rows if row["logprob_bitwise"] is True]
        overheads = [row["overhead_vs_torch"] for row in ok_rows if row["overhead_vs_torch"] is not None]
        batches = sorted({row["batch"] for row in group_rows})
        mean_overhead = sum(overheads) / len(overheads) if overheads else None
        passed = (
            len(group_rows) > 0
            and len(ok_rows) == len(group_rows)
            and len(bitwise_rows) == len(group_rows)
            and mean_overhead is not None
            and mean_overhead <= max_overhead
            and batches == [1, 2, 4, 8, 16]
        )
        results.append(
            {
                "contract": {
                    "dtype": dtype,
                    "allow_tf32": allow_tf32,
                    "seq_len": seq_len,
                    "hidden": hidden,
                    "vocab": vocab,
                    "batches": batches,
                },
                "passed": passed,
                "runs": len(group_rows),
                "ok_runs": len(ok_rows),
                "bitwise_runs": len(bitwise_rows),
                "mean_overhead_vs_torch": mean_overhead,
                "max_invariant_diff": max(
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
    return results


def _validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for item in plan["contracts"]:
        selected_mode = item.get("selected_mode")
        selected_stats = item.get("selected_stats") or {}
        passed = (
            selected_mode == "certified_top_projection"
            and selected_stats.get("bitwise_runs") == selected_stats.get("runs")
        )
        results.append(
            {
                "contract": item["contract"],
                "passed": passed,
                "selected_mode": selected_mode,
                "selected_kind": item.get("selected_kind"),
                "selected_stats": selected_stats,
            }
        )
    return results


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Certified TOP Projection Validation",
        "",
        f"- Overall passed: `{payload['passed']}`",
        f"- Overhead gate: `<={payload['max_overhead']:.2f}x`",
        "",
        "## Operator Contracts",
        "",
        "| dtype | TF32 | shape | runs | bitwise | mean overhead | max invariant diff | max drift vs torch single | passed |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["operator_contracts"]:
        contract = item["contract"]
        shape = f"{contract['seq_len']}x{contract['hidden']}x{contract['vocab']}"
        lines.append(
            f"| `{contract['dtype']}` | `{contract['allow_tf32']}` | `{shape}` | "
            f"{item['runs']} | {item['bitwise_runs']}/{item['runs']} | "
            f"{_fmt(item['mean_overhead_vs_torch'])} | {_fmt(item['max_invariant_diff'], digits=6)} | "
            f"{_fmt(item['max_drift_vs_torch_single'], digits=6)} | `{item['passed']}` |"
        )

    lines.extend(
        [
            "",
            "## Planner Contracts",
            "",
            "| dtype | TF32 | selected mode | selected kind | passed |",
            "|---|---|---|---|---|",
        ]
    )
    for item in payload["planner_contracts"]:
        contract = item["contract"]
        lines.append(
            f"| `{contract['dtype']}` | `{contract['allow_tf32']}` | "
            f"`{item['selected_mode']}` | `{item['selected_kind']}` | `{item['passed']}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the certified TOP projection operator gates.")
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--analysis-tf32", type=Path, default=DEFAULT_ANALYSIS_TF32)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-overhead", type=float, default=1.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_results = _validate_analysis(_load_json(args.analysis), max_overhead=args.max_overhead)
    analysis_results.extend(_validate_analysis(_load_json(args.analysis_tf32), max_overhead=args.max_overhead))
    plan_results = _validate_plan(_load_json(args.plan))
    payload = {
        "max_overhead": args.max_overhead,
        "passed": all(item["passed"] for item in analysis_results + plan_results),
        "operator_contracts": analysis_results,
        "planner_contracts": plan_results,
        "inputs": {
            "analysis": str(args.analysis),
            "analysis_tf32": str(args.analysis_tf32),
            "plan": str(args.plan),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "validation.md").write_text(render_markdown(payload), encoding="utf-8")
    print(args.output_dir / "validation.md")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
