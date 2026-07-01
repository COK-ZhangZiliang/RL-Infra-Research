from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "tp_invariant_all_reduce_bench"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "tp_invariant_all_reduce_analysis"


def _load_payloads(bench_dir: Path) -> list[dict[str, Any]]:
    if not bench_dir.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(bench_dir.glob("tp_invariant_all_reduce_*.json"))]


def _rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        config = payload.get("config", {})
        for record in payload.get("records", []):
            for mode, stats in record.get("modes", {}).items():
                repeatability = stats.get("repeatability") or {}
                vs_torch = stats.get("vs_torch_all_reduce") or {}
                rows.append(
                    {
                        "world_size": config.get("world_size"),
                        "backend": config.get("backend"),
                        "dtype": record.get("dtype"),
                        "shape": record.get("shape"),
                        "rank": record.get("rank"),
                        "mode": mode,
                        "status": stats.get("status"),
                        "latency_ms": stats.get("latency_ms"),
                        "overhead_vs_torch_all_reduce": stats.get("overhead_vs_torch_all_reduce"),
                        "repeatable_bitwise": repeatability.get("bitwise_equal"),
                        "repeatable_max_abs_diff": repeatability.get("max_abs_diff"),
                        "vs_torch_bitwise": vs_torch.get("bitwise_equal"),
                        "vs_torch_max_abs_diff": vs_torch.get("max_abs_diff"),
                        "error": stats.get("error"),
                    }
                )
    return rows


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row["world_size"],
                row["backend"],
                row["dtype"],
                json.dumps(row["shape"], sort_keys=True),
                row["mode"],
            ),
            [],
        ).append(row)

    out = []
    for (world_size, backend, dtype, shape_key, mode), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row["status"] == "ok"]
        out.append(
            {
                "world_size": world_size,
                "backend": backend,
                "dtype": dtype,
                "shape": json.loads(shape_key),
                "mode": mode,
                "ranks": len(group_rows),
                "ok_ranks": len(ok_rows),
                "mean_latency_ms": _mean([row["latency_ms"] for row in ok_rows]),
                "max_latency_ms": max([row["latency_ms"] for row in ok_rows if row["latency_ms"] is not None], default=None),
                "mean_overhead_vs_torch_all_reduce": _mean([row["overhead_vs_torch_all_reduce"] for row in ok_rows]),
                "repeatable_bitwise_ranks": sum(1 for row in ok_rows if row["repeatable_bitwise"] is True),
                "max_repeatable_diff": max(
                    [row["repeatable_max_abs_diff"] for row in ok_rows if row["repeatable_max_abs_diff"] is not None],
                    default=None,
                ),
                "vs_torch_bitwise_ranks": sum(1 for row in ok_rows if row["vs_torch_bitwise"] is True),
                "max_vs_torch_diff": max(
                    [row["vs_torch_max_abs_diff"] for row in ok_rows if row["vs_torch_max_abs_diff"] is not None],
                    default=None,
                ),
            }
        )
    return out


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TP-Invariant All-Reduce Analysis",
        "",
        f"- Benchmark payloads: `{payload['payload_count']}`",
        f"- Rank-mode rows: `{len(payload['rows'])}`",
        "",
        "| world | backend | dtype | shape | mode | ok ranks | mean ms | max ms | mean overhead | repeatable | max repeat diff | vs torch | max vs torch diff |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["summary"]:
        shape = "x".join(str(value) for value in item["shape"])
        lines.append(
            f"| {item['world_size']} | `{item['backend']}` | `{item['dtype']}` | `{shape}` | `{item['mode']}` | "
            f"{item['ok_ranks']}/{item['ranks']} | {_fmt(item['mean_latency_ms'])} | {_fmt(item['max_latency_ms'])} | "
            f"{_fmt(item['mean_overhead_vs_torch_all_reduce'])} | "
            f"{item['repeatable_bitwise_ranks']}/{item['ok_ranks']} | {_fmt(item['max_repeatable_diff'], digits=6)} | "
            f"{item['vs_torch_bitwise_ranks']}/{item['ok_ranks']} | {_fmt(item['max_vs_torch_diff'], digits=6)} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TP-invariant all-reduce benchmark results.")
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
        "summary": _summary(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
