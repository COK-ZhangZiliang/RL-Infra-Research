from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SGLANG_SRT = PROJECT_ROOT / ".data" / "src" / "sglang" / "python" / "sglang" / "srt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_coverage"


CUDA_REGISTERED_BENCHMARKS = {
    "aten::mm": "mm",
    "aten::addmm": "addmm",
    "aten::_log_softmax": "log_softmax",
    "aten::mean.dim": "mean_last_dim",
    "aten::rms_norm": "rms_norm",
    "aten::mm.dtype": "mm",
    "aten::bmm": "bmm",
}

TP_EXPORTED_BENCHMARKS = {
    "matmul_tp_persistent": "tp_row_linear",
    "matmul_tp_inv": "tp_row_linear",
    "moe_sum_tree_reduce": "moe_sum_tree_reduce",
    "tree_all_reduce_sum": "tp_invariant_all_reduce",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cuda_registered_ops(source: str) -> list[str]:
    start = source.index("if not _is_npu:")
    end = source.index("        from sglang.srt.hardware_backend.npu", start)
    cuda_source = source[start:end]
    return sorted(set(re.findall(r'_batch_invariant_LIB\.impl\(\s*"([^"]+)"', cuda_source, flags=re.MULTILINE)))


def _tp_exports(source: str) -> list[str]:
    tree = ast.parse(source)
    exports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for item in node.value.elts:
                            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                exports.append(item.value)
    return sorted(set(exports))


def build_report() -> dict[str, Any]:
    batch_source = _read(SGLANG_SRT / "batch_invariant_ops" / "batch_invariant_ops.py")
    tp_init_source = _read(SGLANG_SRT / "tp_invariant_ops" / "__init__.py")

    cuda_ops = _cuda_registered_ops(batch_source)
    tp_exports = _tp_exports(tp_init_source)
    threshold_match = re.search(
        r'_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE\s*=\s*_get_int_env\(\s*"SGLANG_BATCH_INVARIANT_OPS_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE",\s*(\d+)',
        batch_source,
        flags=re.MULTILINE,
    )
    cuda_rows = [
        {
            "operator": op,
            "benchmark_case": CUDA_REGISTERED_BENCHMARKS.get(op),
            "covered": op in CUDA_REGISTERED_BENCHMARKS,
        }
        for op in cuda_ops
    ]
    tp_rows = [
        {
            "operator": op,
            "benchmark_case": TP_EXPORTED_BENCHMARKS.get(op),
            "covered": op in TP_EXPORTED_BENCHMARKS,
        }
        for op in tp_exports
        if op not in {
            "set_tp_invariant_mode",
            "is_tp_invariant_mode_enabled",
            "disable_tp_invariant_mode",
            "enable_tp_invariant_mode",
        }
    ]
    return {
        "inputs": {
            "batch_invariant_ops": str(SGLANG_SRT / "batch_invariant_ops" / "batch_invariant_ops.py"),
            "tp_invariant_exports": str(SGLANG_SRT / "tp_invariant_ops" / "__init__.py"),
        },
        "cuda_registered_ops": cuda_rows,
        "tp_exported_ops": tp_rows,
        "uncovered_cuda_ops": [row["operator"] for row in cuda_rows if not row["covered"]],
        "uncovered_tp_ops": [row["operator"] for row in tp_rows if not row["covered"]],
        "rowwise_optimization": {
            "env": "SGLANG_BATCH_INVARIANT_OPS_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE",
            "default_max_cols": int(threshold_match.group(1)) if threshold_match else None,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Batch-Invariant Operator Coverage Audit",
        "",
        "## CUDA Registered Ops",
        "",
        "| operator | benchmark case | covered |",
        "|---|---|---|",
    ]
    for row in payload["cuda_registered_ops"]:
        lines.append(f"| `{row['operator']}` | `{row['benchmark_case']}` | `{row['covered']}` |")

    lines.extend(
        [
            "",
            "## TP Exported Ops",
            "",
            "| operator | benchmark case | covered |",
            "|---|---|---|",
        ]
    )
    for row in payload["tp_exported_ops"]:
        lines.append(f"| `{row['operator']}` | `{row['benchmark_case']}` | `{row['covered']}` |")

    lines.extend(
        [
            "",
            "## Row-Wise Optimization Config",
            "",
            f"- Env: `{payload['rowwise_optimization']['env']}`",
            f"- Default max vectorized columns: `{payload['rowwise_optimization']['default_max_cols']}`",
            "",
            "## Uncovered",
            "",
            f"- CUDA registered ops: `{payload['uncovered_cuda_ops']}`",
            f"- TP exported ops: `{payload['uncovered_tp_ops']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit benchmark coverage for current batch/TP-invariant operators.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
