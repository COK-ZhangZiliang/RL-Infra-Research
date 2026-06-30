from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from certified_batch_invariant_projection import certified_top_projection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "low_precision_dtype_probe"


def _dtype_candidates() -> list[tuple[str, torch.dtype | None]]:
    names = ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"]
    out: list[tuple[str, torch.dtype | None]] = []
    for name in names:
        out.append((name, getattr(torch, name, None)))
    return out


def _sample_float8_inputs(
    *,
    dtype: torch.dtype,
    batch: int,
    seq_len: int,
    hidden: int,
    vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.randn(batch, seq_len, hidden, device=device, dtype=torch.float16) * 0.1).to(dtype)
    w = (torch.randn(hidden, vocab, device=device, dtype=torch.float16) / hidden**0.5).to(dtype)
    return x, w


def _probe_dtype(
    *,
    name: str,
    dtype: torch.dtype | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if dtype is None:
        return {
            "dtype": name,
            "input_creation_status": "missing_dtype",
            "torch_matmul_status": "skipped",
            "certified_projection_status": "skipped",
            "note": "This torch build does not expose the dtype.",
        }

    device = torch.device(f"cuda:{args.device}")
    result: dict[str, Any] = {
        "dtype": name,
        "input_creation_status": "unknown",
        "torch_matmul_status": "unknown",
        "certified_projection_status": "unknown",
        "note": "",
    }
    try:
        x, w = _sample_float8_inputs(
            dtype=dtype,
            batch=args.batch,
            seq_len=args.seq_len,
            hidden=args.hidden,
            vocab=args.vocab,
            device=device,
        )
        result["input_creation_status"] = "ok"
    except Exception as exc:
        result["input_creation_status"] = "error"
        result["torch_matmul_status"] = "skipped"
        result["certified_projection_status"] = "skipped"
        result["note"] = repr(exc)
        return result

    try:
        y = x @ w
        torch.cuda.synchronize(device)
        result["torch_matmul_status"] = f"ok:{tuple(y.shape)}:{y.dtype}"
    except Exception as exc:
        result["torch_matmul_status"] = "error"
        result["note"] = repr(exc)

    try:
        y = certified_top_projection(x, w)
        torch.cuda.synchronize(device)
        result["certified_projection_status"] = f"ok:{tuple(y.shape)}:{y.dtype}"
    except Exception as exc:
        result["certified_projection_status"] = "error"
        if result["note"]:
            result["note"] += f"; certified={repr(exc)}"
        else:
            result["note"] = repr(exc)

    if not result["note"]:
        result["note"] = "Float8 projection path appears runnable for this tiny probe."
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe lower-than-FP16 dtype support for projection operators.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for low precision dtype probing.")
    payload = {
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "config": {
            "batch": args.batch,
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "vocab": args.vocab,
        },
        "results": [
            _probe_dtype(name=name, dtype=dtype, args=args)
            for name, dtype in _dtype_candidates()
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output_dir / "probe.json")


if __name__ == "__main__":
    main()
