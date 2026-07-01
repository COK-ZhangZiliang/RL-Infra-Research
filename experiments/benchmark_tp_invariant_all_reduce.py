from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SGLANG_SRC = PROJECT_ROOT / ".data" / "src" / "sglang" / "python"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "tp_invariant_all_reduce_bench"


def _append_sglang_src() -> None:
    if SGLANG_SRC.exists() and str(SGLANG_SRC) not in sys.path:
        sys.path.insert(0, str(SGLANG_SRC))


def _load_tp_ops() -> Any:
    _append_sglang_src()
    from sglang.srt.tp_invariant_ops import tp_invariant_ops

    return tp_invariant_ops


def _dtype(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return mapping[name]


def _time_distributed(
    fn: Callable[[], torch.Tensor],
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    result = fn()
    for _ in range(warmup):
        result = fn()
    dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()
    return (time.perf_counter() - start) * 1000.0 / iters, result


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    diff = (a.float() - b.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "num_different": int((diff != 0).sum().item()),
        "numel": int(diff.numel()),
    }


def _bench_shape(
    *,
    tp_ops: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rank = dist.get_rank()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator).contiguous()

    def torch_all_reduce() -> torch.Tensor:
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def tree_all_reduce() -> torch.Tensor:
        return tp_ops.tree_all_reduce_sum(x, device_group=dist.group.WORLD)

    def tree_all_reduce_optim() -> torch.Tensor:
        return tp_ops.tree_all_reduce_sum_optim(x, device_group=dist.group.WORLD)

    modes = {
        "torch_all_reduce": torch_all_reduce,
        "tree_all_reduce_sum": tree_all_reduce,
        "tree_all_reduce_sum_optim": tree_all_reduce_optim,
    }
    results: dict[str, Any] = {}
    reference = None
    for mode, fn in modes.items():
        try:
            latency_ms, output = _time_distributed(fn, iters=args.iters, warmup=args.warmup, device=device)
            repeat = fn()
            if mode == "torch_all_reduce":
                reference = output
            results[mode] = {
                "status": "ok",
                "latency_ms": latency_ms,
                "repeatability": _compare(output, repeat),
                "vs_torch_all_reduce": None if reference is None else _compare(reference, output),
            }
        except Exception as exc:
            results[mode] = {"status": "error", "error": repr(exc)}

    baseline_ms = results.get("torch_all_reduce", {}).get("latency_ms")
    for payload in results.values():
        if payload.get("status") == "ok" and baseline_ms:
            payload["overhead_vs_torch_all_reduce"] = payload["latency_ms"] / baseline_ms

    return {
        "dtype": dtype_name,
        "shape": list(shape),
        "rank": rank,
        "modes": results,
    }


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.lower().replace("x", ",").split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TP-invariant deterministic tree all-reduce.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtypes", nargs="+", default=["float16", "bfloat16", "float32"])
    parser.add_argument("--shapes", nargs="+", default=["512,1024", "4096,1024", "8192,4096"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL all-reduce benchmarking.")
    dist.init_process_group(backend=args.backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tp_ops = _load_tp_ops()
    rank_payloads = []
    for dtype_name in args.dtypes:
        for shape in [_parse_shape(item) for item in args.shapes]:
            rank_payloads.append(
                _bench_shape(
                    tp_ops=tp_ops,
                    shape=shape,
                    dtype=_dtype(dtype_name),
                    dtype_name=dtype_name,
                    device=device,
                    args=args,
                )
            )

    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, rank_payloads)

    if dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "world_size": dist.get_world_size(),
                "backend": args.backend,
                "dtypes": args.dtypes,
                "shapes": [list(_parse_shape(item)) for item in args.shapes],
                "iters": args.iters,
                "warmup": args.warmup,
                "seed": args.seed,
            },
            "records": [item for rank_items in gathered for item in (rank_items or [])],
        }
        output = args.output_dir / (
            f"tp_invariant_all_reduce_world-{dist.get_world_size()}_"
            f"backend-{args.backend}_seed-{args.seed}.json"
        )
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "manifest.json").write_text(
            json.dumps({"records": [str(output)]}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
