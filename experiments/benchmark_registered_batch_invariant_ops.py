from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SGLANG_SRC = PROJECT_ROOT / ".data" / "src" / "sglang" / "python"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "registered_batch_invariant_operator_bench"


def _append_sglang_src() -> None:
    if SGLANG_SRC.exists() and str(SGLANG_SRC) not in sys.path:
        sys.path.insert(0, str(SGLANG_SRC))


def _load_batch_inv_ops() -> Any:
    _append_sglang_src()
    return importlib.import_module("sglang.srt.batch_invariant_ops.batch_invariant_ops")


def _load_tp_ops() -> Any:
    _append_sglang_src()
    return importlib.import_module("sglang.srt.tp_invariant_ops.tp_invariant_ops")


def _time_cuda(fn: Callable[[], torch.Tensor], *, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    diff = (a.float() - b.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "num_different": int((diff != 0).sum().item()),
        "numel": int(diff.numel()),
    }


def _randn(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _legacy_log_softmax(batch_ops: Any, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty_like(x_2d)
    batch_ops._log_softmax_kernel[(x_2d.shape[0],)](
        x_2d,
        out,
        x_2d.stride(0),
        out.stride(0),
        x_2d.shape[1],
        BLOCK_SIZE=1024,
    )
    return out.reshape(original_shape)


def _legacy_mean_last_dim(batch_ops: Any, x: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
    shape = list(x.shape)
    m = 1
    for value in shape[:-1]:
        m *= value
    n = shape[-1]
    x_3d = x.reshape(m, n, 1)
    output_shape = shape[:-1] + ([1] if keepdim else [])
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    out_2d = out.reshape(m, 1)
    batch_ops.mean_kernel[(m,)](
        x_3d,
        out_2d,
        x_3d.stride(0),
        x_3d.stride(1),
        x_3d.stride(2),
        out_2d.stride(0),
        out_2d.stride(1),
        m,
        n,
        1,
        1024,
    )
    return out


def _legacy_rms_norm(batch_ops: Any, x: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty_like(x_2d)
    batch_ops._rms_norm_kernel[(x_2d.shape[0],)](
        x_2d,
        weight.contiguous(),
        out,
        x_2d.stride(0),
        out.stride(0),
        x_2d.shape[1],
        eps,
        BLOCK_SIZE=1024,
    )
    return out.reshape(original_shape)


def _rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)
    return x.float().mul(torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)).mul(weight.float()).to(x.dtype)


def _record_modes(
    *,
    op_name: str,
    modes: dict[str, Callable[[bool], torch.Tensor]],
    baseline_mode: str,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode, fn in modes.items():
        try:
            mixed = fn(False)
            single = fn(True)
            first = mixed[0] if mixed.ndim >= 3 else mixed
            if op_name in {"mm", "addmm", "tp_row_linear", "moe_sum_tree_reduce"}:
                seq_len = single.shape[0]
                first = mixed[:seq_len]
            elif op_name == "mean_last_dim" and mixed.ndim == 2:
                first = mixed[0]
                single = single[0]
            elif op_name == "bmm":
                first = mixed[0]
                single = single[0]
            else:
                first = mixed[0]
                single = single[0]
            results[mode] = {
                "status": "ok",
                "latency_ms": _time_cuda(lambda fn=fn: fn(False), iters=iters, warmup=warmup, device=device),
                "invariance_vs_single": _compare(single, first),
            }
        except Exception as exc:
            results[mode] = {"status": "error", "error": repr(exc)}

    baseline_ms = results.get(baseline_mode, {}).get("latency_ms")
    for payload in results.values():
        if payload.get("status") == "ok" and baseline_ms:
            payload["overhead_vs_baseline"] = payload["latency_ms"] / baseline_ms
    return results


def _projection_cases(
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    generator: torch.Generator,
    batch_ops: Any,
    tp_ops: Any,
) -> list[dict[str, Any]]:
    records = []
    batch = args.batch
    seq_len = args.seq_len
    hidden = args.hidden
    vocab = args.vocab
    base = _randn((seq_len, hidden), dtype=dtype, device=device, generator=generator)
    mixed = _randn((batch * seq_len, hidden), dtype=dtype, device=device, generator=generator)
    mixed[:seq_len] = base
    weight = _randn((hidden, vocab), dtype=dtype, device=device, generator=generator) / hidden**0.5
    bias = _randn((vocab,), dtype=dtype, device=device, generator=generator)

    projection_modes = {
        "torch_mm": lambda single: torch.mm(base if single else mixed, weight),
        "batch_inv_persistent": lambda single: batch_ops.matmul_persistent(base if single else mixed, weight),
    }
    records.append(
        {
            "op": "mm",
            "dtype": dtype_name,
            "shape": {"batch": batch, "seq_len": seq_len, "hidden": hidden, "vocab": vocab},
            "baseline_mode": "torch_mm",
            "modes": _record_modes(
                op_name="mm",
                modes=projection_modes,
                baseline_mode="torch_mm",
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            ),
        }
    )

    addmm_modes = {
        "torch_addmm": lambda single: torch.addmm(bias, base if single else mixed, weight),
        "batch_inv_addmm": lambda single: batch_ops.addmm_batch_invariant(bias, base if single else mixed, weight),
    }
    records.append(
        {
            "op": "addmm",
            "dtype": dtype_name,
            "shape": {"batch": batch, "seq_len": seq_len, "hidden": hidden, "vocab": vocab},
            "baseline_mode": "torch_addmm",
            "modes": _record_modes(
                op_name="addmm",
                modes=addmm_modes,
                baseline_mode="torch_addmm",
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            ),
        }
    )

    tp_modes = {
        "torch_mm": lambda single: torch.mm(base if single else mixed, weight),
        "tp_inv": lambda single: tp_ops.matmul_tp_inv(base if single else mixed, weight),
        "tp_inv_optim": lambda single: tp_ops.matmul_tp_persistent_optim(base if single else mixed, weight),
    }
    records.append(
        {
            "op": "tp_row_linear",
            "dtype": dtype_name,
            "shape": {"batch": batch, "seq_len": seq_len, "hidden": hidden, "vocab": vocab},
            "baseline_mode": "torch_mm",
            "modes": _record_modes(
                op_name="tp_row_linear",
                modes=tp_modes,
                baseline_mode="torch_mm",
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            ),
        }
    )
    return records


def _moe_sum_case(
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    generator: torch.Generator,
    tp_ops: Any,
    batch: int,
    topk: int,
) -> dict[str, Any]:
    seq_len = args.seq_len
    hidden = args.moe_hidden
    num_experts = args.moe_experts
    scale = args.moe_routed_scaling_factor
    base_input = _randn((seq_len, topk, hidden), dtype=dtype, device=device, generator=generator)
    mixed_input = _randn((batch * seq_len, topk, hidden), dtype=dtype, device=device, generator=generator)
    mixed_input[:seq_len] = base_input

    base_ids = torch.arange(topk, device=device, dtype=torch.int32).expand(seq_len, topk).contiguous()
    mixed_ids = torch.rand(
        (batch * seq_len, num_experts),
        device=device,
        generator=generator,
    ).topk(topk, dim=-1).indices.to(torch.int32)
    mixed_ids[:seq_len] = base_ids

    def _tree_reduce(fn: Callable[..., torch.Tensor], single: bool) -> torch.Tensor:
        x = base_input if single else mixed_input
        ids = base_ids if single else mixed_ids
        out = torch.empty((x.shape[0], hidden), device=device, dtype=dtype)
        result = fn(x.contiguous(), out, ids.contiguous(), scale, num_experts)
        return out if result is None else result

    modes: dict[str, Callable[[bool], torch.Tensor]] = {
        "torch_sum": lambda single: (base_input if single else mixed_input).sum(dim=1) * scale,
        "moe_sum_tree_reduce": lambda single: _tree_reduce(tp_ops.moe_sum_tree_reduce, single),
        "moe_sum_tree_reduce_v1": lambda single: _tree_reduce(tp_ops.moe_sum_tree_reduce_v1, single),
    }
    if topk == 8:
        modes["moe_sum_tree_reduce_v0"] = lambda single: _tree_reduce(tp_ops.moe_sum_tree_reduce_v0, single)
    if topk in (8, 10):
        modes["moe_sum_tree_reduce_v2"] = lambda single: _tree_reduce(tp_ops.moe_sum_tree_reduce_v2, single)

    return {
        "op": "moe_sum_tree_reduce",
        "dtype": dtype_name,
        "shape": {
            "batch": batch,
            "seq_len": seq_len,
            "topk": topk,
            "hidden": hidden,
            "experts": num_experts,
        },
        "baseline_mode": "torch_sum",
        "modes": _record_modes(
            op_name="moe_sum_tree_reduce",
            modes=modes,
            baseline_mode="torch_sum",
            device=device,
            iters=args.iters,
            warmup=args.warmup,
        ),
    }


def _bmm_case(
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    generator: torch.Generator,
    batch_ops: Any,
    batch: int,
) -> dict[str, Any]:
    base_a = _randn((1, args.bmm_m, args.bmm_k), dtype=dtype, device=device, generator=generator)
    base_b = _randn((1, args.bmm_k, args.bmm_n), dtype=dtype, device=device, generator=generator)
    mixed_a = _randn((batch, args.bmm_m, args.bmm_k), dtype=dtype, device=device, generator=generator)
    mixed_b = _randn((batch, args.bmm_k, args.bmm_n), dtype=dtype, device=device, generator=generator)
    mixed_a[0:1] = base_a
    mixed_b[0:1] = base_b
    modes = {
        "torch_bmm": lambda single: torch.bmm(base_a if single else mixed_a, base_b if single else mixed_b),
        "batch_inv_bmm": lambda single: batch_ops.bmm_batch_invariant(base_a if single else mixed_a, base_b if single else mixed_b),
    }
    return {
        "op": "bmm",
        "dtype": dtype_name,
        "shape": {"batch": batch, "m": args.bmm_m, "k": args.bmm_k, "n": args.bmm_n},
        "baseline_mode": "torch_bmm",
        "modes": _record_modes(
            op_name="bmm",
            modes=modes,
            baseline_mode="torch_bmm",
            device=device,
            iters=args.iters,
            warmup=args.warmup,
        ),
    }


def _rowwise_cases(
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    generator: torch.Generator,
    batch_ops: Any,
    batch: int,
    cols: int,
) -> list[dict[str, Any]]:
    base = _randn((1, args.seq_len, cols), dtype=dtype, device=device, generator=generator)
    mixed = _randn((batch, args.seq_len, cols), dtype=dtype, device=device, generator=generator)
    mixed[0:1] = base
    weight = _randn((cols,), dtype=dtype, device=device, generator=generator)
    eps = 1e-6

    log_softmax_modes = {
        "torch_log_softmax": lambda single: F.log_softmax(base if single else mixed, dim=-1),
        "batch_inv_log_softmax": lambda single: batch_ops.log_softmax(base if single else mixed, dim=-1),
        "legacy_chunked_log_softmax": lambda single: _legacy_log_softmax(batch_ops, base if single else mixed, dim=-1),
    }
    mean_modes = {
        "torch_mean": lambda single: torch.mean(base if single else mixed, dim=-1),
        "batch_inv_mean": lambda single: batch_ops.mean_batch_invariant(base if single else mixed, [-1]),
        "legacy_chunked_mean": lambda single: _legacy_mean_last_dim(batch_ops, base if single else mixed),
    }
    rms_modes = {
        "torch_rms_norm": lambda single: _rms_norm_torch(base if single else mixed, weight, eps=eps),
        "batch_inv_rms_norm": lambda single: batch_ops.rms_norm_batch_invariant(base if single else mixed, weight, eps=eps),
        "legacy_chunked_rms_norm": lambda single: _legacy_rms_norm(batch_ops, base if single else mixed, weight, eps=eps),
    }
    return [
        {
            "op": "log_softmax",
            "dtype": dtype_name,
            "shape": {"batch": batch, "seq_len": args.seq_len, "cols": cols},
            "baseline_mode": "torch_log_softmax",
            "modes": _record_modes(
                op_name="log_softmax",
                modes=log_softmax_modes,
                baseline_mode="torch_log_softmax",
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            ),
        },
        {
            "op": "mean_last_dim",
            "dtype": dtype_name,
            "shape": {"batch": batch, "seq_len": args.seq_len, "cols": cols},
            "baseline_mode": "torch_mean",
            "modes": _record_modes(
                op_name="mean_last_dim",
                modes=mean_modes,
                baseline_mode="torch_mean",
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            ),
        },
        {
            "op": "rms_norm",
            "dtype": dtype_name,
            "shape": {"batch": batch, "seq_len": args.seq_len, "cols": cols},
            "baseline_mode": "torch_rms_norm",
            "modes": _record_modes(
                op_name="rms_norm",
                modes=rms_modes,
                baseline_mode="torch_rms_norm",
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark registered CUDA batch-invariant operators.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtypes", nargs="+", default=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch", type=int, default=8, help="Deprecated alias used when --batches is not set.")
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--rowwise-cols", nargs="+", type=int, default=[1024, 4096, 8192, 16384])
    parser.add_argument("--bmm-m", type=int, default=512)
    parser.add_argument("--bmm-k", type=int, default=128)
    parser.add_argument("--bmm-n", type=int, default=128)
    parser.add_argument("--moe-hidden", type=int, default=1024)
    parser.add_argument("--moe-topks", nargs="+", type=int, default=[8, 10])
    parser.add_argument("--moe-experts", type=int, default=64)
    parser.add_argument("--moe-routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for registered batch-invariant operator benchmarking.")

    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    batch_ops = _load_batch_inv_ops()
    tp_ops = _load_tp_ops()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    records = []
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    batches = args.batches or [args.batch]
    if any(topk > args.moe_experts for topk in args.moe_topks):
        raise ValueError(f"All --moe-topks must be <= --moe-experts, got {args.moe_topks=} and {args.moe_experts=}.")

    for dtype_name in args.dtypes:
        dtype = dtype_map[dtype_name]
        for batch in batches:
            args.batch = batch
            records.extend(
                _projection_cases(
                    args=args,
                    dtype=dtype,
                    dtype_name=dtype_name,
                    device=device,
                    generator=generator,
                    batch_ops=batch_ops,
                    tp_ops=tp_ops,
                )
            )
            records.append(
                _bmm_case(
                    args=args,
                    dtype=dtype,
                    dtype_name=dtype_name,
                    device=device,
                    generator=generator,
                    batch_ops=batch_ops,
                    batch=batch,
                )
            )
            for topk in args.moe_topks:
                records.append(
                    _moe_sum_case(
                        args=args,
                        dtype=dtype,
                        dtype_name=dtype_name,
                        device=device,
                        generator=generator,
                        tp_ops=tp_ops,
                        batch=batch,
                        topk=topk,
                    )
                )
            for cols in args.rowwise_cols:
                records.extend(
                    _rowwise_cases(
                        args=args,
                        dtype=dtype,
                        dtype_name=dtype_name,
                        device=device,
                        generator=generator,
                        batch_ops=batch_ops,
                        batch=batch,
                        cols=cols,
                    )
                )

    payload = {
        "config": {
            "batches": batches,
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "rowwise_cols": args.rowwise_cols,
            "bmm_m": args.bmm_m,
            "bmm_k": args.bmm_k,
            "bmm_n": args.bmm_n,
            "moe_hidden": args.moe_hidden,
            "moe_topks": args.moe_topks,
            "moe_experts": args.moe_experts,
            "moe_routed_scaling_factor": args.moe_routed_scaling_factor,
            "batch_inv_max_vectorized_rowwise_block_size": getattr(
                batch_ops,
                "_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE",
                None,
            ),
            "dtypes": args.dtypes,
            "iters": args.iters,
            "warmup": args.warmup,
            "deterministic": args.deterministic,
            "allow_tf32": args.allow_tf32,
            "seed": args.seed,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "SGLANG_BATCH_INVARIANT_OPS_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE": os.environ.get(
                "SGLANG_BATCH_INVARIANT_OPS_MAX_VECTORIZED_ROWWISE_BLOCK_SIZE"
            ),
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches_label = "-".join(str(batch) for batch in batches)
    shape_hash = hashlib.sha1(
        json.dumps(payload["config"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    output = args.output_dir / (
        f"registered_batch_inv_ops_batches-{batches_label}_seq-{args.seq_len}_"
        f"hidden-{args.hidden}_vocab-{args.vocab}_det-{int(args.deterministic)}_"
        f"tf32-{int(args.allow_tf32)}_seed-{args.seed}_{shape_hash}.json"
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"records": [str(output)]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
