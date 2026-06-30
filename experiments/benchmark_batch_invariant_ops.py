from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from certified_batch_invariant_projection import certified_top_projection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SGLANG_SRC = PROJECT_ROOT / ".data" / "src" / "sglang" / "python"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "batch_invariant_operator_bench"


def _append_sglang_src() -> None:
    if SGLANG_SRC.exists():
        sys.path.insert(0, str(SGLANG_SRC))


def _load_tp_ops() -> dict[str, Callable[..., torch.Tensor]]:
    _append_sglang_src()
    try:
        module = importlib.import_module("sglang.srt.tp_invariant_ops.tp_invariant_ops")
    except Exception as exc:
        return {"_import_error": lambda *_args, **_kwargs: (_ for _ in ()).throw(exc)}
    return {
        "tp_inv": module.matmul_tp_inv,
        "tp_inv_optim": module.matmul_tp_persistent_optim,
    }


def _load_batch_inv_ops() -> dict[str, Callable[..., torch.Tensor]]:
    _append_sglang_src()
    try:
        module = importlib.import_module("sglang.srt.batch_invariant_ops.batch_invariant_ops")
    except Exception as exc:
        return {"_import_error": lambda *_args, **_kwargs: (_ for _ in ()).throw(exc)}
    return {
        "batch_inv_persistent": module.matmul_persistent,
    }


def _sample_inputs(
    *,
    batch: int,
    seq_len: int,
    hidden: int,
    vocab: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    base_x = torch.randn(1, seq_len, hidden, device=device, dtype=dtype, generator=generator)
    mixed_x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype, generator=generator)
    mixed_x[0:1] = base_x
    weight = torch.randn(hidden, vocab, device=device, dtype=dtype, generator=generator) / hidden**0.5
    tokens = torch.randint(0, vocab, (batch, seq_len), device=device, generator=generator)
    return mixed_x, weight, tokens


def _sampled_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    diff = (a.float() - b.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "num_different": int((diff != 0).sum().item()),
        "numel": int(diff.numel()),
    }


def _time_cuda(fn: Callable[[], torch.Tensor], *, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _torch_batched_3d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w


def _torch_flattened_2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    return (x.reshape(batch * seq_len, -1) @ w).reshape(batch, seq_len, -1)


def _canonical_per_sample_loop(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[index] @ w for index in range(x.size(0))], dim=0)


def _flattened_custom(
    x: torch.Tensor,
    w: torch.Tensor,
    custom_matmul: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    return custom_matmul(x.reshape(batch * seq_len, -1).contiguous(), w).reshape(batch, seq_len, -1)


def _logits_from_mode(fn: Callable[[], torch.Tensor], *, batch: int, seq_len: int) -> torch.Tensor:
    logits = fn()
    if logits.ndim == 2:
        logits = logits.reshape(batch, seq_len, -1)
    return logits


def _build_modes(x: torch.Tensor, w: torch.Tensor, *, batch: int, seq_len: int) -> dict[str, Callable[[], torch.Tensor]]:
    modes: dict[str, Callable[[], torch.Tensor]] = {
        "torch_batched_3d": lambda: _torch_batched_3d(x, w),
        "torch_flattened_2d": lambda: _torch_flattened_2d(x, w),
        "canonical_per_sample_loop": lambda: _canonical_per_sample_loop(x, w),
        "certified_top_projection": lambda: certified_top_projection(x, w),
        "certified_top_projection_0drift": lambda: certified_top_projection(
            x,
            w,
            require_torch_single_drift_free=True,
        ),
    }
    for name, op in _load_batch_inv_ops().items():
        if name.startswith("_"):
            modes[name] = lambda op=op: op(x.reshape(batch * seq_len, -1).contiguous(), w)
        else:
            modes[name] = lambda op=op: _flattened_custom(x, w, op)
    for name, op in _load_tp_ops().items():
        if name.startswith("_"):
            modes[name] = lambda op=op: op(x.reshape(batch * seq_len, -1).contiguous(), w)
        else:
            modes[name] = lambda op=op: _flattened_custom(x, w, op)
    return modes


def run_one(args: argparse.Namespace, *, batch: int, dtype: torch.dtype) -> dict[str, Any]:
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    x, w, tokens = _sample_inputs(
        batch=batch,
        seq_len=args.seq_len,
        hidden=args.hidden,
        vocab=args.vocab,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    single_x = x[0:1]
    single_tokens = tokens[0:1]
    torch_single_logits = single_x[0] @ w
    torch_single_logprobs = _sampled_logprobs(torch_single_logits.unsqueeze(0), single_tokens)

    modes = _build_modes(x, w, batch=batch, seq_len=args.seq_len)
    single_modes = _build_modes(single_x, w, batch=1, seq_len=args.seq_len)

    results: dict[str, Any] = {}
    for name, fn in modes.items():
        try:
            logits = _logits_from_mode(fn, batch=batch, seq_len=args.seq_len)
            sampled = _sampled_logprobs(logits, tokens)
            single_mode_logits = _logits_from_mode(single_modes[name], batch=1, seq_len=args.seq_len)
            single_mode_logprobs = _sampled_logprobs(single_mode_logits, single_tokens)
            matmul_ms = _time_cuda(fn, iters=args.iters, warmup=args.warmup, device=device)
            matmul_logprob_ms = _time_cuda(
                lambda fn=fn: _sampled_logprobs(_logits_from_mode(fn, batch=batch, seq_len=args.seq_len), tokens),
                iters=args.iters,
                warmup=args.warmup,
                device=device,
            )
            results[name] = {
                "status": "ok",
                "matmul_ms": matmul_ms,
                "matmul_plus_logprob_ms": matmul_logprob_ms,
                "logit_invariance_vs_mode_single": _compare(single_mode_logits, logits[0:1]),
                "sampled_logprob_invariance_vs_mode_single": _compare(single_mode_logprobs, sampled[0:1]),
                "logit_drift_vs_torch_single": _compare(torch_single_logits.unsqueeze(0), logits[0:1]),
                "sampled_logprob_drift_vs_torch_single": _compare(torch_single_logprobs, sampled[0:1]),
            }
        except Exception as exc:
            results[name] = {"status": "error", "error": repr(exc)}

    torch_base = results.get("torch_batched_3d", {})
    base_ms = torch_base.get("matmul_ms")
    for payload in results.values():
        if payload.get("status") == "ok" and base_ms:
            payload["matmul_overhead_vs_torch_batched"] = payload["matmul_ms"] / base_ms

    return {
        "config": {
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "dtype": str(dtype).removeprefix("torch."),
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
        },
        "modes": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batch-invariant/TOP-invariant projection operators.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtypes", nargs="+", default=["float16", "bfloat16", "float32"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batch-invariant operator benchmarking.")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dtype_name in args.dtypes:
        dtype = dtype_map[dtype_name]
        for batch in args.batches:
            record = run_one(args, batch=batch, dtype=dtype)
            out = args.output_dir / (
                f"batch_inv_operator_dtype-{dtype_name}_batch-{batch}_seq-{args.seq_len}_"
                f"hidden-{args.hidden}_vocab-{args.vocab}_det-{int(args.deterministic)}_"
                f"tf32-{int(args.allow_tf32)}_seed-{args.seed}.json"
            )
            out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            records.append(str(out))
            print(out)
    (args.output_dir / "manifest.json").write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
