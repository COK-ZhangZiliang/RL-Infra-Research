from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".data" / "miles-runs" / "canonical_persistent_gemm_tuning"


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _candidate_persistent_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    base_m = tl.arange(0, BLOCK_M)
    base_n = tl.arange(0, BLOCK_N)
    base_k = tl.arange(0, BLOCK_K)
    base_k = tl.max_contiguous(tl.multiple_of(base_k, BLOCK_K), BLOCK_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_M + base_m
        offs_n = pid_n * BLOCK_N + base_n
        mask_m = offs_m < M
        mask_n = offs_n < N
        safe_m = tl.where(mask_m, offs_m, 0)
        safe_n = tl.where(mask_n, offs_n, 0)

        safe_m = tl.max_contiguous(tl.multiple_of(safe_m, BLOCK_M), BLOCK_M)
        safe_n = tl.max_contiguous(tl.multiple_of(safe_n, BLOCK_N), BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + base_k
            a_ptrs = a_ptr + safe_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + safe_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

        c_ptrs = c_ptr + safe_m[:, None] * stride_cm + safe_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _candidate_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    num_warps: int,
    num_stages: int,
    input_precision: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    def grid(meta: dict[str, Any]) -> tuple[int]:
        tiles = triton.cdiv(a.shape[0], meta["BLOCK_M"]) * triton.cdiv(b.shape[1], meta["BLOCK_N"])
        return (min(num_sms, tiles),)

    _candidate_persistent_matmul_kernel[grid](
        a,
        b,
        c,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


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


def _canonical_per_sample_loop(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[index] @ w for index in range(x.size(0))], dim=0)


def _candidate_configs(dtype_name: str, input_precision: str) -> list[dict[str, Any]]:
    if dtype_name == "float32":
        base = [
            (64, 64, 32, 8, 4, 3),
            (64, 128, 32, 8, 4, 3),
            (128, 64, 32, 8, 4, 3),
            (128, 128, 32, 8, 8, 3),
            (64, 64, 64, 8, 4, 3),
            (128, 128, 64, 4, 8, 3),
        ]
    else:
        base = [
            (64, 128, 64, 8, 4, 3),
            (64, 256, 64, 8, 8, 3),
            (128, 128, 64, 8, 8, 3),
            (128, 256, 64, 4, 8, 3),
        ]
    configs = []
    for block_m, block_n, block_k, group_m, num_warps, num_stages in base:
        configs.append(
            {
                "name": (
                    f"candidate_bm{block_m}_bn{block_n}_bk{block_k}_"
                    f"g{group_m}_w{num_warps}_s{num_stages}_{input_precision}"
                ),
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "group_m": group_m,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "input_precision": input_precision,
            }
        )
    return configs


def _build_modes(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    dtype_name: str,
    input_precision: str,
) -> dict[str, Callable[[], torch.Tensor]]:
    modes: dict[str, Callable[[], torch.Tensor]] = {
        "torch_batched_3d": lambda: _torch_batched_3d(x, w),
        "canonical_per_sample_loop": lambda: _canonical_per_sample_loop(x, w),
    }
    flat_x = x.reshape(batch * seq_len, -1).contiguous()
    for config in _candidate_configs(dtype_name, input_precision):
        modes[config["name"]] = lambda config=config: _candidate_matmul(
            flat_x,
            w,
            **{key: value for key, value in config.items() if key != "name"},
        ).reshape(batch, seq_len, -1)
    return modes


def run_one(args: argparse.Namespace, *, batch: int, dtype_name: str, input_precision: str) -> dict[str, Any]:
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[dtype_name]
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

    modes = _build_modes(
        x,
        w,
        batch=batch,
        seq_len=args.seq_len,
        dtype_name=dtype_name,
        input_precision=input_precision,
    )
    single_modes = _build_modes(
        single_x,
        w,
        batch=1,
        seq_len=args.seq_len,
        dtype_name=dtype_name,
        input_precision=input_precision,
    )

    results: dict[str, Any] = {}
    for name, fn in modes.items():
        try:
            logits = fn()
            sampled = _sampled_logprobs(logits, tokens)
            single_mode_logits = single_modes[name]()
            single_mode_logprobs = _sampled_logprobs(single_mode_logits, single_tokens)
            matmul_ms = _time_cuda(fn, iters=args.iters, warmup=args.warmup, device=device)
            results[name] = {
                "status": "ok",
                "matmul_ms": matmul_ms,
                "logit_invariance_vs_mode_single": _compare(single_mode_logits, logits[0:1]),
                "sampled_logprob_invariance_vs_mode_single": _compare(single_mode_logprobs, sampled[0:1]),
                "logit_drift_vs_torch_single": _compare(torch_single_logits.unsqueeze(0), logits[0:1]),
                "sampled_logprob_drift_vs_torch_single": _compare(torch_single_logprobs, sampled[0:1]),
            }
        except Exception as exc:
            results[name] = {"status": "error", "error": repr(exc)}
        torch.cuda.empty_cache()

    base_ms = results.get("torch_batched_3d", {}).get("matmul_ms")
    for payload in results.values():
        if payload.get("status") == "ok" and base_ms:
            payload["matmul_overhead_vs_torch_batched"] = payload["matmul_ms"] / base_ms

    return {
        "config": {
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "dtype": dtype_name,
            "input_precision": input_precision,
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
        },
        "modes": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune candidate canonical persistent GEMM tile configs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtypes", nargs="+", default=["float32"])
    parser.add_argument("--batches", nargs="+", type=int, default=[8])
    parser.add_argument("--input-precisions", nargs="+", default=["ieee", "tf32"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for candidate GEMM tuning.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dtype_name in args.dtypes:
        for input_precision in args.input_precisions:
            for batch in args.batches:
                record = run_one(args, batch=batch, dtype_name=dtype_name, input_precision=input_precision)
                out = args.output_dir / (
                    f"candidate_gemm_dtype-{dtype_name}_batch-{batch}_seq-{args.seq_len}_"
                    f"hidden-{args.hidden}_vocab-{args.vocab}_det-{int(args.deterministic)}_"
                    f"tf32-{int(args.allow_tf32)}_input-{input_precision}_seed-{args.seed}.json"
                )
                out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                records.append(str(out))
                print(out)
    (args.output_dir / "manifest.json").write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
