from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SGLANG_SRC = PROJECT_ROOT / ".data" / "src" / "sglang" / "python"


@dataclass(frozen=True)
class ProjectionDecision:
    mode: str
    reason: str


def _append_sglang_src() -> None:
    if SGLANG_SRC.exists() and str(SGLANG_SRC) not in sys.path:
        sys.path.insert(0, str(SGLANG_SRC))


def _load_batch_inv_matmul() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    _append_sglang_src()
    module = importlib.import_module("sglang.srt.batch_invariant_ops.batch_invariant_ops")
    return module.matmul_persistent


def _torch_projection(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w


def _batch_inv_persistent_projection(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    matmul = _load_batch_inv_matmul()
    flat = x.reshape(batch * seq_len, -1).contiguous()
    return matmul(flat, w).reshape(batch, seq_len, -1)


def _canonical_per_sample_projection(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[index] @ w for index in range(x.size(0))], dim=0)


def select_projection_mode(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    require_torch_single_drift_free: bool = False,
) -> ProjectionDecision:
    """Select the fastest measured batch-invariant projection mode.

    This is intentionally conservative: the rules encode only contracts already
    certified by the local benchmark artifact for shape family
    [*, 512, 1024] x [1024, 8192] on RTX 3090 / PyTorch 2.11 CUDA 13.0.
    Broader production use should load a generated certificate keyed by shape,
    dtype, backend, and GPU architecture.
    """

    if require_torch_single_drift_free:
        return ProjectionDecision(
            mode="canonical_per_sample_loop",
            reason="strict torch-single 0-drift requested",
        )

    if x.dtype in (torch.float16, torch.bfloat16):
        return ProjectionDecision(
            mode="torch_batched_3d",
            reason="torch path is certified batch-invariant for measured FP16/BF16 contracts",
        )

    if x.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        return ProjectionDecision(
            mode="torch_batched_3d",
            reason="torch path is certified batch-invariant for measured FP32+TF32 contract",
        )

    if x.dtype == torch.float32:
        return ProjectionDecision(
            mode="batch_inv_persistent",
            reason="torch FP32/no-TF32 is not batch-invariant; persistent path is certified",
        )

    return ProjectionDecision(
        mode="canonical_per_sample_loop",
        reason=f"uncertified dtype {x.dtype}; use strict fallback",
    )


def certified_top_projection(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    require_torch_single_drift_free: bool = False,
) -> torch.Tensor:
    decision = select_projection_mode(
        x,
        w,
        require_torch_single_drift_free=require_torch_single_drift_free,
    )
    if decision.mode == "torch_batched_3d":
        return _torch_projection(x, w)
    if decision.mode == "batch_inv_persistent":
        return _batch_inv_persistent_projection(x, w)
    if decision.mode == "canonical_per_sample_loop":
        return _canonical_per_sample_projection(x, w)
    raise RuntimeError(f"Unsupported projection decision: {decision}")
