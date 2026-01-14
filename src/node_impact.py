# src/node_impact.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.simulation import run_fpa


KnockoutMode = Literal["set_zero", "set_eps"]


@dataclass(frozen=True)
class NodeImpactResult:
    R_base: float
    delta: np.ndarray        # shape (N,)
    R_knockout: np.ndarray   # shape (N,)
    knockout_mode: str
    eps: float
    # NEW: baseline vulnerability diagnostics
    failed_base: np.ndarray        # shape (N,), bool
    residual_base: np.ndarray      # shape (N,), float


def _as_col(x: np.ndarray, N: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 2 and x.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Expected (N,), (N,1), or (1,N); got shape {x.shape}")

    if N is not None and x.shape != (N, 1):
        raise ValueError(f"Expected shape ({N},1); got {x.shape}")
    return x


def _fpa_from_bhat(V: np.ndarray, b_hat: np.ndarray, *, rel_tol_fpa: float):
    return run_fpa(V, b_hat, rel_tol=rel_tol_fpa, return_paths=False)


def _risk_from_bhat(V: np.ndarray, b_hat: np.ndarray, *, rel_tol_fpa: float) -> float:
    res = run_fpa(V, b_hat, rel_tol=rel_tol_fpa, return_paths=False)
    return float(res.aggregate_shortfall)


def compute_node_impacts(
    V: np.ndarray,
    b_hat: np.ndarray,
    *,
    knockout: KnockoutMode = "set_zero",
    eps: float = 1e-12,
    rel_tol_fpa: float = 1e-8,
    clip_negative_bhat: bool = True,
) -> NodeImpactResult:
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError(f"V must be square (N,N); got shape {V.shape}")
    N = V.shape[0]

    b_hat_col = _as_col(b_hat, N=N)
    if clip_negative_bhat:
        b_hat_col = np.maximum(b_hat_col, 0.0)

    # --- baseline FPA once: risk + vulnerability ---
    base_fpa = _fpa_from_bhat(V, b_hat_col, rel_tol_fpa=rel_tol_fpa)
    R_base = float(base_fpa.aggregate_shortfall)

    residual_base = np.asarray(base_fpa.residual_obligations[:, 0], dtype=float)
    failed_base = residual_base > 0.0

    # --- knockout runs ---
    R_knock = np.empty(N, dtype=float)
    for i in range(N):
        b2 = b_hat_col.copy()
        if knockout == "set_zero":
            b2[i, 0] = 0.0
        elif knockout == "set_eps":
            b2[i, 0] = float(eps)
        else:
            raise ValueError(f"Unknown knockout mode: {knockout}")

        R_knock[i] = _risk_from_bhat(V, b2, rel_tol_fpa=rel_tol_fpa)

    delta = np.maximum(R_knock - R_base, 0.0)

    return NodeImpactResult(
        R_base=R_base,
        delta=delta,
        R_knockout=R_knock,
        knockout_mode=str(knockout),
        eps=float(eps),
        failed_base=failed_base,
        residual_base=residual_base,
    )


def node_impact_under_xi(
    V: np.ndarray,
    b: np.ndarray,
    xi: np.ndarray,
    **kwargs,
):
    b_hat = np.maximum(b - xi, 0.0)
    return compute_node_impacts(V, b_hat, **kwargs)
