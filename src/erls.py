# src/erls.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.buffers import behavioural_base_from_V
from src.experiments import find_min_buffers_for_target_shortfall


@dataclass
class ERLSResult:
    alpha_pre: float
    alpha_post: float
    B_pre: float
    B_post: float
    erls: float
    # diagnostics
    R_pre: float
    R_post: float


def xi_from_uniform_base(
    V_ref: np.ndarray,
    U: np.ndarray,
    lam: float,
    *,
    scale: Literal["row_sum", "col_sum", "total"] = "row_sum",
) -> np.ndarray:
    """
    Deterministic scaling:
        xi(lam) = lam * s(V_ref) * U

    holding U fixed across lam (common random numbers).

    U must be N x 1 with entries in [0, 1].
    """
    V_ref = np.asarray(V_ref, dtype=float)
    U = np.asarray(U, dtype=float).reshape(-1, 1)
    N = V_ref.shape[0]
    if U.shape != (N, 1):
        raise ValueError(f"U must have shape {(N,1)}, got {U.shape}")

    if scale == "row_sum":
        s = V_ref.sum(axis=1).reshape(-1, 1)
    elif scale == "col_sum":
        s = V_ref.sum(axis=0).reshape(-1, 1)
    elif scale == "total":
        s = np.full((N, 1), V_ref.sum() / N, dtype=float)
    else:
        raise ValueError("scale must be one of {'row_sum','col_sum','total'}")

    return float(lam) * s * U


def compute_erls_zero_shortfall(
    V: np.ndarray,
    V_tilde: np.ndarray,
    U_base: np.ndarray,
    lam: float,
    *,
    xi_scale: Literal["row_sum", "col_sum", "total"] = "row_sum",
    buffer_mode: Literal["fixed_shape", "behavioural"] = "behavioural",
    rel_tol_fpa: float = 1e-8,
) -> ERLSResult:
    """
    Compute ERLS for a single draw (V, U_base), at intensity lam, targeting zero shortfall.

    - Construct xi using V as the reference scale (fixed £ shock across compression).
    - Solve for minimal buffers (via scalar search) such that aggregate shortfall is 0:
        pre:  b_pre*
        post: b_post*
    - Define total buffers:
        B_pre  = sum_i b_pre*_i
        B_post = sum_i b_post*_i
    - Define ERLS:
        ERLS = 1 - (B_post / B_pre)  (if B_pre > 0, else 0 by convention)

    buffer_mode:
      - "fixed_shape": post compression uses the same base buffer *shape* as pre (Exp 2).
      - "behavioural": buffer shape is recomputed under compression (Exp 3).
    """
    V = np.asarray(V, dtype=float)
    V_tilde = np.asarray(V_tilde, dtype=float)

    # 1) Build xi from fixed base U and reference V
    xi = xi_from_uniform_base(V_ref=V, U=U_base, lam=lam, scale=xi_scale)

    # 2) Choose buffer shapes
    b_base_pre = behavioural_base_from_V(V)

    if buffer_mode == "fixed_shape":
        # Exp 2: same shape pre and post
        alpha_pre, b_pre, fpa_pre = find_min_buffers_for_target_shortfall(
            V=V,
            xi=xi,
            target_shortfall=0.0,
            b_base=b_base_pre,
            rel_tol_fpa=rel_tol_fpa,
        )
        alpha_post, b_post, fpa_post = find_min_buffers_for_target_shortfall(
            V=V_tilde,
            xi=xi,
            target_shortfall=0.0,
            b_base=b_base_pre,
            rel_tol_fpa=rel_tol_fpa,
        )

    elif buffer_mode == "behavioural":
        # Exp 3: shape recomputed under compression
        alpha_pre, b_pre, fpa_pre = find_min_buffers_for_target_shortfall(
            V=V,
            xi=xi,
            target_shortfall=0.0,
            b_base=b_base_pre,
            rel_tol_fpa=rel_tol_fpa,
        )
        alpha_post, b_post, fpa_post = find_min_buffers_for_target_shortfall(
            V=V_tilde,
            xi=xi,
            target_shortfall=0.0,
            buffer_shape_fn=behavioural_base_from_V,
            rel_tol_fpa=rel_tol_fpa,
        )
    else:
        raise ValueError("buffer_mode must be one of {'fixed_shape','behavioural'}")

    # 3) ERLS based on total buffers (comparable even when shapes differ)
    B_pre = float(b_pre.sum())
    B_post = float(b_post.sum())

    if B_pre <= 0.0:
        erls = 0.0
    else:
        erls = 1.0 - (B_post / B_pre)

    return ERLSResult(
        alpha_pre=float(alpha_pre),
        alpha_post=float(alpha_post),
        B_pre=float(B_pre),
        B_post=float(B_post),
        erls=float(erls),
        R_pre=float(fpa_pre.aggregate_shortfall),
        R_post=float(fpa_post.aggregate_shortfall),
    )
